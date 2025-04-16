import os
import random
import string
import sys
import time  # For retry delays

import matplotlib.font_manager as fm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import ImageSequenceClip

# --- Mistral AI Integration ---
try:
    from mistralai import UserMessage, SystemMessage, Mistral

    MISTRAL_AVAILABLE = True
except ImportError:
    print("Warning: Mistral AI library not found. AI text generation disabled.")
    print("Install it using: pip install mistralai")
    MISTRAL_AVAILABLE = False

# --- Configuration Parameters ---

# Video settings
WIDTH = 1024
HEIGHT = 1024
FPS = 10
DURATION_SECONDS = 5
OUTPUT_FILENAME = "text_match_cut_video_v2.mp4"

# Text & Highlighting settings
HIGHLIGHTED_TEXT = "Mother of Dragons"
HIGHLIGHT_COLOR = "yellow"  # Pillow color name or hex code
TEXT_COLOR = "black"
BACKGROUND_COLOR = "white"
FONT_SIZE_RATIO = 0.05  # Adjusted slightly for multi-line potentially
MIN_LINES = 7  # Min number of text lines per frame
MAX_LINES = 10  # Max number of text lines per frame
VERTICAL_SPREAD_FACTOR = 1.5  # Multiplier for line height (1.0 = tight, 1.5 = looser)

# AI Text Generation Settings
AI_GENERATION_ENABLED = MISTRAL_AVAILABLE  # Auto-disable if library missing
UNIQUE_TEXT_COUNT = 2  # Number of unique text snippets to generate/pre-pool
MISTRAL_MODEL = "mistral-large-latest"  # Or choose another suitable model
# !! IMPORTANT: Load API Key securely !!
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Effect settings
BLUR_TYPE = 'radial'  # Options: 'gaussian', 'radial'
BLUR_RADIUS = 4.0  # Gaussian blur radius, or the radius OUTSIDE which radial blur starts fading strongly
RADIAL_SHARPNESS_RADIUS_FACTOR = 0.3  # For 'radial': Percentage of min(W,H) to keep perfectly sharp around center

# Font settings
FONT_DIR = "fonts"  # Dedicated font folder recommended
MAX_FONT_RETRIES_PER_FRAME = 5
# Generate random words only using ASCII lowercase for fallback/disabled AI
FALLBACK_CHAR_SET = string.ascii_lowercase + " "


# --- Helper Functions ---

def get_random_font(font_paths, exclude_list=None):
    """Selects a random font file path from the list, avoiding excluded ones."""
    available_fonts = list(set(font_paths) - set(exclude_list or []))
    if not available_fonts:
        # Fallback if all fonts failed or list is empty initially
        try:
            prop = fm.FontProperties(family='sans-serif')
            fallback = fm.findfont(prop, fallback_to_default=True)
            print(f"Warning: No usable fonts found from list/system. Using fallback: {fallback}")
            return fallback
        except Exception:
            print("ERROR: No fonts found and fallback failed. Cannot proceed.")
            return None
    return random.choice(available_fonts)


# Fallback random text generator
def generate_random_words(num_words):
    """Generates a string of random 'words' using only FALLBACK_CHAR_SET."""
    words = []
    for _ in range(num_words):
        length = random.randint(3, 8)
        word = ''.join(random.choice(FALLBACK_CHAR_SET.replace(" ", "")) for i in range(length))
        words.append(word)
    return " ".join(words)


def generate_random_text_snippet(highlighted_text, min_lines, max_lines):
    """Generates multiple lines of random text, ensuring MIN_LINES."""
    # Ensure we generate at least min_lines
    num_lines = random.randint(max(1, min_lines), max(min_lines, max_lines))  # Ensure at least min_lines generated
    highlight_line_index = random.randint(0, num_lines - 1)
    lines = []
    min_words_around = 2
    max_words_around = 6
    for i in range(num_lines):
        if i == highlight_line_index:
            words_before = generate_random_words(random.randint(min_words_around, max_words_around))
            words_after = generate_random_words(random.randint(min_words_around, max_words_around))
            lines.append(f"{words_before} {highlighted_text} {words_after}")
        else:
            lines.append(generate_random_words(random.randint(max_words_around, max_words_around * 2)))

    # Double-check final line count (should always pass with the adjusted randint)
    if len(lines) < min_lines:
        print(f"Warning: Random generator created only {len(lines)} lines (min: {min_lines}). This shouldn't happen.")
        return None, -1  # Treat as failure if check fails unexpectedly

    return lines, highlight_line_index


# Mistral AI Text Generation Function
def generate_ai_text_snippet(client, model, highlighted_text, min_lines, max_lines):
    """Generates a text snippet using Mistral AI containing the highlighted text."""
    target_lines = random.randint(min_lines, max_lines)
    prompt = (
        f"Generate a text block of approximately {target_lines} distinct lines (aim for at least {min_lines}). "
        f"One of the lines MUST contain the exact phrase: '{highlighted_text}'. "
        f"The surrounding text should be thematically related to '{highlighted_text}' (e.g., fantasy, power, dragons, leadership). "
        f"Ensure the phrase '{highlighted_text}' fits naturally within its line. "
        f"Format the output ONLY as the text lines, each separated by a single newline character. Do not add any extra explanations or formatting."
        # f"Example line containing the phrase: '...they bowed before the {highlighted_text}, their new queen...'"
    )

    try:
        messages = [UserMessage(content=prompt)]
        chat_response = client.chat.complete(model=model, messages=messages,
                                             temperature=0.5, max_tokens=300)
        content = chat_response.choices[0].message.content.strip()

        # Basic cleanup: remove potential empty lines
        lines = [line for line in content.split('\n') if line.strip()]

        # --- CRITICAL CHECK: Ensure minimum lines ---
        if len(lines) < min_lines:
            print(
                f"Warning: AI returned only {len(lines)} valid lines (minimum requested: {min_lines}). Retrying generation.")
            return None, -1  # Indicate failure due to insufficient lines

        # Find the highlight line
        highlight_line_index = -1
        for i, line in enumerate(lines):
            if highlighted_text in line:
                highlight_line_index = i
                break

        if highlight_line_index == -1:
            print(f"Warning: AI response did not contain the exact phrase '{highlighted_text}'.")
            # Optionally try to insert it into a random line? Or just fail.
            # Let's fail for now to ensure the highlight is always from AI context
            return None, -1  # Indicate failure

        return lines, highlight_line_index
    except Exception as e:
        print(f"An unexpected error occurred during AI text generation: {e}")
        return None, -1  # Indicate failure


def create_radial_blur_mask(width, height, center_x, center_y, sharp_radius, fade_radius):
    """Creates a grayscale mask for radial blur (sharp center, fades out)."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (center_x - sharp_radius, center_y - sharp_radius,
         center_x + sharp_radius, center_y + sharp_radius),
        fill=255
    )
    # Gaussian blur the sharp circle mask for a smooth falloff
    # Ensure fade radius is larger than sharp radius
    blur_amount = max(0.1, (fade_radius - sharp_radius) / 3.5)  # Adjusted divisor for smoothness
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_amount))
    return mask


def create_text_image_frame(width, height, text_lines, highlight_line_index, highlighted_text,
                            font_path, font_size, text_color, bg_color, highlight_color,
                            blur_type, blur_radius, radial_sharp_radius_factor, vertical_spread_factor):
    """Creates a single frame image with centered highlight and multi-line text."""

    # --- Font Loading ---
    try:
        font = ImageFont.truetype(font_path, font_size)
        bold_font = font  # Start with regular as fallback
        # Simple bold variant check (can be improved)
        common_bold_suffixes = ["bd.ttf", "-Bold.ttf", "b.ttf", "_Bold.ttf", " Bold.ttf"]
        base_name, ext = os.path.splitext(font_path)
        for suffix in common_bold_suffixes:
            potential_bold_path = base_name.replace("Regular", "").replace("regular",
                                                                           "") + suffix  # Try removing 'Regular' too
            if os.path.exists(potential_bold_path):
                try:
                    bold_font = ImageFont.truetype(potential_bold_path, font_size)
                    # print(f"    Using bold variant: {os.path.basename(potential_bold_path)}") # Debug
                    break  # Use the first one found
                except IOError:
                    continue  # Try next suffix if loading fails
            # Check without removing Regular if first checks failed
            potential_bold_path = base_name + suffix
            if os.path.exists(potential_bold_path):
                try:
                    bold_font = ImageFont.truetype(potential_bold_path, font_size)
                    # print(f"    Using bold variant: {os.path.basename(potential_bold_path)}") # Debug
                    break
                except IOError:
                    continue

    except IOError as e:
        raise FontLoadError(f"Failed to load font: {font_path}") from e
    except Exception as e:  # Catch other potential font loading issues
        raise FontLoadError(f"Unexpected error loading font {font_path}: {e}") from e

    # --- Calculations ---
    try:
        # Line height using getmetrics()
        try:
             ascent, descent = font.getmetrics()
             metric_height = ascent + abs(descent)
             line_height = int(metric_height * vertical_spread_factor)
        except AttributeError:
             bbox_line_test = font.getbbox("Ay", anchor="lt")
             line_height = int((bbox_line_test[3] - bbox_line_test[1]) * vertical_spread_factor)
        if line_height <= font_size * 0.8:
            line_height = int(font_size * 1.2 * vertical_spread_factor)

        # BOLD font metrics for final highlight placement
        highlight_width_bold = bold_font.getlength(highlighted_text)
        highlight_bbox_h = bold_font.getbbox(highlighted_text, anchor="lt")
        highlight_height_bold = highlight_bbox_h[3] - highlight_bbox_h[1]
        if highlight_width_bold <= 0 or highlight_height_bold <= 0:
             highlight_height_bold = int(font_size * 1.1)
             if highlight_width_bold <=0: highlight_width_bold = len(highlighted_text) * font_size * 0.6

        # Target position for the TOP-LEFT of the final BOLD highlight text (CENTERED)
        highlight_target_x = (width - highlight_width_bold) / 2
        highlight_target_y = (height - highlight_height_bold) / 2

        # Block start Y calculated relative to the centered highlight's top
        block_start_y = highlight_target_y - (highlight_line_index * line_height)

        # Get Prefix and Suffix for background alignment
        highlight_line_full_text = text_lines[highlight_line_index]
        prefix_text = ""
        suffix_text = "" # Also get suffix now
        highlight_found_in_line = False
        try:
            start_index = highlight_line_full_text.index(highlighted_text)
            end_index = start_index + len(highlighted_text)
            prefix_text = highlight_line_full_text[:start_index]
            suffix_text = highlight_line_full_text[end_index:]
            highlight_found_in_line = True
        except ValueError: pass # Treat line normally if not found

        # Measure Prefix Width using REGULAR font (for background positioning)
        prefix_width_regular = font.getlength(prefix_text)
        # Calculate the required starting X for the background highlight line string
        # This is the coordinate used for drawing the *full string* in the background
        bg_highlight_line_start_x = highlight_target_x - prefix_width_regular

    except AttributeError: raise FontDrawError(f"Font lacks methods.")
    except Exception as e: raise FontDrawError(f"Measurement fail: {e}") from e

    # --- Base Image Drawing (Draw FULL lines, use offset for HL line) ---
    # Render onto img_base normally first
    img_base = Image.new('RGB', (width, height), color=bg_color)
    draw_base = ImageDraw.Draw(img_base)
    try:
        current_y = block_start_y
        for i, line in enumerate(text_lines):
            line_x = 0.0
            if i == highlight_line_index and highlight_found_in_line:
                line_x = bg_highlight_line_start_x
            else:
                line_width = font.getlength(line)
                line_x = (width - line_width) / 2
            draw_base.text((line_x, current_y), line, font=font, fill=text_color, anchor="lt")
            current_y += line_height
    except Exception as e: raise FontDrawError(f"Base draw fail: {e}") from e

    # --- Apply Blur (with padding for Gaussian to avoid edge clipping) ---
    img_blurred = None # Initialize
    padding_for_blur = int(blur_radius * 3) # Padding based on blur radius

    if blur_type == 'gaussian' and blur_radius > 0:
        try:
            # Create larger canvas
            padded_width = width + 2 * padding_for_blur
            padded_height = height + 2 * padding_for_blur
            img_padded = Image.new('RGB', (padded_width, padded_height), color=bg_color)
            # Paste original centered onto padded canvas
            img_padded.paste(img_base, (padding_for_blur, padding_for_blur))
            # Blur the padded image
            img_padded_blurred = img_padded.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            # Crop the center back to original size
            img_blurred = img_padded_blurred.crop((padding_for_blur, padding_for_blur,
                                                  padding_for_blur + width, padding_for_blur + height))
        except Exception as e:
            print(f"Error during padded Gaussian blur: {e}. Falling back to direct blur.")
            img_blurred = img_base.filter(ImageFilter.GaussianBlur(radius=blur_radius)) # Fallback


    elif blur_type == 'radial' and blur_radius > 0:
        # For radial, we need img_sharp. Let's try drawing it *in parts* for reliability
        # as the padded blur trick doesn't apply directly here.
        img_sharp = Image.new('RGB', (width, height), color=bg_color)
        draw_sharp = ImageDraw.Draw(img_sharp)
        try:
            current_y = block_start_y
            for i, line in enumerate(text_lines):
                if i == highlight_line_index and highlight_found_in_line:
                    # --- Draw Sharp Highlight Line in Parts ---
                    # Calculate positions relative to the *final* centered highlight target
                    prefix_x = highlight_target_x - prefix_width_regular
                    # Use REGULAR font for the sharp layer (it's just for the mask)
                    draw_sharp.text((prefix_x, current_y), prefix_text, font=font, fill=text_color, anchor="lt")
                    # Highlight part itself starts at highlight_target_x
                    highlight_width_regular = font.getlength(highlighted_text) # Width in regular font
                    draw_sharp.text((highlight_target_x, current_y), highlighted_text, font=font, fill=text_color, anchor="lt")
                    # Suffix starts after the regular highlight width
                    suffix_x = highlight_target_x + highlight_width_regular
                    draw_sharp.text((suffix_x, current_y), suffix_text, font=font, fill=text_color, anchor="lt")
                else:
                    # Draw non-highlight lines centered normally
                    line_width = font.getlength(line)
                    line_x = (width - line_width) / 2
                    draw_sharp.text((line_x, current_y), line, font=font, fill=text_color, anchor="lt")
                current_y += line_height
        except Exception as e:
             raise FontDrawError(f"Failed sharp text draw (parts): {e}") from e

        # Composite blurred base and sharp center
        # Base image (img_base) still uses the offset drawing method for full line
        img_fully_blurred = img_base.filter(ImageFilter.GaussianBlur(radius=blur_radius * 1.5))
        sharp_center_radius = min(width, height) * radial_sharp_radius_factor
        fade_radius = sharp_center_radius + max(width, height) * 0.15
        mask = create_radial_blur_mask(width, height, width / 2, height / 2, sharp_center_radius, fade_radius)
        img_blurred = Image.composite(img_sharp, img_fully_blurred, mask)

    else: # No blur
        img_blurred = img_base.copy()


    # --- Final Image: Draw ONLY Highlight Rectangle & Centered BOLD Text ---
    final_img = img_blurred # Start with the blurred/composited image
    draw_final = ImageDraw.Draw(final_img)
    try:
        # 1. Draw highlight rectangle (centered using bold metrics)
        padding = font_size * 0.10
        draw_final.rectangle(
            [
                (highlight_target_x - padding, highlight_target_y - padding),
                (highlight_target_x + highlight_width_bold + padding, highlight_target_y + highlight_height_bold + padding)
            ],
            fill=highlight_color
        )

        # 2. Draw ONLY the SHARP highlight text using BOLD font at the *perfectly centered* position
        draw_final.text(
            (highlight_target_x, highlight_target_y),
            highlighted_text,
            font=bold_font, # Use BOLD font
            fill=text_color,
            anchor="lt"
        )
        # *** No prefix/suffix drawing here ***

    except Exception as e:
         raise FontDrawError(f"Failed final highlight draw: {e}") from e

    return final_img


# Custom Exceptions for font errors
class FontLoadError(Exception): pass
class FontDrawError(Exception): pass


def main():
    print("Starting video generation...")

    if AI_GENERATION_ENABLED and not MISTRAL_API_KEY:
        print("ERROR: AI Generation is enabled, but MISTRAL_API_KEY environment variable is not set.")
        print("Please set the environment variable or disable AI generation.")
        sys.exit(1)  # Exit cleanly

    # --- Font Discovery ---
    font_paths = []
    # (Font discovery code remains the same)
    if FONT_DIR and os.path.isdir(FONT_DIR):
        print(f"Looking for fonts in specified directory: {FONT_DIR}")
        for filename in os.listdir(FONT_DIR):
            if filename.lower().endswith((".ttf", ".otf")):
                font_paths.append(os.path.join(FONT_DIR, filename))
    else:
        print("FONT_DIR not specified or invalid, searching system fonts...")
        try:
            font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            # Optionally add otf, but ttf often more compatible with basic PIL
            # font_paths.extend(fm.findSystemFonts(fontpaths=None, fontext='otf'))
        except Exception as e:
            print(f"Error finding system fonts: {e}")

    if not font_paths:
        print("ERROR: No fonts found. Please install fonts or specify a valid FONT_DIR.")
        sys.exit(1)
    print(f"Found {len(font_paths)} potential fonts.")

    # --- Pre-generate Text Snippets ---
    text_snippets_pool = []
    print(f"Pre-generating {UNIQUE_TEXT_COUNT} unique text snippets...")

    if AI_GENERATION_ENABLED:
        client = Mistral(api_key=MISTRAL_API_KEY)
        generated_count = 0
        attempts = 0
        max_attempts = UNIQUE_TEXT_COUNT * 3  # Allow for some failures

        while generated_count < UNIQUE_TEXT_COUNT and attempts < max_attempts:
            attempts += 1
            print(f"  Generating snippet {generated_count + 1}/{UNIQUE_TEXT_COUNT} (Attempt {attempts})...")
            lines, hl_index = generate_ai_text_snippet(client, MISTRAL_MODEL, HIGHLIGHTED_TEXT, MIN_LINES, MAX_LINES)
            if lines and hl_index != -1:
                text_snippets_pool.append({"lines": lines, "highlight_index": hl_index})
                generated_count += 1
            else:
                print("    Failed to generate valid snippet, trying again.")
                time.sleep(1)  # Wait a bit before retrying on failure

        if generated_count < UNIQUE_TEXT_COUNT:
            print(
                f"Warning: Only generated {generated_count}/{UNIQUE_TEXT_COUNT} unique AI snippets after {max_attempts} attempts.")
            if generated_count == 0:
                print("ERROR: Failed to generate any AI text snippets. Cannot proceed with AI enabled.")
                sys.exit(1)
    else:
        # Generate random snippets if AI is disabled
        print("AI disabled, generating random text snippets for the pool...")
        for i in range(UNIQUE_TEXT_COUNT):
            lines, hl_index = generate_random_text_snippet(HIGHLIGHTED_TEXT, MIN_LINES, MAX_LINES)
            text_snippets_pool.append({"lines": lines, "highlight_index": hl_index})
        print(f"Generated {len(text_snippets_pool)} random snippets.")

    # --- Calculate Other Parameters ---
    total_frames = int(FPS * DURATION_SECONDS)
    font_size = int(HEIGHT * FONT_SIZE_RATIO)
    print(f"\nVideo Settings: {WIDTH}x{HEIGHT} @ {FPS}fps, {DURATION_SECONDS}s ({total_frames} frames)")
    print(f"Text Settings: Highlight='{HIGHLIGHTED_TEXT}', Size={font_size}px, Spread={VERTICAL_SPREAD_FACTOR}")
    print(f"Effect Settings: BlurType='{BLUR_TYPE}', BlurRadius={BLUR_RADIUS}, HighlightColor='{HIGHLIGHT_COLOR}'")
    print(f"Using {'AI' if AI_GENERATION_ENABLED else 'Random'} Text Pool Size: {len(text_snippets_pool)}")

    # --- Generate Frames ---
    frames = []
    failed_fonts = set()
    print("\nGenerating frames...")
    frame_num = 0
    while frame_num < total_frames:
        print(f"  Attempting Frame {frame_num + 1}/{total_frames}")

        # Select a text snippet from the pool
        snippet = random.choice(text_snippets_pool)
        current_lines = snippet["lines"]
        highlight_idx = snippet["highlight_index"]

        font_retries = 0
        frame_generated = False
        while font_retries < MAX_FONT_RETRIES_PER_FRAME:
            current_font_path = get_random_font(font_paths, exclude_list=failed_fonts)
            if current_font_path is None:
                print("ERROR: Exhausted all available fonts or fallback failed. Stopping.")
                sys.exit(1)  # Exit if no fonts work

            try:
                img = create_text_image_frame(
                    WIDTH, HEIGHT,
                    current_lines, highlight_idx, HIGHLIGHTED_TEXT,
                    current_font_path, font_size,
                    TEXT_COLOR, BACKGROUND_COLOR, HIGHLIGHT_COLOR,
                    BLUR_TYPE, BLUR_RADIUS, RADIAL_SHARPNESS_RADIUS_FACTOR,
                    VERTICAL_SPREAD_FACTOR
                )

                frame_np = np.array(img)
                frames.append(frame_np)
                frame_generated = True
                # print(f"    Frame {frame_num + 1} generated with font: {os.path.basename(current_font_path)}") # Less verbose
                break  # Success, next frame

            except (FontLoadError, FontDrawError) as e:
                # print(f"    Warning: Font '{os.path.basename(current_font_path)}' failed ({e}). Retrying frame.") # Less verbose
                failed_fonts.add(current_font_path)
                font_retries += 1
                # time.sleep(0.05) # Optional small delay
            except Exception as e:
                print(
                    f"    ERROR: Unexpected error generating frame with font {os.path.basename(current_font_path)}: {e}")
                failed_fonts.add(current_font_path)
                font_retries += 1
                # time.sleep(0.05)

        if not frame_generated:
            print(
                f"ERROR: Failed to generate Frame {frame_num + 1} after {MAX_FONT_RETRIES_PER_FRAME} font attempts. Stopping video generation.")
            break  # Stop if a frame repeatedly fails

        frame_num += 1

    # --- Create Video ---
    if not frames:
        print("ERROR: No frames were generated. Cannot create video.")
        sys.exit(1)

    if len(frames) < total_frames:
        print(f"Warning: Only {len(frames)}/{total_frames} frames were generated due to errors. Video will be shorter.")

    print("\nCompiling video...")
    try:
        clip = ImageSequenceClip(frames, fps=FPS)
        # Explicitly use 'libx264' for broad compatibility, 'bar' for progress
        clip.write_videofile(OUTPUT_FILENAME, codec='libx264', fps=FPS, logger='bar')
        print(f"\nVideo saved successfully as '{OUTPUT_FILENAME}'")
    except Exception as e:
        print(f"\nError during video writing: {e}")
        print("Check FFmpeg installation and codec support ('libx264').")

    if failed_fonts:
        print("\nFonts that caused errors (check character support/validity):")
        for ff in sorted(list(failed_fonts)):  # Sort for cleaner output
            print(f" - {os.path.basename(ff)}")

    print("\nScript finished.")


# --- Main Script Logic ---
if __name__ == "__main__":
    if not AI_GENERATION_ENABLED:
        print("NOTE: Mistral AI library not found or AI_GENERATION_ENABLED is False.")
        print("      The script will use RANDOM text generation.")
    elif not MISTRAL_API_KEY:
        print("WARNING: MISTRAL_API_KEY environment variable not found.")
        print("         AI text generation will fail if enabled.")
        # The main function will catch this and exit if AI is enabled.
    main()
