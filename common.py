import io
import logging
import os
import sys
import re
import json
import random
import itertools
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
import torch
import folder_paths


# ----------------------------------------------------------------------------
# Logging. One logger for the pack ("gprompts", children "gprompts.<part>").
# No handler of our own: records go up to ComfyUI's console/log panel handler.
# Level: follows ComfyUI (--verbose) unless debug is switched on, either in
# Settings > Gadzoinks > Debug logging, or with GPROMPTS_LOG=DEBUG (which wins).
# ----------------------------------------------------------------------------
log = logging.getLogger("gprompts")


class _DebugToConsole(logging.Handler):
    """ComfyUI's console handler only shows its own --verbose level (INFO by
    default), so our DEBUG records would never appear. While debug logging is
    on, this prints them. INFO and above still go through ComfyUI's handler,
    so nothing is printed twice; with ComfyUI itself at --verbose DEBUG it
    stays quiet for the same reason."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        try:
            from app.logger import ColoredFormatter        # ComfyUI's [DEBUG] style
            self.setFormatter(ColoredFormatter("%(message)s"))
        except Exception:
            self.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    def emit(self, record):
        if record.levelno >= logging.INFO or logging.getLogger().isEnabledFor(logging.DEBUG):
            return
        try:
            stream = sys.stdout             # ComfyUI swaps stdout to feed its log panel
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)


_DEBUG_HANDLER = _DebugToConsole()


def _show_debug(on):
    if on and _DEBUG_HANDLER not in log.handlers:
        log.addHandler(_DEBUG_HANDLER)
    elif not on and _DEBUG_HANDLER in log.handlers:
        log.removeHandler(_DEBUG_HANDLER)


_ENV_LEVEL = logging.getLevelName(os.environ.get("GPROMPTS_LOG", "").strip().upper() or "NOTSET")
if isinstance(_ENV_LEVEL, int) and _ENV_LEVEL != logging.NOTSET:
    log.setLevel(_ENV_LEVEL)
    _show_debug(_ENV_LEVEL <= logging.DEBUG)
else:
    _ENV_LEVEL = None


def get_logger(part):
    return logging.getLogger(f"gprompts.{part}")


def set_debug(on):
    """Debug logging on/off from the Gadzoinks setting (GPROMPTS_LOG wins)."""
    if _ENV_LEVEL is not None:
        return
    new = logging.DEBUG if on else logging.NOTSET
    if log.level != new:
        log.setLevel(new)
        _show_debug(bool(on))
        log.info("Gadzoinks: debug logging %s", "on" if on else "off")


def dprint(a):
    """Debug detail: shown only with debug logging on."""
    log.debug("%s", a)

# (♩ ♪ ♫ ♬, U+2669–266C)

def pil_to_comfy(img) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)      # [1, H, W, C]


_NOTE_FONT = os.path.join(os.path.dirname(__file__), "note.ttf")
_MUSIC_PNG = os.path.join(os.path.dirname(__file__), "music.png")

def fallback_cover(size=1080):
    img = PILImage.open(_MUSIC_PNG).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), PILImage.LANCZOS)
    return img

def XXfallback_cover(size=1080):
    img = PILImage.new("RGB", (size, size), (24, 26, 32))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(_NOTE_FONT, int(size * 0.55))
    d.text((size / 2, size / 2), "\u266B", font=font, anchor="mm",
           fill=(235, 235, 240))
    return img

def fallback_cover_os(size=1080):
    """Dark card with a music note when no image is provided."""
    img = PILImage.new("RGB", (size, size), (24, 26, 32))
    d = ImageDraw.Draw(img)
    font = None
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",   # macOS
              "C:/Windows/Fonts/seguisym.ttf"):                          # Windows
        if os.path.exists(p):
            font = ImageFont.truetype(p, int(size * 0.5))
            break
    note = "\u266B"  # ♫
    if font:
        bbox = d.textbbox((0, 0), note, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((size - w) / 2 - bbox[0], (size - h) / 2 - bbox[1]),
               note, font=font, fill=(235, 235, 240))
    else:  # no usable font anywhere: simple glyph from shapes
        d.ellipse([size*0.30, size*0.62, size*0.46, size*0.74], fill=(235,235,240))
        d.rectangle([size*0.44, size*0.25, size*0.465, size*0.68], fill=(235,235,240))
    return img
# cover = _cover_from_tensor(image) if image is not None else image_from_character("🎶")

def cover_from_tensor(image):
    """ComfyUI IMAGE tensor [B,H,W,C] float 0-1 -> PIL"""
    return PILImage.fromarray(
        (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))


#####
# DynamicPromptEngine
# The dynamic prompt expansion logic shared by GPrompts ("Dynamic Prompts")
# and GPromptsEnhanced ("Dynamic Prompts with Enhancer").
# Moved here unchanged from GPrompts; the only addition is that each
# expansion records the value chosen for every block (last_values), which
# the enhancer node needs to substitute later expansions into LLM output.
#
class DynamicPromptEngine:
    # Delimiter styles for dynamic blocks.
    #
    # "curly { }"  - the original syntax: {a|b} random, {{a|b}} sequential.
    #   The random pattern is JSON-safe: block content may not contain
    #   braces or double quotes, so structural JSON like {"key": ...} or {}
    #   can never match (every non-empty JSON object has a '"' immediately
    #   inside its braces). Literal '{{' / '}}' adjacency never occurs in
    #   the *structure* of valid JSON, so the sequential pattern is safe too.
    #
    # "angle < >"  - alternate syntax for JSON prompts: <a|b> random,
    #   <<a|b>> sequential. Avoids '{' entirely. require_pipe is True so
    #   that incidental comparisons in prose ("3 < 5 and x > 2") are never
    #   treated as a block: content must contain '|' or be a __wildcard__.
    DELIM_STYLES = {
        "curly { }": {
            "open": "{",
            "seq_token": "{{",
            "seq_re": re.compile(r'\{\{(.*?)\}\}', re.DOTALL),
            "rand_re": re.compile(r'(?<!\{)\{([^{}"]+?)\}'),
            "require_pipe": False,  # legacy behavior: {word} still expands to word
        },
        "angle < >": {
            "open": "<",
            "seq_token": "<<",
            "seq_re": re.compile(r'<<(.*?)>>', re.DOTALL),
            "rand_re": re.compile(r'(?<!<)<([^<>]+?)>'),
            "require_pipe": True,
        },
    }
    DEFAULT_DELIM_STYLE = "curly { }"

    # ------------------------------------------------------------------
    # Register support (named substitution): a computed value can be
    # stored in one of ten registers (0-9) and reused later in the text.
    #   Definition: {{0 monkey|chicken|dog}}  - normal block, but the chosen
    #               value is stored in register 0 (digit + whitespace + options)
    #   Recall:     {{0}}                     - substitutes register 0's value
    # Works with all four block types. Sequential ({{ / <<) and random
    # ({ / <) blocks have SEPARATE register banks, so <<0 and <0 are
    # different registers. Banks reset for every generated prompt.
    # Note: {{0|1|2}} is a plain options block (no whitespace after the
    # digit), NOT a register definition.
    # ------------------------------------------------------------------
    REGISTER_RECALL_RE = re.compile(r'^\s*(\d)\s*$')
    REGISTER_DEF_RE = re.compile(r'^\s*(\d)\s+(\S.*)$', re.DOTALL)

    @classmethod
    def parse_register_block(cls, content):
        """Classify block content.
        Returns (kind, register, body):
          ('recall', '0', None)    for digit-only content
          ('def',    '0', body)    for digit + whitespace + options
          ('plain',  None, content) for everything else
        """
        m = cls.REGISTER_RECALL_RE.match(content)
        if m:
            return ('recall', m.group(1), None)
        m = cls.REGISTER_DEF_RE.match(content)
        if m:
            return ('def', m.group(1), m.group(2))
        return ('plain', None, content)

    def __init__(self):
        self.previous_text = None
        self.previous_delim_style = None
        self.current_iteration = 0
        self.wildcard_cache = {}
        self.sequential_combinations = []
        # Parallel to sequential_combinations: the value chosen for each
        # (non-recall) sequential block in that combination.
        self.sequential_values = []
        # Values chosen by the most recent parse_dynamic_prompt() call:
        # sequential blocks first (in text order), then random blocks
        # (in the order they were replaced).
        self.last_values = []
        self._random_values = []
        self.delims = self.DELIM_STYLES[self.DEFAULT_DELIM_STYLE]

        # Set up folder paths
        self.register_wildcard_path()

    @staticmethod
    def _split_options(content):
        """
        Split block content on unescaped top-level '|'.

        Rules:
        - single quotes are ordinary apostrophes
        - backslash only escapes the next '|' or '\\'
        - any other backslash is literal
        """
        parts = []
        current = []

        i = 0
        n = len(content)

        while i < n:
            ch = content[i]

            # Handle escape sequences:
            # \|  -> literal |
            # \\  -> literal \
            if ch == "\\":
                if i + 1 < n and content[i + 1] in ("|", "\\"):
                    current.append(content[i + 1])
                    i += 2
                    continue
                else:
                    current.append(ch)
                    i += 1
                    continue

            if ch == "|":
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)

            i += 1

        if parts:
            parts.append("".join(current))
            return [part.strip() for part in parts]

        # No top-level pipe found
        return None

    def register_wildcard_path(self):
        """Register wildcards folder path in ComfyUI's folder system if it doesn't exist already"""
        if "wildcards" not in folder_paths.folder_names_and_paths:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            folder_paths.folder_names_and_paths["wildcards"] = ([os.path.join(base_dir, "wildcards")], {".txt", ".json"})

    def parse_dynamic_prompt(self, text):
        d = self.delims
        self.last_values = []
        self._random_values = []
        sequential_values = []
        # First, process all sequential blocks and generate combinations if needed
        if not self.sequential_combinations and d["seq_token"] in text:
            # Initial calculation of all sequential combinations
            self.prepare_sequential_combinations(text)
            dprint(self.sequential_combinations)

        # If we have sequential combinations, use the appropriate one for this iteration
        if self.sequential_combinations and d["seq_token"] in text:
            combination_index = self.current_iteration % len(self.sequential_combinations)
            text = self.sequential_combinations[combination_index]
            if combination_index < len(self.sequential_values):
                sequential_values = list(self.sequential_values[combination_index])

        # Process random blocks (this needs to happen on EVERY iteration)
        if d["open"] in text:
            text = self.process_random_blocks(text)

        self.last_values = sequential_values + self._random_values
        return text

    def prepare_sequential_combinations(self, text):
        d = self.delims
        # Classify every sequential block in order of appearance
        blocks = []  # (kind, register, options-or-None)
        for m in d["seq_re"].finditer(text):
            kind, reg, body = self.parse_register_block(m.group(1))
            if kind == 'recall':
                blocks.append(('recall', reg, None))
                continue
            block_text = body if kind == 'def' else m.group(1)
            # Process wildcards in the block
            if '__' in block_text:
                options = self.resolve_wildcard_references(block_text)
            else:
                # Split by pipe and strip whitespace
                options = self._split_options(block_text)
                if options is None:
                    options = [block_text.strip()]
            blocks.append((kind, reg, options))

        if not blocks:
            return

        option_lists = [b[2] for b in blocks if b[0] != 'recall']

        # Recall-only text (no definitions) still yields one pass-through combo
        all_combinations = list(itertools.product(*option_lists)) if option_lists else [()]

        result_prompts = []
        for combo in all_combinations:
            # Pass 1: assign this combination's value to each non-recall
            # block (in order) and capture register values, so recalls work
            # even if they appear before their definition in the text.
            registers = {}
            values = []      # per-block replacement value; None marks a recall
            combo_iter = iter(combo)
            for kind, reg, _options in blocks:
                if kind == 'recall':
                    values.append(None)
                else:
                    v = next(combo_iter)
                    values.append(v)
                    if reg is not None:
                        registers[reg] = v
            # Pass 2: substitute all blocks in one sweep. Using a function
            # also means backslashes in values (e.g. \" inside JSON strings)
            # are inserted literally.
            value_iter = iter(values)
            def repl(m, _vi=value_iter, _regs=registers):
                v = next(_vi)
                if v is None:  # recall block
                    _kind, reg, _body = self.parse_register_block(m.group(1))
                    # Undefined register: leave the block as-is so the user
                    # can see what went wrong
                    return _regs.get(reg, m.group(0))
                return v
            result_prompts.append(d["seq_re"].sub(repl, text))

        # Store all combinations for future iterations
        self.sequential_combinations = result_prompts
        self.sequential_values = [list(combo) for combo in all_combinations]

    def resolve_wildcard_references(self, text):
        # Find wildcard references like __filename__ or directory__filename__ or nested__dir__file__
        dprint(f"resolve_wildcard_references text:{text}")

        # Look for entire wildcard pattern with the surrounding __
        wildcard_pattern = re.search(r'__(.*?)__', text)
        if wildcard_pattern:
            wildcard_name = wildcard_pattern.group(1)
            dprint(f"resolve_wildcard_references found wildcard name: {wildcard_name}")

            wildcard_options = self.load_wildcard(wildcard_name)
            dprint(f"resolve_wildcard_references wildcard_name:{wildcard_name} wildcard_options:{wildcard_options}")

            if not wildcard_options:
                # No wildcard found, return empty options
                return []

            # Handle weighted options for random selection
            if isinstance(wildcard_options, dict):
                # Create a flat list with repeated items based on weights
                weighted_options = []
                for option, weight in wildcard_options.items():
                    weighted_options.extend([option] * weight)
                return weighted_options
            else:
                return wildcard_options

        # If no wildcard was found, treat as a normal random/sequential block
        return [opt.strip() for opt in text.split('|') if '__' not in opt]

    def load_wildcard(self, wildcard_name):
        """
        Load wildcard file based on path - handles nested paths like attire__headware
        """
        dprint = print

        # Check if we've already loaded this wildcard
        if wildcard_name in self.wildcard_cache:
            return self.wildcard_cache[wildcard_name]

        # For wildcard paths with multiple __ separators (e.g., attire__headware)
        wildcard_path = wildcard_name.replace('__', os.path.sep)
        dprint(f"load_wildcard: wildcard_name:{wildcard_name}")
        dprint(f"wildcard_path:{wildcard_path}")

        # Try to find wildcard using ComfyUI's folder paths
        options = []

        # Try JSON first
        json_path = self.find_wildcard_file(wildcard_path, ".json")
        if json_path:
            dprint(f"Found JSON wildcard at: {json_path}")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list): # ["one","two"]
                    # Handle both simple lists and weighted lists
                    if data and isinstance(data[0], dict): # [ {"one":60},{"two":40} ]
                        # Weighted list format
                        weighted_options = {}
                        for item in data:
                            for option, weight in item.items():
                                weighted_options[option] = weight
                        options = weighted_options
                    else:
                        # Simple list format
                        options = data
                elif isinstance(data, dict): # {"whatever:["one","two"]"}
                    # Handle object format - find the first array value
                    for key, value in data.items():
                        if isinstance(value, list):
                            # Found an array value, use it
                            if value and isinstance(value[0], dict):
                                # Weighted list format
                                weighted_options = {}
                                for item in value:
                                    for option, weight in item.items():
                                        weighted_options[option] = weight
                                options = weighted_options
                            else:
                                # Simple list format
                                options = value
                            break  # Use the first array found
                    else:
                        # No array found in the object
                        log.warning(f"No array value found in JSON object: {json_path}")
                        options = []
            except json.JSONDecodeError:
                log.warning(f"Error parsing JSON file: {json_path}")
        dprint(f"options:{options}")
        # If no JSON or empty result, try TXT
        if not options:
            txt_path = self.find_wildcard_file(wildcard_path, ".txt")
            if txt_path:
                dprint(f"Found TXT wildcard at: {txt_path}")
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # Filter out comments and strip whitespace
                    options = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            options.append(line)
                except Exception as e:
                    log.warning(f"Error reading TXT file: {txt_path}, {str(e)}")

        # Cache the result
        self.wildcard_cache[wildcard_name] = options
        return options

    def process_random_blocks(self, text):
        """
        Process random blocks in the text, handling wildcards correctly.

        Uses the active delimiter style (self.delims). Patterns are JSON-safe:
        - curly: {content} matches only when content contains no braces and no
          double quotes, so JSON structure like {"key": value} or {} is never
          touched. Block content inside a JSON string value works because a
          JSON string cannot contain an unescaped '"'... and escaped quotes
          (\\") would disqualify the block anyway - keep quotes out of options.
        - angle: <content> additionally requires a '|' or a __wildcard__, so
          prose like "3 < 5 and x > 2" is never treated as a block.
        """
        dprint(f"process_random_blocks: {text}")
        d = self.delims
        # Random blocks get their own register bank, separate from the
        # sequential bank ({0 != {{0 and <0 != <<0). Reset per generation.
        registers = {}
        chosen = self._random_values

        def replace_random(match):
            content = match.group(1)
            dprint(f"process_random_blocks content: {content}")
            kind, reg, body = self.parse_register_block(content)

            if kind == 'recall':
                if reg in registers:
                    return registers[reg]
                # Register may be defined later in the text: leave the block
                # untouched so the next pass of the loop can resolve it. If
                # it's never defined, the text stabilizes with the block
                # visible, which surfaces the mistake to the user.
                return match.group(0)

            work = body if kind == 'def' else content
            stripped = work.strip()
            value = None

            # Handle complete wildcard pattern
            if stripped.startswith('__') and stripped.endswith('__') and len(stripped) > 4:
                # Extract the full wildcard path
                wildcard_path = stripped[2:-2]  # Remove the leading and trailing __
                dprint(f"Found full wildcard pattern: {wildcard_path}")

                # Load wildcards directly with the full path
                wildcard_options = self.load_wildcard(wildcard_path)
                dprint(f"Wildcard options: {wildcard_options}")
                if not wildcard_options:
                    value = ""
                elif isinstance(wildcard_options, dict):
                    # Weighted selection - separate keys and weights
                    ks = list(wildcard_options.keys())
                    vs = list(wildcard_options.values())
                    value = random.choices(ks, weights=vs, k=1)[0]
                else:
                    value = random.choice(wildcard_options)
            else:
                # In strict mode (angle delimiters), only pipe-separated
                # content counts as a block; anything else is left untouched.
                # This applies to register definitions too: otherwise prose
                # like "3 < 5 and x > 2" parses as register 5 = "and x".
                # A constant register is still possible: <5 monkey|monkey>.
                if d["require_pipe"] and '|' not in work:
                    return match.group(0)

                # Normal random selection from pipe-separated options
                options = self._split_options(work)
                if options is None:
                    if d["require_pipe"]:
                        return match.group(0)
                if not options:
                    value = ""
                else:
                    value = random.choice(options)

            if kind == 'def':
                registers[reg] = value
            chosen.append(value)
            return value

        # Process all random blocks. The compiled pattern already refuses to
        # match a doubled opening delimiter (sequential blocks), and the loop
        # exits as soon as a pass makes no changes - so leftover structural
        # characters (e.g. JSON braces) cannot cause an infinite loop.
        while d["open"] in text:
            new_text = d["rand_re"].sub(replace_random, text)
            if new_text == text:  # No more replacements made
                break
            text = new_text
        return text

    def find_wildcard_file(self, wildcard_path, extension):
        """Find wildcard file using ComfyUI's folder paths system"""
        try:
            # TODO not working
            # First try to find in wildcards folder
            if folder_paths.folder_names_and_paths.get("wildcards"):
                dprint(f"find_wildcard_file found wildcards")
                wildcard_file = f"{wildcard_path}{extension}"
                full_path = folder_paths.get_full_path("wildcards", wildcard_file)
                dprint(f"find_wildcard_file full_path:{full_path}")
                if full_path:
                    return full_path

            # Fallback: look in the default ComfyUI models/wildcards path
            models_dir = folder_paths.models_dir
            base_dir = os.path.join(models_dir, "wildcards")
            dprint(f"find_wildcard_file base_dir:{base_dir}")
            wildcard_file = os.path.join(base_dir, f"{wildcard_path}{extension}")
            dprint(f"find_wildcard_file wildcard_file:{wildcard_file}")
            if os.path.exists(wildcard_file):
                return wildcard_file
        except Exception as e:
            log.warning(f"Error finding wildcard file: {e}")

        return None

