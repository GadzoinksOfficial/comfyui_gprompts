"""
Dynamic Prompts with Enhancer
From Gadzoinks Official
https://github.com/GadzoinksOfficial/comfyui_gprompts

Two V3 nodes:

  Prompt Enhancer Loader (GGUF)
      Picks a GGUF LLM and a system prompt file from models/LLM and holds the
      generation settings. Outputs a GPROMPT_ENHANCER object. The model itself
      is loaded lazily, the first time a prompt is enhanced.

  Dynamic Prompts with Enhancer
      Expands the dynamic prompt exactly like "Dynamic Prompts", but sends only
      the FIRST expansion to the LLM. Every later expansion (next sequential
      combination, new random picks) is produced by substituting its values
      into the LLM's rewrite in place of the first expansion's values - one
      LLM trip per template instead of one per image.

GGUF inference uses the llama_cpp package (llama-cpp-python), in-process, the
same way xiaowuapple-pixel/ComfyUI-Prompt-Enhancer does: an optional import, so
the rest of the node pack loads without it (see requirements-local-gguf.txt).
Qwen3.5-based models (e.g. the Qwen-Image-2.1 prompt enhancers) need a recent
build: llama-cpp-python 0.3.35+ from PyPI, or the JamePeng fork 0.3.47+, which
publishes prebuilt CUDA wheels.

A future API/Ollama loader only has to output an object implementing
PromptEnhancerBase (cache_key + enhance).
"""
import gc
import json
import math
import os
import inspect
import random
import re
import time
from dataclasses import dataclass

import folder_paths
import comfy.model_management as model_management
import comfy.utils
from comfy_api.latest import io
from server import PromptServer

from .common import dprint, DynamicPromptEngine
from .comfyui_gprompts import promtpForId


# ----------------------------------------------------------------------------
# models/LLM folder
# ----------------------------------------------------------------------------
LLM_FOLDER = "LLM"
MODEL_EXTENSIONS = (".gguf",)
SYSTEM_PROMPT_EXTENSIONS = (".txt", ".md")
NO_MODELS = "(no .gguf files in models/LLM)"
NO_SYSTEM_PROMPTS = "(no .txt/.md files in models/LLM)"


def _register_llm_folder():
    """Register models/LLM (and models/llm if it exists separately)."""
    primary = os.path.join(folder_paths.models_dir, "LLM")
    folder_paths.add_model_folder_path(LLM_FOLDER, primary)
    lower = os.path.join(folder_paths.models_dir, "llm")
    if os.path.isdir(lower):
        same = os.path.isdir(primary) and os.path.samefile(primary, lower)
        if not same:
            folder_paths.add_model_folder_path(LLM_FOLDER, lower)


_register_llm_folder()


def _llm_folders():
    try:
        return [p for p in folder_paths.get_folder_paths(LLM_FOLDER) if os.path.isdir(p)]
    except Exception:
        return []


def list_llm_files(extensions):
    """Relative paths (forward slashes) of files under models/LLM with the
    given extensions. Walks the folders ourselves because other node packs
    may register 'LLM' with a different extension filter."""
    found = set()
    for base in _llm_folders():
        for root, _dirs, files in os.walk(base, followlinks=True):
            for f in files:
                if f.lower().endswith(extensions):
                    rel = os.path.relpath(os.path.join(root, f), base)
                    found.add(rel.replace(os.sep, "/"))
    return sorted(found, key=str.lower)


def resolve_llm_file(rel_path):
    for base in _llm_folders():
        full = os.path.join(base, rel_path.replace("/", os.sep))
        if os.path.isfile(full):
            return full
    return None


# ----------------------------------------------------------------------------
# Enhancer interface
# ----------------------------------------------------------------------------
@dataclass
class EnhanceResult:
    prompt: str          # the rewritten prompt
    wh_ratio: str        # recommended aspect ratio, "" if the model gave none
    thinking: str        # reasoning trace (not used downstream, logged only)
    raw: str             # full model output
    parse_ok: bool       # True if the answer was the expected JSON


class PromptEnhancerBase:
    """What 'Dynamic Prompts with Enhancer' needs from a loader."""

    def cache_key(self) -> str:
        """Changes whenever the same input could produce different output
        (model, system prompt, sampling settings)."""
        raise NotImplementedError

    def enhance(self, user_text: str, seed: int) -> EnhanceResult:
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Output parsing (Qwen-Image-2.1 PE contract:
#   <think>...</think>{"rewritten_prompt": "...", "wh_ratio": "16:9"}
# ----------------------------------------------------------------------------
_PROMPT_KEYS = ("rewritten_prompt", "positive_prompt", "prompt")
_JSON_STRING_FIELD = r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"'


def parse_enhancer_output(text):
    thinking, sep, answer = text.partition("</think>")
    if not sep:
        answer, thinking = text, ""
    thinking = thinking.replace("<think>", "").strip()
    answer = answer.strip()
    # tolerate ```json fences
    answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer).strip()

    obj = None
    try:
        obj = json.loads(answer)
    except Exception:
        start, end = answer.find("{"), answer.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(answer[start:end + 1])
            except Exception:
                obj = None

    if isinstance(obj, dict):
        for key in _PROMPT_KEYS:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return EnhanceResult(val.strip(), str(obj.get("wh_ratio") or ""), thinking, text, True)

    # Malformed JSON (e.g. unescaped quote): pull the string field out directly
    for key in _PROMPT_KEYS:
        m = re.search(_JSON_STRING_FIELD % key, answer, re.DOTALL)
        if m:
            try:
                val = json.loads('"' + m.group(1) + '"')
            except Exception:
                val = m.group(1)
            ratio = re.search(_JSON_STRING_FIELD % "wh_ratio", answer)
            return EnhanceResult(val.strip(), ratio.group(1) if ratio else "", thinking, text, False)

    # Not JSON at all: a generic LLM with a plain system prompt. Use the answer.
    return EnhanceResult(answer, "", thinking, text, False)


# ----------------------------------------------------------------------------
# GGUF backend: the llama_cpp pip package (llama-cpp-python), in-process.
# Modelled on xiaowuapple-pixel/ComfyUI-Prompt-Enhancer's GGUF path:
#   - optional import, so the pack loads without it
#   - explicit ChatML prompt with the <think> block pre-filled
#   - optional cap on thinking ("plan tokens"), then a second pass that hands
#     the plan back and asks for the answer
#   - stop as soon as the JSON answer closes
#   - full runtime reset before every generation (hybrid Qwen3.5 memory
#     cannot be partially reused)
#   - sampling argument names adapted to whichever build is installed
# ----------------------------------------------------------------------------
try:
    import llama_cpp
    from llama_cpp import Llama
    _LLAMA_CPP_IMPORT_ERROR = None
except ImportError as _exc:
    llama_cpp = None
    Llama = None
    _LLAMA_CPP_IMPORT_ERROR = _exc

INSTALL_HELP = (
    "The GGUF prompt enhancer needs the llama_cpp package (llama-cpp-python) in ComfyUI's Python.\n"
    "Easiest: a prebuilt CUDA wheel from https://github.com/JamePeng/llama-cpp-python/releases\n"
    "(pick the file matching your CUDA version and Python, e.g. +cu128 and cp312), then:\n"
    "  python -m pip install <downloaded .whl file>\n"
    "Or build it: CMAKE_ARGS=\"-DGGML_CUDA=on\" python -m pip install llama-cpp-python --no-cache-dir\n"
    "Qwen3.5-based models (e.g. the Qwen-Image-2.1 prompt rewriters) need a recent build "
    "(llama-cpp-python 0.3.35+, or the JamePeng fork 0.3.47+).\n"
    "Use ComfyUI's own python (python_embeded\\python.exe on Windows portable)."
)

# The Qwen-Image-2.1 PE models answer {"rewritten_prompt": "...", "wh_ratio": "..."}.
# With thinking off, opening that answer for the model stops it from planning
# anyway (xiaowuapple measured a 5930-token plan vs 404 tokens with the prefill).
ANSWER_PREFILL = '{"rewritten_prompt": "'
PE_CONTRACT_KEY = '"rewritten_prompt"'
STOP_STRINGS = ["<|im_end|>", "<|endoftext|>"]


def _require_llama_cpp():
    if _LLAMA_CPP_IMPORT_ERROR is not None:
        raise RuntimeError(INSTALL_HELP) from _LLAMA_CPP_IMPORT_ERROR


def _version_tuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", str(v))[:3])


# One GGUF resident at a time; keyed by everything that requires a reload.
_LLM_CACHE = {"key": None, "llm": None}


def unload_llm():
    llm = _LLM_CACHE.get("llm")
    _LLM_CACHE["key"] = None
    _LLM_CACHE["llm"] = None
    if llm is not None:
        try:
            llm.close()
        except Exception as e:
            dprint(f"GPromptsEnhanced: error closing LLM: {e}")
        del llm
        gc.collect()


def _get_llm(model_path, n_ctx, n_gpu_layers, flash_attn):
    _require_llama_cpp()
    key = (model_path, n_ctx, n_gpu_layers, flash_attn)
    if _LLM_CACHE["key"] == key and _LLM_CACHE["llm"] is not None:
        return _LLM_CACHE["llm"]
    unload_llm()
    version = getattr(llama_cpp, "__version__", "0")
    if n_gpu_layers != 0:
        supports_gpu = getattr(llama_cpp, "llama_supports_gpu_offload", lambda: True)
        if not supports_gpu():
            print("GPromptsEnhanced: this llama-cpp-python build is CPU-only; the LLM will "
                  "run on the CPU. Install a CUDA build to use the GPU.")
    print(f"GPromptsEnhanced: loading {os.path.basename(model_path)} (llama-cpp-python {version}, "
          f"n_ctx={n_ctx}, gpu_layers={n_gpu_layers})")
    kwargs = dict(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    init_params = _named_params(Llama.__init__)
    if "flash_attn" in init_params:            # llama-cpp-python (abetlen)
        kwargs["flash_attn"] = flash_attn
    elif "flash_attn_type" in init_params and not flash_attn:   # JamePeng fork, default AUTO
        fa_types = getattr(getattr(llama_cpp, "llama_cpp_lib", llama_cpp), "llama_flash_attn_type", None)
        disabled = getattr(fa_types, "LLAMA_FLASH_ATTN_TYPE_DISABLED", None)
        if disabled is None:
            disabled = getattr(llama_cpp, "LLAMA_FLASH_ATTN_TYPE_DISABLED", None)
        if disabled is not None:
            kwargs["flash_attn_type"] = disabled
    try:
        llm = Llama(**kwargs)
    except Exception as e:
        hint = ""
        if _version_tuple(version) < (0, 3, 35):
            hint = (f"\nllama-cpp-python {version} is probably too old for this model.\n"
                    + INSTALL_HELP)
        raise RuntimeError(f"Could not load GGUF '{model_path}': {e}{hint}") from e
    _LLM_CACHE["key"] = key
    _LLM_CACHE["llm"] = llm
    return llm


def _named_params(fn):
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def _accepted_params(fn):
    """Parameter names fn accepts, or None if it takes **kwargs / can't be inspected."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind == p.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params}


def _adapt_sampling(llm, sampling):
    """Fit our argument names to the installed build. Builds differ silently:
    some expose llama.cpp's own `present_penalty` instead of `presence_penalty`,
    and passing an unknown keyword would fail the whole call."""
    accepted = _accepted_params(llm.create_completion)
    if accepted is None:
        return dict(sampling)
    adapted = {}
    for name, value in sampling.items():
        if name in accepted:
            adapted[name] = value
        elif name == "presence_penalty" and "present_penalty" in accepted:
            adapted["present_penalty"] = value
        else:
            dprint(f"GPromptsEnhanced: this llama-cpp-python build has no '{name}'; skipped")
    return adapted


def _reset_runtime(llm):
    """Clear all model memory before a generation. Qwen3.5 is a hybrid model
    whose recurrent state cannot be partially rewound, and some builds only
    zero the token counter in reset(), so clear the context memory as well."""
    reset = getattr(llm, "reset", None)
    if callable(reset):
        reset()
    ctx = getattr(llm, "_ctx", None)
    clear = getattr(ctx, "memory_clear", None) if ctx is not None else None
    if callable(clear):
        try:
            clear(True)
        except TypeError:
            clear()
        except Exception:
            pass
    try:
        llm.n_tokens = 0
    except Exception:
        pass


def build_chat_prompt(system_prompt, user_text, thinking, contract):
    """ChatML as the Qwen-Image-2.1 PE models were trained on: the assistant
    turn opens the think block itself. With thinking off, an empty think block
    plus (for the PE JSON contract) the opening of the answer."""
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
    if thinking:
        parts.append("<think>\n")
    else:
        parts.append("<think>\n\n</think>\n\n")
        if contract:
            parts.append(ANSWER_PREFILL)
    return "".join(parts)


def restore_prefill(text):
    """Glue the pre-filled answer opening back on, unless the model repeated it."""
    if PE_CONTRACT_KEY in text:
        return text
    return ANSWER_PREFILL + text


def answer_complete(text, thinking):
    """True once the answer's top-level JSON object has closed. Only text after
    </think> counts: the plan quotes JSON examples of its own."""
    marker = text.rfind("</think>")
    if marker >= 0:
        body = text[marker + len("</think>"):]
    elif thinking:
        return False
    else:
        body = text
    depth, started, in_string, escaped = 0, False, False, False
    for ch in body:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            started = True
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and started:
                return True
    return False


class GGUFPromptEnhancer(PromptEnhancerBase):
    def __init__(self, model_path, system_prompt_path, context_length, max_new_tokens,
                 gpu_layers, temperature, top_p, top_k, presence_penalty,
                 enable_thinking, plan_tokens, keep_loaded, flash_attn):
        self.model_path = model_path
        self.system_prompt_path = system_prompt_path
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.gpu_layers = gpu_layers
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.enable_thinking = enable_thinking
        self.plan_tokens = plan_tokens
        self.keep_loaded = keep_loaded
        self.flash_attn = flash_attn

    def _system_prompt(self):
        if not self.system_prompt_path:
            return ""
        with open(self.system_prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def cache_key(self):
        try:
            sp_stamp = os.path.getmtime(self.system_prompt_path) if self.system_prompt_path else 0
        except OSError:
            sp_stamp = 0
        return json.dumps([
            self.model_path, self.system_prompt_path, sp_stamp, self.context_length,
            self.max_new_tokens, self.temperature, self.top_p, self.top_k,
            self.presence_penalty, self.enable_thinking, self.plan_tokens,
        ])

    def _stream(self, llm, prompt, max_tokens, sampling, pbar, stop_when):
        """Generate from a fresh state. stop_when(text, n_tokens) -> True ends
        the stream early. Returns (text, n_tokens, finish_reason)."""
        n_prompt = len(llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
        room = self.context_length - n_prompt
        if room < 256:
            raise RuntimeError(
                f"The prompt is {n_prompt} tokens, leaving only {max(room, 0)} of the "
                f"{self.context_length}-token context for the answer. Increase "
                f"context_length on the Prompt Enhancer Loader.")
        budget = min(max_tokens, room)
        _reset_runtime(llm)
        pieces, n_tokens, finish = [], 0, None
        started = time.perf_counter()
        for chunk in llm.create_completion(prompt, max_tokens=budget, stream=True, **sampling):
            if model_management.processing_interrupted():
                break
            choice = chunk["choices"][0]
            piece = choice.get("text") or ""
            finish = choice.get("finish_reason") or finish
            if piece:
                pieces.append(piece)
                n_tokens += 1
                pbar.update(1)
                if stop_when("".join(pieces), n_tokens):
                    finish = "early"
                    break
        model_management.throw_exception_if_processing_interrupted()
        elapsed = time.perf_counter() - started
        dprint(f"GPromptsEnhanced: {n_tokens} tokens in {elapsed:.1f}s "
               f"({n_tokens / elapsed if elapsed else 0:.0f} tok/s), finish={finish}")
        return "".join(pieces), n_tokens, finish

    def enhance(self, user_text, seed):
        _require_llama_cpp()
        system_prompt = self._system_prompt()
        thinking = self.enable_thinking
        llm = _get_llm(self.model_path, self.context_length, self.gpu_layers, self.flash_attn)

        sampling = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": 0.0,
            "presence_penalty": self.presence_penalty,
            "repeat_penalty": 1.0,
            "seed": seed & 0xFFFFFFFF,
            "stop": STOP_STRINGS,
        }
        stop_on_interrupt = getattr(llama_cpp, "StoppingCriteriaList", None)
        if stop_on_interrupt is not None:
            sampling["stopping_criteria"] = stop_on_interrupt(
                [lambda _ids, _logits: model_management.processing_interrupted()])
        sampling = _adapt_sampling(llm, sampling)
        pbar = comfy.utils.ProgressBar(self.max_new_tokens)

        try:
            raw, finish, contract = run_raw_generation(
                lambda prompt, max_tokens, stop_when: self._stream(
                    llm, prompt, max_tokens, sampling, pbar, stop_when),
                system_prompt, user_text, thinking, self.plan_tokens, self.max_new_tokens)
        finally:
            if not self.keep_loaded:
                unload_llm()
        return finalize_result(raw, finish, thinking, contract, self.max_new_tokens)


def run_raw_generation(stream_raw, system_prompt, user_text, thinking, plan_tokens,
                       max_new_tokens):
    """The raw-prompt flow shared by every backend that accepts a full prompt
    string (GGUF, and the API loader's raw completion modes).

    stream_raw(prompt, max_tokens, stop_when) -> (text, n_tokens, finish_reason)
    must generate from a fresh state and end early (finish "early") when
    stop_when(text_so_far, n_tokens) is True.

    Returns (raw_text, finish_reason, contract)."""
    contract = PE_CONTRACT_KEY in system_prompt

    def done(text):
        # text continues from the prompt: after ANSWER_PREFILL when it was
        # pre-filled, so check the JSON with that opening glued back on.
        return contract and answer_complete(restore_prefill(text), False)

    prompt = build_chat_prompt(system_prompt, user_text, thinking, contract)
    if not thinking:
        raw, _n, finish = stream_raw(prompt, max_new_tokens, lambda t, n: done(t))
        if contract:
            raw = restore_prefill(raw)
        return raw, finish, contract

    def stop_planning(text, n):
        if "</think>" in text:
            return contract and answer_complete(text, True)
        return plan_tokens >= 0 and n >= plan_tokens

    raw, used, finish = stream_raw(prompt, max_new_tokens, stop_planning)
    if "</think>" not in raw and plan_tokens >= 0 and finish == "early" and raw.strip():
        # Plan cut at the budget: hand it back, closed, and ask for the answer.
        plan = raw.rstrip()
        dprint(f"GPromptsEnhanced: plan capped at {used} tokens; writing the answer")
        answer_prompt = prompt + plan + "\n</think>\n\n" + (ANSWER_PREFILL if contract else "")
        answer, _n, finish = stream_raw(answer_prompt, max(256, max_new_tokens - used),
                                        lambda t, n: done(t))
        if contract:
            answer = restore_prefill(answer)
        raw = plan + "\n</think>\n\n" + answer
    return raw, finish, contract


def finalize_result(raw, finish, thinking, contract, max_new_tokens):
    """Check and parse a finished generation (any backend)."""
    if thinking and "</think>" not in raw and finish == "length":
        raise RuntimeError(
            f"The LLM used all {max_new_tokens} tokens while still thinking. "
            f"Set plan_tokens (e.g. 800) or increase max_new_tokens.")
    result = parse_enhancer_output(raw)
    if not result.prompt:
        raise RuntimeError(f"The LLM returned no prompt. Raw output:\n{raw[-2000:]}")
    if contract and not result.parse_ok:
        print("GPromptsEnhanced: warning - the answer was not the expected JSON; using the raw "
              "answer text. Usual causes: a base model instead of a PE model, the wrong "
              "system prompt for the model, or output cut off by the context length.")
    dprint(f"GPromptsEnhanced: parse_ok={result.parse_ok}, wh_ratio={result.wh_ratio!r}")
    return result


# ----------------------------------------------------------------------------
# Substituting later expansions into the enhanced prompt
# ----------------------------------------------------------------------------
def _anchor_pattern(anchor):
    body = r"\s+".join(re.escape(w) for w in anchor.split())
    pre = r"(?<!\w)" if re.match(r"\w", anchor) else ""
    post = r"(?!\w)" if re.search(r"\w$", anchor) else ""
    return f"{pre}({body}){post}"


def substitute_values(text, anchors, values):
    """Replace each anchor (value chosen for a block in the first expansion)
    with the value chosen for the same block now. One regex pass, so a
    replacement is never replaced again (cat->dog, dog->wolf). Matching is
    case-insensitive and whole-word; a capitalised match keeps its capital.
    Returns (new_text, warnings)."""
    warnings = []
    if len(anchors) != len(values):
        warnings.append(f"block count changed ({len(anchors)} -> {len(values)}); "
                        f"only the first {min(len(anchors), len(values))} were substituted")

    mapping = {}  # lower-case anchor -> (anchor, new value)
    for anchor, value in zip(anchors, values):
        anchor = "" if anchor is None else str(anchor).strip()
        value = "" if value is None else str(value).strip()
        if anchor.lower() == value.lower():
            continue
        if not anchor:
            warnings.append(f"first expansion chose an empty option, so '{value}' has "
                            f"nowhere to go in the enhanced prompt")
            continue
        key = anchor.lower()
        if key in mapping and mapping[key][1] != value:
            warnings.append(f"'{anchor}' was chosen by more than one block; "
                            f"using '{mapping[key][1]}' for all of them")
            continue
        mapping[key] = (anchor, value)

    if not mapping:
        return text, warnings

    pairs = sorted(mapping.values(), key=lambda p: len(p[0]), reverse=True)
    regex = re.compile("|".join(_anchor_pattern(a) for a, _ in pairs), re.IGNORECASE)
    hits = [0] * len(pairs)

    def repl(m):
        for i, (_anchor, value) in enumerate(pairs):
            matched = m.group(i + 1)
            if matched is not None:
                hits[i] += 1
                if value and matched[0].isupper() and value[0].islower():
                    return value[0].upper() + value[1:]
                return value
        return m.group(0)

    new_text = regex.sub(repl, text)
    for (anchor, value), n in zip(pairs, hits):
        if n == 0:
            warnings.append(f"'{anchor}' not found in the enhanced prompt, so '{value}' "
                            f"was not applied (the LLM reworded it)")
    if any(value == "" for _a, value in pairs):
        new_text = re.sub(r"[ \t]{2,}", " ", new_text)
        new_text = re.sub(r"\s+([,.;:!?])", r"\1", new_text)
    return new_text, warnings


# ----------------------------------------------------------------------------
# Image size from the recommended aspect ratio
# ----------------------------------------------------------------------------
MEGAPIXEL_OPTIONS = ["0.25", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "4.0"]
DEFAULT_MEGAPIXELS = "1.0"
SIZE_MULTIPLE = 16   # Qwen-Image 2.1's spatial downscale: the latent represents it exactly


def ratio_to_pair(text):
    """'16:9' (also 16x9, 16×9, full-width colon) -> (16, 9); None if unparseable."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*[:：xX×/]\s*(\d+(?:\.\d+)?)\s*$", text or "")
    if not m:
        return None
    w, h = float(m.group(1)), float(m.group(2))
    return (w, h) if w > 0 and h > 0 else None


def square_resolution(megapixels, multiple=32):
    """Side of a square with about megapixels * 1024^2 pixels, rounded to a
    multiple of 32 like Text Encode Qwen Image 2.1's resolution input."""
    try:
        mp = float(megapixels)
    except (TypeError, ValueError):
        mp = float(DEFAULT_MEGAPIXELS)
    return max(multiple, int(round(math.sqrt(mp) * 1024 / multiple)) * multiple)


def canvas_size(wh_ratio, megapixels, multiple=SIZE_MULTIPLE):
    """(width, height) at the ratio with about megapixels * 1024^2 pixels,
    each side rounded to a multiple of `multiple`. 1:1 if the ratio is missing."""
    w_ratio, h_ratio = ratio_to_pair(wh_ratio) or (1.0, 1.0)
    try:
        mp = float(megapixels)
    except (TypeError, ValueError):
        mp = float(DEFAULT_MEGAPIXELS)
    scale = math.sqrt(mp * 1024 * 1024 / (w_ratio * h_ratio))
    width = max(multiple, int(round(w_ratio * scale / multiple)) * multiple)
    height = max(multiple, int(round(h_ratio * scale / multiple)) * multiple)
    return width, height


def preserve_hint(anchors):
    words = []
    for a in anchors:
        a = "" if a is None else str(a).strip()
        if a and a not in words:
            words.append(a)
    if not words:
        return ""
    quoted = ", ".join(f'"{w}"' for w in words)
    return f"\n\n(Keep these exact words unchanged in the rewritten prompt: {quoted})"


# ----------------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------------
PromptEnhancerType = io.Custom("GPROMPT_ENHANCER")


class GPromptEnhancerLoaderGGUF(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = list_llm_files(MODEL_EXTENSIONS) or [NO_MODELS]
        prompts = list_llm_files(SYSTEM_PROMPT_EXTENSIONS) or [NO_SYSTEM_PROMPTS]
        return io.Schema(
            node_id="GPromptEnhancerLoaderGGUF",
            display_name="Prompt Enhancer Loader (GGUF)",
            category="gprompts/enhancer",
            description=(
                "Selects a GGUF LLM and a system prompt file from models/LLM for "
                "'Dynamic Prompts with Enhancer'. Runs in-process with llama-cpp-python. "
                "The model loads the first time a prompt is enhanced."
            ),
            search_aliases=["prompt enhancer", "prompt rewriter", "llm loader", "gguf"],
            inputs=[
                io.Combo.Input("model", options=models,
                               tooltip="GGUF model in models/LLM"),
                io.Combo.Input("system_prompt", options=prompts,
                               tooltip="System prompt file (.txt/.md) in models/LLM. For the "
                                       "Qwen-Image-2.1 PE models this is the shipped system_prompt.txt; "
                                       "the model is useless without it."),
                io.Int.Input("context_length", default=16384, min=2048, max=262144, step=1024,
                             tooltip="Context window in tokens. Must fit the system prompt + your "
                                     "prompt + max_new_tokens. Larger costs more VRAM."),
                io.Int.Input("max_new_tokens", default=8192, min=256, max=65536, step=256,
                             tooltip="Upper limit on generated tokens, thinking included. "
                                     "The Qwen-Image PE models often use 2,000-4,000."),
                io.Int.Input("gpu_layers", default=-1, min=-1, max=999,
                             tooltip="Layers to put on the GPU. -1 = all, 0 = CPU only "
                                     "(keeps VRAM free for the image model, but much slower)."),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=20, min=0, max=200),
                io.Float.Input("presence_penalty", default=1.5, min=0.0, max=2.0, step=0.05,
                               tooltip="Discourages repetition. Use ~1.5 for the T2I rewriter, "
                                       "0 for the edit (I2I) rewriter."),
                io.Boolean.Input("enable_thinking", default=True,
                                 tooltip="Let the model plan before answering. The Qwen-Image PE "
                                         "models were trained with thinking and write richer prompts "
                                         "with it. Off is much faster."),
                io.Int.Input("plan_tokens", default=800, min=-1, max=16384, step=100,
                             tooltip="Cap on thinking (only when enable_thinking is on). When the "
                                     "plan reaches this many tokens it is handed back to the model "
                                     "to write the answer. -1 = no cap (slowest, most detail). "
                                     "xiaowuapple measured ~15s at 400, ~20s at 800, ~90s uncapped "
                                     "on a 4080."),
                io.Boolean.Input("keep_loaded", default=True,
                                 tooltip="Keep the LLM in memory between prompts. Turn off to free "
                                         "its VRAM right after each enhancement."),
                io.Boolean.Input("flash_attn", default=True, advanced=True,
                                 tooltip="Use flash attention (less VRAM for long contexts)."),
            ],
            outputs=[PromptEnhancerType.Output("enhancer", display_name="enhancer")],
        )

    @classmethod
    def validate_inputs(cls, model, system_prompt):
        # Only these two are named, so ComfyUI still range-checks the rest.
        if model == NO_MODELS or resolve_llm_file(model) is None:
            return f"GGUF model not found in models/LLM: {model}"
        if system_prompt == NO_SYSTEM_PROMPTS or resolve_llm_file(system_prompt) is None:
            return f"System prompt file not found in models/LLM: {system_prompt}"
        return True

    @classmethod
    def execute(cls, model, system_prompt, context_length, max_new_tokens, gpu_layers,
                temperature, top_p, top_k, presence_penalty, enable_thinking,
                plan_tokens, keep_loaded, flash_attn=True) -> io.NodeOutput:
        enhancer = GGUFPromptEnhancer(
            model_path=resolve_llm_file(model),
            system_prompt_path=resolve_llm_file(system_prompt),
            context_length=context_length,
            max_new_tokens=max_new_tokens,
            gpu_layers=gpu_layers,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            enable_thinking=enable_thinking,
            plan_tokens=plan_tokens,
            keep_loaded=keep_loaded,
            flash_attn=flash_attn,
        )
        if not keep_loaded:
            unload_llm()
        return io.NodeOutput(enhancer)


# Per-node state. V3 nodes are stateless classes, so the dynamic prompt engine
# (iteration counter, sequential combinations, wildcard cache) and the cached
# LLM result live here, keyed by the node's unique_id.
_NODE_STATE = {}


class GPromptsEnhanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        styles = list(DynamicPromptEngine.DELIM_STYLES.keys())
        return io.Schema(
            node_id="GPromptsEnhanced",
            display_name="Dynamic Prompts with Enhancer",
            category="gprompts/enhancer",
            description=(
                "Dynamic Prompts plus an LLM prompt enhancer. The first expansion is rewritten "
                "by the LLM once; later expansions are made by swapping their chosen values "
                "into that rewrite, so a whole batch costs a single LLM call. Changing the "
                "text, delimiter style, seed or enhancer settings starts over with a new LLM call."
            ),
            search_aliases=["gprompts enhancer", "dynamic prompts llm", "prompt enhancer"],
            inputs=[
                PromptEnhancerType.Input("enhancer",
                                         tooltip="From 'Prompt Enhancer Loader (GGUF)'"),
                io.String.Input("text", multiline=True, default="",
                                tooltip="Dynamic prompt, same syntax as Dynamic Prompts: {a|b} random, "
                                        "{{a|b}} sequential, __wildcard__, registers {{0 a|b}} / {{0}}."),
                io.Combo.Input("delimiter_style", options=styles,
                               default=DynamicPromptEngine.DEFAULT_DELIM_STYLE,
                               tooltip="'curly { }' or 'angle < >' (for JSON prompts)."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             tooltip="Seed for random blocks and for the LLM. 0 = unseeded."),
                io.Boolean.Input("preserve_dynamic_words", default=True,
                                 tooltip="Ask the LLM to keep the first expansion's chosen words "
                                         "verbatim, so later expansions can be swapped in. If the "
                                         "LLM rewords a word anyway, that variation is skipped "
                                         "(see the console)."),
                io.Combo.Input("target_megapixels", options=MEGAPIXEL_OPTIONS,
                               default=DEFAULT_MEGAPIXELS,
                               tooltip="Pixel budget for the width/height outputs, at the LLM's "
                                       "recommended aspect ratio (1:1 if it gave none). 1.0 = "
                                       "1024x1024-sized; 4.0 = Qwen-Image 2.1's native 2K "
                                       "(2048x2048-sized). Sides are multiples of 16."),
                io.String.Input("computed_prompt", multiline=True, default="", optional=True,
                                extra_dict={"readonly": True},
                                tooltip="The final prompt from the last run (read-only)."),
            ],
            outputs=[
                io.String.Output("text", display_name="text",
                                 tooltip="Final prompt: the enhanced prompt with this run's values."),
                io.String.Output("dynamic_prompt", display_name="dynamic_prompt",
                                 tooltip="This run's expansion before LLM enhancement."),
                io.String.Output("computed_prompt", display_name="computed_prompt",
                                 tooltip="Same as text (matches the Dynamic Prompts outputs)."),
                io.Int.Output("seed", display_name="seed"),
                io.String.Output("template", display_name="template",
                                 tooltip="The prompt exactly as typed, dynamic blocks and all."),
                io.String.Output("wh_ratio", display_name="wh_ratio",
                                 tooltip="Aspect ratio recommended by the LLM (e.g. 16:9), "
                                         "empty if it gave none."),
                io.Int.Output("width", display_name="width",
                              tooltip="Image width for wh_ratio at target_megapixels."),
                io.Int.Output("height", display_name="height",
                              tooltip="Image height for wh_ratio at target_megapixels."),
                io.Int.Output("resolution", display_name="resolution",
                              tooltip="Side of a square with about target_megapixels pixels, a "
                                      "multiple of 32 (1.0 -> 1024). For the resolution input of "
                                      "Text Encode Qwen Image 2.1 in edit workflows. For "
                                      "text-to-image use width/height into an Empty Latent Image "
                                      "instead: that node's own latent is always square."),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Every queue advances the iteration, like Dynamic Prompts: always run.
        return float("nan")

    @classmethod
    def execute(cls, enhancer, text, delimiter_style, seed, preserve_dynamic_words=True,
                target_megapixels=DEFAULT_MEGAPIXELS, computed_prompt="") -> io.NodeOutput:
        unique_id = cls.hidden.unique_id
        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo or {}
        workflow_id = extra_pnginfo.get("workflow", {}).get("id")

        if delimiter_style not in DynamicPromptEngine.DELIM_STYLES:
            delimiter_style = DynamicPromptEngine.DEFAULT_DELIM_STYLE
        key = (text, delimiter_style, seed, bool(preserve_dynamic_words), enhancer.cache_key())
        state = _NODE_STATE.get(unique_id)
        if state is None or state["key"] != key:
            engine = DynamicPromptEngine()
            engine.previous_text = text
            engine.previous_delim_style = delimiter_style
            state = {"key": key, "engine": engine, "anchors": None, "result": None}
            _NODE_STATE[unique_id] = state
        engine = state["engine"]
        engine.delims = engine.DELIM_STYLES[delimiter_style]

        if seed > 0:
            random.seed(seed + engine.current_iteration)
        expanded = engine.parse_dynamic_prompt(text)
        values = list(engine.last_values)
        iteration = engine.current_iteration

        if state["result"] is None:
            user_text = expanded
            if preserve_dynamic_words:
                user_text += preserve_hint(values)
            dprint(f"GPromptsEnhanced: enhancing (iteration {iteration}): {user_text}")
            state["result"] = enhancer.enhance(user_text, seed)
            state["anchors"] = values
            final = state["result"].prompt
        else:
            final, warnings = substitute_values(state["result"].prompt, state["anchors"], values)
            for w in warnings:
                print(f"GPromptsEnhanced (node {unique_id}, iteration {iteration}): {w}")

        engine.current_iteration += 1
        result = state["result"]

        if workflow_id:
            promtpForId[workflow_id] = final
        try:
            PromptServer.instance.send_sync("gprompts_executed",
                                            {"node_id": unique_id, "result": final})
        except Exception as e:
            dprint(f"GPromptsEnhanced: could not update UI: {e}")
        if prompt is not None and unique_id in prompt:
            node = prompt[unique_id]
            node.setdefault("inputs", {})["computed_prompt"] = final
            meta = node.get("_meta", {})
            meta["computed_prompt"] = final
            meta["dynamic_prompt"] = expanded
            node["_meta"] = meta

        width, height = canvas_size(result.wh_ratio, target_megapixels)
        return io.NodeOutput(final, expanded, final, seed, text, result.wh_ratio,
                             width, height, square_resolution(target_megapixels))


ENHANCED_NODES = {
    "GPromptEnhancerLoaderGGUF": GPromptEnhancerLoaderGGUF,
    "GPromptsEnhanced": GPromptsEnhanced,
}
ENHANCED_NODE_DISPLAY_NAME_MAPPINGS = {
    "GPromptEnhancerLoaderGGUF": "Prompt Enhancer Loader (GGUF)",
    "GPromptsEnhanced": "Dynamic Prompts with Enhancer",
}
