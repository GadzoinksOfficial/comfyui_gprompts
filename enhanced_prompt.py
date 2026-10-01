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
import base64
import gc
import hashlib
import json
import math
import os
import inspect
import random
import re
import threading
import time
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image as PILImage

import folder_paths
import comfy.model_management as model_management
import comfy.utils
from comfy_api.latest import io
from server import PromptServer

from .common import DynamicPromptEngine, get_logger
from .comfyui_gprompts import promtpForId

log = get_logger("enhancer")


# ----------------------------------------------------------------------------
# Background (prefetch) generation
# The next run's rewrite can be generated on a worker thread while the image
# model works. On that thread, ComfyUI's interrupt flag must not be consumed
# (it belongs to whatever runs on the main thread, and reading it with
# throw_exception_if_processing_interrupted clears it), and progress bars must
# stay quiet (they would draw on whichever node is running). Enhancers call
# interrupted() / check_interrupted() / progress_bar() instead.
# ----------------------------------------------------------------------------
_BACKGROUND = threading.local()


class PrefetchCancelled(Exception):
    """A background rewrite was no longer needed."""


def _background_cancel():
    return getattr(_BACKGROUND, "cancel", None)


def interrupted():
    cancel = _background_cancel()
    if cancel is not None and cancel.is_set():
        return True
    return model_management.processing_interrupted()      # reading only; never clears it


def check_interrupted():
    if _background_cancel() is not None:
        if interrupted():
            raise PrefetchCancelled()
        return
    model_management.throw_exception_if_processing_interrupted()


class _SilentBar:
    def update(self, n):
        pass


def progress_bar(total):
    return _SilentBar() if _background_cancel() is not None else comfy.utils.ProgressBar(total)


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
    ratio_follow: str = ""   # edit models: which input image's shape to keep, e.g. "image1"


TASK_T2I = "text-to-image"
TASK_EDIT = "image edit"
TASK_PAIR = "text-to-image + image edit"


class PromptEnhancerBase:
    """What 'Dynamic Prompts with Enhancer' needs from a loader."""

    task = TASK_T2I

    def cache_key(self) -> str:
        """Changes whenever the same input could produce different output
        (model, system prompt, sampling settings)."""
        raise NotImplementedError

    def cache_key_for(self, has_images):
        return self.cache_key()

    def check_task(self, has_images):
        """Raise a clear error when this enhancer can't handle the request."""
        if has_images and self.task == TASK_T2I:
            raise RuntimeError(
                "Reference images are connected, but the enhancer is a text-to-image loader. "
                "Use an image-edit loader, or an Enhancer Pair with both.")
        if not has_images and self.task == TASK_EDIT:
            raise RuntimeError(
                "The enhancer is an image-edit loader, but no reference images are connected. "
                "Connect image_1 (and image_2, ...), or use a text-to-image loader or an "
                "Enhancer Pair with both.")

    def uses_local_gpu(self, has_images):
        """True when generating would share this machine's GPU with the image
        model (GGUF with GPU layers, or a server on this machine). Decides
        prefetch 'auto'."""
        return False

    def gpu_in_process(self, has_images):
        """True when generation runs CUDA/Metal work inside the ComfyUI process
        (GGUF with GPU layers). Never run in the background: llama.cpp aborts
        the whole process on a GPU error, e.g. when the image model is sampling
        at the same time."""
        return False

    def enhance(self, user_text: str, seed: int, images=None) -> EnhanceResult:
        """images: list of ComfyUI IMAGE tensors [1,H,W,C] (edit enhancers only)."""
        raise NotImplementedError


class EnhancerPair(PromptEnhancerBase):
    """A text-to-image enhancer and an image-edit enhancer behind one output:
    requests with images go to the edit one, the rest to the text one."""

    task = TASK_PAIR

    def __init__(self, text_enhancer, edit_enhancer):
        self.text_enhancer = text_enhancer
        self.edit_enhancer = edit_enhancer

    def _pick(self, has_images):
        chosen = self.edit_enhancer if has_images else self.text_enhancer
        if chosen is None:
            side = "edit_enhancer (reference images are connected)" if has_images else \
                   "text_enhancer (no reference images are connected)"
            raise RuntimeError(f"Enhancer Pair: connect {side}.")
        return chosen

    def check_task(self, has_images):
        self._pick(has_images).check_task(has_images)

    def cache_key(self):
        return json.dumps([e.cache_key() if e else None
                           for e in (self.text_enhancer, self.edit_enhancer)])

    def cache_key_for(self, has_images):
        return self._pick(has_images).cache_key_for(has_images)

    def uses_local_gpu(self, has_images):
        return self._pick(has_images).uses_local_gpu(has_images)

    def gpu_in_process(self, has_images):
        return self._pick(has_images).gpu_in_process(has_images)

    def enhance(self, user_text, seed, images=None):
        chosen = self._pick(bool(images))
        return chosen.enhance(user_text, seed, images=images) if images else chosen.enhance(user_text, seed)


# ----------------------------------------------------------------------------
# Output parsing (Qwen-Image-2.1 PE contract:
#   <think>...</think>{"rewritten_prompt": "...", "wh_ratio": "16:9"}
# ----------------------------------------------------------------------------
_PROMPT_KEYS = ("rewritten_prompt", "positive_prompt", "prompt")
_JSON_STRING_FIELD = r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"'
_CONTRACT_FIELDS = _PROMPT_KEYS + ("wh_ratio", "ratio_follow")
_FIELD_OPENER = re.compile(r'"(%s)"\s*:\s*"' % "|".join(_CONTRACT_FIELDS))


def _json_unquote(value):
    """A JSON string body that may contain unescaped double quotes -> str."""
    fixed = re.sub(r'(?<!\\)"', r'\\"', value).replace("\r", "\\r").replace("\n", "\\n")
    try:
        return json.loads('"' + fixed + '"')
    except Exception:
        return value.replace('\\"', '"')


def _repair_contract(answer):
    """Rebuild {"rewritten_prompt": ..., "wh_ratio": ..., "ratio_follow": ...}
    when the LLM broke the JSON with unescaped double quotes inside the prompt
    (e.g. a sign reading "OPEN"). Each value runs to the last quote before the
    next known key; the last one to the last quote before the closing brace.
    Returns (fields, closed) or (None, False); closed = the object was complete."""
    openers = list(_FIELD_OPENER.finditer(answer))
    if not openers:
        return None, False
    fields, closed = {}, False
    for i, m in enumerate(openers):
        if i + 1 < len(openers):
            seg = answer[m.end():openers[i + 1].start()].rstrip()
            seg = seg[:-1].rstrip() if seg.endswith(",") else seg
            seg = seg[:-1] if seg.endswith('"') else seg
        else:
            seg = answer[m.end():]
            ends = list(re.finditer(r'"\s*\}', seg))
            if ends:
                seg, closed = seg[:ends[-1].start()], True
            else:                       # cut off: keep what there is
                seg = seg.rstrip().rstrip("`").rstrip()
                seg = seg[:-1] if seg.endswith('"') else seg
        fields.setdefault(m.group(1), _json_unquote(seg).strip())
    return fields, closed


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
                return EnhanceResult(val.strip(), str(obj.get("wh_ratio") or ""), thinking, text,
                                     True, str(obj.get("ratio_follow") or ""))

    # Malformed JSON, usually unescaped double quotes inside the prompt: rebuild it
    fields, closed = _repair_contract(answer)
    if fields:
        for key in _PROMPT_KEYS:
            if fields.get(key):
                if closed:
                    log.info("GPromptsEnhanced: the LLM's JSON was malformed (probably unescaped "
                          "quotes in the prompt); repaired it.")
                return EnhanceResult(fields[key], fields.get("wh_ratio", ""), thinking, text,
                                     closed, fields.get("ratio_follow", ""))

    # Last resort: pull the string field out directly
    for key in _PROMPT_KEYS:
        m = re.search(_JSON_STRING_FIELD % key, answer, re.DOTALL)
        if m:
            try:
                val = json.loads('"' + m.group(1) + '"')
            except Exception:
                val = m.group(1)
            ratio = re.search(_JSON_STRING_FIELD % "wh_ratio", answer)
            follow = re.search(_JSON_STRING_FIELD % "ratio_follow", answer)
            return EnhanceResult(val.strip(), ratio.group(1) if ratio else "", thinking, text, False,
                                 follow.group(1) if follow else "")

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
# llama.cpp models are not thread-safe; a prefetch thread and the main thread
# (or two generator nodes) take turns.
_LLM_LOCK = threading.RLock()


def unload_llm():
    with _LLM_LOCK:
        _unload_llm()


def _unload_llm():
    llm = _LLM_CACHE.get("llm")
    _LLM_CACHE["key"] = None
    _LLM_CACHE["llm"] = None
    if llm is not None:
        try:
            llm.close()
        except Exception as e:
            log.warning(f"GPromptsEnhanced: error closing LLM: {e}")
        del llm
        gc.collect()


# Multimodal (edit) models: a chat template that just concatenates the
# message parts. We build the full ChatML prompt ourselves (think prefill,
# plan cap, answer prefill) exactly as for text; image parts become the media
# marker, and llama.cpp's mtmd adds Qwen's <|vision_start|>/<|vision_end|>.
PASSTHROUGH_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message.content is string -%}{{- message.content -}}"
    "{%- else -%}{%- for item in message.content -%}"
    "{%- if item.type == 'image_url' -%}"
    "{{- item.image_url if item.image_url is string else item.image_url.url -}}"
    "{%- else -%}{{- item.text -}}{%- endif -%}"
    "{%- endfor -%}{%- endif -%}"
    "{%- endfor -%}"
)


def _make_mm_handler(mmproj_path, use_gpu=True):
    """MTMD chat handler with the pass-through template, for either build:
    PyPI llama-cpp-python (clip_model_path, template from _get_chat_template)
    or the JamePeng fork (mmproj_path, chat_template_override)."""
    from llama_cpp import llama_chat_format as lcf
    base = getattr(lcf, "MTMDChatHandler", None)
    if base is None:
        raise RuntimeError("This llama-cpp-python build has no multimodal (MTMD) support, which "
                           "image editing needs.\n" + INSTALL_HELP)
    params = _named_params(base.__init__)
    extra = {} if use_gpu or "use_gpu" not in params else {"use_gpu": False}
    if "chat_template_override" in params:
        return base(mmproj_path=mmproj_path, chat_template_override=PASSTHROUGH_TEMPLATE,
                    verbose=False, **extra)

    class _PassthroughHandler(base):
        def _get_chat_template(self, llama_model):
            return PASSTHROUGH_TEMPLATE

    key = "clip_model_path" if "clip_model_path" in params else "mmproj_path"
    return _PassthroughHandler(verbose=False, **{key: mmproj_path}, **extra)


def _get_llm(model_path, n_ctx, n_gpu_layers, flash_attn, mmproj_path=None):
    _require_llama_cpp()
    key = (model_path, n_ctx, n_gpu_layers, flash_attn, mmproj_path)
    if _LLM_CACHE["key"] == key and _LLM_CACHE["llm"] is not None:
        return _LLM_CACHE["llm"]
    unload_llm()
    version = getattr(llama_cpp, "__version__", "0")
    log.info(f"GPromptsEnhanced: loading {os.path.basename(model_path)} (llama-cpp-python {version}, "
          f"n_ctx={n_ctx}, gpu_layers={n_gpu_layers})")
    kwargs = dict(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    if mmproj_path:
        kwargs["chat_handler"] = _make_mm_handler(mmproj_path, use_gpu=n_gpu_layers != 0)
    init_params = _named_params(Llama.__init__)
    if n_gpu_layers == 0 and "op_offload" in init_params:
        # CPU only really means CPU: a CUDA build otherwise still runs large
        # prompt batches on the GPU ("op offload").
        kwargs["op_offload"] = False
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
    # Asked only after loading: builds that load their GPU backend as a plugin
    # (libggml-cuda.so) report no GPU until a model has been loaded.
    if n_gpu_layers != 0:
        supports_gpu = getattr(llama_cpp, "llama_supports_gpu_offload", lambda: True)
        try:
            gpu = bool(supports_gpu())
        except Exception:
            gpu = True
        if not gpu:
            log.warning("GPromptsEnhanced: no GPU backend in this llama-cpp-python build; the LLM "
                  "runs on the CPU. Install a CUDA build to use the GPU.")
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


def _adapt_sampling(fn, sampling):
    """Fit our argument names to the installed build's fn (create_completion or
    create_chat_completion). Builds differ silently: some expose llama.cpp's own
    `present_penalty` instead of `presence_penalty`, and passing an unknown
    keyword would fail the whole call."""
    accepted = _accepted_params(fn)
    if accepted is None:
        return dict(sampling)
    adapted = {}
    for name, value in sampling.items():
        if name in accepted:
            adapted[name] = value
        elif name == "presence_penalty" and "present_penalty" in accepted:
            adapted["present_penalty"] = value
        else:
            log.debug(f"GPromptsEnhanced: this llama-cpp-python build has no '{name}'; skipped")
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


def image_slot(i):
    """Placeholder for image i in a raw prompt; the multimodal path swaps it for the image."""
    return f"\u27e6gprompts-image-{i}\u27e7"


IMAGE_SLOT_RE = re.compile("\u27e6gprompts-image-(\\d+)\u27e7")


def build_chat_prompt(system_prompt, user_text, thinking, contract, image_count=0):
    """ChatML as the Qwen-Image-2.1 PE models were trained on: the assistant
    turn opens the think block itself. With thinking off, an empty think block
    plus (for the PE JSON contract) the opening of the answer. Images (edit)
    come first in the user turn, in order, then the instruction."""
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
    images = "".join(image_slot(i) for i in range(image_count))
    parts.append(f"<|im_start|>user\n{images}{user_text}<|im_end|>\n<|im_start|>assistant\n")
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
                 enable_thinking, plan_tokens, keep_loaded, flash_attn,
                 mmproj_path=None, image_megapixels=None, task=TASK_T2I, second_try=False):
        self.model_path = model_path
        self.second_try = second_try
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
        self.mmproj_path = mmproj_path
        self.image_megapixels = image_megapixels or DEFAULT_LLM_IMAGE_MP
        self.task = task

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
            self.mmproj_path, self.image_megapixels, self.task, self.second_try,
        ])

    def _check_room(self, n_prompt, what="prompt"):
        room = self.context_length - n_prompt
        if room < 256:
            raise RuntimeError(
                f"The {what} is about {n_prompt} tokens, leaving only {max(room, 0)} of the "
                f"{self.context_length}-token context for the answer. Increase context_length "
                f"on the Prompt Enhancer Loader"
                + (", or lower llm_image_megapixels / connect fewer images." if what != "prompt" else "."))
        return room

    @staticmethod
    def _consume(chunks, extract, pbar, stop_when):
        """Read a token stream. extract(chunk) -> (piece, finish_reason)."""
        pieces, n_tokens, finish = [], 0, None
        started = time.perf_counter()
        for chunk in chunks:
            if interrupted():
                break
            piece, fin = extract(chunk)
            finish = fin or finish
            if piece:
                pieces.append(piece)
                n_tokens += 1
                pbar.update(1)
                if stop_when("".join(pieces), n_tokens):
                    finish = "early"
                    break
        check_interrupted()
        elapsed = time.perf_counter() - started
        log.info(f"GPromptsEnhanced: {n_tokens} tokens in {elapsed:.1f}s "
               f"({n_tokens / elapsed if elapsed else 0:.0f} tok/s), finish={finish}")
        return "".join(pieces), n_tokens, finish

    def _stream(self, llm, prompt, max_tokens, sampling, pbar, stop_when):
        """Generate from a fresh state. stop_when(text, n_tokens) -> True ends
        the stream early. Returns (text, n_tokens, finish_reason)."""
        n_prompt = len(llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
        budget = min(max_tokens, self._check_room(n_prompt))
        _reset_runtime(llm)

        def extract(chunk):
            choice = chunk["choices"][0]
            return choice.get("text") or "", choice.get("finish_reason")
        return self._consume(llm.create_completion(prompt, max_tokens=budget, stream=True, **sampling),
                             extract, pbar, stop_when)

    def _stream_mm(self, llm, prompt, max_tokens, sampling, pbar, stop_when, uris, image_tokens):
        """Like _stream, with images: the prompt's image slots become image parts
        and go through the multimodal handler (pass-through template)."""
        text_only = IMAGE_SLOT_RE.sub("", prompt)
        n_prompt = len(llm.tokenize(text_only.encode("utf-8"), add_bos=False, special=True))
        budget = min(max_tokens, self._check_room(n_prompt + image_tokens, "prompt with images"))
        parts, pos = [], 0
        for m in IMAGE_SLOT_RE.finditer(prompt):
            if m.start() > pos:
                parts.append({"type": "text", "text": prompt[pos:m.start()]})
            parts.append({"type": "image_url", "image_url": {"url": uris[int(m.group(1))]}})
            pos = m.end()
        if pos < len(prompt):
            parts.append({"type": "text", "text": prompt[pos:]})
        _reset_runtime(llm)

        def extract(chunk):
            choice = chunk["choices"][0]
            delta = choice.get("delta") or choice.get("message") or {}
            return delta.get("content") or "", choice.get("finish_reason")
        chunks = llm.create_chat_completion(messages=[{"role": "user", "content": parts}],
                                            max_tokens=budget, stream=True, **sampling)
        return self._consume(chunks, extract, pbar, stop_when)

    def uses_local_gpu(self, has_images):
        return self.gpu_layers != 0

    def gpu_in_process(self, has_images):
        return self.gpu_layers != 0

    def enhance(self, user_text, seed, images=None):
        with _LLM_LOCK:
            if not self.second_try:
                return self._enhance(user_text, seed, images)
            try:    # keep the model loaded between the two tries
                return with_second_try(
                    lambda text: self._enhance(text, seed, images, warn=False, unload=False),
                    user_text, self._system_prompt(), len(images or []))
            finally:
                if not self.keep_loaded:
                    unload_llm()

    def _enhance(self, user_text, seed, images=None, warn=True, unload=True):
        _require_llama_cpp()
        images = list(images or [])
        if images and not self.mmproj_path:
            raise RuntimeError("Images need a vision projector: use Prompt Enhancer Loader "
                               "(GGUF, edit) with its mmproj file.")
        system_prompt = self._system_prompt()
        thinking = self.enable_thinking
        llm = _get_llm(self.model_path, self.context_length, self.gpu_layers, self.flash_attn,
                       self.mmproj_path)

        sampling = {
            "temperature": h3_temperature(self.temperature, system_prompt),
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
                [lambda _ids, _logits: interrupted()])
        pbar = progress_bar(self.max_new_tokens)

        if images:
            prepared = [prepared_image(t, self.image_megapixels) for t in images]
            uris = ["data:image/jpeg;base64," + b64 for b64, _n in prepared]
            image_tokens = sum(n for _b64, n in prepared)
            chat_sampling = _adapt_sampling(llm.create_chat_completion, sampling)
            stream = lambda prompt, max_tokens, stop_when: self._stream_mm(
                llm, prompt, max_tokens, chat_sampling, pbar, stop_when, uris, image_tokens)
        else:
            text_sampling = _adapt_sampling(llm.create_completion, sampling)
            stream = lambda prompt, max_tokens, stop_when: self._stream(
                llm, prompt, max_tokens, text_sampling, pbar, stop_when)

        try:
            raw, finish, contract = run_raw_generation(
                stream, system_prompt, user_text, thinking, self.plan_tokens,
                self.max_new_tokens, image_count=len(images))
        finally:
            if unload and not self.keep_loaded:
                unload_llm()
        return finalize_result(raw, finish, thinking, contract, self.max_new_tokens,
                               system_prompt=system_prompt, image_count=len(images), warn=warn)


def run_raw_generation(stream_raw, system_prompt, user_text, thinking, plan_tokens,
                       max_new_tokens, image_count=0):
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

    prompt = build_chat_prompt(system_prompt, user_text, thinking, contract, image_count)
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
        log.info(f"GPromptsEnhanced: plan capped at {used} tokens; writing the answer")
        answer_prompt = prompt + plan + "\n</think>\n\n" + (ANSWER_PREFILL if contract else "")
        answer, _n, finish = stream_raw(answer_prompt, max(256, max_new_tokens - used),
                                        lambda t, n: done(t))
        if contract:
            answer = restore_prefill(answer)
        raw = plan + "\n</think>\n\n" + answer
    return raw, finish, contract


# ----------------------------------------------------------------------------
# MiniMax H3 prompt check. The H3 system prompts ask for MiniMax's official
# structure; a model that ignores them (e.g. a Qwen-Image PE fine-tune) writes
# something that looks plausible but that H3 reads badly. Catch that before a
# long render.
# ----------------------------------------------------------------------------
H3_BASE_FIELDS = ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:")
H3_REF_FIELDS = ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
                 "overall_soundscape:", "non_diegetic_music:")
_H3_I2VA = re.compile(r"^For the target video, at 0\.00 seconds into the target video, <Picture 1> "
                      r"\(from \[Shot 1\]\) is fully referenced\.$")
_H3_FL2VA = re.compile(r"^How the reference pictures align with the target video \u2014 Picture 1 \(from Shot 1\) "
                       r"aligns with the 0\.00-second mark of the target video; Picture 2 \(from Shot (\d+)\) "
                       r"aligns with the (\d+\.\d\d)-second mark of the target video\.$")
_H3_L2VA = re.compile(r"^How the reference pictures align with the target video \u2014 <Picture 1> "
                      r"\(from \[Shot (\d+)\]\) aligns with the (\d+\.\d\d)-second mark of the target video\.$")
_H3_SHOT = re.compile(r"\[Shot (\d+)\](?:\s*At (\d+):(\d+(?:\.\d+)?),)?")
_H3_LABEL = re.compile(r"<(Subject|Picture|Video|Audio) (\d+)>")
# H3 renders 17k+5 frames at 24 fps (ComfyUI's nodes_minimax_h3.py snaps the latent length up to
# that grid); the trained range is 124-362 frames, i.e. these 15 lengths from 5.17 s to 15.08 s.
H3_LENGTHS = tuple((124 + 17 * k) / 24 for k in range(15))
H3_MAX_SECONDS = H3_LENGTHS[-1]


def _h3_seconds(x):
    """Two decimals, half up, as MiniMax writes S.SS (10.125 -> '10.13')."""
    return f"{int(x * 100 + 0.5) / 100:.2f}"


H3_MIN_SHOT = 1.2           # a shorter shot reads as a glitch, not a cut
H3_CUT_PHRASES = ("the camera cuts to", "the shot cuts to", "the shot transitions to",
                  "the shot changes to", "the shot switches to")
_H3_CUT = re.compile("|".join([re.escape(p) for p in H3_CUT_PHRASES]
                              + [r"cross[- ]?dissolve", r"\bfades?\b", r"\bwipes?\b"]), re.IGNORECASE)


def h3_prompt_kind(system_prompt):
    """'ref', 'base' or None, from the system prompt that asked for it."""
    sp = system_prompt or ""
    if "subject_definitions:" in sp and "retention_analysis:" in sp:
        return "ref"
    if all(f in sp for f in H3_BASE_FIELDS):
        return "base"
    return None


def _field_positions(text, fields):
    """Start of each field name at the start of a line, or -1."""
    out = []
    for f in fields:
        m = re.search(r"(?m)^\s*" + re.escape(f), text)
        out.append(m.start() if m else -1)
    return out


def _between(text, start_field, end_field):
    a = re.search(r"(?m)^\s*" + re.escape(start_field), text)
    if not a:
        return ""
    b = re.search(r"(?m)^\s*" + re.escape(end_field), text[a.end():]) if end_field else None
    return text[a.end(): a.end() + b.start()] if b else text[a.end():]


def check_h3_prompt(text, system_prompt, image_count=0):
    """Problems with an H3 prompt written under one of the H3 system prompts
    (empty list = looks right). Not a full validator: it catches the ways a
    model that ignores the system prompt usually goes wrong."""
    kind = h3_prompt_kind(system_prompt)
    if kind is None:
        return []
    text = (text or "").strip().strip('"').strip()
    problems = []
    fields = H3_REF_FIELDS if kind == "ref" else H3_BASE_FIELDS
    pos = _field_positions(text, fields)
    missing = [f.rstrip(":") for f, p in zip(fields, pos) if p < 0]
    if missing:
        problems.append("missing section(s): " + ", ".join(missing))
    present = [p for p in pos if p >= 0]
    if present != sorted(present):
        problems.append("sections are out of order (expected " + " -> ".join(f.rstrip(":") for f in fields) + ")")

    first_line = text.splitlines()[0].strip() if text else ""
    length, last_shot = None, None      # from an FL2VA/L2VA alignment line
    if kind == "base":
        body = _between(text, "integrated_multimodal_description:", "overall_soundscape:")
        if image_count == 0:
            if not first_line.startswith("integrated_multimodal_description:"):
                problems.append("text-to-video prompts start directly with 'integrated_multimodal_description:'")
        elif image_count == 1:
            m = _H3_L2VA.match(first_line)
            if not (_H3_I2VA.match(first_line) or m):
                problems.append("line 1 is not MiniMax's exact first-frame (or last-frame) alignment line with <Picture 1>")
        else:
            m = _H3_FL2VA.match(first_line)
            if not m:
                problems.append("line 1 is not MiniMax's exact first-and-last-frame alignment line (Picture 1 ... Picture 2)")
        if image_count and m:
            last_shot, length = int(m.group(1)), float(m.group(2))
            # Local H3 renders the grid lengths; the online nodes take whole seconds (4-15),
            # which MiniMax's own L2VA example uses (6.00).
            on_grid = any(abs(length - s) < 0.006 for s in H3_LENGTHS)
            whole = length == int(length) and 4 <= length <= 15
            if not (on_grid or whole):
                nearest = min((s for s in H3_LENGTHS if s >= length - 0.006), default=H3_MAX_SECONDS)
                problems.append(f"line 1 says {length:.2f} s, which is not a length H3 renders; locally the "
                                f"next one up is {_h3_seconds(nearest)} s ({round(nearest * 24)} frames), online a "
                                f"whole number of seconds")
        if body and not re.match(r"\s*\[Shot 1\]", body):
            problems.append("integrated_multimodal_description does not start with [Shot 1]")
    else:
        body = _between(text, "detailed_description:", "overall_soundscape:")
        summary = _between(text, "summary:", "retention_analysis:").strip()
        if summary and not summary.startswith("["):
            problems.append("summary does not start with a [task type] prefix")
        if body and "[Shot 1]" not in body:
            problems.append("detailed_description has no [Shot 1]")
        definitions = _between(text, "subject_definitions:", "summary:")
        defined = {m.group(0) for m in _H3_LABEL.finditer(definitions)}
        used = {m.group(0) for m in _H3_LABEL.finditer(text)}
        undefined = sorted(l for l in used - defined if l.startswith("<Subject"))
        if undefined:
            problems.append("used but never defined in subject_definitions: " + ", ".join(undefined))
        if image_count:
            too_high = sorted({m.group(0) for m in _H3_LABEL.finditer(text)
                               if m.group(1) == "Picture" and int(m.group(2)) > image_count})
            if too_high:
                problems.append(f"refers to {', '.join(too_high)} but only {image_count} image(s) are connected")

    # Shots: numbered 1, 2, 3 ... with strictly increasing cut times
    shots = [(int(m.group(1)), (int(m.group(2)) * 60 + float(m.group(3))) if m.group(2) else None)
             for m in _H3_SHOT.finditer(body or "")]
    numbers = [n for n, _t in shots]
    if numbers and numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"shots are not numbered 1, 2, 3 ... ({numbers})")
    if last_shot is not None and numbers and last_shot != numbers[-1]:
        problems.append(f"line 1 aligns the frame with Shot {last_shot}, but the last shot is [Shot {numbers[-1]}]")
    times = [t for n, t in shots if n > 1]
    end = length or H3_MAX_SECONDS
    if any(t is None for t in times):
        problems.append("a shot after [Shot 1] has no 'At MM:SS.mmm,' cut time")
    elif times and (times != sorted(times) or len(set(times)) != len(times)):
        problems.append("shot cut times are not strictly increasing")
    elif times and times[-1] >= end:
        problems.append(f"a cut at {times[-1]:.3f} s is past the end of the video ({end:.2f} s)")
    elif times:
        starts = [0.0] + times + ([length] if length else [])
        short = [(a, b) for a, b in zip(starts, starts[1:]) if b - a < H3_MIN_SHOT - 1e-6]
        if short:
            a, b = short[0]
            problems.append(f"a shot from {a:.3f} s to {b:.3f} s lasts under {H3_MIN_SHOT} s; "
                            "a cut that short reads as a glitch")

    # Every later shot opens with one of MiniMax's cut phrases
    for m in re.finditer(r"\[Shot (\d+)\]\s*At \d+:\d+(?:\.\d+)?,?(.{0,160})", body or "", re.DOTALL):
        if int(m.group(1)) > 1 and not _H3_CUT.search(m.group(2)):
            problems.append(f"[Shot {m.group(1)}] does not open with one of MiniMax's cut phrases ("
                            + ", ".join(f"'{p}'" for p in H3_CUT_PHRASES) + ")")
            break

    # A time that doesn't start a new shot ("... At 12.000, she ...")
    all_times = len(re.findall(r"\bAt \d+(?::\d+)?\.\d+,", body or ""))
    shot_times = len(re.findall(r"\[Shot \d+\]\s*At \d+(?::\d+)?\.\d+,", body or ""))
    if all_times > shot_times:
        problems.append("a time appears inside a shot; only a new shot gets one ('[Shot N] At MM:SS.mmm,')")

    # Dialogue tags
    opens, closes = text.count("<d>"), text.count("</d>")
    if opens != closes:
        problems.append(f"unbalanced dialogue tags (<d> x{opens}, </d> x{closes})")
    if re.search(r"<d>(?!\s*\[)", text):
        problems.append("a <d> block does not start with a [Language] tag")

    # Music belongs in non_diegetic_music, not in the soundscape
    soundscape = _between(text, "overall_soundscape:", "non_diegetic_music:")
    if re.search(r"\b(music|score|soundtrack|melody)\b", soundscape, re.IGNORECASE):
        problems.append("overall_soundscape mentions music; background music belongs in non_diegetic_music")
    return problems


def warn_h3_problems(text, system_prompt, image_count=0):
    problems = check_h3_prompt(text, system_prompt, image_count)
    if problems:
        log.warning("GPromptsEnhanced: the MiniMax H3 prompt does not follow MiniMax's format:\n  - "
                    + "\n  - ".join(problems)
                    + "\nUsually the LLM ignored the system prompt (e.g. a Qwen-Image PE model). Use a "
                      "general instruct model or Claude, or check the text before rendering.")
    return problems


H3_TEMPERATURE = 0.3        # low: the binding constraint on an H3 prompt is keeping the format
LOADER_DEFAULT_TEMPERATURE = 1.0
_h3_temperature_logged = set()


def h3_temperature(temperature, system_prompt, fixed=False):
    """The temperature to sample with. With an H3 system prompt and the loader's temperature
    left at its default (1.0), use H3_TEMPERATURE; any other value on the loader is kept.
    fixed=True: the backend requires 1.0 (Anthropic with thinking on)."""
    if fixed or abs(temperature - LOADER_DEFAULT_TEMPERATURE) > 1e-6 or not h3_prompt_kind(system_prompt):
        return temperature
    key = hash(system_prompt)
    if key not in _h3_temperature_logged:
        _h3_temperature_logged.add(key)
        log.info(f"GPromptsEnhanced: MiniMax H3 system prompt with the loader's default temperature "
                 f"{LOADER_DEFAULT_TEMPERATURE}: using {H3_TEMPERATURE}, which keeps the model on H3's "
                 f"format. Set any other temperature on the loader to use that instead.")
    return H3_TEMPERATURE


def h3_retry_request(user_text, previous, problems):
    """The request for a second try: the problems, the failed answer, then the original
    request last (so a trailing keep-words note stays at the end)."""
    return ("Your previous answer to the request below does not follow the required format:\n- "
            + "\n- ".join(problems)
            + "\n\nYour previous answer was:\n\n" + (previous or "").strip()
            + "\n\nWrite the whole prompt again: fix these problems, keep everything else that was "
              "right, and output only the corrected prompt. The request:\n\n" + user_text)


def with_second_try(once, user_text, system_prompt, image_count):
    """'2nd try on error': once(text) -> EnhanceResult, called with warn=False by the
    enhancer. If the answer fails the H3 format check, ask once more with the problems
    listed, and keep whichever answer has fewer problems (the first on a tie)."""
    first = once(user_text)
    problems = check_h3_prompt(first.prompt, system_prompt, image_count)
    if not problems:
        return first
    check_interrupted()
    log.info(f"GPromptsEnhanced: 2nd try: the H3 prompt has {len(problems)} format problem(s); "
             f"sending them back to the LLM")
    log.debug("GPromptsEnhanced: 2nd try, first answer's problems:\n  - " + "\n  - ".join(problems))
    try:
        second = once(h3_retry_request(user_text, first.prompt, problems))
    except RuntimeError as e:
        log.warning(f"GPromptsEnhanced: 2nd try failed ({e}); keeping the first answer")
        warn_h3_problems(first.prompt, system_prompt, image_count)
        return first
    problems2 = check_h3_prompt(second.prompt, system_prompt, image_count)
    use_second = len(problems2) < len(problems)
    log.info(f"GPromptsEnhanced: 2nd try: {len(problems)} -> {len(problems2)} problem(s); using the "
             + ("second answer" if use_second else "first answer"))
    chosen = second if use_second else first
    warn_h3_problems(chosen.prompt, system_prompt, image_count)
    return chosen


def finalize_result(raw, finish, thinking, contract, max_new_tokens, system_prompt="", image_count=0,
                    warn=True):
    """Check and parse a finished generation (any backend). warn=False: leave the H3
    format warning to the caller (with_second_try)."""
    if thinking and "</think>" not in raw and finish == "length":
        raise RuntimeError(
            f"The LLM used all {max_new_tokens} tokens while still thinking. "
            f"Set plan_tokens (e.g. 800) or increase max_new_tokens.")
    result = parse_enhancer_output(raw)
    if not result.prompt:
        raise RuntimeError(f"The LLM returned no prompt. Raw output:\n{raw[-2000:]}")
    if contract and not result.parse_ok:
        answer = raw.rpartition("</think>")[2].strip() if "</think>" in raw else raw.strip()
        excerpt = answer if len(answer) <= 700 else answer[:500] + " ... " + answer[-150:]
        log.warning("GPromptsEnhanced: the answer was not the expected JSON; using the raw "
              "answer text. Usual causes: the model declined or answered in prose, a base "
              "model instead of a PE model, the wrong system prompt for the model, or output "
              "cut off by max_new_tokens or the context length. The answer was:\n"
              + excerpt)
    log.debug(f"GPromptsEnhanced: parse_ok={result.parse_ok}, wh_ratio={result.wh_ratio!r}")
    if not contract and warn:
        warn_h3_problems(result.prompt, system_prompt, image_count)
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


# ----------------------------------------------------------------------------
# Reference images (image edit)
# ----------------------------------------------------------------------------
LLM_IMAGE_MEGAPIXELS = ["0.25", "0.5", "1.0", "2.0"]
DEFAULT_LLM_IMAGE_MP = "0.5"
MAX_IMAGES = 16          # same as Text Encode Qwen Image 2.1
LLM_PIXELS_PER_TOKEN = 32 * 32   # Qwen vision: 16px patches merged 2x2


def images_from_autogrow(images):
    """Autogrow dict {'image_1': IMAGE, ...} -> list of [1,H,W,C] tensors, in
    socket order, skipping empty sockets - the same order Text Encode Qwen
    Image 2.1 uses, so <image1> means the same picture to both nodes."""
    if not images:
        return []

    def index(name):
        digits = re.findall(r"\d+", name)
        return int(digits[-1]) if digits else 0
    return [images[k][:1] for k in sorted(images, key=index) if images[k] is not None]


def images_fingerprint(images):
    """Stable id for a set of images (cache key: a new image means a new LLM call)."""
    if not images:
        return ""
    h = hashlib.blake2b(digest_size=16)
    for t in images:
        arr = t.detach().cpu().contiguous().numpy()
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


_PREPARED = {}          # (image fingerprint, megapixels) -> (jpeg base64, token estimate)
_PREPARED_MAX = 32


def prepared_image(image, megapixels):
    """Downscaled JPEG (base64) of a reference image and its approximate token
    count, cached: the same reference is sent on every run of a batch, and
    identical bytes also let a server reuse its own cache of the image."""
    key = (images_fingerprint([image]), str(megapixels))
    hit = _PREPARED.get(key)
    if hit is None:
        pil = image_for_llm(image, megapixels)
        hit = (pil_to_jpeg_b64(pil), estimate_image_tokens(pil))
        if len(_PREPARED) >= _PREPARED_MAX:
            _PREPARED.pop(next(iter(_PREPARED)))
        _PREPARED[key] = hit
    return hit


def image_for_llm(image, megapixels):
    """IMAGE tensor [1,H,W,C] -> RGB PIL image with at most ~megapixels pixels
    (never upscaled). Alpha is composited over white, as the encode node does
    for its vision tower."""
    arr = image[0].detach().cpu().float().numpy()
    if arr.shape[-1] == 4:
        arr = arr[..., :3] * arr[..., 3:] + (1.0 - arr[..., 3:])
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = (np.clip(arr[..., :3], 0.0, 1.0) * 255.0).round().astype(np.uint8)
    pil = PILImage.fromarray(arr, "RGB")
    w, h = pil.size
    try:
        target = float(megapixels) * 1024 * 1024
    except (TypeError, ValueError):
        target = float(DEFAULT_LLM_IMAGE_MP) * 1024 * 1024
    if w * h > target:
        scale = math.sqrt(target / (w * h))
        pil = pil.resize((max(32, round(w * scale)), max(32, round(h * scale))), PILImage.LANCZOS)
    return pil


def pil_to_jpeg_b64(pil, quality=92):
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def estimate_image_tokens(pil):
    w, h = pil.size
    return math.ceil(w / 32) * math.ceil(h / 32) + 2


def follow_index(ratio_follow, count):
    """'image2', '<image2>', '2', 'Picture 2' -> 1; None if it names no connected image."""
    m = re.search(r"(\d+)", ratio_follow or "")
    if m:
        i = int(m.group(1)) - 1
        if 0 <= i < count:
            return i
    return None


def edit_canvas(result, images, megapixels):
    """Width/height for an edit: the shape of the image the model says to
    follow (ratio_follow), else its wh_ratio, else the first image's shape."""
    i = follow_index(result.ratio_follow, len(images))
    if i is None and not ratio_to_pair(result.wh_ratio):
        i = 0
    if i is not None:
        h, w = int(images[i].shape[1]), int(images[i].shape[2])
        return canvas_size(f"{w}:{h}", megapixels)
    return canvas_size(result.wh_ratio, megapixels)


def add_trigger_words(prompt, trigger_words):
    """'ohwx woman, tkb_style' + prompt -> 'ohwx woman, tkb_style, <prompt>'.
    Comma-separated entries the prompt already contains (whole words, any case)
    are not added again."""
    entries = [t.strip() for t in (trigger_words or "").split(",") if t.strip()]
    missing = []
    for t in entries:
        if t.lower() in (m.lower() for m in missing):
            continue
        if not re.search(_anchor_pattern(t), prompt or "", re.IGNORECASE):
            missing.append(t)
    if not missing:
        return prompt
    prefix = ", ".join(missing)
    prompt = (prompt or "").lstrip()
    return f"{prefix}, {prompt}" if prompt else prefix


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

SECOND_TRY_TIP = ("MiniMax H3 system prompts only: when the answer fails the H3 format check, send "
                  "the problems back to the LLM once and keep whichever answer has fewer problems. "
                  "Costs a second LLM call, and only when the first answer fails.")


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
                io.Boolean.Input("second_try", display_name="2nd try on error", default=False,
                                 tooltip=SECOND_TRY_TIP),
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
                plan_tokens, keep_loaded, flash_attn=True, second_try=False) -> io.NodeOutput:
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
            second_try=second_try,
        )
        if not keep_loaded:
            unload_llm()
        return io.NodeOutput(enhancer)



NO_MMPROJ = "(no mmproj .gguf files in models/LLM)"


class GPromptEnhancerLoaderGGUFEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        ggufs = list_llm_files(MODEL_EXTENSIONS)
        models = [f for f in ggufs if "mmproj" not in f.lower()] or [NO_MODELS]
        mmprojs = [f for f in ggufs if "mmproj" in f.lower()] or ggufs or [NO_MMPROJ]
        prompts = list_llm_files(SYSTEM_PROMPT_EXTENSIONS) or [NO_SYSTEM_PROMPTS]
        return io.Schema(
            node_id="GPromptEnhancerLoaderGGUFEdit",
            display_name="Prompt Enhancer Loader (GGUF, edit)",
            category="gprompts/enhancer",
            description=(
                "Image-edit prompt rewriter (e.g. Qwen-Image-2.1 PE-I2I) as a GGUF plus its "
                "vision projector (mmproj), from models/LLM. The rewriter sees the reference "
                "images connected to 'Dynamic Prompts with Enhancer'. Runs in-process with "
                "llama-cpp-python."
            ),
            search_aliases=["edit prompt enhancer", "i2i prompt rewriter", "mmproj"],
            inputs=[
                io.Combo.Input("model", options=models,
                               tooltip="Edit rewriter GGUF in models/LLM (e.g. pe_i2i_heretic-Q4_K_M.gguf)."),
                io.Combo.Input("mmproj", options=mmprojs,
                               tooltip="The vision projector shipped with the model "
                                       "(…mmproj….gguf). Lets the LLM see the images."),
                io.Combo.Input("system_prompt", options=prompts,
                               tooltip="The edit model's own system prompt file (about 18 KB for "
                                       "Qwen-Image-2.1 PE-I2I) - not the text-to-image one."),
                io.Int.Input("context_length", default=16384, min=2048, max=262144, step=1024,
                             tooltip="Must fit the system prompt, the images and the answer."),
                io.Int.Input("max_new_tokens", default=8192, min=256, max=65536, step=256),
                io.Int.Input("gpu_layers", default=-1, min=-1, max=999,
                             tooltip="Layers on the GPU. -1 = all, 0 = CPU only."),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=20, min=0, max=200),
                io.Float.Input("presence_penalty", default=0.0, min=0.0, max=2.0, step=0.05,
                               tooltip="Qwen's settings use 0 for the edit rewriter (1.5 is for "
                                       "text-to-image)."),
                io.Boolean.Input("enable_thinking", default=True),
                io.Int.Input("plan_tokens", default=800, min=-1, max=16384, step=100,
                             tooltip="Cap on thinking before the plan is handed back for the "
                                     "answer. -1 = no cap. Edits with several images plan long; "
                                     "the cap matters more here."),
                io.Combo.Input("llm_image_megapixels", options=LLM_IMAGE_MEGAPIXELS,
                               default=DEFAULT_LLM_IMAGE_MP,
                               tooltip="Images are downscaled to about this size before the LLM "
                                       "sees them (~512 tokens each at 0.5). The encode node still "
                                       "gets the originals."),
                io.Boolean.Input("keep_loaded", default=True),
                io.Boolean.Input("flash_attn", default=True, advanced=True),
                io.Boolean.Input("second_try", display_name="2nd try on error", default=False,
                                 tooltip=SECOND_TRY_TIP),
            ],
            outputs=[PromptEnhancerType.Output("enhancer", display_name="enhancer")],
        )

    @classmethod
    def validate_inputs(cls, model, mmproj, system_prompt):
        if model == NO_MODELS or resolve_llm_file(model) is None:
            return f"GGUF model not found in models/LLM: {model}"
        if mmproj == NO_MMPROJ or resolve_llm_file(mmproj) is None:
            return f"mmproj file not found in models/LLM: {mmproj}"
        if mmproj == model:
            return "mmproj must be the vision projector file, not the model itself"
        if system_prompt == NO_SYSTEM_PROMPTS or resolve_llm_file(system_prompt) is None:
            return f"System prompt file not found in models/LLM: {system_prompt}"
        return True

    @classmethod
    def execute(cls, model, mmproj, system_prompt, context_length, max_new_tokens, gpu_layers,
                temperature, top_p, top_k, presence_penalty, enable_thinking, plan_tokens,
                llm_image_megapixels, keep_loaded, flash_attn=True, second_try=False) -> io.NodeOutput:
        enhancer = GGUFPromptEnhancer(
            model_path=resolve_llm_file(model), system_prompt_path=resolve_llm_file(system_prompt),
            context_length=context_length, max_new_tokens=max_new_tokens, gpu_layers=gpu_layers,
            temperature=temperature, top_p=top_p, top_k=top_k, presence_penalty=presence_penalty,
            enable_thinking=enable_thinking, plan_tokens=plan_tokens, keep_loaded=keep_loaded,
            flash_attn=flash_attn, mmproj_path=resolve_llm_file(mmproj),
            image_megapixels=llm_image_megapixels, task=TASK_EDIT, second_try=second_try)
        if not keep_loaded:
            unload_llm()
        return io.NodeOutput(enhancer)


class GPromptEnhancerPair(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GPromptEnhancerPair",
            display_name="Enhancer Pair",
            category="gprompts/enhancer",
            description=(
                "Combines a text-to-image enhancer and an image-edit enhancer into one, so a "
                "workflow can do both: runs with reference images use the edit one, runs "
                "without use the text-to-image one. Either can be a GGUF or API loader."
            ),
            search_aliases=["enhancer switch", "t2i i2i enhancer"],
            inputs=[
                PromptEnhancerType.Input("text_enhancer", optional=True,
                                         tooltip="A text-to-image loader."),
                PromptEnhancerType.Input("edit_enhancer", optional=True,
                                         tooltip="An image-edit loader."),
            ],
            outputs=[PromptEnhancerType.Output("enhancer", display_name="enhancer")],
        )

    @classmethod
    def execute(cls, text_enhancer=None, edit_enhancer=None) -> io.NodeOutput:
        if text_enhancer is None and edit_enhancer is None:
            raise RuntimeError("Enhancer Pair: connect at least one enhancer.")
        if text_enhancer is not None and getattr(text_enhancer, "task", TASK_T2I) != TASK_T2I:
            raise RuntimeError(f"Enhancer Pair: text_enhancer is a {text_enhancer.task} loader; "
                               f"it needs a text-to-image one.")
        if edit_enhancer is not None and getattr(edit_enhancer, "task", TASK_T2I) != TASK_EDIT:
            raise RuntimeError(f"Enhancer Pair: edit_enhancer is a {edit_enhancer.task} loader; "
                               f"it needs an image-edit one.")
        return io.NodeOutput(EnhancerPair(text_enhancer, edit_enhancer))


# Per-node state. V3 nodes are stateless classes, so the dynamic prompt engine
# (iteration counter, sequential combinations, wildcard cache) and the cached
# LLM result live here, keyed by the node's unique_id.
_NODE_STATE = {}
_WARNED = {}


class _Prefetch:
    """The next run's rewrite, generated on a worker thread."""

    def __init__(self, iteration, expanded, values, seed, work):
        self.iteration, self.expanded, self.values, self.seed = iteration, expanded, values, seed
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.result = self.error = None
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, args=(work,), daemon=True,
                                       name="gprompts-prefetch")
        self.thread.start()

    def _run(self, work):
        _BACKGROUND.cancel = self.cancel
        try:
            self.result = work()
        except BaseException as e:          # reported by whoever collects it
            self.error = e
        finally:
            _BACKGROUND.cancel = None
            self.done.set()

    def stop(self):
        """Not needed any more. Don't wait: a local model stops at its next token
        (and holds its lock until then); an API request stops at its next chunk."""
        self.cancel.set()

    def collect(self):
        """Wait for the result (None if it failed); the user's Cancel still works."""
        while not self.done.wait(0.1):
            if model_management.processing_interrupted():
                self.stop()
                model_management.throw_exception_if_processing_interrupted()
        if self.error is not None:
            if not isinstance(self.error, PrefetchCancelled):
                log.warning(f"GPromptsEnhanced: prefetched rewrite failed ({self.error}); retrying now")
            return None
        return self.result


def _stop_prefetch(state):
    pf = state.pop("prefetch", None) if state else None
    if pf is not None:
        pf.stop()


ENHANCE_EVERY_RUN = "every run"
ENHANCE_ONCE = "once, then substitute"
ENHANCE_MODES = [ENHANCE_EVERY_RUN, ENHANCE_ONCE]
PREFETCH_AUTO, PREFETCH_ON, PREFETCH_OFF = "auto", "on", "off"
PREFETCH_MODES = [PREFETCH_AUTO, PREFETCH_ON, PREFETCH_OFF]


class GPromptsEnhanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        styles = list(DynamicPromptEngine.DELIM_STYLES.keys())
        return io.Schema(
            node_id="GPromptsEnhanced",
            display_name="Dynamic Prompts with Enhancer",
            category="gprompts/enhancer",
            description=(
                "Dynamic Prompts plus an LLM prompt enhancer. 'every run' sends each run's "
                "expansion to the LLM. 'once, then substitute' rewrites only the first "
                "expansion and swaps later runs' chosen values into that rewrite: one LLM call "
                "for the whole batch, but it breaks when the LLM rewords a value (dinosaur -> "
                "sauropod). Changing the text, delimiter style, mode, enhancer settings or "
                "reference images starts the batch over; the seed does not."
            ),
            search_aliases=["gprompts enhancer", "dynamic prompts llm", "prompt enhancer"],
            inputs=[
                PromptEnhancerType.Input("enhancer",
                                         tooltip="From a Prompt Enhancer Loader (GGUF or API; "
                                                 "text-to-image, or edit when images are "
                                                 "connected) or an Enhancer Pair."),
                io.String.Input("text", multiline=True, default="",
                                tooltip="Dynamic prompt, same syntax as Dynamic Prompts: {a|b} random, "
                                        "{{a|b}} sequential, __wildcard__, registers {{0 a|b}} / {{0}}."),
                io.String.Input("trigger_words", default="", multiline=False,
                                tooltip="LoRA trigger words, e.g. 'ohwx woman, tkb_style'. Added "
                                        "to the start of the final prompt exactly as typed; never "
                                        "sent to the LLM, so it can't drop or change them. Leave "
                                        "them out of the text. Words the prompt already "
                                        "contains are not added twice."),
                io.Combo.Input("delimiter_style", options=styles,
                               default=DynamicPromptEngine.DEFAULT_DELIM_STYLE,
                               tooltip="'curly { }' or 'angle < >' (for JSON prompts)."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             tooltip="Seed for random blocks (seed + run number, as in Dynamic Prompts) "
                                     "and for the batch's LLM call. 0 = unseeded. Changing it does "
                                     "not restart the batch."),
                io.Boolean.Input("preserve_dynamic_words", default=True,
                                 tooltip="'once, then substitute' only: ask the LLM to keep the "
                                         "first expansion's chosen words verbatim, so later "
                                         "expansions can be swapped in. If the LLM rewords a word "
                                         "anyway, that variation is skipped (see the console)."),
                io.Combo.Input("target_megapixels", options=MEGAPIXEL_OPTIONS,
                               default=DEFAULT_MEGAPIXELS,
                               tooltip="Pixel budget for the width/height outputs, at the LLM's "
                                       "recommended aspect ratio (1:1 if it gave none; for edits, "
                                       "the shape of the image it names in ratio_follow). 1.0 = "
                                       "1024x1024-sized; 4.0 = Qwen-Image 2.1's native 2K "
                                       "(2048x2048-sized). Sides are multiples of 16."),
                io.Combo.Input("enhance", options=ENHANCE_MODES, default=ENHANCE_EVERY_RUN,
                               tooltip="'every run': the LLM rewrites each run's expansion (one "
                                       "LLM call per image; always correct). 'once, then "
                                       "substitute': one LLM call per batch, later runs swap "
                                       "their values into the first rewrite (fast, but fails "
                                       "when the LLM rewords a value, e.g. dinosaur -> sauropod)."),
                io.Combo.Input("prefetch_next", options=PREFETCH_MODES, default=PREFETCH_AUTO,
                               tooltip="'every run' only: as soon as a run's prompt is out, start "
                                       "rewriting the next run's prompt in the background, so "
                                       "the LLM works while the image renders. auto = on for an "
                                       "API on another machine and a CPU-only GGUF (gpu_layers "
                                       "0); off for an API server on this machine (it shares the "
                                       "GPU). on = also for a server on this machine. A GGUF on "
                                       "the GPU never prefetches (it can crash ComfyUI). After "
                                       "the last run of a batch one rewrite goes unused (a "
                                       "wasted call on paid APIs)."),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, MAX_IMAGES + 1)],
                        min=0,
                    ),
                    tooltip="Reference images for image editing, shown to the LLM. Wire the same "
                            "images, in the same order, to Text Encode Qwen Image 2.1. Refer to "
                            "them in the text as <image1>, <image2>, ...",
                ),
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
                io.String.Output("ratio_follow", display_name="ratio_follow",
                                 tooltip="Edit: which reference image's shape the output keeps "
                                         "(e.g. image1), as chosen by the LLM. Empty otherwise."),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Every queue advances the iteration, like Dynamic Prompts: always run.
        return float("nan")

    @staticmethod
    def _prefetch_wanted(mode, enhancer, has_images):
        if mode == PREFETCH_OFF:
            return False
        if enhancer.gpu_in_process(has_images):
            # Even when forced on: a GPU error in llama.cpp aborts ComfyUI.
            if mode == PREFETCH_ON and not _WARNED.get("gpu_prefetch"):
                _WARNED["gpu_prefetch"] = True
                log.warning("GPromptsEnhanced: prefetch_next is on, but the GGUF runs on this "
                      "machine's GPU; running it in the background next to the image model "
                      "can crash ComfyUI, so prefetch is skipped. Set gpu_layers to 0 (CPU) "
                      "or use an API loader to prefetch.")
            return False
        if mode == PREFETCH_ON:
            return True
        return not enhancer.uses_local_gpu(has_images)

    @staticmethod
    def _start_prefetch(engine, text, seed, enhancer, refs):
        """Expand the next run now (without disturbing Python's random state) and
        start its rewrite on a worker thread."""
        iteration = engine.current_iteration
        saved = random.getstate()
        try:
            if seed > 0:
                random.seed(seed + iteration)
            expanded = engine.parse_dynamic_prompt(text)
            values = list(engine.last_values)
        finally:
            random.setstate(saved)
        llm_seed = seed + iteration if seed > 0 else seed
        refs = list(refs)
        if refs:
            work = lambda: enhancer.enhance(expanded, llm_seed, images=refs)
        else:
            work = lambda: enhancer.enhance(expanded, llm_seed)
        log.debug(f"GPromptsEnhanced: prefetching iteration {iteration}: {expanded}")
        return _Prefetch(iteration, expanded, values, seed, work)

    @classmethod
    def execute(cls, enhancer, text, delimiter_style, seed, trigger_words="",
                preserve_dynamic_words=True, target_megapixels=DEFAULT_MEGAPIXELS,
                enhance=ENHANCE_EVERY_RUN, prefetch_next=PREFETCH_AUTO,
                images: io.Autogrow.Type = None) -> io.NodeOutput:
        unique_id = cls.hidden.unique_id
        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo or {}
        workflow_id = extra_pnginfo.get("workflow", {}).get("id")

        if delimiter_style not in DynamicPromptEngine.DELIM_STYLES:
            delimiter_style = DynamicPromptEngine.DEFAULT_DELIM_STYLE
        if enhance not in ENHANCE_MODES:
            enhance = ENHANCE_EVERY_RUN
        every_run = enhance == ENHANCE_EVERY_RUN
        refs = images_from_autogrow(images)
        enhancer.check_task(bool(refs))
        # Like Dynamic Prompts, the seed is not part of the key: it only drives
        # random blocks (seed + iteration). A seed set to randomize must not
        # restart the batch, or every run would be the first expansion.
        key = (text, delimiter_style, enhance, bool(preserve_dynamic_words) and not every_run,
               enhancer.cache_key_for(bool(refs)), images_fingerprint(refs))
        state = _NODE_STATE.get(unique_id)
        if state is None or state["key"] != key:
            _stop_prefetch(state)
            engine = DynamicPromptEngine()
            engine.previous_text = text
            engine.previous_delim_style = delimiter_style
            state = {"key": key, "engine": engine, "anchors": None, "result": None}
            _NODE_STATE[unique_id] = state
        engine = state["engine"]
        engine.delims = engine.DELIM_STYLES[delimiter_style]

        iteration = engine.current_iteration
        pf = state.pop("prefetch", None)
        if pf is not None and pf.iteration == iteration and seed == 0 and pf.seed == 0:
            # Unseeded: the prefetch's random draw is as good as a new one, so use it.
            expanded, values = pf.expanded, list(pf.values)
        else:
            if seed > 0:
                random.seed(seed + iteration)
            expanded = engine.parse_dynamic_prompt(text)
            values = list(engine.last_values)
        if pf is not None and (pf.iteration != iteration or pf.expanded != expanded
                               or not every_run):
            pf.stop()               # predicted a different prompt (e.g. the seed changed)
            pf = None

        prefetched = None
        if pf is not None:
            waited = time.perf_counter()
            prefetched = pf.collect()
            if prefetched is not None:
                log.info(f"GPromptsEnhanced: using the rewrite prefetched during the last run "
                      f"(waited {time.perf_counter() - waited:.1f}s of "
                      f"{time.perf_counter() - pf.started:.1f}s)")

        if prefetched is not None:
            state["result"] = prefetched
            state["anchors"] = values
            final = prefetched.prompt
        elif every_run or state["result"] is None:
            user_text = expanded
            if preserve_dynamic_words and not every_run:
                user_text += preserve_hint(values)
            # every run: seed + iteration, so a repeated combination still gets a fresh,
            # reproducible rewrite (0 stays unseeded)
            llm_seed = seed + iteration if (every_run and seed > 0) else seed
            log.debug(f"GPromptsEnhanced: enhancing (iteration {iteration}, {enhance}): {user_text}")
            if refs:
                log.debug(f"GPromptsEnhanced: with {len(refs)} reference image(s)")
                state["result"] = enhancer.enhance(user_text, llm_seed, images=refs)
            else:
                state["result"] = enhancer.enhance(user_text, llm_seed)
            state["anchors"] = values
            final = state["result"].prompt
        else:
            final, warnings = substitute_values(state["result"].prompt, state["anchors"], values)
            for w in warnings:
                log.warning(f"GPromptsEnhanced (node {unique_id}, iteration {iteration}): {w}")

        # LoRA trigger words: in front, outside the LLM's reach. Not part of the
        # batch key, so changing them never costs a new LLM call.
        final = add_trigger_words(final, trigger_words)
        expanded = add_trigger_words(expanded, trigger_words)

        engine.current_iteration += 1
        result = state["result"]
        if every_run and cls._prefetch_wanted(prefetch_next, enhancer, bool(refs)):
            state["prefetch"] = cls._start_prefetch(engine, text, seed, enhancer, refs)

        if workflow_id:
            promtpForId[workflow_id] = final
        try:
            PromptServer.instance.send_sync("gprompts_executed",
                                            {"node_id": unique_id, "result": final})
        except Exception as e:
            log.debug(f"GPromptsEnhanced: could not update UI: {e}")
        if prompt is not None and unique_id in prompt:
            node = prompt[unique_id]
            node.setdefault("inputs", {})["computed_prompt"] = final
            meta = node.get("_meta", {})
            meta["computed_prompt"] = final
            meta["dynamic_prompt"] = expanded
            node["_meta"] = meta

        if refs:
            width, height = edit_canvas(result, refs, target_megapixels)
        else:
            width, height = canvas_size(result.wh_ratio, target_megapixels)
        return io.NodeOutput(final, expanded, final, seed, text, result.wh_ratio,
                             width, height, square_resolution(target_megapixels),
                             result.ratio_follow)


ENHANCED_NODES = {
    "GPromptEnhancerLoaderGGUF": GPromptEnhancerLoaderGGUF,
    "GPromptEnhancerLoaderGGUFEdit": GPromptEnhancerLoaderGGUFEdit,
    "GPromptEnhancerPair": GPromptEnhancerPair,
    "GPromptsEnhanced": GPromptsEnhanced,
}
ENHANCED_NODE_DISPLAY_NAME_MAPPINGS = {
    "GPromptEnhancerLoaderGGUF": "Prompt Enhancer Loader (GGUF)",
    "GPromptEnhancerLoaderGGUFEdit": "Prompt Enhancer Loader (GGUF, edit)",
    "GPromptEnhancerPair": "Enhancer Pair",
    "GPromptsEnhanced": "Dynamic Prompts with Enhancer",
}
