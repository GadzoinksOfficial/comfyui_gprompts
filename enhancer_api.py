"""
Prompt Enhancer Loader (API)
From Gadzoinks Official
https://github.com/GadzoinksOfficial/comfyui_gprompts

The API counterpart of "Prompt Enhancer Loader (GGUF)": the same GPROMPT_ENHANCER
output, so "Dynamic Prompts with Enhancer" works unchanged, but the LLM runs
behind an HTTP API: a local server (Ollama, llama-server, vLLM, SGLang,
LM Studio, ...) or a hosted provider (OpenAI, OpenRouter, DashScope, Anthropic,
or anything else with an OpenAI-compatible endpoint).

Five request styles:

  OpenAI-compatible chat       /chat/completions. Works nearly everywhere.
                               Thinking on/off is sent in whatever field the
                               provider understands (auto-detected, or chosen).
  OpenAI-compatible completions (raw prompt)
                               /completions with the full ChatML prompt built
                               here - same prompt, <think> prefill, plan cap and
                               early stop as the GGUF loader. vLLM, SGLang,
                               llama-server, LM Studio.
  Ollama chat                  /api/chat (native: think, keep_alive, num_ctx).
  Ollama generate (raw prompt) /api/generate with raw=true: GGUF-loader parity
                               on Ollama.
  Anthropic messages           /v1/messages.

Portability: providers reject different parameters (top_k, min_p, seed,
max_tokens vs max_completion_tokens, thinking fields...). When a request fails
with HTTP 400/422 and the error names a parameter this node sent, the node
drops (or renames) it and retries, and remembers that for the server + model.

API keys never go on the node (widget values are saved into workflows and
image metadata). They live in Settings > Gadzoinks > LLM; the node only names
which one to use. No extra Python packages: plain urllib.
"""
import functools
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import comfy.model_management as model_management
import comfy.utils
from comfy_api.latest import io

from .common import get_logger
from .comfyui_gprompts import get_setting
from .enhanced_prompt import (
    PromptEnhancerType, PromptEnhancerBase, PE_CONTRACT_KEY, STOP_STRINGS,
    answer_complete, run_raw_generation, finalize_result,
    list_llm_files, resolve_llm_file, SYSTEM_PROMPT_EXTENSIONS,
    TASK_T2I, TASK_EDIT, LLM_IMAGE_MEGAPIXELS, DEFAULT_LLM_IMAGE_MP,
    image_for_llm, pil_to_jpeg_b64, prepared_image,
    interrupted, check_interrupted, progress_bar, h3_temperature,
    with_second_try, SECOND_TRY_TIP,
)

log = get_logger("enhancer.api")

API_OPENAI_CHAT = "OpenAI-compatible chat"
API_OPENAI_RAW = "OpenAI-compatible completions (raw prompt)"
API_OLLAMA_CHAT = "Ollama chat"
API_OLLAMA_RAW = "Ollama generate (raw prompt)"
API_ANTHROPIC = "Anthropic messages"
API_STYLES = [API_OPENAI_CHAT, API_OPENAI_RAW, API_OLLAMA_CHAT, API_OLLAMA_RAW, API_ANTHROPIC]
RAW_STYLES = {API_OPENAI_RAW, API_OLLAMA_RAW}
CHAT_STYLES = [API_OPENAI_CHAT, API_OLLAMA_CHAT, API_ANTHROPIC]   # the ones that can carry images
OLLAMA_STYLES = {API_OLLAMA_CHAT, API_OLLAMA_RAW}

DEFAULT_BASE_URLS = {
    API_OPENAI_CHAT: "http://127.0.0.1:11434/v1",      # Ollama's OpenAI endpoint
    API_OPENAI_RAW: "http://127.0.0.1:8080/v1",        # llama-server default port
    API_OLLAMA_CHAT: "http://127.0.0.1:11434",
    API_OLLAMA_RAW: "http://127.0.0.1:11434",
    API_ANTHROPIC: "https://api.anthropic.com",
}

THINKING_MODES = ["model default", "on", "off"]
THINKING_FIELDS = ["auto", "chat_template_kwargs", "enable_thinking", "reasoning_effort",
                   "reasoning", "think", "none"]
NO_SYSTEM_PROMPT = "(none)"
ANTHROPIC_VERSION = "2023-06-01"


# ----------------------------------------------------------------------------
# API keys (Settings > Gadzoinks > LLM)
# ----------------------------------------------------------------------------
_KEY_LOOKUP = {"missing_since": 0.0}


def parse_named_keys(text):
    """'openrouter=sk-1; dashscope=sk-2' (also newline/comma separated) -> dict"""
    keys = {}
    for part in re.split(r"[;\n,]", text or ""):
        name, sep, value = part.partition("=")
        if sep and name.strip() and value.strip():
            keys[name.strip()] = value.strip()
    return keys


def resolve_api_key(name):
    """''          -> the default key from settings (may be empty: local servers)
       'none'      -> no key
       'env:VAR'   -> environment variable VAR
       'openrouter'-> the named key from settings"""
    name = (name or "").strip()
    if name.lower() == "none":
        return ""
    if name.lower().startswith("env:"):
        var = name[4:].strip()
        value = os.environ.get(var, "")
        if not value:
            raise RuntimeError(f"Environment variable {var} is not set (api_key_name '{name}').")
        return value
    # Don't stall every run waiting on a key that was never set (local servers).
    recently_missing = time.time() - _KEY_LOOKUP["missing_since"] < 60
    if not name:
        value = get_setting("llm_api_key", wait_seconds=0 if recently_missing else 2.0) or ""
        _KEY_LOOKUP["missing_since"] = 0.0 if value else time.time()
        return value
    named = parse_named_keys(get_setting("llm_api_keys", wait_seconds=2.0) or "")
    if name in named:
        return named[name]
    raise RuntimeError(
        f"No API key named '{name}'. Add it in Settings > Gadzoinks > LLM > "
        f"'LLM API keys, named' as {name}=your-key (separate several with ';'). "
        f"Known names: {', '.join(sorted(named)) or 'none'}.")


# ----------------------------------------------------------------------------
# HTTP with parameter adaptation
# ----------------------------------------------------------------------------
class HTTPStatusError(RuntimeError):
    def __init__(self, status, body, url):
        super().__init__(f"{url} returned HTTP {status}: {body[:800]}")
        self.status, self.body, self.url = status, body, url


# (url, model) -> {"drop": set(), "rename": {old: new}} learned from 400s
_LEARNED = {}
RETRYABLE = {429, 500, 502, 503, 504}


def _open(url, payload, headers, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        err = HTTPStatusError(e.code, body, url)
        err.retry_after = e.headers.get("Retry-After") if e.headers else None
        raise err
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}. Is the server running, and is "
                           f"base_url right?") from e


# Errors about a thinking field often name the feature, not the field
# (Ollama: "... does not support thinking").
THINKING_KEYS = {"think", "thinking", "enable_thinking", "reasoning", "reasoning_effort",
                 "chat_template_kwargs"}
_MENTION_ALIASES = {
    "think": ("thinking",), "enable_thinking": ("thinking",), "reasoning": ("thinking",),
    "reasoning_effort": ("reasoning",), "chat_template_kwargs": ("enable_thinking",),
}


def _mentions(body, name):
    for word in (name,) + _MENTION_ALIASES.get(name, ()):
        if re.search(r"(?<![\w.])" + re.escape(word) + r"(?![\w])", body):
            return True
    return False


def _apply_learned(payload, learned):
    payload = dict(payload)
    for old, new in learned["rename"].items():
        if old in payload:
            payload[new] = payload.pop(old)
    for key in learned["drop"]:
        payload.pop(key, None)
    return payload


def post_adaptive(url, payload, headers, timeout, model, optional, renames=None):
    """POST, adapting to what this server accepts. `optional` lists top-level
    keys that may be dropped; `renames` maps key -> alternative name to try
    when the error mentions the alternative (max_tokens -> max_completion_tokens).
    Returns the open response."""
    renames = renames or {}
    learned = _LEARNED.setdefault((url, model), {"drop": set(), "rename": {}})
    body = _apply_learned(payload, learned)
    transient = 0
    for _attempt in range(8):
        try:
            return _open(url, body, headers, timeout)
        except HTTPStatusError as e:
            if e.status in RETRYABLE and transient < 2:
                transient += 1
                try:
                    wait = min(float(e.retry_after), 30.0)
                except (TypeError, ValueError):
                    wait = 2.0 * transient
                log.warning(f"GPromptsEnhanced: HTTP {e.status} from {url}, retrying in {wait:.0f}s")
                time.sleep(wait)
                check_interrupted()
                continue
            if e.status not in (400, 422):
                raise
            changed = False
            for old, new in renames.items():
                if old in body and _mentions(e.body, new):
                    body[new] = body.pop(old)
                    learned["rename"][old] = new
                    changed = True
            if not changed:
                named = [k for k in body if k in optional and _mentions(e.body, k)]
                # "temperature may only be 1 when thinking is enabled" names both:
                # drop the ordinary parameter and keep thinking if possible.
                plain = [k for k in named if k not in THINKING_KEYS]
                for key in plain or named:
                    body.pop(key)
                    learned["drop"].add(key)
                    changed = True
                    log.info(f"GPromptsEnhanced: {url} rejected '{key}'; retrying without it")
            if not changed:
                raise
    raise RuntimeError(f"{url}: gave up adapting the request")


def iter_stream(resp, kind):
    """Yield JSON events from a streaming response. kind: 'sse' or 'ndjson'.
    A plain JSON body (server ignored stream=true) is yielded as one event."""
    ctype = (resp.headers.get("Content-Type") or "").lower()
    try:
        if ctype.startswith("application/json"):
            yield json.loads(resp.read().decode("utf-8"))
            return
        for raw in resp:
            if interrupted():
                return
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if kind == "sse" or line.startswith("data:"):
                if not line.startswith("data:"):
                    continue          # event:, id:, :comment
                line = line[5:].strip()
                if line == "[DONE]":
                    return
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("error"):
                err = event["error"]
                raise RuntimeError(f"Server error: {err.get('message', err) if isinstance(err, dict) else err}")
            if isinstance(event, dict) and event.get("type") == "error":
                raise RuntimeError(f"Server error: {event}")
            yield event
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _normalize_finish(reason):
    if reason in ("length", "max_tokens", "model_length_context_window_exceeded"):
        return "length"
    return reason


_NO_PROMPT_CACHE = set()       # Anthropic-style base URLs that rejected cache_control


def _log_cache_usage(usage):
    """Anthropic reports how much of the prompt came from its cache."""
    read = usage.get("cache_read_input_tokens") or 0
    written = usage.get("cache_creation_input_tokens") or 0
    if read or written:
        log.info(f"GPromptsEnhanced: prompt cache - {read} tokens reused, {written} tokens stored, "
              f"{usage.get('input_tokens') or 0} new")


@functools.lru_cache(maxsize=64)
def is_this_machine(url):
    """True when url points at the ComfyUI machine itself (loopback, or this
    host's own name or address)."""
    host = (urllib.parse.urlsplit(url).hostname or "").lower().rstrip(".")
    if host in ("localhost", "0.0.0.0", "::", "::1") or host.startswith("127."):
        return True
    names = set()
    try:
        name = socket.gethostname().lower()
        names |= {name, name.split(".")[0], name.split(".")[0] + ".local"}
        names.add(socket.getfqdn().lower())
    except OSError:
        pass
    if host in names:
        return True
    try:
        own = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None)}
        own.add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        own = set()
    return host in own


def normalize_base_url(url, style):
    url = (url or "").strip() or DEFAULT_BASE_URLS[style]
    url = url.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/api/generate", "/api/chat",
                   "/v1/messages", "/messages"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    if style in (API_OPENAI_CHAT, API_OPENAI_RAW):
        if urllib.parse.urlparse(url).path in ("", "/"):
            url += "/v1"
    elif style == API_ANTHROPIC:
        if url.endswith("/v1"):
            url = url[:-3]
    return url


def auto_thinking_field(base_url, style):
    if style == API_OLLAMA_CHAT:
        return "think"
    host = urllib.parse.urlparse(base_url).netloc.lower()
    if "openrouter" in host:
        return "reasoning"
    if "dashscope" in host or "aliyuncs" in host:
        return "enable_thinking"
    if "api.openai.com" in host or host.endswith(":11434") or "ollama" in host:
        return "reasoning_effort"
    return "chat_template_kwargs"   # vLLM, SGLang, llama-server


def thinking_payload(field, on):
    """Top-level fields that switch thinking on/off in the given convention."""
    if field == "chat_template_kwargs":
        return {"chat_template_kwargs": {"enable_thinking": on}}
    if field == "enable_thinking":
        return {"enable_thinking": on}
    if field == "reasoning_effort":
        return {"reasoning_effort": "medium" if on else "none"}
    if field == "reasoning":
        return {"reasoning": {"enabled": on}}
    if field == "think":
        return {"think": on}
    return {}


def _deep_merge(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ----------------------------------------------------------------------------
# The enhancer
# ----------------------------------------------------------------------------
class APIPromptEnhancer(PromptEnhancerBase):
    def __init__(self, api, base_url, model, api_key_name, system_prompt_path, thinking,
                 thinking_field, plan_tokens, max_new_tokens, context_length, temperature,
                 top_p, top_k, min_p, presence_penalty, keep_alive, timeout, extra_json,
                 task=TASK_T2I, image_megapixels=None, second_try=False):
        self.task = task
        self.second_try = second_try
        self.image_megapixels = image_megapixels or DEFAULT_LLM_IMAGE_MP
        self.api = api
        self.base_url = normalize_base_url(base_url, api)
        self.model = model.strip()
        self.api_key_name = api_key_name
        self.system_prompt_path = system_prompt_path
        self.thinking = thinking
        self.thinking_field = thinking_field
        self.plan_tokens = plan_tokens
        self.max_new_tokens = max_new_tokens
        self.context_length = context_length
        self.temperature = temperature
        self._run_temperature = temperature     # per call: see h3_temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.keep_alive = keep_alive.strip()
        self.timeout = timeout
        self.extra = json.loads(extra_json) if extra_json.strip() else {}

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
            self.api, self.base_url, self.model, self.api_key_name, self.system_prompt_path,
            sp_stamp, self.thinking, self.thinking_field, self.plan_tokens, self.max_new_tokens,
            self.context_length, self.temperature, self.top_p, self.top_k, self.min_p,
            self.presence_penalty, self.extra, self.task, self.image_megapixels, self.second_try,
        ], sort_keys=True)

    # -- request building --------------------------------------------------
    def _headers(self, api_key):
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream, application/x-ndjson, application/json"}
        if self.api == API_ANTHROPIC:
            headers["anthropic-version"] = ANTHROPIC_VERSION
            if api_key:
                headers["x-api-key"] = api_key
        elif api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _keep_alive(self):
        ka = self.keep_alive
        if not ka:
            return None
        return int(ka) if re.fullmatch(r"-?\d+", ka) else ka

    def _ollama_options(self, max_tokens, seed, stop=None):
        options = {
            "num_ctx": self.context_length, "num_predict": max_tokens,
            "temperature": self._run_temperature, "top_p": self.top_p, "top_k": self.top_k,
            "min_p": self.min_p, "presence_penalty": self.presence_penalty,
            "repeat_penalty": 1.0, "seed": seed,
        }
        if stop:
            options["stop"] = stop
        return options

    def _finish_payload(self, payload):
        if self.api in OLLAMA_STYLES and self._keep_alive() is not None:
            payload["keep_alive"] = self._keep_alive()
        return _deep_merge(payload, json.loads(json.dumps(self.extra)))

    # -- raw prompt (GGUF-loader parity) -----------------------------------
    def _raw_stream(self, prompt, max_tokens, stop_when, seed, api_key, pbar):
        seed = seed & 0x7FFFFFFF
        if self.api == API_OLLAMA_RAW:
            url = self.base_url + "/api/generate"
            payload = {"model": self.model, "prompt": prompt, "raw": True, "stream": True,
                       "options": self._ollama_options(max_tokens, seed, STOP_STRINGS)}
            optional = {"keep_alive"}
            kind = "ndjson"
        else:
            url = self.base_url + "/completions"
            payload = {"model": self.model, "prompt": prompt, "stream": True,
                       "max_tokens": max_tokens, "temperature": self._run_temperature,
                       "top_p": self.top_p, "top_k": self.top_k, "min_p": self.min_p,
                       "presence_penalty": self.presence_penalty, "seed": seed,
                       "stop": STOP_STRINGS}
            optional = {"top_k", "min_p", "presence_penalty", "seed", "stop", "top_p"}
            kind = "sse"
        payload = self._finish_payload(payload)
        resp = post_adaptive(url, payload, self._headers(api_key), self.timeout, self.model,
                             optional | set(self.extra))
        thinking_buf, text_buf, n, finish = [], [], 0, None
        started = time.perf_counter()
        events = iter_stream(resp, kind)
        for ev in events:
            if self.api == API_OLLAMA_RAW:
                think_piece, piece = ev.get("thinking") or "", ev.get("response") or ""
                if ev.get("done"):
                    finish = ev.get("done_reason") or "stop"
            else:
                choice = (ev.get("choices") or [{}])[0]
                think_piece, piece = "", choice.get("text") or ""
                finish = choice.get("finish_reason") or finish
            if not (think_piece or piece):
                continue
            thinking_buf.append(think_piece)
            text_buf.append(piece)
            n += 1
            pbar.update(1)
            text = self._combine("".join(thinking_buf), "".join(text_buf))
            if stop_when(text, n):
                finish = "early"
                break
        events.close()      # disconnect now, so the server stops generating
        check_interrupted()
        self._log_speed(n, started, finish)
        return self._combine("".join(thinking_buf), "".join(text_buf)), n, _normalize_finish(finish)

    @staticmethod
    def _combine(thinking, text):
        # A server that splits thinking out even in raw mode: put it back in
        # front of the answer the way the raw prompt expects (<think> is open).
        if thinking and "</think>" not in text:
            return thinking + ("\n</think>\n\n" + text if text else "")
        return thinking + text

    # -- chat APIs ---------------------------------------------------------
    def _user_message(self, user_text, images_b64):
        """The user turn with images first, in order, then the instruction -
        in the image format of this API."""
        if not images_b64:
            return {"role": "user", "content": user_text}
        if self.api == API_OLLAMA_CHAT:
            return {"role": "user", "content": user_text, "images": list(images_b64)}
        if self.api == API_ANTHROPIC:
            parts = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                  "data": b}} for b in images_b64]
            if self._anthropic_cache():
                # Everything up to the last image is identical on every run: cache it.
                parts[-1]["cache_control"] = {"type": "ephemeral"}
        else:
            parts = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}}
                     for b in images_b64]
        return {"role": "user", "content": parts + [{"type": "text", "text": user_text}]}

    def _chat(self, system_prompt, user_text, seed, api_key, pbar, stop_when, images_b64=None):
        seed = seed & 0x7FFFFFFF
        on = {"on": True, "off": False}.get(self.thinking)
        messages = [self._user_message(user_text, images_b64)]
        renames = {}
        if self.api == API_ANTHROPIC:
            url = self.base_url + "/v1/messages"
            payload = {"model": self.model, "max_tokens": self.max_new_tokens, "messages": messages,
                       "stream": True, "temperature": self._run_temperature, "top_p": self.top_p,
                       "top_k": self.top_k}
            if system_prompt:
                payload["system"] = system_prompt
                if self._anthropic_cache():
                    payload["system"] = [{"type": "text", "text": system_prompt,
                                          "cache_control": {"type": "ephemeral"}}]
            if on:
                budget = self.plan_tokens if self.plan_tokens >= 1024 else 1024
                budget = min(budget, max(1024, self.max_new_tokens - 1024))
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            optional = {"temperature", "top_p", "top_k", "thinking"}
            kind = "sse"
        else:
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            if self.api == API_OLLAMA_CHAT:
                url = self.base_url + "/api/chat"
                payload = {"model": self.model, "messages": messages, "stream": True,
                           "options": self._ollama_options(self.max_new_tokens, seed)}
                if on is not None:
                    payload["think"] = on
                optional = {"think", "keep_alive"}
                kind = "ndjson"
            else:
                url = self.base_url + "/chat/completions"
                payload = {"model": self.model, "messages": messages, "stream": True,
                           "max_tokens": self.max_new_tokens, "temperature": self._run_temperature,
                           "top_p": self.top_p, "top_k": self.top_k, "min_p": self.min_p,
                           "presence_penalty": self.presence_penalty, "seed": seed}
                field = self.thinking_field
                if field == "auto":
                    field = auto_thinking_field(self.base_url, self.api)
                if on is not None:
                    payload.update(thinking_payload(field, on))
                optional = {"top_k", "min_p", "presence_penalty", "seed", "top_p", "temperature",
                            "chat_template_kwargs", "enable_thinking", "reasoning_effort",
                            "reasoning", "think"}
                renames = {"max_tokens": "max_completion_tokens"}
                kind = "sse"
        payload = self._finish_payload(payload)
        try:
            resp = post_adaptive(url, payload, self._headers(api_key), self.timeout, self.model,
                                 optional | set(self.extra), renames)
        except HTTPStatusError as e:
            # An Anthropic-style server that doesn't know prompt caching: once without it.
            if not (self.api == API_ANTHROPIC and e.status in (400, 422)
                    and "cache_control" in (e.body or "") and self._anthropic_cache()):
                raise
            _NO_PROMPT_CACHE.add(self.base_url)
            log.info(f"GPromptsEnhanced: {self.base_url} does not accept prompt caching; "
                  f"sending without it.")
            return self._chat(system_prompt, user_text, seed, api_key, pbar, stop_when,
                              images_b64)

        content, reasoning, n, finish = [], [], 0, None
        started = time.perf_counter()
        events = iter_stream(resp, kind)
        for ev in events:
            piece, think_piece = "", ""
            if self.api == API_ANTHROPIC:
                etype = ev.get("type")
                if etype == "message_start":
                    _log_cache_usage((ev.get("message") or {}).get("usage") or {})
                elif etype == "content_block_delta":
                    delta = ev.get("delta") or {}
                    piece = delta.get("text") or ""
                    think_piece = delta.get("thinking") or ""
                elif etype == "message_delta":
                    finish = (ev.get("delta") or {}).get("stop_reason") or finish
                elif etype is None and "content" in ev:        # non-streaming body
                    for block in ev.get("content") or []:
                        if block.get("type") == "text":
                            piece += block.get("text") or ""
                        elif block.get("type") == "thinking":
                            think_piece += block.get("thinking") or ""
                    finish = ev.get("stop_reason") or finish
            elif self.api == API_OLLAMA_CHAT:
                msg = ev.get("message") or {}
                piece, think_piece = msg.get("content") or "", msg.get("thinking") or ""
                if ev.get("done"):
                    finish = ev.get("done_reason") or "stop"
            else:
                choice = (ev.get("choices") or [{}])[0]
                delta = choice.get("delta") or choice.get("message") or {}
                piece = delta.get("content") or ""
                think_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if not isinstance(think_piece, str):
                    think_piece = ""
                finish = choice.get("finish_reason") or finish
            if not (piece or think_piece):
                continue
            content.append(piece)
            reasoning.append(think_piece)
            n += 1
            pbar.update(1)
            if piece and stop_when("".join(content)):
                finish = "early"
                break
        events.close()      # disconnect now, so the server stops generating
        check_interrupted()
        self._log_speed(n, started, finish)
        return "".join(content), "".join(reasoning), _normalize_finish(finish)

    @staticmethod
    def _log_speed(n, started, finish):
        elapsed = time.perf_counter() - started
        log.info(f"GPromptsEnhanced: {n} chunks in {elapsed:.1f}s, finish={finish}")

    def uses_local_gpu(self, has_images):
        """A server on this machine shares its GPU with the image model."""
        return is_this_machine(self.base_url)

    def _anthropic_cache(self):
        return self.api == API_ANTHROPIC and self.base_url not in _NO_PROMPT_CACHE

    # -- entry point ---------------------------------------------------------
    def enhance(self, user_text, seed, images=None):
        if not self.second_try:
            return self._enhance(user_text, seed, images)
        return with_second_try(lambda text: self._enhance(text, seed, images, warn=False),
                               user_text, self._system_prompt(), len(images or []))

    def _enhance(self, user_text, seed, images=None, warn=True):
        images = list(images or [])
        if images and self.api in RAW_STYLES:
            raise RuntimeError(f"'{self.api}' can't send images; use a chat style "
                               f"({', '.join(CHAT_STYLES)}) for image editing.")
        system_prompt = self._system_prompt()
        self._run_temperature = h3_temperature(
            self.temperature, system_prompt,
            fixed=(self.api == API_ANTHROPIC and self.thinking == "on"))
        api_key = resolve_api_key(self.api_key_name)
        pbar = progress_bar(self.max_new_tokens)
        log.info(f"GPromptsEnhanced: {self.api} -> {self.base_url} model={self.model}"
              + (f" with {len(images)} image(s)" if images else ""))
        images_b64 = [prepared_image(t, self.image_megapixels)[0] for t in images]

        if self.api in RAW_STYLES:
            thinking = self.thinking != "off"
            raw, finish, contract = run_raw_generation(
                lambda prompt, max_tokens, stop_when: self._raw_stream(
                    prompt, max_tokens, stop_when, seed, api_key, pbar),
                system_prompt, user_text, thinking, self.plan_tokens, self.max_new_tokens)
            return finalize_result(raw, finish, thinking, contract, self.max_new_tokens,
                                   system_prompt=system_prompt, image_count=len(images), warn=warn)

        contract = PE_CONTRACT_KEY in system_prompt

        def stop_when(content):
            if not contract or ("<think>" in content and "</think>" not in content):
                return False
            return answer_complete(content, False)

        content, reasoning, finish = self._chat(system_prompt, user_text, seed, api_key, pbar,
                                                stop_when, images_b64)
        raw = content
        if reasoning and "</think>" not in content:
            raw = f"<think>{reasoning}</think>\n\n{content}"
        return finalize_result(raw, finish, self.thinking == "on", contract, self.max_new_tokens,
                               system_prompt=system_prompt, image_count=len(images), warn=warn)


# ----------------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------------
def _api_schema(edit):
    """Schema for the text-to-image API loader, or (edit=True) its image-edit sibling:
    chat styles only (raw prompts can't carry images), edit defaults, image size."""
    prompts = [NO_SYSTEM_PROMPT] + list_llm_files(SYSTEM_PROMPT_EXTENSIONS)
    if edit:
        api_input = io.Combo.Input(
            "api", options=CHAT_STYLES, default=API_OPENAI_CHAT,
            tooltip="Request style (chat styles only: they can carry images). The server must "
                    "serve a vision model: llama-server with --mmproj, an Ollama vision model, "
                    "or a hosted vision model.")
        plan_tip = "Anthropic only: thinking budget (minimum 1024). Ignored otherwise."
        penalty_default, penalty_tip = 0.0, "Qwen's settings use 0 for the edit rewriter."
        model_tip = "Model name as the server knows it, e.g. qwen2.1-pe-i2i, qwen3-vl:8b, gpt-4.1."
    else:
        api_input = io.Combo.Input(
            "api", options=API_STYLES, default=API_OPENAI_CHAT,
            tooltip="Request style. 'OpenAI-compatible chat' works almost everywhere. The two "
                    "'(raw prompt)' styles send the exact ChatML prompt the GGUF loader builds "
                    "(think prefill, plan_tokens, early stop) - use them for Qwen-family models "
                    "such as the Qwen-Image prompt rewriters.")
        plan_tip = ("Raw prompt styles: cap on thinking before the plan is handed back for the "
                    "answer (-1 = no cap). Anthropic: thinking budget (minimum 1024). Ignored "
                    "otherwise.")
        penalty_default, penalty_tip = 1.5, "~1.5 for the Qwen-Image T2I rewriter."
        model_tip = ("Model name as the server knows it, e.g. qwen2.1-pe-t2i, qwen3:8b, "
                     "gpt-4.1-mini, anthropic/claude-sonnet-4.")
    extra_inputs = []
    if edit:
        extra_inputs.append(io.Combo.Input(
            "llm_image_megapixels", options=LLM_IMAGE_MEGAPIXELS, default=DEFAULT_LLM_IMAGE_MP,
            tooltip="Images are downscaled to about this size before they are sent (smaller is "
                    "cheaper and faster). The encode node still gets the originals."))
    return io.Schema(
            node_id="GPromptEnhancerLoaderAPIEdit" if edit else "GPromptEnhancerLoaderAPI",
            display_name="Prompt Enhancer Loader (API, edit)" if edit else "Prompt Enhancer Loader (API)",
            category="gprompts/enhancer",
            description=(
                ("Image-edit prompt rewriter behind an HTTP API; it sees the reference images "
                 "connected to 'Dynamic Prompts with Enhancer'. " if edit else
                 "An LLM behind an HTTP API (Ollama, llama-server, vLLM, SGLang, LM Studio, "
                 "OpenAI, OpenRouter, DashScope, Anthropic, or any OpenAI-compatible endpoint) for "
                 "'Dynamic Prompts with Enhancer'. ")
                + "API keys come from Settings > Gadzoinks > LLM, never from the node, so they "
                  "are not saved into workflows or images."
            ),
            search_aliases=(["edit prompt enhancer", "i2i", "vision llm api"] if edit else
                            ["prompt enhancer", "ollama", "openai", "llm api", "openrouter"]),
            inputs=[
                api_input,
                io.String.Input("base_url", default="",
                                tooltip="Server address. Blank = the style's local default "
                                        "(Ollama 127.0.0.1:11434, llama-server 127.0.0.1:8080, "
                                        "api.anthropic.com). Examples: http://192.168.1.20:11434/v1, "
                                        "https://api.openai.com/v1, https://openrouter.ai/api/v1, "
                                        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1."),
                io.String.Input("model", default="", tooltip=model_tip),
                io.String.Input("api_key_name", default="",
                                tooltip="Which API key to use from Settings > Gadzoinks > LLM. "
                                        "Blank = the default key (fine to leave unset for local "
                                        "servers). A name = that entry in the named keys list "
                                        "(name=key; name=key). 'env:VAR' = environment variable. "
                                        "'none' = send no key."),
                io.Combo.Input("system_prompt", options=prompts,
                               tooltip="System prompt file (.txt/.md) in models/LLM, or (none). "
                                       "Qwen-Image PE models need their shipped system_prompt.txt."),
                io.Combo.Input("thinking", options=THINKING_MODES, default="model default",
                               tooltip="Turn the model's thinking on or off, or leave it to the "
                                       "server. Raw prompt styles: 'model default' = on."),
                io.Combo.Input("thinking_field", options=THINKING_FIELDS, default="auto",
                               advanced=True,
                               tooltip="OpenAI-compatible chat only: which request field carries "
                                       "thinking on/off. auto picks by server: reasoning_effort "
                                       "(OpenAI, Ollama), reasoning (OpenRouter), enable_thinking "
                                       "(DashScope), chat_template_kwargs (vLLM, SGLang, llama-server)."),
                io.Int.Input("plan_tokens", default=800, min=-1, max=65536, step=100,
                             tooltip=plan_tip),
                io.Int.Input("max_new_tokens", default=8192, min=256, max=131072, step=256),
                io.Int.Input("context_length", default=16384, min=2048, max=1048576, step=1024,
                             tooltip="Ollama only (num_ctx). Ollama's small default context "
                                     "silently truncates long system prompts."),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=20, min=0, max=500),
                io.Float.Input("min_p", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("presence_penalty", default=penalty_default, min=-2.0, max=2.0,
                               step=0.05, tooltip=penalty_tip + " Parameters a server rejects are "
                                                  "dropped automatically."),
                io.String.Input("keep_alive", default="", advanced=True,
                                tooltip="Ollama only: how long the model stays loaded after a call "
                                        "(e.g. 5m, 1h, 0 = unload at once to free VRAM, -1 = forever). "
                                        "Blank = server default."),
                io.Int.Input("timeout", default=600, min=10, max=7200, advanced=True,
                             tooltip="Seconds to wait for the server between streamed chunks."),
                io.String.Input("extra_json", default="", multiline=True, advanced=True,
                                tooltip="JSON merged into the request body for anything provider-"
                                        'specific, e.g. {"reasoning_effort": "low"} or '
                                        '{"options": {"num_gpu": 20}}. Do not put API keys here.'),
            ] + extra_inputs + [
                # last, so saved workflows keep their widget positions
                io.Boolean.Input("second_try", display_name="2nd try on error", default=False,
                                 tooltip=SECOND_TRY_TIP),
            ],
            outputs=[PromptEnhancerType.Output("enhancer", display_name="enhancer")],
        )


class GPromptEnhancerLoaderAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _api_schema(edit=False)

    @classmethod
    def validate_inputs(cls, model, system_prompt, extra_json, api_key_name):
        if not (model or "").strip():
            return "model is required (the model name as the server knows it)"
        if system_prompt != NO_SYSTEM_PROMPT and resolve_llm_file(system_prompt) is None:
            return f"System prompt file not found in models/LLM: {system_prompt}"
        if (extra_json or "").strip():
            try:
                extra = json.loads(extra_json)
            except json.JSONDecodeError as e:
                return f"extra_json is not valid JSON: {e}"
            if not isinstance(extra, dict):
                return "extra_json must be a JSON object ({...})"
            if re.search(r"api[_-]?key|authorization|bearer|x-api-key", json.dumps(extra).lower()):
                return ("extra_json looks like it contains an API key. Put keys in Settings > "
                        "Gadzoinks > LLM instead; node values are saved into workflows and images.")
        if re.match(r"^(sk-|sk_|Bearer\s)", (api_key_name or "").strip()):
            return ("api_key_name looks like an actual key. Put the key in Settings > Gadzoinks > "
                    "LLM and type only its name here.")
        return True

    @classmethod
    def execute(cls, api, base_url, model, api_key_name, system_prompt, thinking, thinking_field,
                plan_tokens, max_new_tokens, context_length, temperature, top_p, top_k, min_p,
                presence_penalty, keep_alive, timeout, extra_json, second_try=False) -> io.NodeOutput:
        sp_path = None if system_prompt == NO_SYSTEM_PROMPT else resolve_llm_file(system_prompt)
        return io.NodeOutput(APIPromptEnhancer(
            api=api, base_url=base_url, model=model, api_key_name=api_key_name,
            system_prompt_path=sp_path, thinking=thinking, thinking_field=thinking_field,
            plan_tokens=plan_tokens, max_new_tokens=max_new_tokens, context_length=context_length,
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            presence_penalty=presence_penalty, keep_alive=keep_alive, timeout=timeout,
            extra_json=extra_json, second_try=second_try))


class GPromptEnhancerLoaderAPIEdit(GPromptEnhancerLoaderAPI):
    @classmethod
    def define_schema(cls):
        return _api_schema(edit=True)

    @classmethod
    def execute(cls, api, base_url, model, api_key_name, system_prompt, thinking, thinking_field,
                plan_tokens, max_new_tokens, context_length, temperature, top_p, top_k, min_p,
                presence_penalty, keep_alive, timeout, extra_json,
                llm_image_megapixels=DEFAULT_LLM_IMAGE_MP, second_try=False) -> io.NodeOutput:
        sp_path = None if system_prompt == NO_SYSTEM_PROMPT else resolve_llm_file(system_prompt)
        return io.NodeOutput(APIPromptEnhancer(
            api=api, base_url=base_url, model=model, api_key_name=api_key_name,
            system_prompt_path=sp_path, thinking=thinking, thinking_field=thinking_field,
            plan_tokens=plan_tokens, max_new_tokens=max_new_tokens, context_length=context_length,
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            presence_penalty=presence_penalty, keep_alive=keep_alive, timeout=timeout,
            extra_json=extra_json, task=TASK_EDIT, image_megapixels=llm_image_megapixels,
            second_try=second_try))


API_NODES = {"GPromptEnhancerLoaderAPI": GPromptEnhancerLoaderAPI,
             "GPromptEnhancerLoaderAPIEdit": GPromptEnhancerLoaderAPIEdit}
API_NODE_DISPLAY_NAME_MAPPINGS = {"GPromptEnhancerLoaderAPI": "Prompt Enhancer Loader (API)",
                                  "GPromptEnhancerLoaderAPIEdit": "Prompt Enhancer Loader (API, edit)"}
