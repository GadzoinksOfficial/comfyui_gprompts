# ComfyUI GPrompts Nodes

## Introduction
This package provides custom nodes for ComfyUI that enhance prompt generation, string formatting, and saving images, video, and audio.

## Nodes Overview
- **GPrompts** - Create dynamic prompts with random or sequential selection. Also supports wildcard files.
- **Dynamic Prompts with Enhancer** + **Prompt Enhancer Loaders (GGUF / API, text-to-image / edit)** + **Enhancer Pair** - GPrompts plus an LLM prompt enhancer (e.g. the Qwen-Image-2.1 prompt rewriters), local GGUF or any Ollama/OpenAI-compatible/Anthropic API, for text-to-image and image editing, calling the LLM every run or once per batch.
- **String Formatter** - Build custom output strings from multiple inputs and system variables.
- **Save Image With Notes** - Save images with embedded workflow notes and metadata.
- **Load Images From Folder** - Load images from a folder one at a time, in order, or randomly, and read back the prompt they were created with.
- **Save Image To Immich Server** - Save images with embedded workflow notes to an Immich server.
- **Save Video To Immich Server** - Save videos to an Immich server.
- **Save Audio To Immich Server** - Save audio to an Immich server (as an mp4 with a cover image).

---

## GPrompts Node

### Description
This is another dynamic prompts node for ComfyUI. I found most of the ones out there to be either
too complicated or too limiting, so I wrote my own.

### Basics
Create a GPrompts node and connect its output to a CLIP node text input.

Format of a dynamic prompt:
```
{ cat | dog | jackalope }        random selection
{{ green | yellow | red }}       sequential selection
```

If you generate 4 images with `a stop light showing {{ green | yellow | red }}`
you will get a green light image, yellow, red, and then another green.

If you use `a stop light showing { green | yellow | red }`, each image will have a 33% chance of any color.

The sequential cycle starts at the first combination and starts over whenever the prompt text changes (or ComfyUI is restarted).

The normal delimiters are `{` and `}`, but you can change them to `<` and `>`, which is useful for JSON prompts:
```
< cat | dog >          random selection
<< green | red >>      sequential selection
```

### Registers
`{0 cat | dog }` will store the chosen value in register 0, which can be referenced later with `{0}`.
`{` and `{{` have their own separate registers. Registers 0..9 are available.

```
a {0 green | blue} {{0 dog|monkey}} standing on top of a {0} {{0}}
```
will generate something like `a green dog standing on top of a green dog`.

Note: `{{0|1|2}}` (no space after the digit) is a normal options block, not a register.

### Quotes and special characters inside blocks
**Avoid using `"` and `'` inside `{ }` and `{{ }}` blocks.** Double quotes in particular will stop a random block from being recognized, and quotes can break JSON prompts.

Two ways around this:

1. **Keep the quotes outside the block.** This works fine:
   ```
   wearing a T shirt that says "{ hello | goodbye }"
   ```
2. **Use a Unicode character that looks like a quote.** Copy and paste one of these into your options:

   **Instead of `"` (double quote)**

   | Character | Name | Code point |
   |-----------|------|------------|
   | `“` | Left double quotation mark | U+201C |
   | `”` | Right double quotation mark | U+201D |
   | `″` | Double prime | U+2033 |
   | `＂` | Fullwidth quotation mark | U+FF02 |
   | `„` | Double low-9 quotation mark | U+201E |
   | `〃` | Ditto mark | U+3003 |

   **Instead of `'` (single quote / apostrophe)**

   | Character | Name | Code point |
   |-----------|------|------------|
   | `‘` | Left single quotation mark | U+2018 |
   | `’` | Right single quotation mark | U+2019 |
   | `′` | Prime | U+2032 |
   | `＇` | Fullwidth apostrophe | U+FF07 |
   | `ʼ` | Modifier letter apostrophe | U+02BC |
   | `‚` | Single low-9 quotation mark | U+201A |

   Example:
   ```
   a sign that reads { “OPEN” | “CLOSED” | “BACK IN 5” }
   a { cat’s | dog’s } toy
   ```

If you need a literal `|` inside an option, escape it with a backslash: `\|`.

### Wildcards
Wildcard files are either .txt or .json and go in `comfyui/models/wildcards`.

You can use a wildcard file with a list of options:
```
a woman with {{__hair_color__}} {{__hair_style__}} hair
```
This will use the contents of `comfyui/models/wildcards/hair_color.txt` and `hair_style.txt`.

Assuming the files are
```
blonde
red
brown
```
and
```
long
short
pixie
mohawk
```
you will have 12 combinations. Set ComfyUI to generate 12 images and you will see all combinations.

A wildcard reference to `__hair__hairstyles__` will use the file `models/wildcards/hair/hairstyles.txt` (or .json).

#### JSON wildcard files
Instead of text you can use a JSON file. For example `seasons.json`:

Simple:
```json
{ "doesnotmatter": ["summer", "winter", "fall", "spring"] }
```

Weighted:
```json
{ "whatever": [ { "summer": 6 }, { "spring": 4 }, { "fall": 3 }, { "winter": 1 } ] }
```

Weights are only relevant to `{}` random selection. With random, the odds of getting a choice are weight / total weight, so for summer the odds are 6 out of 14.
For `{{}}` sequential you will get all 4 seasons.


### TODO
- Add support for wildcard files that include other wildcard files.

---

## Dynamic Prompts with Enhancer

### Description
Same dynamic prompt syntax as GPrompts, plus an LLM that rewrites the prompt into the long, detailed
form image models like Qwen-Image 2.1 work best with.

The **enhance** setting picks how often the LLM runs:

- **every run** (default) - each run's expansion is sent to the LLM. One LLM call per image, and
  every prompt is rewritten around its own values. With a seed above 0 the LLM gets `seed + run
  number`, so a combination that comes round again gets a fresh (but reproducible) rewrite.
- **once, then substitute** - only the **first** expansion goes to the LLM. For every later run the
  node swaps that run's chosen values into the LLM's rewrite, so a whole batch costs one LLM call.
  Example: `a {{tiki bar|beach hut}} with {red|amber} lanterns` is rewritten once as "Cinematic
  photo of a tiki bar with amber lanterns at dusk", and the next run becomes "Cinematic photo of a
  beach hut with red lanterns at dusk". This only works while the LLM keeps the values as words
  that can be swapped. If it rewords or elaborates them ("add a dinosaur" becomes "a towering
  sauropod..."), later runs can't swap in their value; use **every run** for prompts like that.

With **every run**, **prefetch_next** hides most of the LLM's time: as soon as a run's prompt is
out, the node expands the next run and starts rewriting it on a background thread, so the LLM works
while the image renders, and the next run usually finds its prompt ready.
- `auto` (default) - on for an API loader pointing at another machine or a hosted service, and for a
  GGUF on the CPU (`gpu_layers` = 0). Off for an API server on this machine (`localhost`, a blank
  `base_url` for the local styles, or this machine's own name or address), which shares the GPU.
- `on` - also prefetch from a server on this machine (it may slow the image a little; a failure
  there only fails that rewrite).
- `off` - never.

A **GGUF on the GPU never prefetches, not even with `on`**. It runs inside the ComfyUI process, and
llama.cpp aborts the whole process on a GPU error, such as running out of memory while the image
model samples at the same time. That takes ComfyUI down (`Fatal Python error: Aborted`, core
dumped). To prefetch with a local model, run it on the CPU (`gpu_layers` 0; the vision projector
then runs on the CPU too) or behind a server such as Ollama or llama-server.

The next run's expansion is predictable for sequential `{{ }}` blocks and for random `{ }` blocks
with a fixed seed. With the seed set to randomize and random blocks in the text, the prediction is
usually wrong: the background rewrite is dropped and that run calls the LLM itself (no gain, no
harm). After the last run of a batch one background rewrite goes unused, which on a paid API is one
wasted call.

In substitute mode, **preserve_dynamic_words** (on by default) asks the LLM to keep the chosen words
verbatim. If it rewords one anyway ("cat" becomes "kitten"), that variation can't be swapped in; the
console says so and that run keeps the first run's value.

Changing the text, delimiter style, `enhance` mode, any enhancer setting or a reference image starts
the batch over. The seed does not: as in Dynamic Prompts it only picks the random blocks, so a seed
set to randomize still steps through the combinations.

### Outputs
- `text` / `computed_prompt` - the final prompt for this run
- `dynamic_prompt` - this run's expansion before enhancement
- `seed`
- `template` - the prompt exactly as typed, dynamic blocks and all
- `wh_ratio` - aspect ratio recommended by the LLM (e.g. `16:9`), empty if none
- `width` / `height` - an image size at `wh_ratio` (1:1 if none) with about `target_megapixels`
  pixels (1.0 = 1024x1024-sized, 4.0 = Qwen-Image 2.1's native 2K), sides rounded to multiples of 16.
  Wire them into an Empty Latent Image for text-to-image.
- `resolution` - the side of a square with about `target_megapixels` pixels (multiple of 32;
  1.0 = 1024), for the `resolution` input of Text Encode Qwen Image 2.1 in edit workflows. For
  text-to-image, don't use that node's latent output: it is always square. Use width/height into
  an Empty Latent Image instead; the encode node's `resolution` then has no effect.
- `ratio_follow` - for edits, the reference image whose shape the output keeps (e.g. `image1`),
  as chosen by the LLM; width/height follow that image's shape.

### Getting the models
The Qwen-Image-2.1 prompt rewriters come as two separate models, one for text-to-image (**PE-T2I**)
and one for image editing (**PE-I2I**), each with its own system prompt. The edit model also needs
its vision projector (**mmproj**), the file that lets the LLM see images. Any quantization works
(Q4_K_M ~6 GB, Q6_K ~7.5 GB). For example, with the `hf` command from `huggingface_hub`, run from
the ComfyUI folder:

```
hf download pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF \
  pe_t2i_heretic-Q4_K_M.gguf system_prompt.txt --local-dir models/LLM/pe_t2i

hf download pottokao/Qwen-Image-2.1-PE-I2I-Heretic-GGUF \
  pe_i2i_heretic-Q4_K_M.gguf pe_i2i_heretic.mmproj-bf16.gguf system_prompt.txt \
  --local-dir models/LLM/pe_i2i
```

or download the same files in a browser from the repos' *Files* tab. Keep each model in its own
subfolder: both ship a file called `system_prompt.txt`, and the edit one (~18 KB) is different from
the text-to-image one (~10 KB).

```
ComfyUI/models/LLM/
    pe_t2i/  pe_t2i_heretic-Q4_K_M.gguf, system_prompt.txt
    pe_i2i/  pe_i2i_heretic-Q4_K_M.gguf, pe_i2i_heretic.mmproj-bf16.gguf, system_prompt.txt
```

Other GGUF builds work too, but the mmproj must come from the same repo as its model. After adding
files, refresh the browser page (or restart ComfyUI) so the node lists pick them up.

### Prompt Enhancer Loader (GGUF)
Put the `.gguf` model and its system prompt file (`.txt` or `.md`) in `ComfyUI/models/LLM/`
(subfolders are fine), then pick them on the node. The model loads the first time a prompt is
enhanced; turn `keep_loaded` off to free its VRAM after each use, or set `gpu_layers` to 0 to run it
on the CPU and leave the GPU to the image model.

Note that `gpu_layers` defaults to -1: the **whole LLM goes on the GPU** and, with `keep_loaded` on,
stays there next to the image model. A 9B rewriter at Q4-Q6 holds roughly 6-8 GB of VRAM plus its
context. If the image model runs short of memory, turn `keep_loaded` off or set `gpu_layers` to 0
(slower rewrites, but no VRAM used).

For the Qwen-Image-2.1 prompt rewriters, use the `system_prompt.txt` that ships with the model; they
produce nothing useful without it. Keep `enable_thinking` on, and use `presence_penalty` ~1.5 for the
text-to-image rewriter.

### Requirements
The loader runs the model in-process with the `llama_cpp` package
([llama-cpp-python](https://pypi.org/project/llama-cpp-python/)), the same way
[ComfyUI-Prompt-Enhancer](https://github.com/xiaowuapple-pixel/ComfyUI-Prompt-Enhancer) does. It is
optional and not installed automatically (see `requirements-local-gguf.txt`), because it has to match
your CUDA version. Install it into ComfyUI's own Python.

Easiest, no compiling: download the prebuilt CUDA wheel for your CUDA and Python version from
[JamePeng/llama-cpp-python releases](https://github.com/JamePeng/llama-cpp-python/releases)
(the file name carries both, e.g. `+cu128` and `cp312`) and `python -m pip install` the `.whl`.
Or build it yourself:

```
CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install -U llama-cpp-python --no-cache-dir
```

Qwen3.5-based models, including the Qwen-Image-2.1 prompt rewriters, need llama-cpp-python 0.3.35+
(PyPI) or the JamePeng fork 0.3.47+.

### Thinking and plan_tokens
With `enable_thinking` on, the model plans before it answers. That plan is where the time goes, so
`plan_tokens` caps it: when the plan reaches that length it is handed back to the model, closed, and
the model writes the answer from it. -1 removes the cap (slowest, most detail). With thinking off the
answer is written directly (fastest, shallower prompts).

### Prompt Enhancer Loader (API)
The same enhancer, but the LLM runs behind an HTTP API: a local or LAN server (Ollama,
llama-server, vLLM, SGLang, LM Studio) or a hosted one (OpenAI, OpenRouter, DashScope, Anthropic, or
anything OpenAI-compatible). Running the LLM on another machine leaves all your GPU memory to the
image model. No extra Python packages are needed.

Pick the request style with `api`:

| api | endpoint | use it for |
|---|---|---|
| OpenAI-compatible chat | `/chat/completions` | almost everything |
| OpenAI-compatible completions (raw prompt) | `/completions` | Qwen-family models on vLLM, SGLang, llama-server, LM Studio |
| Ollama chat | `/api/chat` | any Ollama model |
| Ollama generate (raw prompt) | `/api/generate` | Qwen-family models on Ollama |
| Anthropic messages | `/v1/messages` | Claude |

The two **raw prompt** styles send exactly the prompt the GGUF loader builds (think prefill,
`plan_tokens`, early stop), so a Qwen-Image prompt rewriter behaves the same as it does locally. Chat
styles let the server apply the model's own template; thinking is switched on or off with whatever
field the server understands (`thinking_field`, auto-detected by default).

Servers differ in which parameters they accept. If a request is rejected because of a parameter
(`top_k`, `min_p`, `seed`, `max_tokens`, a thinking field...), the node drops or renames it, retries,
and remembers that for the server and model. `extra_json` adds anything provider-specific to the
request body.

**API keys** go in Settings > Gadzoinks > LLM, never on the node (node values are saved into
workflows and image metadata):
- *LLM API key (default)* - used when `api_key_name` is blank. Leave it empty for local servers.
- *LLM API keys, named* - `openrouter=sk-or-...; dashscope=sk-...`; select one with
  `api_key_name` (e.g. `openrouter`).
- `api_key_name` also accepts `env:VARIABLE` (read an environment variable) and `none`.

For Ollama, set `context_length` (Ollama's small default context silently truncates the 10 KB
Qwen-Image system prompt) and use `keep_alive` = `0` to unload the model right after each call.

#### Quick start: a Qwen-Image rewriter on Ollama
Useful for running the LLM on another machine (a Mac, a second PC) so the ComfyUI GPU is left to the
image model.

1. On the LLM machine, import the GGUF into Ollama. No template or system prompt is needed in the
   Modelfile: the node sends both.
   ```
   echo 'FROM ./pe_t2i_heretic-Q4_K_M.gguf' > Modelfile
   ollama create qwen-pe-t2i -f Modelfile
   ollama list
   ```
2. If ComfyUI is on a different machine, make Ollama listen on the network: `OLLAMA_HOST=0.0.0.0
   ollama serve`, or for the macOS app `launchctl setenv OLLAMA_HOST 0.0.0.0` and restart the app.
   Check from the ComfyUI machine: `curl http://<llm-host>:11434/api/tags` should list the model.
   Ollama has no authentication; only do this on a network you trust.
3. Keep the matching `system_prompt.txt` in `models/LLM` on the **ComfyUI** machine.
4. On **Prompt Enhancer Loader (API)** set `api` = `Ollama generate (raw prompt)`, `base_url` =
   blank (same machine) or `http://<llm-host>:11434`, `model` = `qwen-pe-t2i`, `system_prompt` = the
   T2I `system_prompt.txt`, `context_length` = 16384. Leave `api_key_name` blank.
5. Queue. The console shows `GPromptsEnhanced: Ollama generate (raw prompt) -> ... model=qwen-pe-t2i`
   and then `... finish=early` when the answer is complete.

If `ollama create` rejects the GGUF's architecture, update Ollama. Or serve the GGUF with
llama.cpp's own server instead (`llama-server -m pe_t2i_heretic-Q4_K_M.gguf -c 16384 --host 0.0.0.0
--port 8080`) and use `api` = `OpenAI-compatible completions (raw prompt)` with `base_url` =
`http://<llm-host>:8080/v1`.

For **edits** over an API, the server must see images: use a chat style on the edit loader and a
server that serves the PE-I2I model together with its mmproj (e.g. `llama-server -m
pe_i2i_heretic-Q4_K_M.gguf --mmproj pe_i2i_heretic.mmproj-bf16.gguf ...`).

### Image editing
Connect reference images to **Dynamic Prompts with Enhancer**: its `image_1` socket grows a new one
each time you connect an image (up to 16, like Text Encode Qwen Image 2.1). Refer to them in your text
as `<image1>`, `<image2>`, ... When images are connected, the node uses an **edit** loader; without
them, a **text-to-image** loader.

| Loader | Use |
|---|---|
| Prompt Enhancer Loader (GGUF) / (API) | text-to-image |
| Prompt Enhancer Loader (GGUF, edit) / (API, edit) | image edit (the LLM sees the images) |
| Enhancer Pair | one text-to-image + one edit loader behind a single `enhancer` output |

An edit model needs its own files: for Qwen-Image-2.1 the **PE-I2I** GGUF, its **mmproj** vision
projector (also in `models/LLM`), and its own ~18 KB system prompt - not the text-to-image ones. Use
`presence_penalty` 0 for it. Images are downscaled for the LLM (`llm_image_megapixels`, ~512 tokens per
image at 0.5); the encode node still receives the originals. The API edit loader offers only the chat
styles, since raw prompts can't carry images; the server must serve a vision model (llama-server with
`--mmproj`, an Ollama vision model, or a hosted one).

**Wiring an edit:** each Load Image goes to *two* places, in the same order:
1. `image_1`, `image_2`, ... on Dynamic Prompts with Enhancer (so the rewriter sees them), and
2. `image_1`, `image_2`, ... on Text Encode Qwen Image 2.1, with the VAE connected (so the image
   model edits them).

Then `text` -> the encode node's prompt, `resolution` -> its `resolution`, and use **the encode node's
own latent output** for the sampler (it is sized from the first reference; any other size shifts the
edit). In **once, then substitute** mode the images stay fixed for the batch while the dynamic
blocks vary the instruction, so the batch costs one LLM call; changing an image starts a new one.

### Warnings
- **every run costs one LLM call per image.** With thinking on, that can take longer than the image
  itself; use `prefetch_next` (API or CPU), a lower `plan_tokens`, or thinking off. On a paid API,
  every image is a paid request.
- **once, then substitute is only safe for simple swaps** (colours, plain nouns). If the LLM
  rewrites a value into something else, later runs keep the first run's word; the console prints
  `'<word>' not found in the enhanced prompt, so '<new word>' was not applied` when that happens.
- **Local GGUF uses the GPU by default** (`gpu_layers` -1) and competes with the image model for
  VRAM. It is never run in the background (see `prefetch_next`): a GPU error there would abort
  ComfyUI.
- **prefetch_next makes one extra call per batch** (the rewrite for a run that never comes). Set it
  to `off` for paid APIs if that matters.
- **Your prompts leave the machine** when the API loader points at a hosted service (OpenAI,
  OpenRouter, Anthropic, DashScope, ...), and reference images do too with the edit loader.
- **Node values are saved into workflows and image metadata.** That is why API keys only go in
  Settings > Gadzoinks > LLM; the node refuses `extra_json` or `api_key_name` values that look like
  a key.
- **The edit and text-to-image files are not interchangeable.** The PE-I2I model with the T2I system
  prompt (or the other way round) gives poor or broken prompts.

### Troubleshooting
| Message or symptom | Cause and fix |
|---|---|
| `mmproj must be the vision projector file, not the model itself` (shown under model, mmproj and system_prompt) | There is no mmproj file in `models/LLM`, so the list fell back to all GGUFs. Download the model's `...mmproj...gguf` (see *Getting the models*), refresh, and select it. ComfyUI repeats the one error under every input it checks. |
| `GGUF model / mmproj / System prompt file not found in models/LLM` | The file was moved or renamed; refresh the page and pick it again. |
| `The GGUF prompt enhancer needs the llama_cpp package` | Install llama-cpp-python (see *Requirements*), or use the API loader. |
| `Reference images are connected, but the enhancer is a text-to-image loader` (or the reverse) | Use the matching loader, or an Enhancer Pair with both. |
| API: connection refused or timeout | The server isn't running or isn't listening on the network (`OLLAMA_HOST`), or `base_url` is wrong. |
| API: HTTP 404 | `model` doesn't match the server's name for it (`ollama list`). |
| API: HTTP 401/403 | Missing or wrong key in Settings > Gadzoinks > LLM, or the wrong `api_key_name`. |
| Output ignores the rewriting rules, or no JSON / no `wh_ratio` | No system prompt selected, the wrong one, or (Ollama) `context_length` too small for it. |
| `'<word>' not found in the enhanced prompt, so ... was not applied` | once, then substitute mode and the LLM reworded that value. Switch `enhance` to every run. |
| ComfyUI dies with `Fatal Python error: Aborted` in `llama_cpp` ... `ggml_abort` | llama.cpp hit a GPU error, usually out of VRAM next to the image model. Turn `keep_loaded` off, use a smaller quant or `context_length`, or run the LLM on the CPU (`gpu_layers` 0) or on another machine. |
| Very slow rewrites | Thinking is on with no cap: lower `plan_tokens` or turn thinking off. Check that a GGUF is really on the GPU (`gpu_layers` -1) or that `ollama ps` shows GPU. |

---

# String Formatter

## Description
Builds an output string from supplied inputs and from system variables.
For example, if you connect prompt (or computed_prompt) to A and seed to B, then the format string `generating $a with seed $b on $hostname` will generate a string like `generating a smiling cat with seed 12345 on hal2000`.

## 📝 System Variables Reference

This node provides access to various system variables that can be used in your workflows. Below is a complete list of available variables:

### 📅 Date & Time Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `datetime` | Full date and time | `2024-01-15 14:30:25` |
| `date` | Current date | `2024-01-15` |
| `time` | Current time | `14:30:25` |
| `time_24h` | 24-hour format time | `14:30` |
| `time_12h` | 12-hour format time | `02:30 PM` |
| `iso_datetime` | ISO format datetime | `2024-01-15T14:30:25.123456` |
| `timestamp` | Unix timestamp (seconds) | `1705329025` |
| `timestamp_ms` | Unix timestamp (milliseconds) | `1705329025123` |
| `year` | Full year | `2024` |
| `year_short` | Short year | `24` |
| `month` | Full month name | `January` |
| `month_num` | Month number | `01` |
| `day` | Day of month | `15` |
| `day_num` | Day number (alias for `day`) | `15` |
| `hour` | Hour | `14` |
| `minute` | Minute | `30` |
| `second` | Second | `25` |
| `am_pm` | AM/PM indicator | `PM` |
| `weekday` | Full weekday name | `Monday` |
| `weekday_short` | Short weekday name | `Mon` |

### 🎲 Random & Unique Identifiers
| Variable | Description | Example |
|----------|-------------|---------|
| `uuid` | Full UUID v4 | `123e4567-e89b-12d3-a456-426614174000` |
| `uuid_short` | First 8 chars of UUID | `123e4567` |
| `random_hex` | Random hex string (32-bit) | `a1b2c3d4` |
| `random_int` | Random 4-digit number | `7352` |
| `counter` | Sequential counter (6-digit, increments per execution) | `000042` |
| `batch_id` | Batch identifier (last 8 digits of timestamp_ms) | `90251234` |

### 🗂️ Path & Directory Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `date_path` | Date formatted as path | `2024/01/15` |
| `datetime_path` | Datetime formatted as path | `20240115_143025` |
| `cwd` | Current working directory | `/path/to/comfyui` |
| `model_dir` | ComfyUI models directory | `/path/to/models` |
| `input_dir` | ComfyUI input directory | `/path/to/input` |
| `output_dir` | ComfyUI output directory | `/path/to/output` |
| `temp_dir` | ComfyUI temp directory | `/path/to/temp` |

### 💻 System Information
| Variable | Description | Example |
|----------|-------------|---------|
| `hostname` | Computer hostname | `my-workstation` |
| `node` | Network node name | `my-workstation` |
| `os` | Operating system with release | `Windows 10` |
| `system` | System name | `Windows` |
| `release` | System release | `10.0.19045` |
| `platform` | Full platform info | `Windows-10-10.0.19045` |
| `machine` | Machine type | `AMD64` |
| `processor` | Processor info | `Intel64 Family 6 Model 158` |
| `architecture` | System architecture | `64bit` |
| `cpu_count` | Number of CPU cores | `16` |
| `pid` | Process ID | `12345` |
| `python_version` | Python version | `3.10.12` |
| `user` | Current username | `username` |

### 🎮 GPU Information (if available)
| Variable | Description | Example |
|----------|-------------|---------|
| `cuda_available` | CUDA availability | `True` / `False` |
| `gpu_name` | GPU device name | `NVIDIA GeForce RTX 4090` |
| `gpu_count` | Number of GPUs detected | `1` |

---

# Save Image With Notes

## Description
This node modifies a copy of your workflow, adding a Note node to the new workflow that is then saved inside the image.
You can add your own text with the `notes` input,
or wire `computed_prompt` from GPrompts, which creates a Note and saves the computed prompt in the image metadata.

Note: This node uses the standard ComfyUI Save Image node to do the actual saving.

---

# Load Images From Folder

## Description
Loads images from a folder on the ComfyUI server. It can load a single image, step through the folder one image per run, or pick images at random.

It can also read the image metadata. If an image was saved with **Save Image With Notes** or **Save Image To Immich Server** with a computed prompt, the `prompt` output returns that prompt, so you can re-run or vary old generations.

Supported file types: png, jpg, jpeg, bmp, tiff, webp. Files are sorted by path, and EXIF orientation is applied automatically.

**Credit:** This node is forked from the Load Image Batch node in [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui) by WASasquatch. Thanks for the original work.

## Node Settings
- **mode**:
  - `single_image`: loads the image at `index`. If `index` is larger than the number of images, it wraps around.
  - `incremental_image`: loads the next image each time the workflow runs, and goes back to the first image after the last one. The position resets when ComfyUI restarts.
  - `random`: picks a random image based on `seed`.
- **seed**: seed used by `random` mode.
- **index**: image number to load in `single_image` mode (starts at 0).
- **path**: folder to load images from.
- **pattern**: filename pattern, default `*`. For example `*.png` for PNG files only, or `**/*` to include subfolders.
- **allow_RGBA_output**: if `false`, images with transparency are converted to RGB.
- **filename_text_extension**: if `true`, the `filename_text` output includes the file extension.
- **load_exif**: if `true`, reads metadata from the image so the `prompt` output can be filled in.

## Outputs
- **image**: the loaded image.
- **filename_text**: the file name.
- **width** / **height**: image size in pixels.
- **prompt**: the computed prompt stored in the image, or empty if there isn't one.

---

# Immich Nodes

Save images, video, and audio to an Immich server: https://immich.app

## Configuration (shared by all Immich nodes)
Create an API Key in your Immich server.

Install the node in ComfyUI and go to Settings. In Settings look for the **Gadzoinks** section.
Enter the API Key, the Hostname, and the Port.

- **Save to Disk**: if disabled, the file is deleted from the ComfyUI server file system after uploading to Immich.
- **Default Album**: album to use if none is specified in the node.
- **Default Tags**: these tags are combined with the tags in the node.

Settings are remembered on the ComfyUI server, so saving keeps working after a restart or if the browser is disconnected.

## Save Image To Immich Server

### Description
This node modifies a copy of your workflow, adding a Note node to the new workflow that is then saved inside the image.
You can add your own text with the `notes` input,
or wire `computed_prompt` from GPrompts, which creates a Note and saves the computed prompt in the image metadata.
Supports adding images to albums, and adding tags.

### Node Settings
- **notes**: takes a string and creates a Note node that is added to the workflow saved with the image. Often used with the String Formatter node.
- **computed_prompt**: ignore, will probably be removed.
- **album**: add the image to this album. The album is created if it does not exist.
- **tags**: comma separated tags, merged with the Default Tags from Settings.
- **save_also**: if enabled, the image is saved as normal with ComfyUI. If disabled, the image on ComfyUI is deleted after upload.

## Save Video To Immich Server

### Description
Saves a video generated in ComfyUI and uploads it to your Immich server.
Uses the same Gadzoinks settings as the image node, and supports albums, tags, and optionally keeping a copy on the ComfyUI server.

## Save Audio To Immich Server

### Description
Saves audio generated in ComfyUI and uploads it to your Immich server.

Immich only handles photos and videos, so it does not support audio files such as mp3. To get around this, the audio is saved as an **mp4 video with a still cover image**, which Immich can store and play.

- **Cover image**: optionally connect your own image to use as the cover. If no image is connected, a default cover is used.
- Supports albums, tags, and optionally keeping a copy on the ComfyUI server, the same as the image node.

### Requirement: ffmpeg
**ffmpeg must be installed** and available on the system PATH of the machine running ComfyUI, or this node will not work.

- Windows: `winget install ffmpeg`
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`

You can check that it is installed by running `ffmpeg -version` in a terminal.

** Immich support **
All of these nodes work with standard Immich, but I have my own fork of Immich with extra features such as the ability to see the prompt and metadata of an Image, and the ability to search for text in a prompt ( find all images of dragons )
The installation is still rough, https://github.com/neal3000/immich_gadzoinks/tree/immich_gadzoinks

---

# Logging and debugging
All nodes in this pack write to the ComfyUI console: the terminal window ComfyUI runs in, and the
**Logs** tab in ComfyUI's bottom panel (terminal icon). Messages are tagged by level:

| Level | What you see | Shown by default |
|---|---|---|
| `[INFO]` | one short line per action: which LLM is called, prompt cache use, speed | yes |
| `[WARNING]` | something went wrong but the run continued: an LLM answer that wasn't the expected JSON, an HTTP retry, a failed Immich upload step | yes |
| `[ERROR]` | the run stopped; ComfyUI shows the traceback | yes |
| `[DEBUG]` | step-by-step detail: expansions, parsing, prefetch, settings sync, Immich upload steps | **no** - turn on below |

API keys are never written to the log, not even in debug output.

## Turning on debug logging
Pick one of these.

### 1. In ComfyUI's settings (easiest)
1. Open **Settings** (gear icon).
2. Go to **Gadzoinks > Debug**.
3. Switch on **Debug logging (console)**.

It takes effect immediately, no restart needed. ComfyUI remembers the setting; after a restart it
applies again as soon as the ComfyUI page is open in a browser. Switch it off the same way when you're
done: debug output is verbose.

### 2. With an environment variable (from startup, and without a browser)
Set `GPROMPTS_LOG=DEBUG` before starting ComfyUI. It overrides the setting above, and is useful for
problems during startup or for headless/API use.

- **Linux / macOS**
  ```
  GPROMPTS_LOG=DEBUG python main.py
  ```
- **Windows, Command Prompt**
  ```
  set GPROMPTS_LOG=DEBUG
  python main.py
  ```
- **Windows, PowerShell**
  ```
  $env:GPROMPTS_LOG = "DEBUG"
  python main.py
  ```
- **Windows portable build**: edit `run_nvidia_gpu.bat` (or `run_cpu.bat`) and add the line
  `set GPROMPTS_LOG=DEBUG` above the line that starts ComfyUI.

Other values: `INFO` (the default), `WARNING` (only warnings and errors from this pack).

### 3. With ComfyUI's own `--verbose` flag
`python main.py --verbose DEBUG` turns on debug output for all of ComfyUI and every node pack, not
just this one. It works, but expect a lot of output.

## Saving the output to a file
- **Linux / macOS**: `GPROMPTS_LOG=DEBUG python main.py 2>&1 | tee comfyui-debug.log`
- **Windows**: `python main.py > comfyui-debug.log 2>&1` (the console then stays empty; open the file
  in a text editor)
- Or copy it from the **Logs** tab in ComfyUI.

## Reporting a problem
1. Turn on debug logging (option 1 is enough).
2. Reproduce the problem: queue the workflow again.
3. Copy the console output from just before you queued until the error or wrong result.
4. Include it in your issue, along with which nodes and loader settings you used.

## For developers
Use the pack's logger rather than `print`:
```python
from .common import get_logger
log = get_logger("mypart")          # appears as gprompts.mypart
log.debug("detail %s", value)       # %-style: the text is only built when debug is on
log.info("one line per action")
log.warning("recovered from a problem")
```
`dprint(...)` from `common` still works and logs at debug level.
