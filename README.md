# ComfyUI GPrompts Nodes

## Introduction
This package provides custom nodes for ComfyUI that enhance prompt generation, string formatting, and saving images, video, and audio.

## Nodes Overview
- **GPrompts** - Create dynamic prompts with random or sequential selection. Also supports wildcard files.
- **Dynamic Prompts with Enhancer** + **Prompt Enhancer Loader (GGUF / API)** - GPrompts plus an LLM prompt enhancer (e.g. the Qwen-Image-2.1 prompt rewriter), local GGUF or any Ollama/OpenAI-compatible/Anthropic API, with one LLM call per batch.
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

Only the **first** expansion goes to the LLM. For every later run the node swaps that run's chosen
values into the LLM's rewrite in place of the first run's values, so a whole batch costs one LLM
call. Example: `a {{tiki bar|beach hut}} with {red|amber} lanterns` is rewritten once as
"Cinematic photo of a tiki bar with amber lanterns at dusk", and the next run becomes
"Cinematic photo of a beach hut with red lanterns at dusk".

With **preserve_dynamic_words** on (the default) the LLM is asked to keep those words verbatim. If it
rewords one anyway ("cat" becomes "kitten"), that variation can't be swapped in; the console says so
and that run keeps the first run's value. Changing the text, delimiter style, seed or any enhancer
setting starts over with a new LLM call.

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

### Prompt Enhancer Loader (GGUF)
Put the `.gguf` model and its system prompt file (`.txt` or `.md`) in `ComfyUI/models/LLM/`
(subfolders are fine), then pick them on the node. The model loads the first time a prompt is
enhanced; turn `keep_loaded` off to free its VRAM after each use, or set `gpu_layers` to 0 to run it
on the CPU and leave the GPU to the image model.

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
