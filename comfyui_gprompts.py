"""
Comfyui Nodes Pack
From Gadzoinks Official
https://github.com/GadzoinksOfficial/comfyui_gprompts

"""
import time
import os
import re
import json
import random
import socket
import uuid
import sys
import time
from typing_extensions import override
from fractions import Fraction
from comfy_api.latest import ComfyExtension, io, ui, Input, InputImpl, Types
from comfy.cli_args import args
from pathlib import Path
from collections import defaultdict
import folder_paths
import comfy.model_management as model_management
import server
import comfy
from datetime import datetime
from server import PromptServer
import aiohttp
import platform
import torch
import traceback
import numpy as np
from aiohttp import web
from nodes import PreviewImage, SaveImage
from comfy_extras.nodes_video import SaveVideo
from comfy_execution.graph import ExecutionBlocker
import folder_paths
import os
from PIL import Image
from PIL.ExifTags import TAGS
import json
import threading
from .immich_importer import ImmichImporter
from .common import dprint, DynamicPromptEngine, get_logger, set_debug

log = get_logger("settings")


# Web directory for documentation files
WEB_DIRECTORY = "./web/js"

temp_dir = folder_paths.get_temp_directory()
delete_queue_file = os.path.join(temp_dir, "gz_delete_queue.txt")

last_processed_result = ""
last_processed_result_workflow_id = 0
promtpForId = {}
filename_counter = 0

def get_last_prompt(workflow_id):
    global promtpForId
    return promtpForId.get(workflow_id)

# ----------------------------------------------------------------------------
# Settings handling
#
# Settings originate in the browser frontend (ComfyUI settings screen) and are
# pushed to us via GET /gprompts/setting. Previously they lived only in an
# in-memory dict, which had two failure modes:
#   1. "gadzoinks.request_settings" is broadcast to ALL connected clients, and
#      any client (second tab, half-connected VPN session, client whose
#      settings store hadn't loaded) could answer with blank values and
#      clobber the good ones -> settings "forgotten" mid-batch.
#   2. Every save depended on a live frontend round-trip (send_sync + fixed
#      0.5s sleep). Over a weak/VPN connection the resend arrives late or
#      never, and the upload fails even though we knew the settings moments
#      earlier.
# Now: blank values never overwrite non-blank ones, any complete set of
# settings is persisted to disk and reloaded at startup, and the wait for a
# frontend response is event-driven with a bounded timeout instead of a
# fixed sleep. The frontend becomes a refresh source, not a dependency.
# ----------------------------------------------------------------------------
the_settings = {}
settings_lock = threading.Lock()
settings_updated = threading.Event()

# Placeholder defaults from older versions of the frontend settings screen.
# A fresh client (new browser/incognito) would answer the settings broadcast
# with these and clobber real values, so treat them as "unset".
PLACEHOLDER_VALUES = {"spoon", "example.local"}

# dprint moved to common.py (imported above)

def parse_bool_setting(value, default=False):
    """Settings arrive as strings from the frontend; bool('false') is True,
    so parse explicitly."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("true", "1", "on", "yes")

def apply_settings(params):
    """Merge a settings payload from a frontend client into the_settings.
    Blank values and known placeholder defaults never overwrite good values.
    Persists to disk when the result is a complete Immich set, or holds LLM
    API keys. Returns list of ignored keys."""
    skipped = []
    with settings_lock:
        for key, value in params.items():
            sval = "" if value is None else str(value).strip()
            if sval == "" or sval in PLACEHOLDER_VALUES:
                if the_settings.get(key):
                    skipped.append(key)
                    continue
            the_settings[key] = value
            shown = "***" if is_secret_setting(key) and sval else value
            dprint(f"setting [{key}]={shown}")
        if not get_missing(the_settings) or any(the_settings.get(k) for k in LLM_SETTING_KEYS):
            persist_settings(dict(the_settings))
    if "debug_logging" in params:
        set_debug(parse_bool_setting(params.get("debug_logging")))
    # Wake any execute() currently waiting on a settings refresh
    settings_updated.set()
    return skipped

# LLM API keys for the API prompt enhancer loader (Settings > Gadzoinks > LLM)
LLM_SETTING_KEYS = ("llm_api_key", "llm_api_keys")

def is_secret_setting(key):
    k = key.lower()
    return "apikey" in k or "api_key" in k

def masked_settings(settings):
    """Copy of a settings dict that is safe to print (API keys hidden)."""
    return {k: ("***" if is_secret_setting(k) and v else v) for k, v in (settings or {}).items()}

def get_setting(key, wait_seconds=4.0):
    """One setting value. If it is not known yet (server just started, no
    browser has synced), ask connected frontends to resend and wait briefly,
    then fall back to the persisted settings file."""
    with settings_lock:
        value = the_settings.get(key)
    if value not in (None, ""):
        return value
    settings_updated.clear()
    try:
        PromptServer.instance.send_sync("gadzoinks.request_settings", {})
    except Exception as e:
        dprint(f"get_setting: request_settings send failed: {e}")
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        settings_updated.wait(0.25)
        settings_updated.clear()
        with settings_lock:
            value = the_settings.get(key)
        if value not in (None, ""):
            return value
    value = load_persisted_settings().get(key)
    if value not in (None, ""):
        with settings_lock:
            the_settings.setdefault(key, value)
    return value

def get_settings_file():
    """Where we persist the last known-good settings."""
    try:
        base = folder_paths.get_user_directory()
    except Exception:
        # Older ComfyUI without get_user_directory: keep it next to this node
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "gadzoinks_settings.json")

def load_persisted_settings():
    try:
        with open(get_settings_file(), "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Gadzoinks: could not read persisted settings: {e}")
    return {}

def persist_settings(settings):
    """Atomically write settings to disk (write temp file, then rename)."""
    try:
        path = get_settings_file()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log.warning(f"Gadzoinks: could not persist settings: {e}")

# Load last known-good settings at startup so a server restart (or a frontend
# that never connects, e.g. headless/API usage) still has working values.
the_settings.update(load_persisted_settings())

def get_immich_settings(wait_seconds=4.0):
    """
    Return a consistent snapshot of settings for one save operation.

    If the current settings are complete, return immediately (no sleep, no
    frontend round-trip - this is the fast path for every image in a batch).
    If incomplete, ask connected frontends to resend and wait, event-driven,
    up to wait_seconds. If the frontend is unreachable (weak VPN link, no
    browser attached), fall back to the persisted settings from disk.
    """
    with settings_lock:
        snap = dict(the_settings)
    if not get_missing(snap):
        return snap

    # Ask frontends to resend. send_sync can fail if no client is attached.
    settings_updated.clear()
    try:
        PromptServer.instance.send_sync("gadzoinks.request_settings", {})
    except Exception as e:
        dprint(f"get_immich_settings: request_settings send failed: {e}")

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        settings_updated.wait(0.25)
        settings_updated.clear()
        with settings_lock:
            snap = dict(the_settings)
        if not get_missing(snap):
            return snap

    # Frontend didn't answer in time - fall back to last persisted good values
    disk = load_persisted_settings()
    if disk:
        merged = dict(disk)
        # current non-blank values still win over older persisted ones
        for k, v in snap.items():
            if v not in (None, ""):
                merged[k] = v
        if not get_missing(merged):
            log.info("Gadzoinks: frontend unreachable, using persisted settings")
            with settings_lock:
                the_settings.update(merged)
            return merged
    return snap

dprint("LOADING GPROMPTS")


###
# Utility

def add_to_delete_queue(filepath):
    """Queue a file for deletion"""
    global delete_queue_file
    if isinstance(filepath, Path):
        filepath = str(filepath)
    with open(delete_queue_file, 'w') as f:
        f.write(filepath)

def process_deletion_queue():
    """Call this at plugin start to delete queued files"""
    global delete_queue_file
    dprint(f"delete_queue_file:{delete_queue_file}")
    if not os.path.exists(delete_queue_file):
        return
    with open(delete_queue_file, 'r') as f:
        filepath = f.read().strip()
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except:
            pass
    os.remove(delete_queue_file)


def get_missing(settings):
    missing = []
    if not settings.get("immich_hostname"): missing.append("Hostname")
    if not settings.get("immich_port"):     missing.append("Port")
    if not settings.get("immich_apikey"):   missing.append("Api Key")
    return missing

async def request_settings_from_frontend():
    await PromptServer.instance.send("gadzoinks.request_settings", {}, sid=None)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL Hex
def pil2hex(image):
    return hashlib.sha256(np.array(tensor2pil(image)).astype(np.uint16).tobytes()).hexdigest()

# PIL to Mask
def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask

def extract_computed_prompt(data):
    """
    Traverse a nested dictionary to find class_type == 'GPrompts'
    and extract _meta['computed_prompt']
    """
    # If data is a dictionary, check its values
    if isinstance(data, dict):
        # Check if current node has class_type == 'GPrompts'
        if data.get('class_type') == 'GPrompts':
            # Extract computed_prompt from _meta
            if '_meta' in data and 'computed_prompt' in data['_meta']:
                return data['_meta']['computed_prompt']
        
        # Recursively traverse all values in the dictionary
        for key, value in data.items():
            result = extract_computed_prompt(value)
            if result is not None:
                return result
    
    # If data is a list, traverse each item
    elif isinstance(data, list):
        for item in data:
            result = extract_computed_prompt(item)
            if result is not None:
                return result
    
    return None

def add_note_node_to_workflow( workflow, note_text=None):
    """Helper to add a note node to workflow"""
    nodes = workflow.get("nodes", [])

    # Get next available node ID
    max_id = 0
    for node in nodes:
        node_id = node.get("id", "0")
        try:
            node_id_int = int(node_id)
            max_id = max(max_id, node_id_int)
        except (ValueError, TypeError):
            continue

    note_node_id = str(max_id + 1)
    # Create note node
    note_node = {
        "id": note_node_id,
        "type": "Note",
        "pos": [50, 50],  # Top-left corner
        "size": {"0": 425, "1": 180},
        "flags": {},
        "order": len(nodes) + 1,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {"Node name for S&R": "Note"},
        "widgets_values": [note_text]
    }

    workflow["nodes"].append(note_node)

def extract_exif(image):
    """Extract metadata from image including ComfyUI prompt data"""
    metadata = {}
    
    # Try to get standard EXIF data (for JPEGs, etc.)
    try:
        if hasattr(image, '_getexif') and image._getexif():
            exif = image._getexif()
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                metadata[f"EXIF_{tag}"] = str(value)
    except Exception as e:
        metadata["EXIF_Error"] = f"Failed to read EXIF: {str(e)}"
    
    # Get PNG text chunks (where ComfyUI stores metadata)
    if hasattr(image, 'info') and image.info:
        png_info = image.info
        
        # Extract specific ComfyUI fields
        comfy_fields = ['Prompt', 'Workflow', 'computed_prompt', 'prompt', 'workflow']
        for field in comfy_fields:
            if field in png_info:
                metadata[field] = png_info[field]
        
        # Try to extract just the positive prompt from the Prompt JSON
        if 'Prompt' in png_info:
            try:
                prompt_data = json.loads(png_info['Prompt'])
                # Look for CLIPTextEncode nodes that might contain the prompt
                for node_id, node_data in prompt_data.items():
                    if node_data.get('class_type') == 'CLIPTextEncode':
                        if 'title' in node_data.get('_meta', {}):
                            if node_data['_meta']['title'] in ['pos', 'positive']:
                                if 'inputs' in node_data and 'text' in node_data['inputs']:
                                    metadata['Positive_Prompt'] = node_data['inputs']['text']
            except:
                pass
        
        # Also include any other PNG text chunks (for debugging)
        other_fields = [k for k in png_info.keys() 
                       if k not in ['Prompt', 'Workflow', 'Computed prompt', 'prompt', 'workflow']]
        for field in other_fields:
            if isinstance(png_info[field], str) and len(png_info[field]) < 1000:
                metadata[f"{field}"] = png_info[field]
    
    # Format as readable string
    if metadata:
        metadata_str = "\n".join([f"{k}: {v[:200]}..." if len(str(v)) > 200 else f"{k}: {v}" 
                                  for k, v in metadata.items()])
    else:
        metadata_str = "No metadata found"
    
    return metadata, metadata_str

def OLDextract_exif( image):
    """Extract EXIF data from image and return date and  formatted string"""
    exif_data = {}
    exif_str = ""
    try:
        if hasattr(image, '_getexif') and image._getexif():
            exif = image._getexif()
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = str(value)
    except Exception as e:
        exif_data["Error"] = f"Failed to read EXIF: {str(e)}"
    
    # Format as readable string
    if exif_data:
        exif_str =  "\n".join([f"{k}: {v}" for k, v in exif_data.items()])
    return exif_data,exif_str
########
## Load Image Batch
## forked from was-ns custom node
import os
import glob
import random
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

class LoadImagesBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["single_image", "incremental_image", "random"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "pattern": ("STRING", {"default": '*', "multiline": False}),
                "allow_RGBA_output": (["false", "true"],),
            },
            "optional": {
                "filename_text_extension": (["true", "false"],),
                "load_exif": (["true", "false"],),  # New option to enable/disable EXIF loading
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "filename_text", "width", "height", "prompt")
    FUNCTION = "load_batch_images"

    CATEGORY = "gprompts/image"


    def load_batch_images(self, path, pattern='*', index=0, mode="single_image", 
                         seed=0, allow_RGBA_output='false', 
                         filename_text_extension='true', load_exif='true'):
        
        allow_RGBA = (allow_RGBA_output == 'true')
        load_exif_data = (load_exif == 'true')

        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")
            
        # Load all image paths
        image_paths = []
        for file_name in glob.glob(os.path.join(path, pattern), recursive=True):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                image_paths.append(os.path.abspath(file_name))
        
        image_paths.sort()
        
        if not image_paths:
            raise ValueError(f"No images found in {path} with pattern {pattern}")

        # Select image based on mode
        if mode == 'single_image':
            selected_index = index % len(image_paths)  # Wrap around if index too high
            image_path = image_paths[selected_index]
        elif mode == 'incremental_image':
            # For incremental, we'll just cycle through based on a simple counter
            # You can modify this to store state if needed
            if not hasattr(self, '_incremental_counter'):
                self._incremental_counter = {}
            
            counter_key = f"{path}_{pattern}"
            if counter_key not in self._incremental_counter:
                self._incremental_counter[counter_key] = 0
            
            selected_index = self._incremental_counter[counter_key]
            self._incremental_counter[counter_key] = (selected_index + 1) % len(image_paths)
            image_path = image_paths[selected_index]
        else:  # random mode
            random.seed(seed)
            selected_index = int(random.random() * len(image_paths))
            image_path = image_paths[selected_index]

        # Load the image
        try:
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image)  # Apply orientation from EXIF
            
            # Get dimensions
            width, height = image.size
            
            # Extract EXIF data if requested
            exif_data = {}
            exif_string = None
            if load_exif_data:
                exif_data, exif_string = extract_exif(image)
                dprint(f"Loaded exif:\n{exif_data}")

            exif_prompt = exif_data.get("computed_prompt","")
            #TODO backtrace to find pos prompt

            
            # Convert RGBA if needed
            if not allow_RGBA and image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode != 'RGB' and image.mode != 'RGBA':
                image = image.convert('RGB')
            
            # Get filename
            filename = os.path.basename(image_path)
            if filename_text_extension == "false":
                filename = os.path.splitext(filename)[0]
            
            image_tensor = pil2tensor(image)
            
            return (image_tensor, filename, width, height, exif_prompt)
            
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Error loading image {image_path}: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution for incremental and random modes
        if kwargs.get('mode') != 'single_image':
            return float("NaN")
        
        # For single_image, check if file has changed
        if 'path' in kwargs and 'pattern' in kwargs:
            path = kwargs['path']
            pattern = kwargs.get('pattern', '*')
            index = kwargs.get('index', 0)
            
            # Get the specific image file
            image_paths = []
            for file_name in glob.glob(os.path.join(path, pattern), recursive=True):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    image_paths.append(os.path.abspath(file_name))
            
            if image_paths and index < len(image_paths):
                # Return file hash to detect changes
                import hashlib
                with open(image_paths[index], 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()
        
        return float("NaN")

#######
### Save with Notes
class GImageSaveWithExtraMetadata(SaveImage):
    def __init__(self):
        super().__init__()
        self.data_cached = None
        self.data_cached_text = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to save with metadata"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Prefix for the output filename (supports $variables for dynamic naming)"
                })
            },
            "optional": {
                "notes": ("*", {
                    "default": "",
                    "tooltip": "Text for notes node that is embedded in saved image"
                }),
                "computed_prompt": ("*", {
                    "default": "",
                    "tooltip": "The computed prompt from Gprompts Node to embed in saved image (use this OR notes)"
                })
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "gprompts/image"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"

    DESCRIPTION = (
        "Saves an image with additional metadata embedded in the PNG info. "
        "Can include computed prompts and notes that will be stored in the image file "
        "creates a new Notes node and then saves with that in the workflow."
    )

    def execute(self, image=None, filename_prefix="ComfyUI", notes=None, computed_prompt=None, prompt = None,extra_pnginfo=None):
        if not extra_pnginfo:
            extra_pnginfo_new = {}
        else:
            extra_pnginfo_new = extra_pnginfo.copy()
        note_text = None
        if computed_prompt:
            extra_pnginfo_new["computed_prompt"] = computed_prompt
            note_text = f'Image created with prompt "{computed_prompt}"'
        if notes and not computed_prompt:
            note_text = notes

        if not "%" in filename_prefix:
            filename_prefix = datetime.now().strftime("%Y-%m-%d") + os.path.sep + filename_prefix

        # Add note node to workflow
        if note_text and prompt and "workflow" in extra_pnginfo_new:
            workflow = extra_pnginfo_new["workflow"]
            add_note_node_to_workflow(workflow, note_text)

        # Save image
        saved = super().save_images(image, filename_prefix, prompt, extra_pnginfo_new)
        return saved


############ 
# Save to Immich server
class GImageSaveImmich(SaveImage):
    def __init__(self):
        super().__init__()
        self.data_cached = None
        self.data_cached_text = None
    """
    # DO NOT USE VALIDATE - it deactivates the node, but provides no user feedback as to why 
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if not the_settings.get("immich_apikey"):
            return "Immich api key is not set — please configure it in settings"

        if not the_settings.get("immich_hostname"):
            return "Immich server hostname is not set — please configure it in settings"

        if not the_settings.get("immich_port"):
            return "Immich server port is not set — please configure it in settings"

        return True
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to save with metadata"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Prefix for the output filename (supports $variables for dynamic naming)"
                }),
                "album" : ("STRING", {
                    "default": "",
                    "tooltip": "optional album name, leave blank to use default value from settings"
                }),
                "tags" : ("STRING", {
                    "default": "",
                    "tooltip": "optional tags for images. Comma seperated. Merged with Tags from Settings."
                }),
                "save_also" : ("BOOLEAN", {
                    "default": parse_bool_setting(the_settings.get("immich_save_also"), default=True),
                    "tooltip": "Also save image to disk on the comfyui server"
                }),
            },
            "optional": {
                "notes": ("*", {
                    "default": "",
                    "tooltip": "Optional text for notes node that is embedded in saved image"
                }),
                "computed_prompt": ("*", {
                    "default": "",
                    "tooltip": "The computed prompt from Gprompts Node to embed in saved image (use this OR notes)"
                })
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "gprompts/immich"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    
    DESCRIPTION = (
        "Saves an image to Immich server ( and optionaly to file system ).",
        "with additional metadata embedded in the PNG info. "
        "Can include computed prompts and notes that will be stored in the image file "
        "creates a new Notes node and then saves with that in the workflow."
    )

    def execute(self, image=None, filename_prefix="ComfyUI", album=None,tags="",save_also=True, notes=None, computed_prompt=None,unique_id=0, prompt = None,extra_pnginfo=None):
        global last_processed_result,last_processed_result_workflow_id,promtpForId,filename_counter
        filename_counter = filename_counter+1
        workflow_id = extra_pnginfo.get("workflow",{}).get("id")
        last_prompt = promtpForId.get(workflow_id)
        process_deletion_queue()
        # One consistent settings snapshot for this whole save. Fast path is
        # instant when settings are already known; only blocks (bounded) when
        # they are genuinely missing. Falls back to persisted settings if the
        # frontend is unreachable (e.g. semi-disconnected VPN client).
        settings = get_immich_settings()
        #dprint(f"workflow_id {workflow_id}   last_prompt:{last_prompt}")
        #dprint(f"entry extra_pnginfo:\n{extra_pnginfo}")
        #dprint(f"prompt:\n{prompt}")
        if not extra_pnginfo:
            extra_pnginfo_new = {}
        else:
            extra_pnginfo_new = extra_pnginfo.copy()
        note_text = None
        # Try in order notes,computed_prompt, global value of gprompt node
        if notes:
            note_text = notes
        if not note_text and computed_prompt:
            note_text = f'Image created with prompt "{computed_prompt}"'
        if not note_text and last_prompt:
            note_text = f'Image created with prompt "{last_prompt}"'
        if "computed_prompt" not in extra_pnginfo_new:
            if computed_prompt:
                extra_pnginfo_new["computed_prompt"] = computed_prompt
            else:
                extra_pnginfo_new["computed_prompt"] = last_prompt
        if not album:
            album = settings.get('immich_default_album')
        basetags = settings.get('immich_base_tags') or ""
        tags = tags or ""
        all_tags = (tags + ',' + basetags).split(',')
        user_tags = list({tag.strip() for tag in all_tags if tag.strip()})
        
        # still testing this. maybe extract model and loras for tag values
        # gen_tags = [ "gz|software|comfyui","gz|basemodel|flux2", "gz|lora|realism", "gz|lora|scifi" ]
        gen_tags = [ "gz|software|comfyui" ]
        structured_tags = user_tags + gen_tags
        imm_fullpath = ""
        if not "%" in filename_prefix:
            filename_prefix = datetime.now().strftime("%Y-%m-%d") + os.path.sep + filename_prefix
        imm_filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        workflow = None

        extracted_prompt = extract_computed_prompt(prompt)
        #dprint(f">>>>>>>>>> last_prompt: {last_prompt}")
        #dprint(f">>>>>>>>>> extracted_prompt: {extracted_prompt}")

        # Add note node to workflow
        if note_text and  "workflow" in extra_pnginfo_new:
            workflow = extra_pnginfo_new["workflow"]
            add_note_node_to_workflow(workflow, note_text)

        # Save image (always use save_images() to create EXIF and workflow data)
        #dprint(f"extra_pnginfo_new:\n:{extra_pnginfo_new}")
        dprint(f"filename_prefix:{filename_prefix}")
        saved = super().save_images(image, filename_prefix, prompt, extra_pnginfo_new)
        #saved:{'ui': {'images': [{'filename': 'itest_00001_.png', 'subfolder': '2026-02-21', 'type': 'output'}]}}
        rc = saved

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, image.shape[1], image.shape[0])
        dprint(f"full_output_folder:{full_output_folder}")
        images = saved.get('ui',{}).get('images')
        imm_fullpath = None
        if images:
            imm_filename = images[0].get('filename', '') # itest_00001_.png
            imm_fullpath = full_output_folder + os.sep + imm_filename # file to delete (maybe)
            imm_filename = imm_filename.replace("_.png", ".png") # not sure why extra '_'
        if not save_also:
            # if not saving we get stuck at 00001 , so use a counter intead
            imm_filename = re.sub(r'(\d+)(?=[^_]*$)', str(filename_counter), imm_filename)
        dprint(f"saved:{saved.get('ui') if isinstance(saved, dict) else saved}")
        dprint(f"imm_filename:{imm_filename}")
        dprint(f"imm_fullpath:{imm_fullpath}")
        # Validation, I am putting this after the image is saved to file system
        server = settings.get("immich_hostname")
        port = settings.get("immich_port")
        api_key = settings.get('immich_apikey')
        flag_includehostname = parse_bool_setting(settings.get('immich_includehostname'))
        dprint(f"flag_includehostname:{flag_includehostname}")
        url = f"http://{server}:{port}"
        missing = get_missing(settings)
        if missing:
            log.error("IMMICH CONFIGURATION ERROR - Save to Immich Server node is missing: %s. "
                      "Please configure in Settings > Gadzoinks.", ", ".join(missing))
            log.debug("settings snapshot: %s", masked_settings(settings))
            # Generate an error so the user gets alerted to what is wrong
            error_msg = f": Missing {', '.join(missing)}. Open Settings, Gadzoinks to configure."
            raise ValueError(error_msg)
        upload_rc = None
        try:
            if imm_filename:
                importer = ImmichImporter(url, api_key)
                ext=['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif', '.avif']
                rating = None
                upload_rc = None
                if not save_also:
                    directory = os.path.dirname(imm_fullpath)
                    original_filename = os.path.basename(imm_fullpath)
                    temp_fullpath = os.path.join(directory, imm_filename)
                    os.rename(imm_fullpath, temp_fullpath)
                    try:
                        upload_rc = importer.upload_photo(temp_fullpath, album, tags = user_tags,structured_tags=structured_tags,rating=rating, comfy_workflow=workflow)
                    finally:
                        os.rename(temp_fullpath, imm_fullpath)
                else:
                    upload_rc = importer.upload_photo(imm_fullpath, album, tags = user_tags,structured_tags=structured_tags,rating=rating, comfy_workflow=workflow)
                dprint(f"upload_rc:{upload_rc}")
        finally:
            # mark file for Deletion if we're not keeping it
            # cannot delete immediate or UI has nothing to show
            if not save_also and imm_fullpath and os.path.exists(imm_fullpath):
                add_to_delete_queue(imm_fullpath)
        try:
            dprint(f"upload_rc:{upload_rc}")
            if upload_rc and upload_rc.get("success"):
                d = {}
                if extra_pnginfo_new:
                    if 'workflow' in extra_pnginfo_new:
                        d['workflow'] = extra_pnginfo_new['workflow']
                    if 'computed_prompt' in extra_pnginfo_new:
                        d['computed_prompt'] = extra_pnginfo_new['computed_prompt']
                if prompt:
                    d['promptflow'] = prompt
                if flag_includehostname:
                    d["source_host"] =  socket.gethostname()
                importer.gz_ingest(upload_rc.get("id"),upload_rc.get("album_id"),d)
        except Exception as e:
            dprint(f"Error /gz/ingest: {e}")
        return rc

# ============================================================================
# StringFormatter Node - Acts like sprintf
# ============================================================================

class StringFormatter:
    def __init__(self):
        self.counter = 0
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format_string": ("STRING", {
                    "multiline": True, 
                    "default": "The prompt is $a, the seed is $b, width is $c, height is $d. Generated at $datetime on $hostname, Operating system $os",
                    "tooltip": "Template string with $variables (e.g., $a, $datetime, $hostname). Use $a through $h for inputs."
                }),
            },
            "optional": {
                "a": ("*", {"default": "", "tooltip": "Input A - any value (automatically converted to string)"}),
                "b": ("*", {"default": "", "tooltip": "Input B - any value (automatically converted to string)"}),
                "c": ("*", {"default": "", "tooltip": "Input C - any value (automatically converted to string)"}),
                "d": ("*", {"default": "", "tooltip": "Input D - any value (automatically converted to string)"}),
                "e": ("*", {"default": "", "tooltip": "Input E - any value (automatically converted to string)"}),
                "f": ("*", {"default": "", "tooltip": "Input F - any value (automatically converted to string)"}),
                "g": ("*", {"default": "", "tooltip": "Input G - any value (automatically converted to string)"}),
                "h": ("*", {"default": "", "tooltip": "Input H - any value (automatically converted to string)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_string",)
    FUNCTION = "format_string"
    CATEGORY = "gprompts/text"
    
    DESCRIPTION = (
        "Node that builds strings by replacing $variables. "
        "Supports custom inputs a-h and system variables like $datetime, $hostname, $os, etc. "
        "Use with GPrompts Save Image to create notes, or anywhere else for creating dynamic filenames, prompts , etc."
    )

    def get_system_variables(self):
        """Get predefined system variables"""
        now = datetime.now()
        self.counter += 1
        
        # Base variables
        variables = {
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "hostname": socket.gethostname(),
            "os": f"{platform.system()} {platform.release()}",
            "month": now.strftime("%B"),
            "year": str(now.year),
            "day": now.strftime("%d"),
            "timestamp": str(int(now.timestamp())),
            "timestamp_ms": str(int(now.timestamp() * 1000)),
            "iso_datetime": now.isoformat(),
            "time_24h": now.strftime("%H:%M"),
            "time_12h": now.strftime("%I:%M %p"),
            "weekday": now.strftime("%A"),
            "weekday_short": now.strftime("%a"),
            "month_num": now.strftime("%m"),
            "day_num": now.strftime("%d"),
            "year_short": now.strftime("%y"),
            "hour": now.strftime("%H"),
            "minute": now.strftime("%M"),
            "second": now.strftime("%S"),
            "am_pm": now.strftime("%p"),
            
            # Random/Unique
            "uuid": str(uuid.uuid4()),
            "uuid_short": str(uuid.uuid4())[:8],
            "random_hex": format(random.getrandbits(32), '08x'),
            "random_int": str(random.randint(1000, 9999)),
            "counter": "{:06d}".format(self.counter),
            "date_path": now.strftime("%Y/%m/%d"),
            "datetime_path": now.strftime("%Y%m%d_%H%M%S"),
            "batch_id": str(int(now.timestamp() * 1000))[-8:],
            
            # System
            "cpu_count": str(os.cpu_count()),
            "pid": str(os.getpid()),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
            "node": platform.node(),
            "architecture": platform.architecture()[0],
            "user": os.getenv('USERNAME') or os.getenv('USER') or 'unknown',
            "cwd": os.getcwd(),
        }
        
        # Add ComfyUI paths if available
        try:
            variables.update({
                "model_dir": folder_paths.models_dir,
                "input_dir": folder_paths.input_directory,
                "output_dir": folder_paths.output_directory,
                "temp_dir": folder_paths.temp_directory,
            })
        except:
            pass
        
        # Add GPU info if torch is available
        if 'torch' in sys.modules:
            import torch
            variables["cuda_available"] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                variables["gpu_name"] = torch.cuda.get_device_name(0)
                variables["gpu_count"] = str(torch.cuda.device_count())
            else:
                variables["gpu_name"] = "none"
                variables["gpu_count"] = "0"
        
        return variables

    def format_string(self, format_string, a="", b="", c="", d="", e="", f="", g="", h=""):
        """
        Main execution function for the StringFormatter node
        Acts like sprintf by replacing $variables with their values
        """
        # Create dictionary of user inputs
        user_vars = {
            "a": str(a) if a != "" else "",
            "b": str(b) if b != "" else "",
            "c": str(c) if c != "" else "",
            "d": str(d) if d != "" else "",
            "e": str(e) if e != "" else "",
            "f": str(f) if f != "" else "",
            "g": str(g) if g != "" else "",
            "h": str(h) if h != "" else "",
        }
        
        # Get system variables
        system_vars = self.get_system_variables()
        
        # Combine all variables
        all_vars = {**user_vars, **system_vars}
        
        # Replace all $variables in the format string
        result = format_string
        
        # Sort keys by length (longest first) to avoid partial replacements
        # e.g., $datetime should be replaced before $date
        sorted_keys = sorted(all_vars.keys(), key=len, reverse=True)
        
        for key in sorted_keys:
            placeholder = f"${key}"
            if placeholder in result:
                result = result.replace(placeholder, all_vars[key])
        
        return (result,)
           
#####
# GPrompts
# Dynamic prompt text genereration
#
class GPrompts(DynamicPromptEngine):
    # The expansion engine (delimiter styles, registers, sequential/random
    # blocks, wildcards) lives in common.DynamicPromptEngine so the
    # 'Dynamic Prompts with Enhancer' node can share it.

    def beforeQueued(self, args):
        dprint(f"beforeQueued args:{args}")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "tooltip": "Input text with dynamic blocks. Curly style: {option1|option2} random, {{option1|option2}} sequential. Angle style: <option1|option2> random, <<option1|option2>> sequential. __wildcard__ for wildcard files. Registers 0-9 reuse a chosen value: {{0 monkey|dog}} stores the pick, {{0}} recalls it (sequential and random banks are separate)."
                }),
            },
            "optional": {
                "delimiter_style": (list(cls.DELIM_STYLES.keys()), {
                    "default": cls.DEFAULT_DELIM_STYLE,
                    "tooltip": "Delimiters for dynamic blocks. 'curly { }' is the classic {a|b} / {{a|b}} syntax (now JSON-safe). 'angle < >' uses <a|b> / <<a|b>> instead, handy when the prompt itself is JSON."
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for random generation (combined with iteration for reproducibility)"
                }),
                "computed_prompt": ("STRING", {
                    "multiline": True, 
                    "readonly": True,
                    "default": "",
                    "tooltip": "Output of the processed prompt (read-only)"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "dynamic_prompt", "computed_prompt", "seed")
    FUNCTION = "process_dynamic_prompt"
    CATEGORY = "gprompts"
    
    DESCRIPTION = (
        "Dynamic prompt generator with support for random {}, sequential {{}}, and wildcard __word__ syntax. "
        "A delimiter_style toggle switches to <> / <<>> delimiters for use inside JSON prompts. "
        "Registers 0-9 let a chosen value be reused: {{0 monkey|dog}} stores the pick and {{0}} recalls it; "
        "sequential and random blocks have separate register banks. "
        "Perfect for creating variations in prompts, batch processing, and A/B testing different prompt combinations. "
        "Supports weighted options and nested wildcards from text/json files."
    )

    def process_dynamic_prompt(self, text, seed=0, computed_prompt="",delimiter_style=None,unique_id=0,prompt=None,extra_pnginfo=None):
        global last_processed_result,last_processed_result_workflow_id,promtpForId
        dynamic_text = text
        workflow_id = extra_pnginfo.get("workflow",{}).get("id")
        if delimiter_style not in self.DELIM_STYLES:
            delimiter_style = self.DEFAULT_DELIM_STYLE
        self.delims = self.DELIM_STYLES[delimiter_style]
        # If text or delimiter style changed, reset the iteration counter
        if self.previous_text != text or self.previous_delim_style != delimiter_style:
            self.previous_text = text
            self.previous_delim_style = delimiter_style
            self.current_iteration = 0
            self.sequential_combinations = []
        
        # Set the random seed for reproducibility
        if seed > 0:
            random.seed(seed + self.current_iteration)
        
        # Process the prompt with dynamic elements
        processed_text = self.parse_dynamic_prompt(text)
        
        # Increment the iteration counter for next time
        self.current_iteration += 1
        
        last_processed_result = processed_text
        if workflow_id:
            promtpForId[workflow_id] = processed_text
            dprint(f"promtpForId: {promtpForId}")
        dprint(f"process_dynamic_prompt:     unique_id:{unique_id}  last_processed_result:{last_processed_result} QAQ")
        # send a message to front end so we can update the text in UI
        PromptServer.instance.send_sync("gprompts_executed", 
                                {"node_id": unique_id, "result": processed_text} )
        # Create and populate a DynamicPrompt object with our data
        # have to wire the node together to pass DynamicPrompt
        dynamic_data = None
        
        if prompt is not None:
            from comfy_execution.graph import DynamicPrompt
            dynamic_data = DynamicPrompt(prompt)
            # Store our processed data as an ephemeral node
            metadata_node_id = "computed_prompt"
            nodeInfo = {
                "class_type": "GPromptsData",
                "data": {
                    "original_text": text,
                    "computed_prompt": processed_text,
                    "seed": seed,
                    "iteration": self.current_iteration
                }
            }
            dprint(f"nodeInfo:{nodeInfo}")
            dynamic_data.add_ephemeral_node(
                node_id=metadata_node_id,
                node_info=nodeInfo,
                parent_id=unique_id,
                display_id=unique_id
            )
        # This is a lot simpler just stick in node._meta, problem is if we have multiple gprompts in a workflow
        if prompt is not None:
            dprint(f"processed_text: {processed_text}")
            node = prompt[unique_id]
            node["inputs"]["computed_prompt"] = processed_text  # update computed_prompt, value passed to us was previous version
            meta = node.get("_meta",{})
            meta["computed_prompt"] = processed_text
            node["_meta"] = meta
            dprint(f"updated node:\n{node}")
        return (processed_text,dynamic_text,processed_text,seed)

    @PromptServer.instance.routes.get("/gprompts/prompt")
    async def prompt(request):
        global last_processed_result
        #dprint(f"@PromptServer.instance.routes.get(/gprompts/prompt)   last_processed_result:{last_processed_result} ")
        return web.json_response( { "prompt":"dummy value" } );
        #eturn web.json_response( { "prompt":last_processed_result } );
######
    @PromptServer.instance.routes.get("/gprompts/setting")
    async def setting(request):
        # Legacy GET endpoint (query params). Kept for backwards compatibility
        # with older frontend JS; new JS uses the POST endpoint below so the
        # API key doesn't travel in the URL / server logs.
        dprint(f"/gprompts/setting GET")
        params = dict(request.rel_url.query)
        skipped = apply_settings(params)
        if skipped:
            dprint(f"setting: ignored blank/placeholder values for {skipped}")
        dprint(f"setting the_settings {masked_settings(the_settings)}")
        return web.Response(text=f"Parameters received {masked_settings(params)}")

    @PromptServer.instance.routes.post("/gprompts/setting")
    async def setting_post(request):
        dprint(f"/gprompts/setting POST")
        try:
            params = await request.json()
            if not isinstance(params, dict):
                raise ValueError("expected JSON object")
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        skipped = apply_settings(params)
        if skipped:
            dprint(f"setting: ignored blank/placeholder values for {skipped}")
        return web.json_response({"ok": True, "ignored": skipped})
#######
# Node registration
NODE_CLASS_MAPPINGS = {
    "GPrompts": GPrompts,
    "GImageSaveImmich" : GImageSaveImmich,
    "GImageSaveWithExtraMetadata": GImageSaveWithExtraMetadata,
    "StringFormatter": StringFormatter,
    "LoadImagesBatch":LoadImagesBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPrompts": "Dynamic Prompts",
    "GImageSaveImmich" : "Save the image to a Immich Server",
    "GImageSaveWithExtraMetadata": "Save Image and add Note node to embedded workflow",
    "StringFormatter": "String Formatter (sprintf)",
    "LoadImagesBatch" : "Load images from folder"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']




