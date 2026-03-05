"""
Comfyui Nodes Pack
From Gadzoinks Official
https://github.com/GadzoinksOfficial/comfyui_gprompts

"""
import os
import re
import json
import random
import socket
import uuid
import sys
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
from comfy_execution.graph import ExecutionBlocker
from .immich_importer import ImmichImporter


# Web directory for documentation files
WEB_DIRECTORY = "./web/js"

last_processed_result = ""
the_settings = {}

print("LOADING GPROMPTS")

def dprint(a):
    print(a)
    pass

###
# Utility
# Tensor to PIL

def get_missing(settings):
    missing = []
    if not settings.get("immich_hostname"): missing.append("Hostname")
    if not settings.get("immich_port"):     missing.append("Port")
    if not settings.get("immich_apikey"):   missing.append("Api Key")
    return missing

async def request_settings_from_frontend():
    await PromptServer.instance.send("gadzoinks.request_settings", {}, sid=None)

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
from PIL import Image
from PIL.ExifTags import TAGS
import json

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

    CATEGORY = "Image/Loaders"


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

    CATEGORY = "image"
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
                    "default": the_settings.get("immich_save_also",True),
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
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "image"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    
    DESCRIPTION = (
        "Saves an image to Immich server ( and optionaly to file system ).",
        "with additional metadata embedded in the PNG info. "
        "Can include computed prompts and notes that will be stored in the image file "
        "creates a new Notes node and then saves with that in the workflow."
    )

    def execute(self, image=None, filename_prefix="ComfyUI", album=None,tags="",save_also=True, notes=None, computed_prompt=None, prompt = None,extra_pnginfo=None):
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
        if not album:
            album = the_settings.get('immich_default_album')
        basetags = the_settings.get('immich_base_tags') or ""
        tags = tags or ""
        all_tags = (tags + ',' + basetags).split(',')
        tags_list = list({tag.strip() for tag in all_tags if tag.strip()})

        imm_fullpath = ""
        if not "%" in filename_prefix:
            filename_prefix = datetime.now().strftime("%Y-%m-%d") + os.path.sep + filename_prefix
        imm_filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        workflow = None

        # try and find gprompts node (will not handle multiple instances reliably)
        computed_prompt = None
        if note_text and prompt and "workflow" in extra_pnginfo_new:
            workflow = extra_pnginfo_new["workflow"] 
            nodes = workflow.get("nodes", [])
            for node in nodes:
                if node.get("type") == "GPrompts": 
                    dprint(f"GPrompts node:{node}")
                    computed_prompt = node.get("properties",{}).get("_meta",{}).get("computed_result")
        # add computed_prompt as top level in embedded in json
        if computed_prompt:
            extra_pnginfo_new['computed_prompt'] = computed_prompt
        # Add note node to workflow
        if note_text and prompt and "workflow" in extra_pnginfo_new:
            workflow = extra_pnginfo_new["workflow"]
            add_note_node_to_workflow(workflow, note_text)

        # Save image (always use save_images() to create EXIF and workflow data)

        
        dprint(f"extra_pnginfo_new:\n:{extra_pnginfo_new}")

        saved = super().save_images(image, filename_prefix, prompt, extra_pnginfo_new)
        #saved:{'ui': {'images': [{'filename': 'itest_00001_.png', 'subfolder': '2026-02-21', 'type': 'output'}]}}
        rc = saved

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, image.shape[1], image.shape[0])
        dprint(f"full_output_folder:{full_output_folder}")
        images = saved.get('ui',{}).get('images')
        if images:
            imm_filename = images[0].get('filename', '')
        dprint(f"saved:{saved}")
        dprint(f"imm_filename:{imm_filename}")
        imm_fullpath = full_output_folder + os.sep + imm_filename
        dprint(f"imm_fullpath:{imm_fullpath}")
        # Validation, I am putting this after the image is saved to file system
        server = the_settings.get("immich_hostname")
        port = the_settings.get("immich_port")
        api_key = the_settings.get('immich_apikey')
        url = f"http://{server}:{port}"
        missing = get_missing(the_settings)
        if missing:
            print(f"\n🟡 IMMICH settings missing ({', '.join(missing)}), requesting from frontend...")
            # this call, forces javascript to resend settings to this code async via "/gprompts/setting". which is why we sleep after
            PromptServer.instance.send_sync("gadzoinks.request_settings", {})
            time.sleep(0.3)
            server  = the_settings.get("immich_hostname")
            port    = the_settings.get("immich_port")
            api_key = the_settings.get('immich_apikey')
            url     = f"http://{server}:{port}"
            missing = get_missing(the_settings)
        if missing:
            print("\n🔴 IMMICH CONFIGURATION ERROR")
            print(f"Save Image to Immich Server Node Missing: {', '.join(missing)}")
            print("Please configure in Settings:Gadzoinks")
            print(f"global settings:{the_settings}")
            # Generate an error so the user gets alerted to what is wrong
            error_msg = f": Missing {', '.join(missing)}. Open Settings, Gadzoinks to configure."
            raise ValueError(error_msg)
        try:
            if imm_filename:
                importer = ImmichImporter(url, api_key)
                ext=['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif', '.avif']
                rating = None
                importer.upload_photo(imm_fullpath, album, tags_list, rating, workflow)
        finally:
            # Delete the file if we're not keeping it
            if not save_also and imm_fullpath and os.path.exists(imm_fullpath):
                rc = {} 
                try:
                    os.unlink(imm_fullpath)
                    dprint(f"Deleted file (save_also=False): {imm_fullpath}")
                except Exception as e:
                    dprint(f"Error deleting file {imm_fullpath}: {e}")
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
    CATEGORY = "utils/text"
    
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
class GPrompts:
    def __init__(self):
        self.previous_text = None
        self.current_iteration = 0
        self.wildcard_cache = {}
        self.sequential_combinations = []
        
        # Set up folder paths
        self.register_wildcard_path()
    
    def beforeQueued(self, args):
        dprint(f"beforeQueued args:{args}")
        
    def register_wildcard_path(self):
        """Register wildcards folder path in ComfyUI's folder system if it doesn't exist already"""
        if "wildcards" not in folder_paths.folder_names_and_paths:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            folder_paths.folder_names_and_paths["wildcards"] = ([os.path.join(base_dir, "wildcards")], {".txt", ".json"})

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "tooltip": "Input text with dynamic blocks: {option1|option2} for random, {{option1|option2}} for sequential, __wildcard__ for wildcard files"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for random generation (combined with iteration for reproducibility)"
                }),
                "iteration": ("INT", {
                    "default": -1,
                    "tooltip": "Iteration counter (-1 for auto-increment, 0+ for fixed iteration)"
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
                "prompt": "PROMPT"
            }
        }

    RETURN_TYPES = ("STRING", "DYNPROMPT", "STRING", "INT")
    RETURN_NAMES = ("text", "dynprompt", "computed_prompt", "seed")
    FUNCTION = "process_dynamic_prompt"
    CATEGORY = "text"
    
    DESCRIPTION = (
        "Dynamic prompt generator with support for random {}, sequential {{}}, and wildcard __word__ syntax. "
        "Perfect for creating variations in prompts, batch processing, and A/B testing different prompt combinations. "
        "Supports weighted options and nested wildcards from text/json files."
    )

    def process_dynamic_prompt(self, text, seed=0, iteration=-1,computed_prompt="",unique_id=0,prompt=None):
        global last_processed_result
        # If text changed, reset the iteration counter
        if self.previous_text != text:
            self.previous_text = text
            self.current_iteration = 0
            self.sequential_combinations = []
        
        # If iteration is provided (not -1), use it directly
        if iteration >= 0:
            self.current_iteration = iteration
        
        # Set the random seed for reproducibility
        if seed > 0:
            random.seed(seed + self.current_iteration)
        
        # Process the prompt with dynamic elements
        processed_text = self.parse_dynamic_prompt(text)
        
        # Increment the iteration counter for next time
        if iteration == -1:
            self.current_iteration += 1
        
        last_processed_result = processed_text
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
            node = prompt[unique_id]
            node["inputs"]["computed_prompt"] = processed_text  # update computed_prompt, value passed to us was previous version
            meta = node.get("_meta",{})
            meta["computed_prompt"] = processed_text
            node["_meta"] = meta
        return (processed_text,dynamic_data,processed_text,seed)

    def parse_dynamic_prompt(self, text):
        # First, process all sequential blocks and generate combinations if needed
        if not self.sequential_combinations and '{{' in text:
            # Initial calculation of all sequential combinations
            self.prepare_sequential_combinations(text)
            dprint(self.sequential_combinations)
            
        # If we have sequential combinations, use the appropriate one for this iteration
        if self.sequential_combinations and '{{' in text:
            combination_index = self.current_iteration % len(self.sequential_combinations)
            text = self.sequential_combinations[combination_index]
        
        # Process random blocks (this needs to happen on EVERY iteration)
        if '{' in text:
            text = self.process_random_blocks(text)
        
        return text

    def prepare_sequential_combinations(self, text):
        # Find all sequential blocks
        sequential_blocks = re.finditer(r'\{\{(.*?)\}\}', text)
        block_options = []
        
        # Extract options from each sequential block
        for block in sequential_blocks:
            block_text = block.group(1)
            # Process wildcards in the block
            if '__' in block_text:
                options = self.resolve_wildcard_references(block_text)
            else:
                # Split by pipe and strip whitespace
                options = [opt.strip() for opt in block_text.split('|')]
            block_options.append(options)
        
        # Generate all combinations if there are sequential blocks
        if block_options:
            import itertools
            all_combinations = list(itertools.product(*block_options))
            
            # Generate each prompt variant with the combinations
            result_prompts = []
            for combo in all_combinations:
                temp_text = text
                for i, replacement in enumerate(combo):
                    pattern = r'\{\{.*?\}\}'  # Find the first sequential block
                    temp_text = re.sub(pattern, replacement, temp_text, count=1)
                result_prompts.append(temp_text)
            
            # Store all combinations for future iterations
            self.sequential_combinations = result_prompts

    def resolve_wildcard_references(self, text):
        """
        Handle wildcards like __attire__headware__ correctly by treating the entire string as one path
        """
        dprint(f"resolve_wildcard_references text:{text}")
        
        # Check if this contains a wildcard pattern (text between __ __)
        if '__' in text:
            # For patterns like __attire__headware__, we need to get the full string between the outermost __
            match = re.match(r'^__(.+?)__$', text)
            if match:
                wildcard_name = match.group(1)
                dprint(f"resolve_wildcard_references found FULL wildcard path: {wildcard_name}")
                
                wildcard_options = self.load_wildcard(wildcard_name)
                dprint(f"resolve_wildcard_references wildcard_options:{wildcard_options}")
                
                if wildcard_options:
                    # Handle weighted options for random selection
                    if isinstance(wildcard_options, dict):
                        # Create a flat list with repeated items based on weights
                        weighted_options = []
                        for option, weight in wildcard_options.items():
                            weighted_options.extend([option] * weight)
                        return weighted_options
                    else:
                        return wildcard_options
        
        # If no wildcard pattern or no options found, treat as normal random block
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
                        print(f"No array value found in JSON object: {json_path}")
                        options = []
            except json.JSONDecodeError:
                print(f"Error parsing JSON file: {json_path}")
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
                    print(f"Error reading TXT file: {txt_path}, {str(e)}")
        
        # Cache the result
        self.wildcard_cache[wildcard_name] = options
        return options

    def process_random_blocks(self, text):
        """
        Process random blocks in the text, handling wildcards correctly
        """
        dprint(f"process_random_blocks: {text}")
        
        def replace_random(match):
            content = match.group(1)
            dprint(f"process_random_blocks content: {content}")
            
            # Handle complete wildcard pattern
            if content.startswith('__') and content.endswith('__'):
                # Extract the full wildcard path
                wildcard_path = content[2:-2]  # Remove the leading and trailing __
                dprint(f"Found full wildcard pattern: {wildcard_path}")
                
                # Load wildcards directly with the full path
                wildcard_options = self.load_wildcard(wildcard_path)
                dprint(f"Wildcard options: {wildcard_options}")
                if not wildcard_options:
                    return ""
                
                if isinstance(wildcard_options, dict):
                    # Weighted selection - separate keys and weights
                    ks = list(wildcard_options.keys())
                    vs = list(wildcard_options.values())
                    return random.choices(ks, weights=vs, k=1)[0]
                else:
                    return random.choice(wildcard_options)
                return ""
            else:
                # Normal random selection from pipe-separated options
                options = [opt.strip() for opt in content.split('|')]
                if not options:
                    return ""
                return random.choice(options)
        
        # Process all random blocks
        has_sequential = '{{' in text
        
        # Process all non-sequential random blocks
        while '{' in text:
            # Use the appropriate regex pattern
            if has_sequential:
                new_text = re.sub(r'(?<!\{)\{([^\{].*?)\}', replace_random, text)
            else:
                new_text = re.sub(r'\{([^\{].*?)\}', replace_random, text)
            if new_text == text:  # No more replacements made
                break
            text = new_text
        return text


    # Replace the resolve_wildcard_references method with this:
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


 
    def find_wildcard_file(self, wildcard_path, extension):
        """Find wildcard file using ComfyUI's folder paths system"""
        try:
            # Use debug print function if available, otherwise use regular print
            
            dprint(f"find_wildcard_file looking for: {wildcard_path}{extension}")
            
            # First try to find in wildcards folder
            if folder_paths.folder_names_and_paths.get("wildcards"):
                dprint(f"find_wildcard_file found wildcards folder path")
            
            # Try direct path first (with wildcards/ prefix)
            wildcard_file = f"{wildcard_path}{extension}"
            full_path = folder_paths.get_full_path("wildcards", wildcard_file)
            dprint(f"find_wildcard_file direct path:{full_path}")
            
            if full_path:
                return full_path
                
            # If not found with direct path, might need to check subdirectories
            # using os.path.join for correct path construction
            models_dir = folder_paths.models_dir
            wildcards_dir = os.path.join(models_dir, "wildcards")
            
            # For nested directories, create the full file path
            wildcard_file_path = os.path.join(wildcards_dir, f"{wildcard_path}{extension}")
            dprint(f"find_wildcard_file checking nested path: {wildcard_file_path}")
            
            if os.path.exists(wildcard_file_path):
                return wildcard_file_path
                
            # Additional check for directory structure - if wildcard_path contains separators
            if os.path.sep in wildcard_path:
                # The path might be a nested directory structure
                dprint(f"Checking nested directory structure")
                dir_path = os.path.dirname(wildcard_path)
                base_name = os.path.basename(wildcard_path)
                
                # Check if the directory exists
                full_dir_path = os.path.join(wildcards_dir, dir_path)
                if os.path.exists(full_dir_path):
                    dprint(f"Directory exists: {full_dir_path}")
                    # Check for the file in that directory
                    file_path = os.path.join(full_dir_path, f"{base_name}{extension}")
                    if os.path.exists(file_path):
                        return file_path
            
        except Exception as e:
            print(f"Error finding wildcard file: {e}")
        
        return None

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
            print(f"Error finding wildcard file: {e}")
        
        return None

    @PromptServer.instance.routes.get("/gprompts/prompt")
    async def prompt(request):
        global last_processed_result
        #dprint(f"@PromptServer.instance.routes.get(/gprompts/prompt)   last_processed_result:{last_processed_result} ")
        return web.json_response( { "prompt":"dummy value" } );
        #eturn web.json_response( { "prompt":last_processed_result } );
######
    @PromptServer.instance.routes.get("/gprompts/setting")
    async def setting(request):
        global the_settings
        params = request.rel_url.query
        for key, value in params.items():
            the_settings[key] = value
            dprint(f"setting [{key}]={value}")
        dprint(f"setting the_settings {the_settings}")
        return web.Response(text=f"Parameters received {params}")
#######
# Node registration
NODE_CLASS_MAPPINGS = {
    "GPrompts": GPrompts,
    "Save Image to Immich Server" : GImageSaveImmich,
    "Save Image With Notes": GImageSaveWithExtraMetadata,
    "String Formatter": StringFormatter,
    "Batch Image Loader":LoadImagesBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPrompts": "Dynamic Prompts",
    "Save Image to Immich Server" : "Save the to a Immich Server",
    "Save Image With Notes": "Save Image and add Note node to embedded workflow",
    "String Formatter": "String Formatter (sprintf)",
    "Batch Image Loader" : "Load images from folder"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

