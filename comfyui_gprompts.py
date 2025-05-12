import os
import re
import json
import random
from pathlib import Path
from collections import defaultdict
import folder_paths
import comfy.model_management as model_management
import server
import comfy
from server import PromptServer
import aiohttp
from aiohttp import web

last_processed_result = ""

print("LOADING GPROMPTS")

def dprint(a):
    #print(a)
    pass

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
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "iteration": ("INT", {"default": -1}),  # -1 means auto-increment
                "computed_prompt": ("STRING", {"multiline": True, "readonly": True,"default": ""}),
            },
            "hidden": {
            "unique_id": "UNIQUE_ID",
            "prompt": "PROMPT"  # Add this to get access to the workflow graph
        }
        }

    RETURN_TYPES = ("STRING","DYNPROMPT")
    RETURN_NAMES = ("text","dynprompt")
    FUNCTION = "process_dynamic_prompt"
    CATEGORY = "text"

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
            dprint(f"nodeInfo:{nodeInfo} QAQ")
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
        return (processed_text,dynamic_data)

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
        dprint = print
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
        dprint = print
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
            dprint = print
            
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

# Node registration
NODE_CLASS_MAPPINGS = {
    "GPrompts": GPrompts
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPrompts": "Dynamic Prompts"
}

