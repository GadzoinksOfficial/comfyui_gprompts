"""
@author: gadzoinksofficial
@title: Gprompts
@nickname: Gprompts
@description: Another dynamic prompt node, designed to be easy to use and support wildcards

Dynamic Prompts extension for ComfyUI
Allows for random and sequential substitutions in prompts
"""
import sys, os
from .comfyui_gprompts import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, GPrompts
from .v3_nodes import V3NODES, V3NODE_DISPLAY_NAME_MAPPINGS
from .enhanced_prompt import ENHANCED_NODES, ENHANCED_NODE_DISPLAY_NAME_MAPPINGS
from .enhancer_api import API_NODES, API_NODE_DISPLAY_NAME_MAPPINGS
from .immich_loader import IMMICH_LOADER_NODES, IMMICH_LOADER_DISPLAY_NAME_MAPPINGS

# Merge V3 nodes into the V1 registration path
# (V3 io.ComfyNode classes are backwards-compatible with NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **V3NODES, **ENHANCED_NODES, **API_NODES,
                       **IMMICH_LOADER_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **V3NODE_DISPLAY_NAME_MAPPINGS,
                              **ENHANCED_NODE_DISPLAY_NAME_MAPPINGS, **API_NODE_DISPLAY_NAME_MAPPINGS,
                              **IMMICH_LOADER_DISPLAY_NAME_MAPPINGS}

module_root_directory = os.path.dirname(os.path.realpath(__file__))
module_js_directory = os.path.join(module_root_directory, "js")
WEB_DIRECTORY = "./web/js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

import logging
logging.getLogger("gprompts").info(f"Dynamic Prompts for ComfyUI loaded: {list(NODE_CLASS_MAPPINGS.keys())}")


