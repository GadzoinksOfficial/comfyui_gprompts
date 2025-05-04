"""
@author: gadzoinksofficial
@title: Gprompts
@nickname: Gprompts
@description: Another dynamic prompt node, designed to be easy to use and support wildcards

Dynamic Prompts extension for ComfyUI
Allows for random and sequential substitutions in prompts
"""
import sys, os
from .comfyui_gprompts import NODE_CLASS_MAPPINGS, GPrompts
module_root_directory = os.path.dirname(os.path.realpath(__file__))
module_js_directory = os.path.join(module_root_directory, "js")

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS',"WEB_DIRECTORY"]

print(f"Dynamic Prompts for ComfyUI loaded: {list(NODE_CLASS_MAPPINGS.keys())}")

