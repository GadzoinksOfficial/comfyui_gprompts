# Dynamic Prompts Node

## Description
The Dynamic Prompts node generates text with random, sequential, and wildcard-based variations. Perfect for creating prompt variations, batch processing, and A/B testing different combinations.

## Syntax

### Random Selection `{}`
Use curly braces with pipe-separated options for random selection:
This will randomly select one of the three options each time the node executes.

### Sequential Selection `{{}}`
Use double curly braces for sequential cycling through options:

### wildcard Selection 
A __color__ __animal__ wearing a __clothing__hats__cap__
This will use the wildcard file color.txt , animal.txt and clothing/hats/cap.txt
wildcard files are either .txt `(` one entry per line `)` or .json and go in comfyui/models/wildcards
See https://github.com/GadzoinksOfficial/comfyui_gprompts for full documentation
