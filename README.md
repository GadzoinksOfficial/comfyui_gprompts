# ComfyUI GPrompts Nodes

## Introduction
This package provides three custom nodes for ComfyUI that enhance prompt generation, string formatting, and image saving capabilities.

## Nodes Overview
- **GPrompts** - Create dynamic prompts with random or sequential selection. Also support wildcard files.
- **String Formatter** - Build custom output strings from multiple inputs and system variables
- **Save Image With Notes** - Save images with embedded workflow notes and metadata

---

## GPrompts Node

### Description
This is another dyanmic prompts node for Comfyui, I found most of the ones out there to be either
too complicated or too limiting, so I wrote my own.

### Prompt Format Syntax

### Basics
```

create a gprompts node and connect its output
to a clip node text input
(you may need to enable text input on the clip node)

Format of a dynamic prompt
{ cat | dog | jackalope  }     random selection
{{ green | yellow | red }}      sequential seletion

If you generate 4 images with "a stop light showing {{ green | yellow | red }}"
you will get a green light image, yellow, red, and then another green.

If you use "a stop light showing { green | yellow | red }", each image will have a 33% chance of any color.


Wildcards
wildcard files are either .txt or .json and go in comfyui/models/wildcards

you can use a wildcard file with a list of options
"a woman with {{__hair_color__}} {{__hair_style__}} hair"

This will use the contents of comfyui/models/wildcards/hair_color.txt
and hair_style.txt

assuming the files are
blonde
red
brown

and

long
short
pixie
mohawk

you will have 12 combinations. Set comfyui to generate 12 images and you will
see all combinations

json wildcard files
instead of text you can use a json file. For example  seasons.json
 simple --   { "doesnotmatter" : ["summer","winter","fall","spring" ] } 
 weighted --   { "whatever" : [ { "summer" : 6 } ,  { "spring" : 4 } ,  
          {"fall" : 3 } ,  { "winter" : 1 }  ] }
weighted is only relevant to {} random selection, with random the odds of 
getting  a choice are weight/total weights. so for summer odds are 6 out of 14.
for {{}} sequential you will get all 4 seasons.

A wildcard reference to __hair__hairstyles__ , will use the file models/wildcards/hair/hairstyles.txt ( or .json )

Version 2.0 Update
Nothing has changed in regards to core functionality, but the node now outputs the computed_prompt and the seed.

Advanced:
there is an optional dynaprompt output that can connect to a node that understands how to utiltize it to extract the dynamic prompt and the computed prompt.   see the gadzoinks custom node as an example .
Note: Dynaprompts don't seem to be used much so ignore this.

TODO:
 need to add support for wildcard files that include other wildcard files
```


# String Formatter

## Description
Builds an output strng from supplied inputs and from system variables.
for example if use connect prompt to A and seed to B, then the format string "generating $a with seed $b on $hostname" you will generate a string a like "generating a smiling cat with seed 12345 on hal2000"

# Save Image With Notes

## Description
This node modifies a copy of you workflow adding a notes node in the new workflow that is then saved inside the image.
You can add your own text with 'notes' input
or wire the  'computed_node' from Gprompts which creates a Note and saves the computed prompt in the exif json

Note: This node uses the standard comfyui Save Image node to do the actual saving.


