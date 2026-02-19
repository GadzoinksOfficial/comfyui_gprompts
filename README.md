```
This is another dyanmic prompts node for Comfyui
I found most of the ones out there to be either
too complicated or too limiting, so I wrote my own.

Basics

create a gprompts node and connect its output
to a clip node text input
(you may need to enable text input on the clip node)

Format of a dynamic prompt
{ cat | dog | jackalope  }     random selection
{{ green | yellow | red }}      sequential seletion

If you generate 4 images with "a stop light showing {{ green | yellow | red }}"
you will get a green light image, yellow, red, and then another green.

If you use "a stop light showing { green | yellow | red }", each image will have a 33% chance of any color.

Advanced: there is an optional dynaprompt output that can connect to a node
that understands how to utiltize it to extract the dynamic prompt and
the computed prompt.  ( see the gadzoinks custom node as an example }
Note: Dynaprompts don't seem to be used much so ignore this.

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

There is a new Node called "GPrompts Save Image". This is an extension of the standard Save Image (i.e. it does some stuff and then calls Save Image).
It has an input for the computed_prompt output from GPrompts node (it will accept any text).
When the image is saved the computed_prompt is saved in the image's embedded metadata and a Notes node is inserted in the workflow with the computed prompt. Making it easy to see what prompt values were used for a specific image.
Another feature is images are automatically saved in a date folder YYYY-MM-DD ( if you don't like this include '%' in the filename prefix will disable)

TODO:
 need to add support for wildcard files that include other wildcard files

```

