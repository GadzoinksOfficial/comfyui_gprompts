This is another dyanmic prompts node for Comfyui
I found most of the ones out there to be either too complicated or too limiting,
so I wrote my own

basics

create a grpompt node and connect its output to a clip node text input (you may need to nable text input)

format of a dynamic prompt
{ cat | dog | jackalope  }     random selection 
{{ green | yellow | red }}      sequential seletion

If you generate 4 images with " a stop light showing {{ green | yellow | red }}"
you will get a green light image, yellow, red, and then another green.

wildcards

you can use a wildcard file with a list of options
"a woman with {{__hair_color__}} {{__hair_style__}} hair"

This will use the contents of comfyui/models/wildcards/hair_color.txt  ( and hair_style.txt)

assuming the files are
blonde
red
brown

and

long
short
pixie
mohawk


you will have 12 combinations. Set comfyui to generate 12 images and you will see all combinations

json wildcard files
instead of text you can use a json file,  season.json
 simple --   { [“summer”,”winter”,”fall”,”spring” ] ) 
 weighted --   { [ { “summer” : 6 } ,  { “spring” : 4 } ,  { “fall ” : 3 } ,  { “winter” : 1 }  ] }
weighted is only relevant to {} random selection, for {{}} sequential you will get all 4 seasons.


TODO:
 need to add support for wildcard files that include other wildcard files
 trying to find a way to save the computed prompt with the extra data written to EXIF


