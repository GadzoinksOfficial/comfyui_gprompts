# v3_nodes.py
import os
import socket
import uuid
import sys
import time
import folder_paths
import subprocess, tempfile
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from comfy_api.latest import io
from comfy_extras.nodes_video import SaveVideo
from comfy_extras.nodes_audio import SaveAudioAdvanced
from .immich_importer import ImmichImporter
from .comfyui_gprompts import get_last_prompt,parse_bool_setting,add_note_node_to_workflow,extract_computed_prompt,get_missing,get_immich_settings,get_settings_file,apply_settings,dprint
from  .common import  pil_to_comfy, cover_from_tensor, fallback_cover,fallback_cover_os
#
#
#

def make_mp4(mp3_path, image=None, notes="",metadata={}):
    cover = cover_from_tensor(image) if image is not None else fallback_cover()
    # libx264 requires even dimensions
    w, h = cover.size
    cover = cover.resize((w - w % 2 or 2, h - h % 2 or 2))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        cover.save(tf.name)
        png_path = tf.name

    mp4_path = os.path.splitext(mp3_path)[0] + ".mp4"
    cmd = ["ffmpeg", "-y",
           "-loop", "1", "-i", png_path,
           "-i", mp3_path,
           "-c:v", "libx264", "-tune", "stillimage", "-r", "1",
           "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "256k",
           "-shortest",
           "-movflags", "use_metadata_tags+faststart"]   # <- the key change
    dprint(f"make_mp4 mp4_path:{mp4_path}")
    if metadata:
        for k, v in metadata.items():
            cmd += ["-metadata", f"{k}={json.dumps(v)}"]  # JSON value, same as Comfy
    if notes:
        cmd += ["-metadata", f"comment={notes}"]
    cmd.append(mp4_path)
    dprint(f"cmd : {cmd}")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        dprint(f"subprocess finished")
    finally:
        os.unlink(png_path)
    return mp4_path
#
#
#
class GSaveAudioToImmich(SaveAudioAdvanced):
    @classmethod
    def define_schema(cls):
        # Start from the real SaveAudio schema
        schema = SaveAudioAdvanced.define_schema()
        schema.node_id = "GSaveAudioToImmich"
        schema.display_name = "Save Audio To Immich Server"
        schema.category = "gprompts/immich"
        schema.description = (
            "Saves the mp3 audio to the ComfyUI output directory, "
            "then uploads it to an Immich server as an mp4."
            "optionally add a cover image"
        )
        schema.search_aliases = []
        schema.inputs = list(schema.inputs) + [
            io.Image.Input("image",  optional=True,
                            tooltip="Optional cover image to use with MP4."),
            io.String.Input("notes", default="", optional=True,
                            tooltip="Optional text for Notes Node in embedded workflow."),
            io.String.Input("album_name", default="", optional=True,
                            tooltip="Optional album to add the video to."),
            io.String.Input("tags", default="", optional=True,
                            tooltip="Optional tags. Comma seperated. Merged with Tags from Settings."),

        ]
        return schema

    @classmethod
    def execute(cls, audio, filename_prefix, format, codec=None,
        image=None,notes="",album_name="",tags="") -> io.NodeOutput:
        # temp
        save_also=True
        notes=None
        computed_prompt=None
        unique_id= cls.hidden.unique_id
        prompt = cls.hidden.prompt
        extra_pnginfo=cls.hidden.extra_pnginfo
        md = {}
        if cls.hidden.extra_pnginfo is not None:
            md.update(cls.hidden.extra_pnginfo)      # your modified copy rides along here
        if cls.hidden.prompt is not None:
            md["prompt"] = cls.hidden.prompt


        extra_pnginfo_new = dict(cls.hidden.extra_pnginfo or {})
        workflow_id = extra_pnginfo.get("workflow",{}).get("id")
        workflow = None
        note_text = None
        #TODO extracted_prompt = extract_computed_prompt(prompt)
        #dprint(f">>>>>>>>>> last_prompt: {last_prompt}")
        #dprint(f">>>>>>>>>> extracted_prompt: {extracted_prompt}")

        settings = get_immich_settings()
        if  extra_pnginfo:
            extra_pnginfo_new = extra_pnginfo.copy()
        note_text = None
        # Try in order notes,, global value of gprompt node
        if notes:
            note_text = notes
        last_prompt = get_last_prompt(workflow_id)
        if not note_text and last_prompt:
            note_text = f'Image created with prompt "{last_prompt}"'
        #if not note_text and last_prompt:
        #    note_text = f'Image created with prompt "{last_prompt}"'
        #if "computed_prompt" not in extra_pnginfo_new:
        #    if computed_prompt:
        #        extra_pnginfo_new["computed_prompt"] = computed_prompt
        #    else:
        #        extra_pnginfo_new["computed_prompt"] = last_prompt
        if not album_name:
            album_name = settings.get('immich_default_album')
        # Add note node to workflow
        if note_text and  "workflow" in extra_pnginfo_new:
            workflow = extra_pnginfo_new["workflow"]
            add_note_node_to_workflow(workflow, note_text)
        dprint(f"unique_id:{unique_id} workflow_id:{workflow_id} prompt:{prompt} ")
        dprint(f"extra_pnginfo:{extra_pnginfo}")
        dprint(f"workflow:{workflow}")
        dprint(f"note_text: {note_text}")
# Let core SaveVideo do the encoding/metadata/counter work.
        cls.hidden.extra_pnginfo = extra_pnginfo_new
        result = super().execute(audio=audio, filename_prefix=filename_prefix,
                                 format={"format": "mp3", "quality": "V0"})
        dprint(f"result:{result}")
        # Recover the path it just wrote: result.ui is a ui.??
        # whose .values is a list of ui.SavedResult dicts.
        saved = result.ui.results[0]
        filepath = os.path.join(folder_paths.get_output_directory(),
                                saved.subfolder, saved.filename)
        imm_fullpath = os.path.join(folder_paths.get_output_directory(),saved.subfolder,saved.filename)
        imm_filename = saved.filename
        basetags = settings.get('immich_base_tags') or ""
        tags = tags or ""
        all_tags = (tags + ',' + basetags).split(',')
        user_tags = list({tag.strip() for tag in all_tags if tag.strip()})
        structured_tags = user_tags # future: extract some tags from metadata

        dprint(f"saved:{saved}")
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
            print("\nIMMICH CONFIGURATION ERROR")
            print(f"Save Image to Immich Server Node Missing: {', '.join(missing)}")
            print("Please configure in Settings:Gadzoinks")
            print(f"settings snapshot:{settings}")
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
                        mp4_path = make_mp4(temp_fullpath,image = image,notes = "")
                        dprint(f"mp4_path:{mp4_path}")
                        upload_rc = importer.upload_photo(mp4_path, album, tags = user_tags,structured_tags=structured_tags,rating=rating, comfy_workflow=workflow)
                    finally:
                        os.rename(temp_fullpath, imm_fullpath)
                else:
                    dprint(f"B1")
                    mp4_path = make_mp4(imm_fullpath,image = image,notes = "")
                    upload_rc = importer.upload_photo(mp4_path, album_name, tags = user_tags,structured_tags=structured_tags,rating=rating, comfy_workflow=workflow)
                    dprint(f"B2 upload_rc={upload_rc}")
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

        return result

#
#
#
class GSaveVideoToImmich(SaveVideo):
    @classmethod
    def define_schema(cls):
        # Start from the real SaveVideo schema (format/codec combos, hidden
        # prompt/pnginfo, output-node flag) so we never drift from core.
        schema = SaveVideo.define_schema()
        schema.node_id = "GSaveVideoToImmich"   # Save Video To Immich Server"
        schema.display_name = "Save Video To Immich Server"
        schema.category = "gprompts/immich"
        schema.description = (
            "Saves the video to the ComfyUI output directory, "
            "then uploads it to an Immich server."
        )
        schema.search_aliases = []
        schema.inputs = list(schema.inputs) + [
            io.String.Input("notes", default="", optional=True,
                            tooltip="Optional text for Notes Node in embedded workflow."),
            io.String.Input("album_name", default="", optional=True,
                            tooltip="Optional album to add the video to."),
            io.String.Input("tags", default="", optional=True,
                            tooltip="Optional tags. Comma seperated. Merged with Tags from Settings."),

        ]
        return schema

    @classmethod
    def execute(cls, video, filename_prefix, format, codec=None,
                 notes="",album_name="",tags="") -> io.NodeOutput:
        # temp
        save_also=True
        notes=None
        computed_prompt=None
        unique_id= cls.hidden.unique_id
        prompt = cls.hidden.prompt
        extra_pnginfo=cls.hidden.extra_pnginfo
        extra_pnginfo_new = dict(cls.hidden.extra_pnginfo or {})
        workflow_id = extra_pnginfo.get("workflow",{}).get("id")
        workflow = None
        note_text = None
        #TODO extracted_prompt = extract_computed_prompt(prompt)
        #dprint(f">>>>>>>>>> last_prompt: {last_prompt}")
        #dprint(f">>>>>>>>>> extracted_prompt: {extracted_prompt}")



        settings = get_immich_settings()
        if  extra_pnginfo:
            extra_pnginfo_new = extra_pnginfo.copy()
        note_text = None
        # Try in order notes,, global value of gprompt node
        if notes:
            note_text = notes
        last_prompt = get_last_prompt(workflow_id)
        if not note_text and last_prompt:
            note_text = f'Image created with prompt "{last_prompt}"'
        #if not note_text and last_prompt:
        #    note_text = f'Image created with prompt "{last_prompt}"'
        #if "computed_prompt" not in extra_pnginfo_new:
        #    if computed_prompt:
        #        extra_pnginfo_new["computed_prompt"] = computed_prompt
        #    else:
        #        extra_pnginfo_new["computed_prompt"] = last_prompt
        if not album_name:
            album_name = settings.get('immich_default_album')



        # Add note node to workflow
        if note_text and  "workflow" in extra_pnginfo_new:
            workflow = extra_pnginfo_new["workflow"]
            add_note_node_to_workflow(workflow, note_text)


        #TODO last_prompt = promtpForId.get(workflow_id)
        dprint(f"unique_id:{unique_id} workflow_id:{workflow_id} prompt:{prompt} ")
        dprint(f"extra_pnginfo:{extra_pnginfo}")
        dprint(f"workflow:{workflow}")
        dprint(f"note_text: {note_text}")
# Let core SaveVideo do the encoding/metadata/counter work.
        cls.hidden.extra_pnginfo = extra_pnginfo_new
        result = super().execute(video=video, filename_prefix=filename_prefix,
                                 format=format, codec=codec)

        # Recover the path it just wrote: result.ui is a ui.PreviewVideo,
        # whose .values is a list of ui.SavedResult dicts.
        saved = result.ui.values[0]
        filepath = os.path.join(folder_paths.get_output_directory(),
                                saved.subfolder, saved.filename)
        imm_fullpath = os.path.join(folder_paths.get_output_directory(),saved.subfolder,saved.filename)
        imm_filename = saved.filename
        basetags = settings.get('immich_base_tags') or ""
        tags = tags or ""
        all_tags = (tags + ',' + basetags).split(',')
        user_tags = list({tag.strip() for tag in all_tags if tag.strip()})
        structured_tags = user_tags # future: extract some tags from metadata
        dprint(f"saved:{saved}")
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
            print("\nIMMICH CONFIGURATION ERROR")
            print(f"Save Image to Immich Server Node Missing: {', '.join(missing)}")
            print("Please configure in Settings:Gadzoinks")
            print(f"settings snapshot:{settings}")
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
                    dprint(f"B1")
                    upload_rc = importer.upload_photo(imm_fullpath, album_name, tags = user_tags,structured_tags=structured_tags,rating=rating, comfy_workflow=workflow)
                    dprint(f"B2 upload_rc={upload_rc}")
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

        return result
#####


#####
def upload_to_immich(filepath, immich_url, immich_api_key, album_name):
    pass

V3NODES = {"GSaveVideoToImmich": GSaveVideoToImmich, "GSaveAudioToImmich":GSaveAudioToImmich}
V3NODE_DISPLAY_NAME_MAPPINGS = {"GSaveVideoToImmich": "Save video To Immich Server",
    "GSaveAudioToImmich" : "Save audio to immich server" }


