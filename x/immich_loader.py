"""Load Image From Immich: steps through the Immich images that pass a set of
filters (album, tag, favorite, minimum rating), one per run, and reads the
prompt back out of each original file.

The images are listed and downloaded on the ComfyUI server, with the Immich
address and API key from Settings > Gadzoinks (the same ones the save nodes use).
"""
import asyncio
import io
import json
import re
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from aiohttp import web
from comfy_api.latest import io as cio
from server import PromptServer

from .common import get_logger
from .comfyui_gprompts import get_immich_settings, get_missing, masked_settings
from .immich_importer import ImmichImporter, ImmichError

log = get_logger("immich.loader")

try:                                    # HEIC/HEIF originals from phones, if installed
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except Exception:
    HEIF_SUPPORT = False

# 'increment' preselected where this ComfyUI has the enum; older ones show the
# control with its default.
_INDEX_CONTROL = getattr(getattr(cio, "ControlAfterGenerate", None), "increment", True)

RATING_OPTIONS = ["any", "1+", "2+", "3+", "4+", "5"]

# Camera RAW: Pillow would open only a small embedded thumbnail, if anything.
_RAW_MIME = re.compile(r"image/(x-|.*raw|.*dng)", re.IGNORECASE)

# (server url, filters) -> list of assets, sorted. Re-read at index 0, so a
# batch sees one fixed list and photos added mid-batch don't shift the indexes.
_ASSET_LISTS = {}


# ----------------------------------------------------------------------------
# Finding the album or tag, and its images
# ----------------------------------------------------------------------------
def _names(items, key, limit=25):
    names = sorted({str(i.get(key) or "") for i in items if i.get(key)}, key=str.lower)
    more = f" ... and {len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(f"'{n}'" for n in names[:limit]) + more


def find_album(importer, name):
    albums = importer.list_albums()
    wanted = name.strip().lower()
    matches = [a for a in albums if str(a.get("albumName") or "").strip().lower() == wanted]
    if not matches:
        raise ValueError(f"Load From Immich: no album named '{name}'. Albums: "
                         + (_names(albums, "albumName") or "(none)"))
    if len(matches) > 1:
        matches.sort(key=lambda a: a.get("assetCount") or 0, reverse=True)
        log.warning(f"Load From Immich: {len(matches)} albums are named '{name}'; using the one "
                    f"with the most photos ({matches[0].get('assetCount')}).")
    return matches[0]


def find_tag(importer, name):
    tags = importer.list_tags()
    wanted = name.strip().lower()
    full = [t for t in tags if str(t.get("value") or "").lower() == wanted]
    if full:
        return full[0]
    leaf = [t for t in tags if str(t.get("name") or "").lower() == wanted]
    if len(leaf) == 1:
        return leaf[0]
    if len(leaf) > 1:
        raise ValueError(f"Load From Immich: several tags are named '{name}': "
                         + _names(leaf, "value") + ". Use the full name.")
    raise ValueError(f"Load From Immich: no tag named '{name}'. Tags: "
                     + (_names(tags, "value") or "(none)"))


def _sort_key(asset):
    exif = asset.get("exifInfo") or {}
    when = exif.get("dateTimeOriginal") or asset.get("localDateTime") or asset.get("fileCreatedAt") or ""
    return (str(when), str(asset.get("originalFileName") or "").lower(), str(asset.get("id")))


def _album_assets(importer, album):
    try:
        assets = importer.search_images(album_id=album["id"])
    except ImmichError as e:
        if "HTTP 400" not in str(e):
            raise
        assets = []             # an older server without album search: fall back below
    if not assets and (album.get("assetCount") or 0) > 0:
        # Older servers list the album's assets with the album (and search
        # may not cover albums shared by someone else).
        full = importer.get_album(album["id"])
        assets = [a for a in (full.get("assets") or []) if a.get("type") == "IMAGE"]
    return assets


def _rating(asset):
    try:
        return int((asset.get("exifInfo") or {}).get("rating") or 0)
    except (TypeError, ValueError):
        return 0


def parse_min_rating(value):
    """'any' -> 0, '2+' -> 2, '5' -> 5."""
    m = re.match(r"\s*(\d)", str(value or ""))
    return min(5, int(m.group(1))) if m else 0


def describe_filters(album, tag, favorites, min_rating):
    parts = []
    if album:
        parts.append(f"album '{album}'")
    if tag:
        parts.append(f"tag '{tag}'")
    if favorites:
        parts.append("favorites")
    if min_rating:
        parts.append(f"{min_rating}+ stars" if min_rating < 5 else "5 stars")
    return " + ".join(parts)


def list_images(importer, album="", tag="", favorites=False, min_rating=0):
    """(label, assets): the images that pass every filter that is set, oldest first.
    Album and tag are each one server search and the results are intersected;
    favorite and rating are then checked on each image. With neither album nor
    tag, the server narrows by favorite, or by each rating from min_rating to 5
    (Immich's rating filter matches one value only)."""
    sets, names = [], {}
    if album:
        a = find_album(importer, album)
        names["album"] = a.get("albumName") or album
        sets.append(_album_assets(importer, a))
    if tag:
        t = find_tag(importer, tag)
        names["tag"] = t.get("value") or tag
        sets.append(importer.search_images(tag_id=t["id"]))
    if not sets:
        if favorites:
            sets.append(importer.search_images(is_favorite=True))
        elif min_rating:
            sets.append([a for r in range(min_rating, 6) for a in importer.search_images(rating=r)])
        else:
            raise ValueError("Load From Immich: set at least one filter (album, tag, favorites "
                             "or rating).")
    ids = set.intersection(*[{a["id"] for a in s} for s in sets])
    records = {}
    for s in sets:              # prefer a record that carries EXIF (rating, date)
        for a in s:
            if a["id"] in ids and (a["id"] not in records or not records[a["id"]].get("exifInfo")):
                records[a["id"]] = a
    images = [a for a in records.values()
              if a.get("type", "IMAGE") == "IMAGE" and not a.get("isTrashed")
              and (not favorites or a.get("isFavorite"))
              and (not min_rating or _rating(a) >= min_rating)]
    images.sort(key=_sort_key)
    label = describe_filters(names.get("album"), names.get("tag"), favorites, min_rating)
    return label, images


# ----------------------------------------------------------------------------
# Reading the prompt back out of a file
# ----------------------------------------------------------------------------
_TEXT_KEYS = ("text", "prompt", "t5xxl", "clip_l", "text_g", "text_l", "positive_prompt",
              "string", "value")
_NO_FOLLOW = {"clip", "model", "vae", "image", "images", "pixels", "latent_image", "samples",
              "mask", "control_net", "clip_vision", "style_model", "noise", "sigmas", "sampler"}


def _is_link(v):
    return (isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int))
            and isinstance(v[1], int))


def _trace_text(graph, link, seen=None, depth=0):
    """Follow a conditioning link back to the text that made it."""
    if depth > 12:
        return ""
    seen = seen if seen is not None else set()
    node_id = str(link[0])
    if node_id in seen:
        return ""
    seen.add(node_id)
    node = graph.get(node_id) or {}
    computed = (node.get("_meta") or {}).get("computed_prompt")
    if isinstance(computed, str) and computed.strip():
        return computed.strip()
    inputs = node.get("inputs") or {}
    for key in _TEXT_KEYS:
        v = inputs.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if _is_link(v):
            found = _trace_text(graph, v, seen, depth + 1)
            if found:
                return found
    for key, v in inputs.items():          # conditioning passed through other nodes
        if key not in _NO_FOLLOW and key not in _TEXT_KEYS and _is_link(v):
            found = _trace_text(graph, v, seen, depth + 1)
            if found:
                return found
    return ""


def prompts_from_comfy_graph(graph):
    """(positive, negative) from a ComfyUI API prompt (the 'prompt' PNG chunk)."""
    if not isinstance(graph, dict):
        return "", ""
    def by_id(item):
        try:
            return (0, int(item[0]))
        except (TypeError, ValueError):
            return (1, str(item[0]))
    nodes = sorted(graph.items(), key=by_id)
    for _nid, node in nodes:                       # samplers: positive + negative
        inputs = (node or {}).get("inputs") or {}
        if _is_link(inputs.get("positive")):
            pos = _trace_text(graph, inputs["positive"])
            neg = _trace_text(graph, inputs["negative"]) if _is_link(inputs.get("negative")) else ""
            if pos:
                return pos, neg
    for _nid, node in nodes:                       # guiders (FLUX BasicGuider etc.)
        inputs = (node or {}).get("inputs") or {}
        if "guider" in str(node.get("class_type", "")).lower() and _is_link(inputs.get("conditioning")):
            pos = _trace_text(graph, inputs["conditioning"])
            if pos:
                return pos, ""
    for _nid, node in nodes:                       # no sampler found: any prompt node's result
        computed = ((node or {}).get("_meta") or {}).get("computed_prompt")
        if isinstance(computed, str) and computed.strip():
            return computed.strip(), ""
    return "", ""


def parse_a1111(text):
    """(positive, negative) from A1111/Forge 'parameters' text."""
    text = (text or "").strip()
    neg_m = re.search(r"\n\s*Negative prompt:\s*", text)
    steps_m = re.search(r"\n\s*Steps:\s*\d", text)
    if neg_m:
        end = steps_m.start() if steps_m and steps_m.start() > neg_m.end() else len(text)
        return text[:neg_m.start()].strip(), text[neg_m.end():end].strip()
    if steps_m:
        return text[:steps_m.start()].strip(), ""
    return text, ""


def _json_text(value):
    """A PNG text chunk ComfyUI wrote with json.dumps -> the value."""
    try:
        return json.loads(value)
    except Exception:
        return value


def _exif_user_comment(img):
    try:
        raw = img.getexif().get_ifd(0x8769).get(0x9286)
    except Exception:
        return ""
    if isinstance(raw, bytes):
        head, body = raw[:8], raw[8:]
        if head.startswith(b"UNICODE"):
            for enc in ("utf-16-be", "utf-16-le"):
                try:
                    text = body.decode(enc)
                    if text.isprintable() or "\n" in text:
                        return text.strip("\x00").strip()
                except UnicodeDecodeError:
                    continue
        return body.decode("utf-8", errors="ignore").strip("\x00").strip()
    return str(raw or "").strip()


def read_prompt(info, img=None):
    """{'prompt', 'negative_prompt', 'source', 'parameters'} from an image's metadata.
    Order: the computed_prompt our save nodes write, the ComfyUI prompt graph,
    A1111 'parameters' (PNG text or JPEG EXIF UserComment)."""
    out = {"prompt": "", "negative_prompt": "", "source": "", "parameters": ""}
    info = info or {}
    graph = _json_text(info["prompt"]) if isinstance(info.get("prompt"), str) else None
    computed = _json_text(info["computed_prompt"]) if isinstance(info.get("computed_prompt"), str) else None
    graph_pos, graph_neg = prompts_from_comfy_graph(graph) if isinstance(graph, dict) else ("", "")
    if isinstance(computed, str) and computed.strip():
        out.update(prompt=computed.strip(), negative_prompt=graph_neg, source="computed_prompt")
        return out
    if graph_pos:
        out.update(prompt=graph_pos, negative_prompt=graph_neg, source="comfyui workflow")
        return out
    params = info.get("parameters") if isinstance(info.get("parameters"), str) else ""
    if not params and img is not None:
        params = _exif_user_comment(img)
    if params:
        pos, neg = parse_a1111(params)
        out.update(prompt=pos, negative_prompt=neg, source="a1111 parameters", parameters=params)
    return out


# ----------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------
def decode_image(data):
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def to_tensors(img):
    """(IMAGE [1,H,W,3], MASK [1,H,W]) the way Load Image makes them."""
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = ImageOps.exif_transpose(img)
    if img.mode == "I":
        img = img.point(lambda i: i * (1 / 255))
    rgb = img.convert("RGB")
    image = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0).unsqueeze(0)
    if "A" in img.getbands():
        alpha = np.array(img.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(alpha)
    else:
        mask = torch.zeros((rgb.height, rgb.width), dtype=torch.float32)
    return image, mask.unsqueeze(0), rgb.width, rgb.height


def load_asset(importer, asset):
    """(PIL image, metadata info dict, which file was used). The original when
    Pillow can read it; Immich's preview JPEG for RAW, or HEIC without pillow-heif."""
    name = asset.get("originalFileName") or asset.get("id")
    mime = str(asset.get("originalMimeType") or "")
    if not _RAW_MIME.match(mime):
        data = importer.download_original(asset["id"])
        try:
            img = decode_image(data)
            return img, dict(img.info), "original"
        except Exception as e:
            hint = (" (install pillow-heif in ComfyUI's Python to read HEIC originals)"
                    if re.search(r"hei[cf]", mime + name, re.IGNORECASE) and not HEIF_SUPPORT else "")
            log.warning(f"Load From Immich: can't decode the original {name}: {e}{hint}; "
                        f"using Immich's preview instead.")
    img = decode_image(importer.download_preview(asset["id"]))
    return img, {}, "preview"


# ----------------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------------
class GLoadImageFromImmich(cio.ComfyNode):
    @classmethod
    def define_schema(cls):
        return cio.Schema(
            node_id="GLoadImageFromImmich",
            display_name="Load Image From Immich",
            category="gprompts/immich",
            description=(
                "Steps through the Immich images that pass every filter you set (album, tag, "
                "favorites, minimum rating), one per run (index with 'increment'). Downloads the "
                "original on the ComfyUI server and reads "
                "the prompt it was made with (GPrompts/ComfyUI metadata, or A1111 parameters). "
                "Server and API key come from Settings > Gadzoinks."
            ),
            search_aliases=["immich", "load from immich", "immich album", "immich batch"],
            inputs=[
                cio.String.Input("album", default="",
                                 tooltip="Only images in this album. Blank = any album. Not "
                                         "case-sensitive."),
                cio.String.Input("tag", default="",
                                 tooltip="Only images with this tag. Blank = any. A nested tag by "
                                         "its full name, e.g. 'Trips/Japan'. Not case-sensitive."),
                cio.Boolean.Input("favorites_only", default=False,
                                  tooltip="Only images marked as favorite."),
                cio.Combo.Input("min_rating", options=RATING_OPTIONS, default="any",
                                tooltip="Only images with at least this many stars: '2+' takes "
                                        "2, 3, 4 and 5 stars. Unrated images count as 0."),
                cio.Int.Input("index", default=0, min=0, max=1_000_000,
                              control_after_generate=_INDEX_CONTROL,
                              tooltip="Which image, oldest first (wraps around). With 'increment', "
                                      "each queued run loads the next one; queue 'count' runs to do "
                                      "them all. At 0 the list is read again from Immich."),
            ],
            outputs=[
                cio.Image.Output("image", display_name="image"),
                cio.Mask.Output("mask", display_name="mask"),
                cio.String.Output("prompt", display_name="prompt",
                                  tooltip="The prompt the image was made with, if its file has one."),
                cio.String.Output("negative_prompt", display_name="negative_prompt"),
                cio.String.Output("description", display_name="description",
                                  tooltip="The description in Immich."),
                cio.String.Output("filename", display_name="filename"),
                cio.Int.Output("width", display_name="width"),
                cio.Int.Output("height", display_name="height"),
                cio.Int.Output("index", display_name="index",
                               tooltip="The index actually used (after wrapping)."),
                cio.Int.Output("count", display_name="count",
                               tooltip="How many images pass the filters."),
                cio.String.Output("asset_id", display_name="asset_id"),
                cio.String.Output("metadata", display_name="metadata",
                                  tooltip="JSON: Immich details (date, people, tags) and where the "
                                          "prompt came from."),
            ],
        )

    @classmethod
    def validate_inputs(cls, album="", tag="", favorites_only=False, min_rating="any", **kwargs):
        if not ((album or "").strip() or (tag or "").strip() or favorites_only
                or parse_min_rating(min_rating)):
            return "Set at least one filter: album, tag, favorites_only or min_rating."
        return True

    @classmethod
    def execute(cls, album, tag, favorites_only, min_rating, index) -> cio.NodeOutput:
        settings = get_immich_settings()
        missing = get_missing(settings)
        if missing:
            log.debug("settings snapshot: %s", masked_settings(settings))
            raise ValueError(f"Load From Immich: missing {', '.join(missing)}. "
                             f"Open Settings, Gadzoinks to configure.")
        url = f"http://{settings.get('immich_hostname')}:{settings.get('immich_port')}"
        importer = ImmichImporter(url, settings.get("immich_apikey"), importer_name="GLoadImageFromImmich")
        album, tag = (album or "").strip(), (tag or "").strip()
        stars = parse_min_rating(min_rating)
        key = (url, album.lower(), tag.lower(), bool(favorites_only), stars)
        try:
            if index == 0 or key not in _ASSET_LISTS:
                _ASSET_LISTS[key] = list_images(importer, album, tag, bool(favorites_only), stars)
            label, assets = _ASSET_LISTS[key]
            if not assets:
                raise ValueError(f"Load From Immich: no images match {label}.")
            used = index % len(assets)
            asset = assets[used]
            img, info, used_file = load_asset(importer, asset)
        except ImmichError as e:
            raise ValueError(f"Load From Immich: {e}") from e

        found = read_prompt(info, img)
        image, mask, width, height = to_tensors(img)
        exif = asset.get("exifInfo") or {}
        filename = asset.get("originalFileName") or asset["id"]
        meta = {
            "asset_id": asset["id"],
            "file_name": filename,
            "filters": label,
            "index": used,
            "count": len(assets),
            "taken": exif.get("dateTimeOriginal") or asset.get("localDateTime"),
            "description": exif.get("description") or "",
            "people": [p.get("name") for p in asset.get("people") or [] if p.get("name")],
            "tags": [t.get("value") for t in asset.get("tags") or [] if t.get("value")],
            "favorite": bool(asset.get("isFavorite")),
            "rating": _rating(asset) or None,
            "loaded": used_file,
            "prompt_source": found["source"] or None,
        }
        if found["parameters"]:
            meta["parameters"] = found["parameters"]
        log.info(f"Load From Immich: {label}: {used + 1}/{len(assets)} {filename}"
                 + (f", prompt from {found['source']}" if found["source"] else ", no prompt in the file"))
        return cio.NodeOutput(image, mask, found["prompt"], found["negative_prompt"],
                              meta["description"], filename, width, height, used, len(assets),
                              asset["id"], json.dumps(meta, ensure_ascii=False))


# ----------------------------------------------------------------------------
# Album and tag names for the node's pickers (web/js/gadzoinks.js)
# ----------------------------------------------------------------------------
NAMES_TTL = 60          # seconds; 'refresh' in the picker skips the cache
_names_cache = {"at": 0.0, "key": None, "data": None}


def immich_names(refresh=False):
    """{'albums': [...], 'tags': [...], 'error': str|None}, sorted by name."""
    settings = get_immich_settings()
    missing = get_missing(settings)
    if missing:
        return {"albums": [], "tags": [], "error": f"Immich is not set up: missing {', '.join(missing)} "
                                                   f"(Settings > Gadzoinks)."}
    url = f"http://{settings.get('immich_hostname')}:{settings.get('immich_port')}"
    key = (url, settings.get("immich_apikey"))
    now = time.time()
    if (not refresh and _names_cache["key"] == key and _names_cache["data"] is not None
            and now - _names_cache["at"] < NAMES_TTL):
        return _names_cache["data"]
    importer = ImmichImporter(url, settings.get("immich_apikey"), importer_name="GLoadImageFromImmich")
    data, errors = {"albums": [], "tags": [], "error": None}, []
    try:
        data["albums"] = sorted({str(a.get("albumName")) for a in importer.list_albums()
                                 if a.get("albumName")}, key=str.lower)
    except ImmichError as e:
        errors.append(str(e))
        if "cannot reach" in str(e) or "did not answer" in str(e):
            data["error"] = str(e)          # the server is down: don't wait for it twice
            return data
    try:
        data["tags"] = sorted({str(t.get("value") or t.get("name")) for t in importer.list_tags()
                               if t.get("value") or t.get("name")}, key=str.lower)
    except ImmichError as e:
        errors.append(str(e))
    data["error"] = "; ".join(dict.fromkeys(errors)) or None
    if not errors:
        _names_cache.update(at=now, key=key, data=data)
    return data


@PromptServer.instance.routes.get("/gadzoinks/immich/names")
async def immich_names_route(request):
    refresh = request.rel_url.query.get("refresh") in ("1", "true")
    loop = asyncio.get_running_loop()
    try:            # blocking HTTP and the settings wait stay off the event loop
        data = await loop.run_in_executor(None, immich_names, refresh)
    except Exception as e:
        log.warning(f"Load From Immich: listing albums and tags failed: {e}")
        data = {"albums": [], "tags": [], "error": str(e)}
    return web.json_response(data)


IMMICH_LOADER_NODES = {"GLoadImageFromImmich": GLoadImageFromImmich}
IMMICH_LOADER_DISPLAY_NAME_MAPPINGS = {"GLoadImageFromImmich": "Load Image From Immich"}
