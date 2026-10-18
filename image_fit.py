"""Shared sizing for the gprompts image loaders: an optional aspect-ratio
center crop and a megapixel budget, applied while the image is decoded.

    img = open_image(path_or_bytes)     # lazy: only the header is read
    out, fit = fit_image(img, max_megapixels=2.0, aspect="16:9")

Order: crop first, then downscale, so the budget applies to the pixels you
keep. JPEGs are decoded at reduced scale (libjpeg's DCT scaling via
Image.draft), so a 48 MP photo bound for 2 MP never exists at full size in
memory. Every image is also clamped to HARD_CAP_MEGAPIXELS, even with
max_megapixels off, so one huge panorama can't stall a batch.
"""
import io
import logging
import math

from PIL import Image, ImageOps

log = logging.getLogger("gprompts.image_fit")

MIB = 1024 * 1024

# Output ceiling, applied even when max_megapixels is 0 (off). Far above
# anything a diffusion model wants; it only catches panoramas and scans.
HARD_CAP_MEGAPIXELS = 64

# Pillow refuses to open anything over 2 x Image.MAX_IMAGE_PIXELS (about
# 179 MP by default) as a decompression bomb. Files the user chose may be
# opened up to this size instead; beyond it they are refused.
OPEN_LIMIT_MEGAPIXELS = 1024

ASPECT_OPTIONS = ["free", "1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2",
                  "9:16", "16:9", "9:21", "21:9"]

MAX_MEGAPIXELS_TOOLTIP = (
    "Downscale so the output is at most this many megapixels (after any "
    "crop), keeping the aspect ratio. 0 = off. Images are never upscaled. "
    f"Anything over {HARD_CAP_MEGAPIXELS} MP is always scaled down.")
ASPECT_TOOLTIP = ("Center-crop to this width:height ratio before scaling. "
                  "'free' keeps the whole image.")

# EXIF orientations that swap width and height.
_SWAPPED = {5, 6, 7, 8}


def aspect_ratio(aspect):
    """'16:9' -> 16/9; 'free', '' or None -> None."""
    text = str(aspect or "free").strip().lower()
    if text in ("", "free"):
        return None
    try:
        w, h = (float(p) for p in text.split(":"))
    except ValueError:
        raise ValueError(f"Aspect must be 'free' or 'W:H', got '{aspect}'") from None
    if not (w > 0 and h > 0 and math.isfinite(w) and math.isfinite(h)):
        raise ValueError(f"Aspect must be 'free' or 'W:H', got '{aspect}'")
    return w / h


def open_image(source):
    """Image.open on a path, bytes or file object, allowing user-chosen files
    past Pillow's default decompression-bomb limit (up to
    OPEN_LIMIT_MEGAPIXELS). Only the header is read; fit_image decodes."""
    fp = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        return Image.open(fp)
    except Image.DecompressionBombError:
        pass
    if hasattr(fp, "seek"):
        fp.seek(0)
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = OPEN_LIMIT_MEGAPIXELS * MIB // 2  # error at 2x
    try:
        img = Image.open(fp)
    except Image.DecompressionBombError:
        raise ValueError(f"Image is over the {OPEN_LIMIT_MEGAPIXELS} megapixel "
                         "limit; resize it before loading") from None
    finally:
        Image.MAX_IMAGE_PIXELS = previous
    # Pillow repeats the bomb check when the pixels are decoded; mark this
    # one as checked so load() does not refuse it.
    img._gprompts_open_limit = True
    return img


def _oriented_size(img):
    w, h = img.size
    try:
        orientation = img.getexif().get(0x0112, 1)
    except Exception:
        orientation = 1
    return (h, w) if orientation in _SWAPPED else (w, h)


def _center_box(width, height, ratio):
    """Largest centered box of the given ratio, as fractions of the image."""
    if ratio is None:
        return None
    if width / height > ratio:          # too wide: trim the sides
        frac = ratio * height / width
        return ((1 - frac) / 2, 0.0, (1 + frac) / 2, 1.0)
    frac = (width / ratio) / height     # too tall: trim top and bottom
    if frac >= 1.0:
        return None
    return (0.0, (1 - frac) / 2, 1.0, (1 + frac) / 2)


def _pixel_box(box, width, height):
    x0, y0, x1, y1 = box
    left, top = round(x0 * width), round(y0 * height)
    right = max(left + 1, round(x1 * width))
    bottom = max(top + 1, round(y1 * height))
    return (left, top, min(right, width), min(bottom, height))


def _resizable(img):
    """A mode that Image.resize can filter with LANCZOS."""
    if img.mode in ("RGB", "RGBA", "L", "LA", "I", "F"):
        return img
    if img.mode.startswith("I;16"):
        return img.convert("I")
    if img.mode == "P" and "transparency" in img.info:
        return img.convert("RGBA")
    if img.mode in ("PA", "RGBa", "La"):
        return img.convert("RGBA")
    return img.convert("RGB")


def fit_image(img, max_megapixels=0.0, aspect="free"):
    """(PIL image, info) with EXIF orientation applied, center-cropped to
    `aspect` and scaled to at most `max_megapixels` (0 = off; always at most
    HARD_CAP_MEGAPIXELS). First frame only. `img` should come from
    open_image and not be loaded yet, so JPEGs can decode at reduced scale.

    info: original_width/height (oriented), width/height (output),
    cropped (bool), scaled (bool), clamped (bool: the hard cap applied).
    """
    try:
        max_megapixels = float(max_megapixels or 0)
    except (TypeError, ValueError):
        raise ValueError("max_megapixels must be a number") from None
    if not math.isfinite(max_megapixels) or max_megapixels < 0:
        raise ValueError("max_megapixels must be 0 (off) or more")
    ratio = aspect_ratio(aspect)

    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    orig_w, orig_h = _oriented_size(img)
    box = _center_box(orig_w, orig_h, ratio)
    if box:
        keep_w = (box[2] - box[0]) * orig_w
        keep_h = (box[3] - box[1]) * orig_h
    else:
        keep_w, keep_h = orig_w, orig_h

    budget = HARD_CAP_MEGAPIXELS * MIB
    if max_megapixels > 0:
        budget = min(budget, max_megapixels * MIB)
    scale = min(1.0, math.sqrt(budget / (keep_w * keep_h)))
    clamped = scale < 1.0 and (max_megapixels <= 0 or max_megapixels > HARD_CAP_MEGAPIXELS)
    if clamped:
        log.warning("image is %dx%d (%.0f MP); scaling it to the %d MP safety cap",
                    orig_w, orig_h, orig_w * orig_h / MIB, HARD_CAP_MEGAPIXELS)

    # JPEG: let libjpeg decode at 1/2, 1/4 or 1/8 scale, never below what
    # the final resize needs. draft() works in stored (unrotated) pixels;
    # the scale is the same either way.
    if scale < 0.5 and img.format == "JPEG":
        stored_w, stored_h = img.size
        img.draft(img.mode if img.mode in ("RGB", "L") else "RGB",
                  (math.ceil(stored_w * scale), math.ceil(stored_h * scale)))

    if getattr(img, "_gprompts_open_limit", False):
        previous = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            img.load()
        finally:
            Image.MAX_IMAGE_PIXELS = previous
    else:
        img.load()

    out = ImageOps.exif_transpose(img)
    if box:
        out = out.crop(_pixel_box(box, *out.size))
    target_w = max(1, round(keep_w * scale))
    target_h = max(1, round(keep_h * scale))
    # Rounding can overshoot the budget by a row or column: trim the side
    # that strays furthest from the aspect ratio until it fits.
    while target_w * target_h > budget and target_w > 1 and target_h > 1:
        if target_w / target_h > keep_w / keep_h:
            target_w -= 1
        else:
            target_h -= 1
    scaled = (target_w, target_h) != out.size and target_w * target_h < out.size[0] * out.size[1]
    if scaled:
        out = _resizable(out).resize((target_w, target_h), Image.LANCZOS)

    return out, {
        "original_width": orig_w,
        "original_height": orig_h,
        "width": out.size[0],
        "height": out.size[1],
        "cropped": bool(box),
        "scaled": scaled,
        "clamped": clamped,
    }
