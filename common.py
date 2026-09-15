import io
import os
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
import torch

# (♩ ♪ ♫ ♬, U+2669–266C)

def pil_to_comfy(img) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)      # [1, H, W, C]


_NOTE_FONT = os.path.join(os.path.dirname(__file__), "note.ttf")
_MUSIC_PNG = os.path.join(os.path.dirname(__file__), "music.png")

def fallback_cover(size=1080):
    img = PILImage.open(_MUSIC_PNG).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), PILImage.LANCZOS)
    return img

def XXfallback_cover(size=1080):
    img = PILImage.new("RGB", (size, size), (24, 26, 32))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(_NOTE_FONT, int(size * 0.55))
    d.text((size / 2, size / 2), "\u266B", font=font, anchor="mm",
           fill=(235, 235, 240))
    return img

def fallback_cover_os(size=1080):
    """Dark card with a music note when no image is provided."""
    img = PILImage.new("RGB", (size, size), (24, 26, 32))
    d = ImageDraw.Draw(img)
    font = None
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",   # macOS
              "C:/Windows/Fonts/seguisym.ttf"):                          # Windows
        if os.path.exists(p):
            font = ImageFont.truetype(p, int(size * 0.5))
            break
    note = "\u266B"  # ♫
    if font:
        bbox = d.textbbox((0, 0), note, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((size - w) / 2 - bbox[0], (size - h) / 2 - bbox[1]),
               note, font=font, fill=(235, 235, 240))
    else:  # no usable font anywhere: simple glyph from shapes
        d.ellipse([size*0.30, size*0.62, size*0.46, size*0.74], fill=(235,235,240))
        d.rectangle([size*0.44, size*0.25, size*0.465, size*0.68], fill=(235,235,240))
    return img
# cover = _cover_from_tensor(image) if image is not None else image_from_character("🎶")

def cover_from_tensor(image):
    """ComfyUI IMAGE tensor [B,H,W,C] float 0-1 -> PIL"""
    return PILImage.fromarray(
        (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))

