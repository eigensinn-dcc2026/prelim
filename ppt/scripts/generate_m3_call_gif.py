#!/usr/bin/env python3
"""Generate a small, presentation-safe looping M3 call/waveform GIF.

Uses only an existing repository-owned icon and Pillow-drawn wave bars.
Frame 0 is fully informative for PDF/static preview.
Usage: python ppt/scripts/generate_m3_call_gif.py
"""
from math import sin, pi
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures/icons/flow_call.png"
OUTPUT = ROOT / "figures/anim/m3_call_wave.gif"
BG = (234, 243, 250, 255)  # Existing Mission 3 C.light.
FRAMES = 12
SIZE = (360, 144)

def frame(index: int) -> Image.Image:
    t = 2 * pi * index / FRAMES
    canvas = Image.new("RGBA", SIZE, BG)
    icon = Image.open(SOURCE).convert("RGBA").resize((112, 112), Image.Resampling.LANCZOS)
    icon = icon.rotate(5.5 * sin(t), resample=Image.Resampling.BICUBIC, expand=True)
    canvas.alpha_composite(icon, (17 + (128 - icon.width) // 2, (SIZE[1] - icon.height) // 2))
    draw = ImageDraw.Draw(canvas)
    # Animated soundwave: existing brand blues, no synthetic data or labels.
    blue = (47, 78, 145, 255)
    sky = (0, 157, 218, 255)
    line = (126, 177, 216, 255)
    draw.line((140, 72, 164, 72), fill=line, width=3)
    bars = [12, 26, 42, 57, 36, 21, 42, 30, 16]
    for j, base in enumerate(bars):
        pulse = 0.78 + 0.22 * sin(t + j * 0.72)
        height = round(base * pulse)
        x = 175 + 18 * j
        col = blue if j == 3 else sky
        draw.rounded_rectangle((x, 72 - height / 2, x + 9, 72 + height / 2), radius=4, fill=col)
    return canvas.convert("RGB")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
frames = [frame(i) for i in range(FRAMES)]
# The first frame is an upright telephone with a visible waveform.
frames[0].save(OUTPUT, save_all=True, append_images=frames[1:],
               duration=100, loop=0, optimize=True, disposal=2)
assert OUTPUT.stat().st_size < 700_000, OUTPUT.stat().st_size
with Image.open(OUTPUT) as check:
    assert check.n_frames >= 8, check.n_frames
print(f"Generated {OUTPUT} ({OUTPUT.stat().st_size} bytes, {FRAMES} frames)")
