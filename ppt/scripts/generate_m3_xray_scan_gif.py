#!/usr/bin/env python3
"""Generate a presentation-safe looping GIF for the M3 quotation card.

This is an explanatory scan animation, NOT actual model inference. It
uses the team's existing flow_call.png icon and Pillow-drawn decorations.
First frame conveys the same concept in PDF/static slide previews.
"""
from math import pi, sin
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
PHONE = ROOT / "figures" / "icons" / "flow_call.png"
DEST = ROOT / "figures" / "anim" / "m3_xray_scan.gif"
W, H, N = 510, 192, 14
BG = (234, 243, 250, 255)
BLUE = (47, 78, 145, 255)
SKY = (0, 157, 218, 255)
GRID = (215, 226, 240, 255)


def make_frame(i: int):
    phase = 2 * pi * i / N
    im = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((10, 18, 140, 175), radius=24,
                        fill=(245, 248, 252, 255), outline=GRID, width=2)
    phone = Image.open(PHONE).convert("RGBA")
    phone.thumbnail((109, 124), Image.Resampling.LANCZOS)
    phone = phone.rotate(4.5 * sin(phase), Image.Resampling.BICUBIC, expand=True)
    im.alpha_composite(phone, (76 - phone.width // 2, 95 - phone.height // 2))

    d.rounded_rectangle((156, 42, 293, 150), radius=22,
                        fill=(255, 255, 255, 255), outline=GRID, width=2)
    bars = [18, 29, 53, 76, 42, 60, 35, 18]
    for j, size in enumerate(bars):
        h = int(size * (0.78 + 0.22 * sin(phase + j * 0.62)))
        x = 172 + j * 14
        c = BLUE if j in (3, 5) else SKY
        d.rounded_rectangle((x, 96-h//2, x+8, 96+h//2), radius=4, fill=c)
    d.line((296, 96, 318, 96), fill=SKY, width=4)
    d.polygon([(318, 96), (308, 90), (308, 102)], fill=SKY)

    d.rounded_rectangle((326, 25, 496, 164), radius=22,
                        fill=(255, 255, 255, 255), outline=GRID, width=2)
    for yy, length in ((57, 125), (82, 138), (107, 118), (132, 133)):
        d.rounded_rectangle((344, yy, 344+length, yy+7), radius=3,
                            fill=(221, 232, 245, 255))
    scan_x = int(340 + i / (N-1) * 140)
    glow = Image.new("RGBA", (W, H), (0,0,0,0))
    g = ImageDraw.Draw(glow)
    g.rectangle((scan_x-8, 40, scan_x+8, 155), fill=(0, 157, 218, 92))
    im = Image.alpha_composite(im, glow.filter(ImageFilter.GaussianBlur(5)))
    d = ImageDraw.Draw(im)
    d.line((scan_x, 42, scan_x, 155), fill=SKY, width=3)
    return im.convert("RGB")


DEST.parent.mkdir(parents=True, exist_ok=True)
frames = [make_frame(i) for i in range(N)]
frames[0].save(DEST, save_all=True, append_images=frames[1:],
               duration=95, loop=0, optimize=True, disposal=2)
with Image.open(DEST) as check:
    assert check.n_frames >= 12
    assert check.info.get("loop") == 0
    assert check.size == (W, H)
assert DEST.stat().st_size < 600_000
print(f"Generated {DEST}, frames={N}, size={DEST.stat().st_size} bytes")
