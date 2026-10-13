"""19장용 GIF: 신고 전화 → M2 → M1 → M3 → 상황실 화면이 순서대로 켜진다.
- 첫 프레임은 전부 켜진 완성 상태(PDF·썸네일에서도 뜻이 통하게), 이어서 어둡게 리셋 → 순서대로 점등 → 완성 상태 유지 → 반복.
- 아이콘은 ppt/figures/icons/flow_*.png(make_icons.py 산출물). 실제 데이터는 쓰지 않는다.
실행: python ppt/tools/make_gif_flow.py → ppt/figures/anim/flow_pipeline.gif
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ICONS = os.path.join(HERE, "..", "figures", "icons")
OUT = os.path.join(HERE, "..", "figures", "anim")
os.makedirs(OUT, exist_ok=True)

W, H = 1920, 232
ICON = 124
BLUE, GREEN, SUB, SKY = (0x2F, 0x4E, 0x91), (0x5E, 0x8E, 0x2A), (0x61, 0x61, 0x61), (0x00, 0x9D, 0xDA)
OFF = (0xE0, 0xE0, 0xE0)
FB = ImageFont.truetype(r"C:\Windows\Fonts\malgunbd.ttf", 33)
FR = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 27)

NODES = [("flow_call", "신고 전화", "음성 입력", BLUE), ("flow_m2", "Mission 2", "신고자 발화 분리", BLUE),
         ("flow_m1", "Mission 1", "신고자 성별", BLUE), ("flow_m3", "Mission 3", "증상 태깅", BLUE),
         ("flow_screen", "상황실 화면", "자동 기록", GREEN)]
icons = [Image.open(os.path.join(ICONS, n[0] + ".png")).convert("RGBA").resize((ICON, ICON), Image.LANCZOS) for n in NODES]


def mix(c1, c2, a):
    return tuple(int(c1[i] * (1 - a) + c2[i] * a) for i in range(3))


def clamp(v):
    return max(0.0, min(1.0, v))


def frame(t):
    """t: 0~4.6 진행도. 노드 i는 [i, i+0.6]에 점등, 화살표 i는 [i+0.6, i+1.0]에 점등."""
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    gap = W / len(NODES)
    for i, (_, title, sub, col) in enumerate(NODES):
        cx = int(gap * i + gap / 2)
        a = clamp((t - i) / 0.6)
        ic = icons[i]
        dim = ic.convert("LA").convert("RGBA")
        dim.putalpha(dim.getchannel("A").point(lambda v: int(v * 0.35)))
        base = Image.alpha_composite(Image.new("RGBA", ic.size, (255, 255, 255, 255)), dim)
        lit = Image.alpha_composite(Image.new("RGBA", ic.size, (255, 255, 255, 255)), ic)
        im.paste(Image.blend(base, lit, a).convert("RGB"), (cx - ICON // 2, 8))
        d.text((cx, 140), title, font=FB, fill=mix(OFF, col, a), anchor="ma")
        d.text((cx, 184), sub, font=FR, fill=mix(OFF, SUB, a), anchor="ma")
        if i < len(NODES) - 1:
            aa = clamp((t - i - 0.6) / 0.4)
            x, y = int(cx + gap / 2 - 18), 8 + ICON // 2 - 23
            d.polygon([(x, y), (x + 20, y), (x + 36, y + 23), (x + 20, y + 46), (x, y + 46), (x + 16, y + 23)], fill=mix(OFF, SKY, aa))
    return im


frames, durs = [frame(4.6)], [900]
frames.append(frame(-0.0)); durs.append(500)
n = 46
for k in range(1, n + 1):
    frames.append(frame(4.6 * k / n)); durs.append(60)
durs[-1] = 2200
pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
path = os.path.join(OUT, "flow_pipeline.gif")
pal[0].save(path, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True, disposal=1)
print(path, os.path.getsize(path) // 1024, "KB,", len(frames), "frames")
