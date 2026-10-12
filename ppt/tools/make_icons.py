"""2단계 아이콘 생성: Segoe MDL2 Assets 글리프를 원형 배지 PNG로 렌더링 (투명 배경, 팔레트 고정).
실행: python ppt/tools/make_icons.py  → ppt/figures/icons/*.png
"""
import os
from PIL import Image, ImageDraw, ImageFont

BLUE, SKY, MINT, RED, GRAY = "#2F4E91", "#009DDA", "#8BC34A", "#E53935", "#616161"
FONT = r"C:\Windows\Fonts\segmdl2.ttf"
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "icons")
os.makedirs(OUT, exist_ok=True)
S = 512  # 출력 크기(px). 슬라이드에서 0.5~0.9in로 축소해 선명하게 보임

# 이름: (글리프 코드포인트, 배지 색)
ICONS = {
    # 3장 목차
    "sec_prep": (0xE7BE, BLUE), "sec_data": (0xE9D9, SKY), "sec_mission": (0xE7C1, BLUE), "sec_wrap": (0xE930, MINT),
    # 4장 로드맵 W1~W6
    "w1": (0xE8F1, SKY), "w2": (0xE9F9, BLUE), "w3": (0xE930, BLUE), "w4": (0xE713, BLUE), "w5": (0xEC4A, BLUE), "w6": (0xE8E5, BLUE),
    # 6장 EDA
    "m1": (0xE77B, BLUE), "m2": (0xE716, BLUE), "m3": (0xE9D9, BLUE),
    # 19장 흐름
    "flow_call": (0xE717, BLUE), "flow_m2": (0xE716, BLUE), "flow_m1": (0xE77B, BLUE), "flow_m3": (0xE8BD, BLUE), "flow_screen": (0xE7F4, MINT),
}


def render(name, cp, color):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, S - 1, S - 1), fill=color)
    f = ImageFont.truetype(FONT, int(S * 0.52))
    ch = chr(cp)
    bb = d.textbbox((0, 0), ch, font=f)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((S - w) / 2 - bb[0], (S - h) / 2 - bb[1]), ch, font=f, fill="white")
    img.save(os.path.join(OUT, name + ".png"))


for n, (cp, col) in ICONS.items():
    render(n, cp, col)
print(len(ICONS), "icons ->", os.path.abspath(OUT))
