"""Canva 원본 PNG를 슬라이드용으로 정리한다. (원본은 그대로 두고 _crop / sym_N 파일을 새로 만든다)
- 거의 흰색(>=246) 배경을 순백으로 만들고, 내용 경계에 맞춰 여백을 잘라낸다.
- symptoms_icons.png(3x3)는 아이콘 9개로 잘라 sym_0..8.png로 저장한다 (증상 순서: 고열·구토·두통·복통·어지러움·열상·오심·전신쇠약·호흡곤란).
실행: python ppt/tools/crop_canva.py
"""
import os
import numpy as np
from PIL import Image

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "canva")


def load_white(name):
    a = np.array(Image.open(os.path.join(D, name)).convert("RGB"))
    a[(a >= 246).all(axis=2)] = 255
    return a


def bbox(a, thr=235):
    ys, xs = np.where(a.min(axis=2) < thr)
    return xs.min(), xs.max() + 1, ys.min(), ys.max() + 1


def crop(a, pad=24):
    x0, x1, y0, y1 = bbox(a)
    h, w = a.shape[:2]
    return a[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]


for name in ("hero_call_flow", "speakers_overlap"):
    out = crop(load_white(name + ".png"))
    Image.fromarray(out).save(os.path.join(D, name + "_crop.png"))
    print(name + "_crop.png", out.shape[1], "x", out.shape[0], "ratio", round(out.shape[1] / out.shape[0], 3))

# 증상 아이콘 3x3 → 9개
a = load_white("symptoms_icons.png")
x0, x1, y0, y1 = bbox(a)
cw, ch = (x1 - x0) / 3, (y1 - y0) / 3
n = 0
for r in range(3):
    for c in range(3):
        cell = a[int(y0 + r * ch):int(y0 + (r + 1) * ch), int(x0 + c * cw):int(x0 + (c + 1) * cw)]
        icon = crop(cell, pad=6)
        # 정사각형으로 맞춤
        h, w = icon.shape[:2]
        s = max(h, w)
        sq = np.full((s, s, 3), 255, dtype=np.uint8)
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = icon
        Image.fromarray(sq).save(os.path.join(D, f"sym_{n}.png"))
        n += 1
print("sym icons:", n)
