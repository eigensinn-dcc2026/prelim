"""Higgsfield MP4 → 슬라이드용 자동 반복 GIF.
pptxgenjs의 MP4는 클릭해야 재생되고 자동 재생·반복을 지원하지 않아, 슬라이드쇼에서 저절로 반복되는 GIF로 바꾼다. 원본 MP4는 그대로 둔다.
- 내용 경계에 맞춰 흰 여백을 잘라내고, 전체에서 하나의 팔레트를 만들어 프레임 깜빡임을 막는다.
- 첫 프레임은 완성된 장면(마지막 프레임)으로 둔다 (PDF·썸네일용). 마지막→처음은 크로스페이드로 이어 반복 때 튀지 않게 한다.
실행: python ppt/tools/mp4_to_gif.py → ppt/figures/anim/*.gif
"""
import os
import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "videos", "higgsfield")
OUT = os.path.join(HERE, "..", "figures", "anim")
os.makedirs(OUT, exist_ok=True)

JOBS = [  # (파일명, 출력 가로 px, fps)
    ("hero_call_flow", 900, 10),
    ("speakers_separation", 900, 10),
]


def read_frames(path):
    c = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr, c.get(cv2.CAP_PROP_FPS)


for name, width, fps_out in JOBS:
    frames, fps = read_frames(os.path.join(SRC, name + ".mp4"))
    # 모든 프레임의 내용 경계(흰색이 아닌 영역)의 합집합
    x0 = y0 = 10**9; x1 = y1 = 0
    for f in frames[:: max(1, len(frames) // 30)]:
        ys, xs = np.where(f.min(axis=2) < 235)
        x0, x1, y0, y1 = min(x0, xs.min()), max(x1, xs.max() + 1), min(y0, ys.min()), max(y1, ys.max() + 1)
    pad = 24
    h, w = frames[0].shape[:2]
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    step = max(1, round(fps / fps_out))
    seq = [f[y0:y1, x0:x1] for f in frames[::step]]
    ow, oh = width, round(width * (y1 - y0) / (x1 - x0))
    seq = [cv2.resize(f, (ow, oh), interpolation=cv2.INTER_AREA) for f in seq]
    for f in seq:  # 거의 흰 배경을 순백으로
        f[(f >= 246).all(axis=2)] = 255
    # 시간 방향 노이즈 제거: 직전 프레임과 거의 같은 픽셀은 직전 값을 유지(영상 압축 노이즈가 GIF 용량을 키우는 것을 막는다)
    for k in range(1, len(seq)):
        same = (np.abs(seq[k].astype(int) - seq[k - 1].astype(int)).max(axis=2) < 14)
        seq[k][same] = seq[k - 1][same]
    # 마지막 → 처음 크로스페이드 6프레임
    fade = [(seq[-1] * (1 - k / 7) + seq[0] * (k / 7)).astype(np.uint8) for k in range(1, 7)]
    full = [seq[-1]] + seq + fade           # 첫 프레임 = 완성 장면
    durs = [800] + [int(1000 / fps_out)] * len(seq) + [int(1000 / fps_out)] * len(fade)
    durs[len(seq)] = 1500                   # 시퀀스 끝(완성 장면)에서 잠시 멈춤
    # 전역 팔레트
    pick = [Image.fromarray(full[i]) for i in np.linspace(0, len(full) - 1, 12, dtype=int)]
    mont = Image.new("RGB", (ow, oh * len(pick)))
    for i, im in enumerate(pick):
        mont.paste(im, (0, i * oh))
    pal = mont.quantize(colors=64, method=Image.MEDIANCUT, dither=Image.NONE)
    out = [Image.fromarray(f).quantize(palette=pal, dither=Image.NONE) for f in full]
    path = os.path.join(OUT, name + ".gif")
    out[0].save(path, save_all=True, append_images=out[1:], duration=durs, loop=0, optimize=False, disposal=1)
    print(name + ".gif", f"{ow}x{oh}", f"ratio {ow / oh:.3f}", len(out), "frames", os.path.getsize(path) // 1024, "KB")
