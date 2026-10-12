"""11장용 GIF: 두 화자의 파형이 흘러가며 log-mel 스펙트로그램이 오른쪽에 쌓인다 (합성 신호, 실제 통화 아님).
첫 프레임은 완성된 그림, 이어서 재생 막대가 0→6초로 이동하며 파형과 스펙트로그램이 채워지고, 완성 상태로 유지한 뒤 반복한다.
신호 생성은 make_m2_synthetic.py와 같은 난수 시드(7)를 쓴다.
실행: python ppt/tools/make_gif_spectrogram.py → ppt/figures/anim/m2_waveform_to_mel.gif
"""
import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.signal import spectrogram
from PIL import Image

fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf"); fm.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
BLUE, RED, LINE, SUB = "#2F4E91", "#E53935", "#E0E0E0", "#616161"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "anim")
os.makedirs(OUT, exist_ok=True)

sr, T = 8000, 6.0
t = np.arange(int(sr * T)) / sr
rng = np.random.default_rng(7)


def voice(f0, segs, formants):
    env = np.zeros_like(t)
    for a, b in segs:
        m = (t >= a) & (t < b); x = (t[m] - a) / (b - a)
        env[m] = np.sin(np.pi * x) ** 0.7 * (0.8 + 0.2 * np.sin(2 * np.pi * 3.5 * t[m]))
    f = f0 * (1 + 0.06 * np.sin(2 * np.pi * 0.7 * t)); ph = 2 * np.pi * np.cumsum(f) / sr
    sig = sum(np.sin(k * ph) * np.exp(-k / (fm_ / f0)) for k, fm_ in [(1, formants[0]), (2, formants[0]), (3, formants[1]), (4, formants[1]), (5, formants[2]), (6, formants[2])])
    return env * sig * 0.5 + env * 0.04 * rng.standard_normal(len(t))


a = voice(130, [(0.2, 2.4), (3.6, 5.8)], (700, 1500, 2600)); b = voice(210, [(1.9, 3.9)], (500, 1900, 3000))
f, tt, S = spectrogram(a + b, sr, nperseg=256, noverlap=192)
SdB = 10 * np.log10(S + 1e-9)
OVER = [(1.9, 2.4), (3.6, 3.9)]


def render(tp):
    fig = plt.figure(figsize=(8.0, 2.325), dpi=160)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.15, wspace=0.28)
    axes = []
    for i, (sig, col, name) in enumerate([(a, BLUE, "대원"), (b, RED, "신고자")]):
        ax = fig.add_subplot(gs[i, 0]); axes.append(ax)
        m = t <= tp
        ax.plot(t[m], sig[m], color=col, lw=0.4)
        ax.set_xlim(0, T); ax.set_ylim(-1.9, 1.9); ax.set_yticks([])
        ax.set_ylabel(name, color=col, fontweight="bold", rotation=0, labelpad=22, va="center", fontsize=11)
        for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
        ax.spines["bottom"].set_color(LINE); ax.tick_params(colors=SUB, labelsize=8)
        if i == 0: ax.set_xticklabels([])
        for lo, hi in OVER:
            if tp >= lo: ax.axvspan(lo, min(hi, tp), color="#FFCDD2", alpha=0.8, lw=0)
        if tp < T: ax.axvline(tp, color=SUB, lw=1)
    axes[0].set_title("두 사람 목소리 (분홍 = 겹친 구간)", color=RED, fontsize=9.5, fontweight="bold", loc="left")
    ax2 = fig.add_subplot(gs[:, 1])
    M = np.ma.masked_where(np.broadcast_to(tt[None, :] > tp, SdB.shape), SdB)
    ax2.pcolormesh(tt, f, M, cmap="Blues", shading="auto", vmin=-70, vmax=-15)
    ax2.set_xlim(0, T); ax2.set_ylim(0, 4000); ax2.set_ylabel("Hz", color=SUB, fontsize=8); ax2.tick_params(colors=SUB, labelsize=8)
    ax2.set_title("log-mel 스펙트로그램 (CNN 입력)", color=BLUE, fontsize=9.5, fontweight="bold", loc="left")
    for lo, hi in OVER:
        if tp >= hi: ax2.axvspan(lo, hi, facecolor="none", edgecolor=RED, lw=1.1, ls="--")
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="white"); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames, durs = [render(T)], [900]
N = 30
for k in range(N + 1):
    frames.append(render(T * k / N)); durs.append(90)
durs[-1] = 2200
size = frames[0].size
frames = [fr.resize(size) for fr in frames]
pal = [fr.convert("P", palette=Image.ADAPTIVE, colors=96) for fr in frames]
path = os.path.join(OUT, "m2_waveform_to_mel.gif")
pal[0].save(path, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True, disposal=1)
print(path, os.path.getsize(path) // 1024, "KB,", len(frames), "frames,", size)
