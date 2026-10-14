"""11장용 합성 도식: 실제 통화 데이터를 쓰지 않고 난수 시드로 만든 신호. 두 화자 파형 + 겹침 구간 + log-mel 질감."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.signal import spectrogram
fm.fontManager.addfont("C:/Windows/Fonts/malgun.ttf"); fm.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
BLUE, SKY, RED, LINE, SUB = "#2F4E91", "#009DDA", "#E53935", "#E0E0E0", "#616161"
sr, T = 8000, 6.0
t = np.arange(int(sr*T))/sr
rng = np.random.default_rng(7)
def voice(f0, segs, formants):
    env = np.zeros_like(t)
    for a, b in segs:
        m = (t >= a) & (t < b); x = (t[m]-a)/(b-a)
        env[m] = np.sin(np.pi*x)**0.7 * (0.8+0.2*np.sin(2*np.pi*3.5*t[m]))
    f = f0*(1+0.06*np.sin(2*np.pi*0.7*t))
    ph = 2*np.pi*np.cumsum(f)/sr
    sig = sum(np.sin(k*ph)*np.exp(-k/ (fm_/f0)) for k, fm_ in [(1, formants[0]), (2, formants[0]), (3, formants[1]), (4, formants[1]), (5, formants[2]), (6, formants[2])])
    return env*sig*0.5 + env*0.04*rng.standard_normal(len(t))
segA = [(0.2, 2.4), (3.6, 5.8)]          # 대원
segB = [(1.9, 3.9)]                      # 신고자 — A와 0.5s, 3.6~3.9s 겹침
a = voice(130, segA, (700, 1500, 2600)); b = voice(210, segB, (500, 1900, 3000))
mix = a + b
fig = plt.figure(figsize=(8.8, 2.1), dpi=220)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.15, wspace=0.28)
axes = []
for i, (sig, col, name) in enumerate([(a, BLUE, "대원"), (b, RED, "신고자")]):
    ax = fig.add_subplot(gs[i, 0]); axes.append(ax); ax.plot(t, sig, color=col, lw=0.4); ax.set_xlim(0, T); ax.set_ylim(-1.9, 1.9)
    ax.set_yticks([]); ax.set_ylabel(name, color=col, fontweight="bold", rotation=0, labelpad=22, va="center", fontsize=10)
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(LINE); ax.tick_params(colors=SUB, labelsize=7)
    if i == 0: ax.set_xticklabels([])
    for lo, hi in [(1.9, 2.4), (3.6, 3.9)]: ax.axvspan(lo, hi, color="#FFCDD2", alpha=0.8, lw=0)
axes[0].set_title("두 사람 목소리 (분홍 = 겹친 구간)", color=RED, fontsize=8.5, fontweight="bold", loc="left")
ax2 = fig.add_subplot(gs[:, 1])
f, tt, S = spectrogram(mix, sr, nperseg=256, noverlap=192)
ax2.pcolormesh(tt, f, 10*np.log10(S+1e-9), cmap="Blues", shading="auto", vmin=-70, vmax=-15)
ax2.set_ylim(0, 4000); ax2.set_ylabel("Hz", color=SUB, fontsize=7); ax2.tick_params(colors=SUB, labelsize=7)
ax2.set_title("log-mel 스펙트로그램 (CNN 입력)", color=BLUE, fontsize=8.5, fontweight="bold", loc="left")
for lo, hi in [(1.9, 2.4), (3.6, 3.9)]: ax2.axvspan(lo, hi, facecolor="none", edgecolor=RED, lw=1, ls="--")
fig.savefig("ppt/figures/m2_synthetic_waveform.png", bbox_inches="tight", facecolor="white")
