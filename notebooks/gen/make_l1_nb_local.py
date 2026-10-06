"""Generator for notebooks/mission2_l1_retrain.ipynb (M2 L1 retrain, local CPU).

Usage (from repo root): python notebooks/gen/make_l1_nb_local.py [output.ipynb]
Default output: notebooks/mission2_l1_retrain.ipynb
"""
import json, sys
cells=[]
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(True)})
md("""
# M2 보험 제출본 (L1 재학습 노트북, 로컬 CPU) · 코드 = 실행 시점 `origin/main` (0번 셀에서 commit 출력)

- 용도: M2 보험 제출본 재현 (L1: 발화 구간 음향 통계 53차원 + LightGBM 기본값). Training(서울) 9,000통화 샘플(`--n-calls 9000 --seed 42`)
- 참고 기준: 03:43 KST Colab 실행 = 공식 Val **84.26%** (0.8425504926438404), ckpt sha256 `3f1e872a…8780` — 이것은 **L2 대조용 baseline (제출 후보 아님)**
- 판정: |acc − 0.8425504926438404| ≤ 0.003 → **보험 확정**, 아니면 **원인 확인**. ckpt SHA가 달라도 허용(버전은 pip_freeze로 고정 기록)
- 에폭/조기종료 없음. 내부 검증은 call_id 그룹 80/20(`--seed 42`). **공식 Validation(서울 3,640통화)은 마지막 평가만**
- 입력 규칙(#41): 대상 발화 startAt~endAt 음성만. text·발화 순서·통화 단위 정보 미사용. 제공 라벨 그대로
- 팀 규칙: 0번 셀에서 `origin/main` 코드 zip을 만들고(sha256·commit·파일 목록 출력) 그 압축을 푼 코드로만 실행 (공유 작업 트리 미사용)
- 로컬 CPU (feature 추출 workers=4, M1/M3와 CPU 공유). 개인 절대경로 없음: 저장소 루트는 `git rev-parse --show-toplevel`(노트북 위치 기준) 또는 환경변수 `DCC_REPO_ROOT`
- 경로: `DATA_ROOT = <repo>/data/raw`, `OUT = <repo>/outputs/m2_l1_retrain` (gitignore 대상), manifest = `<repo>/data/manifests/dcc_manifest_v1.csv`
""")
md("## 0. 코드 고정 (origin/main zip → 압축 해제 → 그 코드로 실행)")
code(r"""
from pathlib import Path
import os, sys, subprocess, json, hashlib, time, zipfile, shutil

def _toplevel(start):
    r = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0: raise RuntimeError(f"git 저장소 안에서 실행하거나 DCC_REPO_ROOT를 지정하세요: {r.stderr}")
    return Path(r.stdout.strip())
WT = Path(os.environ["DCC_REPO_ROOT"]) if os.environ.get("DCC_REPO_ROOT") else _toplevel(Path.cwd())   # 공유 작업 트리 (읽기만: git fetch/archive)
WT = WT.resolve()
DATA_ROOT = WT / "data" / "raw"          # data/raw_merged 아님 (manifest 상대경로 기준)
OUT       = WT / "outputs" / "m2_l1_retrain"   # outputs/ 는 .gitignore 대상
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run_log.txt"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def git(*a):
    r = subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0: raise RuntimeError(f"git {a} failed: {r.stderr}")
    return r.stdout.strip()

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

log("=== M2 보험 제출본 (L1 재학습 노트북, 로컬 CPU) start ===")
git("fetch", "origin")                       # 원격 ref만 갱신 (작업 트리/브랜치 변경 없음)
COMMIT = git("rev-parse", "origin/main")
print("작업 트리 브랜치(참고, 실행 코드와 무관):", git("branch", "--show-current"), "| cwd:", Path.cwd())
ZIP = OUT / f"code_main_{COMMIT[:7]}.zip"
subprocess.run(["git", "-C", str(WT), "archive", "--format=zip", "-o", str(ZIP), "origin/main"], check=True)
with zipfile.ZipFile(ZIP) as z:
    zip_comment = z.comment.decode("ascii", "replace")
    names = z.namelist()
    CODE = OUT / "code"
    if CODE.exists(): shutil.rmtree(CODE)
    z.extractall(CODE)
assert zip_comment == COMMIT, f"zip comment {zip_comment!r} != {COMMIT}"
ZIP_SHA = sha256(ZIP)
log(f"code zip: {ZIP}")
log(f"code zip sha256: {ZIP_SHA}")
log(f"commit (origin/main): {COMMIT}")
log(f"zip comment (git archive embedded commit): {zip_comment}")
files = [n for n in names if not n.endswith("/")]
print(f"zip files total: {len(files)}")
for n in files:
    if n.startswith(("scripts/", "src/")) or "/" not in n: print("  ", n)

REPO = CODE                                   # 이후 모든 스크립트는 압축 해제한 코드에서 실행
MANIFEST = REPO / "data/manifests/dcc_manifest_v1.csv"
if not MANIFEST.exists():
    MANIFEST = WT / "data/manifests/dcc_manifest_v1.csv"   # manifest는 git에 없음 → 작업 트리의 data/ 사본(읽기만)
import pandas as pd
_m = pd.read_csv(MANIFEST)
MANIFEST_SHA = sha256(MANIFEST)
log(f"manifest: {MANIFEST} | sha256 {MANIFEST_SHA} | rows {len(_m)} | split {_m['split'].value_counts().to_dict()}")
assert len(_m) == 32840, len(_m)
assert DATA_ROOT.exists(), DATA_ROOT
os.chdir(REPO); sys.path.insert(0, str(REPO / "src"))
os.environ["DCC_DATA_ROOT"] = str(DATA_ROOT)
log(f"resolved paths: WT(repo root)={WT} | REPO(code run dir)={REPO} | DATA_ROOT={DATA_ROOT} | OUT={OUT} | MANIFEST={MANIFEST}")
""")
md("## 0-1. 설정 · 실행 헬퍼")
code(r"""
N_CALLS, SEED, WORKERS = 9000, 42, 4
REF_VAL_ACC = 0.8425504926438404      # 03:43 KST Colab 실행 (L2 대조용 baseline, 제출 후보 아님)
REF_CKPT_SHA = "3f1e872a5d1025f0630fb0327dac551711ed56dd40409153c5756c6bd0ef8780"
RUN_BUILD = RUN_TRAIN = RUN_EVAL = True

CHILD_ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
             "TQDM_MININTERVAL": "30"}   # tqdm 진행 표시는 30초마다 (로그/노트북 출력 크기 제한)

def run(cmd):
    cmd = [str(c) for c in cmd]           # list args (Windows 따옴표 문제 없음)
    line = "$ " + " ".join(cmd); print(line, flush=True)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}\n"); f.flush()
        p = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", env=CHILD_ENV)
        for l in p.stdout:
            print(l, end="", flush=True)
            f.write(f"[{time.strftime('%H:%M:%S')}] {l}"); f.flush()
        rc = p.wait()
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] exit {rc} ({time.time()-t0:.0f}s)\n")
    if rc != 0: raise RuntimeError(f"exit {rc}: {line}")
""")
md("## 0-2. 공식 Validation 폴더 (서울 3,640통화) — 분할 압축 폴더에 흩어진 WAV/JSON을 하드링크로 한 폴더에 모음")
code(r"""
VAL_DIR = OUT / "val_seoul"
VAL_AUDIO_DIR, VAL_LABEL_DIR = VAL_DIR / "wav", VAL_DIR / "json"
src_wav  = sorted(DATA_ROOT.glob("*/대학부 데이터/Validation/1.원천데이터/VS_서울_구급/*.wav"))
src_json = sorted(DATA_ROOT.glob("*/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급/*.json"))
_seoul = set(src_wav) | set(src_json)
other = [p for p in DATA_ROOT.glob("*/대학부 데이터/Validation/**/*") if p.suffix.lower() in (".wav", ".json") and p not in _seoul]
print("Seoul Val WAV", len(src_wav), "| JSON", len(src_json), "| 그 외 Validation wav/json", len(other))
assert len(src_wav) == 3640 and len(src_json) == 3640, (len(src_wav), len(src_json))
ws, js = {p.stem: p for p in src_wav}, {p.stem: p for p in src_json}
assert len(ws) == 3640 and len(js) == 3640 and set(ws) == set(js), "stem 중복/불일치"
val_manifest_stems = set(_m.loc[_m["split"] == "Validation", "stem"])
assert set(ws) == val_manifest_stems, "manifest Validation stem과 불일치"
method = {}
for d, srcs in ((VAL_AUDIO_DIR, ws), (VAL_LABEL_DIR, js)):
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    for stem, sp in srcs.items():
        dst = d / sp.name
        try: os.link(sp, dst); method["hardlink"] = method.get("hardlink", 0) + 1
        except OSError: shutil.copy2(sp, dst); method["copy"] = method.get("copy", 0) + 1
n_w, n_j = len(list(VAL_AUDIO_DIR.glob("*.wav"))), len(list(VAL_LABEL_DIR.glob("*.json")))
assert n_w == 3640 and n_j == 3640
log(f"VAL folder: {VAL_DIR} | made from DATA_ROOT/*/대학부 데이터/Validation/{{1.원천데이터/VS_서울_구급,2.라벨링데이터/VL_서울_구급}} "
    f"(4 split parts) | wav {n_w} json {n_j} | method {method} | stems == manifest Validation stems")
""")
md("## 1. 환경 기록 (설치 없음 · 버전만 기록)")
code(r"""
import platform, numpy, pandas, sklearn, lightgbm, librosa, pyarrow, tqdm, scipy, soundfile
env = {"commit": COMMIT, "code_zip_sha256": ZIP_SHA, "manifest_sha256": MANIFEST_SHA, "python": sys.version, "platform": platform.platform(),
       "numpy": numpy.__version__, "pandas": pandas.__version__, "pyarrow": pyarrow.__version__, "sklearn": sklearn.__version__,
       "lightgbm": lightgbm.__version__, "librosa": librosa.__version__, "scipy": scipy.__version__, "soundfile": soundfile.__version__,
       "tqdm": tqdm.__version__, "cpus": os.cpu_count(), "workers": WORKERS}
log("env " + json.dumps(env, ensure_ascii=False))
json.dump(env, open(OUT / "env.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
def pip_freeze(path):
    r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    Path(path).write_text(r.stdout, encoding="utf-8"); return len(r.stdout.splitlines())
print("pip freeze lines", pip_freeze(OUT / "pip_freeze_start.txt"))
""")
md("## 2. Feature 추출 (Training 9,000통화 · workers=4 · 200통화마다 중간 저장)")
code(r"""
FEAT, IDX, FAIL = OUT/"m2_features_train_9k.parquet", OUT/"m2_utterances_train_9k.parquet", OUT/"m2_build_failures_9k.jsonl"
if RUN_BUILD:
    # 주의: --resume은 --n-calls 0에서만 허용됨. 9k 샘플은 끊기면 처음부터 다시
    run([sys.executable, "scripts/build_mission2_dataset.py", "--manifest", MANIFEST, "--data-root", DATA_ROOT,
         "--n-calls", N_CALLS, "--seed", SEED, "--workers", WORKERS, "--flush-every", 200,
         "--index-output", IDX, "--features-output", FEAT, "--failures-log", FAIL])
f = pd.read_parquet(FEAT); log(f"features: 발화 {len(f)} | 통화 {f['call_id'].nunique() if 'call_id' in f else '?'} | sha256 {sha256(FEAT)}")
""")
md("## 3. 학습 (LightGBM 기본값 · 내부 call_id 그룹 80/20)")
code(r"""
CKPT, TRAIN_LOG = OUT/"m2_baseline_lgbm_9k.pkl", OUT/"m2_exp_log_9k.json"
if RUN_TRAIN:
    run([sys.executable, "scripts/train_mission2_baseline.py", "--features", FEAT, "--seed", SEED,
         "--test-size", 0.2, "--model-output", CKPT, "--log-output", TRAIN_LOG])
print(json.dumps(json.load(open(TRAIN_LOG, encoding="utf-8")), ensure_ascii=False, indent=1)[:3000])
CKPT_SHA = sha256(CKPT)
log(f"ckpt sha256 {CKPT_SHA} " + ("| 기존 실행과 동일" if CKPT_SHA == REF_CKPT_SHA else "| 기존과 다름(허용 · 버전은 pip_freeze 참고)"))
""")
md("## 4. 공식 Validation — 제출용 `mission2_inference.py` 그대로 (평가만)")
code(r"""
PRED = OUT/"mission2_val.csv"
if RUN_EVAL:
    run([sys.executable, "scripts/mission2_inference.py", "--audio_dir", VAL_AUDIO_DIR, "--label_dir", VAL_LABEL_DIR,
         "--ckpt_path", CKPT, "--output", PRED])
""")
code(r"""
# 정답 대조: inference.py는 라벨 파일의 발화 순서대로 행을 씀 → 파일별 순서로 맞춤
import numpy as np
pred = pd.read_csv(PRED)
pred["stem"] = pred["audio_file_name"].map(lambda s: Path(str(s)).stem)
rows = []
for jp in sorted(Path(VAL_LABEL_DIR).rglob("*.json")):
    doc = json.load(open(jp, encoding="utf-8"))
    for i, u in enumerate(doc["utterances"]):      # startAt/endAt/speaker만 읽음 (text 미사용)
        rows.append({"stem": jp.stem, "i": i, "startAt": u["startAt"], "endAt": u["endAt"], "y": int(u["speaker"])})
gt = pd.DataFrame(rows)
pred["i"] = pred.groupby("stem").cumcount()
df = gt.merge(pred[["stem", "i", "speaker"]], on=["stem", "i"], how="left", validate="one_to_one")
assert df["speaker"].notna().all() and len(df) == len(pred), "행 수/정렬 불일치"
df["ok"] = df["y"] == df["speaker"]
acc = df.ok.mean()
cm = pd.crosstab(df.y, df.speaker).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values.tolist()
log(f"공식 Val Acc {acc:.4%} ({acc!r}) | 발화 {len(df)} | 통화 {df.stem.nunique()} | 혼동행렬 {cm}")
""")
md("## 5. 슬라이스 (분석용)")
code(r"""
df["dur"] = df["endAt"] - df["startAt"]          # 단위가 초인지 확인 (ms면 /1000)
def call_overlap_ratio(g):
    iv = g[["startAt", "endAt"]].to_numpy(float); iv = iv[iv[:, 0].argsort()]
    ov = 0.0; end = -np.inf
    for s, e in iv:                                # 통화 안에서 두 발화 이상 겹치는 시간
        if s < end: ov += min(e, end) - s
        end = max(end, e)
    span = iv[:, 1].max() - iv[:, 0].min()
    return ov / span if span > 0 else 0.0
ratio = df.groupby("stem")[["startAt", "endAt"]].apply(call_overlap_ratio); p90 = ratio.quantile(0.9)
top = set(ratio[ratio >= p90].index)
S = {"overall": df, f"overlap_top10pct_calls(p90={p90:.4f})": df[df.stem.isin(top)], "overlap_rest": df[~df.stem.isin(top)],
     "dur_<0.5s": df[df.dur < 0.5], "dur_0.5-1s": df[(df.dur >= 0.5) & (df.dur < 1)], "dur_<1s": df[df.dur < 1],
     "dur_>=1s": df[df.dur >= 1], "recall_대원(0)": df[df.y == 0], "recall_신고자(1)": df[df.y == 1]}
slices = {k: {"n": int(len(d)), "acc": float(d.ok.mean())} for k, d in S.items()}
json.dump(slices, open(OUT/"m2_l1_retrain_official_val_slices.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
pd.DataFrame(slices).T
""")
md("## 6. 판정 · 환경 고정")
code(r"""
diff = acc - REF_VAL_ACC
verdict = "보험 확정" if abs(diff) <= 0.003 else "원인 확인"
summary = {"candidate": "M2 보험 제출본 (L1 재학습 노트북, 로컬 CPU)", "commit": COMMIT, "code_zip_sha256": ZIP_SHA,
           "manifest_sha256": MANIFEST_SHA, "accuracy": acc, "ref_accuracy": REF_VAL_ACC,
           "ref_note": "03:43 KST Colab run = L2 대조용 baseline (제출 후보 아님)", "diff_pp": diff * 100,
           "verdict": verdict, "n_utt": int(len(df)), "n_calls": int(df.stem.nunique()), "confusion_matrix": cm,
           "ckpt_sha256": CKPT_SHA, "same_ckpt_as_ref": CKPT_SHA == REF_CKPT_SHA, "env": env}
json.dump(summary, open(OUT/"m2_l1_retrain_official_val_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("pip freeze lines", pip_freeze(OUT / "pip_freeze.txt"))
log(f"=== done: verdict {verdict} | acc {acc:.6f} | diff {diff*100:+.3f}pp ===")
""")
for _i, _c in enumerate(cells): _c["id"] = f"m2l1-{_i:02d}"
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "notebooks/mission2_l1_retrain.ipynb"
json.dump(nb,open(OUT_PATH,"w",encoding="utf-8",newline="\n"),ensure_ascii=False,indent=1)
