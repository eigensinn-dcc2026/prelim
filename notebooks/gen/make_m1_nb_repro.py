import json
from pathlib import Path
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(keepends=True)}
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)}
cells = [
md("""
# Mission 1 학습 노트북 — 신고자 성별 분류 (Accuracy)

팀 eigensinn / DCC 2026 대학부 예선. **원본부터 재현 실행 (M1-EXP-001-repro)**.

- 입력: 통화 WAV + 라벨 JSON → speaker=1(신고자) 구간만 잘라 **52차원 음향 통계**(MFCC 20 mean/std, RMS, ZCR, spectral centroid/rolloff/bandwidth/flatness mean/std)
- 모델: `StandardScaler + LogisticRegression` (제출), LightGBM (비교용)
- 데이터: Seoul Training 29,200통화로만 fit. 공식 Validation 3,640통화는 평가만 (fit·scaler 통계 사용 금지, `SplitContaminationError`로 강제)
- seed = 42, 하이퍼파라미터는 M1-EXP-001과 동일 (변경 없음)
- 판정: 공식 Val LR 95.11% ±0.3%p → 재현 성공, 이 실행의 ckpt를 제출 가중치로 사용
"""),
md("## 0. 환경 · 코드 버전"),
code("""
import json, os, subprocess, sys, hashlib, platform, datetime
from pathlib import Path

REPO = Path(os.environ.get("M1_REPO", Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent)).resolve()
sys.path.insert(0, str(REPO / "src"))
DATA_ROOT = Path(os.environ.get("M1_DATA_ROOT", REPO / "data" / "raw"))   # manifest 상대경로 기준
MANIFEST = REPO / "data/manifests/dcc_manifest_v1.csv"
OUT = Path(os.environ.get("M1_OUT", REPO / "outputs" / "m1_repro"))
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
RUN_PIPELINE = True

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()

print("start:", datetime.datetime.now().isoformat(timespec="seconds"))
print("python:", sys.version.split()[0], platform.platform())
print("REPO:", REPO); print("DATA_ROOT:", DATA_ROOT); print("OUT:", OUT)
try:
    print("git HEAD:", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
except Exception as e:
    print("git HEAD: n/a", e)
for rel in ["scripts/build_mission1_dataset.py", "scripts/train_mission1_baseline.py",
            "scripts/eval_mission1_validation.py", "scripts/bootstrap_ci_mission1.py",
            "src/dcc_data/mission1_features.py", "src/dcc_data/mission1_train.py",
            "scripts/mission1_inference.py", "data/manifests/dcc_manifest_v1.csv"]:
    print(f"sha256 {rel}: {sha256(REPO / rel)}")
import numpy, pandas, sklearn, librosa, lightgbm
for m in (numpy, pandas, sklearn, librosa, lightgbm):
    print(m.__name__, m.__version__)
"""),
code("""
def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen([sys.executable, "-u", *map(str, cmd)], cwd=REPO, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    with open(OUT / "run_log.txt", "a", encoding="utf-8") as log:
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
    rc = p.wait()
    assert rc == 0, f"failed ({rc}): {cmd}"
"""),
md("## 1. 통화 단위 52-d feature 추출 (원본 WAV/JSON → parquet)\n\nValidation 먼저, 다음 Training. 두 캐시는 분리되어 Validation이 Training parquet에 섞이지 않는다."),
code("""
if RUN_PIPELINE:
    run(["scripts/build_mission1_dataset.py", "--manifest", MANIFEST, "--data-root", DATA_ROOT,
         "--split", "Validation", "--features-output", OUT / "m1_features_validation.parquet",
         "--failures-log", OUT / "m1_build_failures_validation.jsonl", "--flush-every", 500])
"""),
code("""
if RUN_PIPELINE:
    run(["scripts/build_mission1_dataset.py", "--manifest", MANIFEST, "--data-root", DATA_ROOT,
         "--split", "Training", "--features-output", OUT / "m1_features_train.parquet",
         "--failures-log", OUT / "m1_build_failures_train.jsonl", "--flush-every", 500])
"""),
md("## 2. 학습 (Training 내부 8:2 sanity → 전체 Training refit)"),
code("""
if RUN_PIPELINE:
    run(["scripts/train_mission1_baseline.py", "--features", OUT / "m1_features_train.parquet",
         "--seed", SEED,
         "--gbm-output", OUT / "m1_baseline_lgbm.pkl",
         "--lr-output", OUT / "m1_baseline_lr.pkl",
         "--log-output", OUT / "m1_exp_log.json"])
"""),
md("## 3. 공식 Validation 평가 (frozen, fit 없음) + bootstrap CI"),
code("""
if RUN_PIPELINE:
    run(["scripts/eval_mission1_validation.py", "--features", OUT / "m1_features_validation.parquet",
         "--gbm-ckpt", OUT / "m1_baseline_lgbm.pkl", "--lr-ckpt", OUT / "m1_baseline_lr.pkl",
         "--log-output", OUT / "m1_val_eval.json"])
    run(["scripts/bootstrap_ci_mission1.py", "--log", OUT / "m1_val_eval.json",
         "--output", OUT / "m1_val_bootstrap_ci.json"])
"""),
md("## 4. 결과 · 재현 판정 (기준: M1-EXP-001 LR 95.11%, 혼동행렬 [[1878,81],[97,1584]])"),
code("""
REF_ACC, REF_CM, TOL = 0.951098901098901, [[1878, 81], [97, 1584]], 0.003
REF_SHA = "924B5620E1DE723D6B7821EB4849B4EA4B981C4BFD6621B469C5D0753F7934F8"
ev = json.loads((OUT / "m1_val_eval.json").read_text(encoding="utf-8"))["measured"]
ci = json.loads((OUT / "m1_val_bootstrap_ci.json").read_text(encoding="utf-8"))
for name in ("lr", "gbm"):
    r = ev[name]
    print(f"[{name.upper()}] Val acc={r['accuracy']:.4f} CI95={[round(x,4) for x in ci[name]['accuracy_ci_95']]} "
          f"recall F/M={r['recall_F']:.4f}/{r['recall_M']:.4f} cm={r['confusion_matrix']}")
lr = ev["lr"]
new_sha = sha256(OUT / "m1_baseline_lr.pkl")
fails = {s: sum(1 for _ in open(OUT / f"m1_build_failures_{s}.jsonl", encoding="utf-8")) for s in ("train", "validation")}
diff = lr["accuracy"] - REF_ACC
verdict = {
    "experiment": "M1-EXP-001-repro",
    "val_acc_lr": lr["accuracy"], "ref_acc": REF_ACC, "diff_pp": round(diff * 100, 3),
    "confusion_matrix_lr": lr["confusion_matrix"], "cm_identical": lr["confusion_matrix"] == REF_CM,
    "lr_ckpt_sha256": new_sha, "ref_sha256": REF_SHA, "sha_identical": new_sha == REF_SHA,
    "build_failures": fails,
    "verdict": "exact" if lr["confusion_matrix"] == REF_CM else ("within_tol" if abs(diff) <= TOL else "out_of_tol"),
}
(OUT / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(verdict, ensure_ascii=False, indent=2))
"""),
md("## 5. 패키지 버전 고정"),
code("""
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(OUT / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
print(freeze)
print("end:", datetime.datetime.now().isoformat(timespec="seconds"))
"""),
md("""
## 6. 추론

제출 루트 `inference.py` = `scripts/mission1_inference.py` 복사본. 출력 CSV 열: `audio file name`, `gender` (utf-8-sig).

```bash
python inference.py --audio_dir <wav_dir> --label_dir <json_dir> --ckpt_path m1_baseline_lr.pkl --output <out.csv>
```
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path("notebooks/mission1_train.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok", len(cells))
