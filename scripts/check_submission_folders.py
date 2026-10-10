"""Check independent official CLIs with allowed-input-only Validation samples.

M1/M3 outputs are compared with their previously verified full-validation CSVs.
M2 uses random test weights solely to validate packaging, never as a submission.
"""
from __future__ import annotations
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/submission_check"
DATA = ROOT.parent / "dcc2026-prelim/data/raw_merged/Validation"


def main():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    audio = OUT / "audio"
    audio.mkdir(exist_ok=True)
    labels = {mission: OUT / f"labels{mission}" for mission in (1, 2, 3)}
    for folder in labels.values():
        folder.mkdir(exist_ok=True)
    jsons = sorted((DATA / "2.라벨링데이터/VL_서울_구급").rglob("*.json"))[:2]
    wav_index = {p.stem: p for p in (DATA / "1.원천데이터/VS_서울_구급").rglob("*.wav")}
    for path in jsons:
        copy_wav = audio / wav_index[path.stem].name
        if not copy_wav.exists():
            shutil.copy2(wav_index[path.stem], copy_wav)
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        for mission, allowed in ((1, ("startAt", "endAt", "speaker")),
                                 (2, ("startAt", "endAt")), (3, ("text",))):
            sanitized = {"utterances": [{key: u[key] for key in allowed} for u in doc["utterances"]]}
            (labels[mission] / path.name).write_text(json.dumps(sanitized, ensure_ascii=False), encoding="utf-8")
    env = os.environ.copy()
    env.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               LOKY_MAX_CPU_COUNT="1", PYTHONPATH=str(ROOT / "outputs/eval_deps"))
    checks = {}
    for mission, ckpt, reference in ((1, "m1_xvec001_combined.ckpt", ROOT / "outputs/model_selection/m1_xvec001_gpu/mission1.csv"),
                                      (3, "m3_roberta003_seed42.pt", ROOT / "outputs/model_selection/mission3.csv")):
        folder = ROOT / "submission" / f"mission{mission}"
        result_path = OUT / f"mission{mission}.csv"
        started = time.perf_counter()
        process = subprocess.run([sys.executable, str(folder / "inference.py"), "--audio_dir", str(audio),
                                  "--label_dir", str(labels[mission]), "--ckpt_path", str(folder / ckpt),
                                  "--output", str(result_path)], cwd=OUT, env=env, capture_output=True, text=True, timeout=300)
        (OUT / f"mission{mission}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"Mission {mission} failed: {process.stderr[-3000:]}")
        actual = pd.read_csv(result_path)
        previous = pd.read_csv(reference)
        column = actual.columns[0]
        expected = previous[previous[column].isin(actual[column])].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)
        if mission == 3:
            for cell in actual["symptom"]:
                assert isinstance(ast.literal_eval(cell), list)
        checks[f"mission{mission}"] = dict(status="passed", rows=len(actual), elapsed_seconds=time.perf_counter()-started,
                                           allowed_inputs_only=True, matches_full_validation_csv=True, offline=True)
        print(f"Mission {mission} independent CLI: passed", flush=True)
    # Use the repository's known random-weight fixture metadata only for a functional M2 check.
    fixture_path = ROOT / "tests/test_inference_mission2.py"
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("m2_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    import torch
    from dcc_data.mission2_checkpoint import ENSEMBLE_STATE_DICTS_KEY, build_ensemble_metadata
    from dcc_data.mission2_cnn import SpeakerCNNLarge
    torch.manual_seed(42)
    test_ckpt = OUT / "M2_RANDOM_TEST_ONLY.pt"
    torch.save({ENSEMBLE_STATE_DICTS_KEY: [SpeakerCNNLarge().state_dict() for _ in range(3)],
                "metadata": build_ensemble_metadata([fixture._l2_metadata(torch_seed=s) for s in (1, 2, 3)])}, test_ckpt)
    folder = ROOT / "submission/mission2"
    process = subprocess.run([sys.executable, str(folder / "inference.py"), "--audio_dir", str(audio),
                              "--label_dir", str(labels[2]), "--ckpt_path", str(test_ckpt),
                              "--output", str(OUT / "mission2_random.csv")], cwd=OUT, env=env, capture_output=True, text=True, timeout=300)
    (OUT / "mission2_random.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(process.stderr[-3000:])
    actual = pd.read_csv(OUT / "mission2_random.csv")
    expected_rows = sum(len(json.loads(path.read_text(encoding="utf-8"))["utterances"]) for path in labels[2].glob("*.json"))
    assert list(actual.columns) == ["audio file name", "startAt", "endAt", "speaker"]
    assert len(actual) == expected_rows and actual.speaker.isin([0, 1]).all()
    assert "fallback=0" in process.stderr
    checks["mission2"] = dict(status="code_only_passed", rows=len(actual), random_test_weights=True,
                               final_checkpoint_verified=False, allowed_inputs_only=True, fallback_rows=0)
    (OUT / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
