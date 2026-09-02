"""Phase 2A: 타임스탬프-오디오 길이 정합성 검증 스크립트.

------------------------------------------------------------------
무엇을 확인하는가? (여기서 꼭 이해하고 넘어갈 점)
------------------------------------------------------------------
JSON 라벨 파일에는 통화 전체 길이(`endAt`)와, 그 안에 있는 각 발화(utterance)의
시작/끝 시각(`startAt`/`endAt`, 단위: ms)이 적혀 있다. 이 값들이 실제 WAV
오디오 파일의 재생 길이와 앞뒤가 맞는지 확인하지 않으면, 나중에 "화자
구간을 오디오에서 잘라내는" Mission 2/3 전처리에서 범위를 벗어난 index로
잘라내려다 조용히 틀린 데이터를 학습하게 될 위험이 있다.

이 스크립트는 두 가지를 검사한다.
  1) wav_duration_ms - call_endAt        : 통화 전체 길이 == WAV 길이여야 함 (완전 일치 기대)
  2) wav_duration_ms - max(utterance_endAt): 마지막 발화가 통화가 끝나기 전에
     끝나야 하므로 항상 0 이상이어야 함 (통화 종료 후 무음 구간이 있는 게 정상)

------------------------------------------------------------------
검증 범위 (표본, 전수 아님)
------------------------------------------------------------------
`reports/eda/eda_summary.json`을 만들 때 쓴 것과 "동일한" 앞쪽 800개
(Training) / 250개(Validation) 행을 그대로 재사용한다. 전수(32,840개)를
다 열어서 확인하면 표본 300개 처리에 47초가 걸린 것을 기준으로 약 85분이
걸려서, 지금 단계에서는 비효율적이라 판단해 표본으로 검증한다.

주의: 이 800/250개는 무작위 표본이 아니라 매니페스트 파일 순서상 "앞에서부터
N개"다 (scripts/eda_dcc_dataset.py의 limit_rows()와 동일한 방식). 즉 진짜
random sampling이 아니라는 점을 감안해서 결과를 읽어야 한다 — 완전한 대표성
보장은 아니고, "이 범위에서는 문제가 없었다"는 확인 정도로 이해하자.

사용법 (팀 저장소 루트, dcc2026-prelim/ 에서 실행):
    python scripts/verify_timestamps.py
"""

from __future__ import annotations

import csv
import json
import statistics
import wave
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_PATH = Path("data/manifests/dcc_manifest_v1.csv")
DATA_ROOT = Path("data/raw")
OUTPUT_PATH = Path("reports/eda/phase2a_check.json")

# eda_summary.json과 동일한 검증 범위 (scripts/eda_dcc_dataset.py의 limit_rows와 동일 로직)
N_TRAIN = 800
N_VAL = 250

TOLERANCE_MS = 50  # 인코딩/반올림 오차를 감안한 허용치


def load_manifest_rows(split: str, limit: int) -> list[dict]:
    rows = []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                rows.append(row)
    if limit <= 0 or limit >= len(rows):
        return rows
    return rows[:limit]  # limit_rows()와 동일: 매니페스트 순서상 앞에서부터 N개 (무작위 아님)


def wav_duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
    return frames / rate * 1000.0


def stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "min": round(min(values), 1),
        "median": round(statistics.median(values), 1),
        "max": round(max(values), 1),
    }


def check_split(rows: list[dict], split_label: str) -> dict:
    diffs_call: list[float] = []
    diffs_utt: list[float] = []
    violations: list[dict] = []
    fail_reasons = {
        "json_read_error": 0,
        "missing_call_endAt": 0,
        "wav_read_error": 0,
    }

    for row in rows:
        json_path = DATA_ROOT / row["json_path"]
        wav_path = DATA_ROOT / row["wav_path"]

        try:
            with open(json_path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            fail_reasons["json_read_error"] += 1
            continue

        call_end = doc.get("endAt")
        if call_end is None:
            fail_reasons["missing_call_endAt"] += 1
            continue

        utterances = doc.get("utterances", [])
        max_utt_end = max((u.get("endAt", 0) for u in utterances), default=0)

        try:
            wav_dur = wav_duration_ms(wav_path)
        except (OSError, wave.Error):
            fail_reasons["wav_read_error"] += 1
            continue

        diff_call = wav_dur - call_end
        diff_utt = wav_dur - max_utt_end
        diffs_call.append(diff_call)
        diffs_utt.append(diff_utt)

        if abs(diff_call) > TOLERANCE_MS or diff_utt < -TOLERANCE_MS:
            violations.append(
                {
                    "stem": row["stem"],
                    "wav_duration_ms": round(wav_dur, 1),
                    "call_endAt": call_end,
                    "max_utterance_endAt": max_utt_end,
                    "diff_call_ms": round(diff_call, 1),
                    "diff_utt_ms": round(diff_utt, 1),
                }
            )

    read_failed = sum(fail_reasons.values())
    return {
        "split": split_label,
        "n_checked": len(rows) - read_failed,
        "read_failed": read_failed,
        "read_failed_reasons": fail_reasons,
        "violation_count": len(violations),
        "diff_call_vs_wav_ms": stats(diffs_call),
        "diff_utterance_end_vs_wav_ms": stats(diffs_utt),
        "violations": violations[:20],  # 너무 길어지지 않게 최대 20건만 예시 저장
    }


def main() -> None:
    train_rows = load_manifest_rows("Training", N_TRAIN)
    val_rows = load_manifest_rows("Validation", N_VAL)

    train_result = check_split(train_rows, "Training")
    val_result = check_split(val_rows, "Validation")

    result = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "startAt/endAt(ms) 라벨과 실제 WAV 재생 길이의 정합성 검증 (Phase 2A)",
            "scope_note": (
                "표본 검증 (전수 아님). eda_summary.json과 동일한 앞쪽 "
                f"Training {N_TRAIN}개 / Validation {N_VAL}개. "
                "무작위 표본이 아니라 매니페스트 파일 순서 기준 앞부분임에 유의."
            ),
            "tolerance_ms": TOLERANCE_MS,
        },
        "training": train_result,
        "validation": val_result,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Training: {train_result['n_checked']}건 확인, 위반 {train_result['violation_count']}건")
    print(f"Validation: {val_result['n_checked']}건 확인, 위반 {val_result['violation_count']}건")
    print(f"결과 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
