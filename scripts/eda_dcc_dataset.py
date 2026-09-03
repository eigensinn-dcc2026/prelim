"""DCC 2026 예선 데이터 EDA (표본 기반) 스크립트.

------------------------------------------------------------------
왜 "표본 기반"인가? (여기서 꼭 이해하고 넘어갈 점)
------------------------------------------------------------------
전체 Training(서울)에는 JSON 파일이 29,200개 있다. 파일 하나를 열어서
JSON을 파싱하는 데 걸리는 시간을 재보니 약 30ms였다. 전부 다 읽으면
29,200 * 30ms = 약 15분이 걸린다는 뜻이다.

EDA(탐색적 데이터 분석)의 목적은 "완벽한 숫자"가 아니라 "다음 결정을
내리기에 충분한 감"을 얻는 것이다. 예를 들어 "gender 비율이 대략
6:4인지 9:1인지"만 알아도 Accuracy를 그대로 믿을지, 클래스 가중치를
줄지 결정할 수 있다. 그래서 이 스크립트는 무작위로 일부 파일만 뽑아
(표본 추출, sampling) 통계를 낸다. 표본 수가 수백 개만 되어도 비율
추정치는 꽤 안정적이다 (통계학에서 "큰 수의 법�칙"이라고 부르는 것과
관련이 있다).

나중에 시간이 있을 때 --sample-train, --sample-val 을 늘려서(또는
0으로 주면 전수 조사) 다시 돌리면 된다.

------------------------------------------------------------------
사용법 (팀 저장소 루트, 즉 dcc2026-prelim/ 에서 실행)
------------------------------------------------------------------
Phase 1 (기본): Mission 1~3 골격 JSON + WAV header cache.
기존 표본 결과 reports/eda/eda_summary.json 은 덮어쓰지 않는다.

    python scripts/eda_dcc_dataset.py

개발용 smoke test (JSON/WAV 처리 행 수를 제한):

    python scripts/eda_dcc_dataset.py --json-max-train 10 --json-max-val 10

레거시 표본 집계 함수 run_eda() 는 보존되어 있으나, 기본 실행 경로에서는
호출하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# src/ 폴더를 파이썬이 찾을 수 있도록 경로에 추가한다.
# (노트북에서는 보통 !pip install -e . 로 설치하지만, 스크립트에서는
#  이렇게 경로만 잠깐 추가해도 충분하다.)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dcc_data.audio import read_wav_info  # noqa: E402
from dcc_data.manifest import ManifestRow, load_manifest  # noqa: E402
from dcc_data.mission3 import SYMPTOM_CLASSES  # noqa: E402

SYMPTOM_SET = set(SYMPTOM_CLASSES)
LEGACY_EDA_PATH = REPO_ROOT / "reports" / "eda" / "eda_summary.json"
DEFAULT_MISSION_OUTPUT = REPO_ROOT / "reports" / "eda" / "eda_mission_summary.json"
DEFAULT_WAV_HEADER_CACHE = REPO_ROOT / "reports" / "eda" / "wav_header_cache.csv"
WAV_HEADER_FIELDS = (
    "split",
    "stem",
    "wav_path",
    "file_size",
    "mtime_ns",
    "channels",
    "sample_rate",
    "sample_width",
    "frame_count",
    "duration_s",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data" / "manifests" / "dcc_manifest_v1.csv"),
        help="scripts/build_manifest.py 로 만든 manifest CSV 경로",
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "raw"),
        help="manifest 안의 상대경로가 기준으로 삼는 폴더",
    )
    parser.add_argument(
        "--sample-train",
        type=int,
        default=800,
        help="(레거시 run_eda) Training 표본 크기. Phase 1 기본 실행에서는 사용하지 않음",
    )
    parser.add_argument(
        "--sample-val",
        type=int,
        default=300,
        help="(레거시 run_eda) Validation 표본 크기. Phase 1 기본 실행에서는 사용하지 않음",
    )
    parser.add_argument("--seed", type=int, default=42, help="재현성을 위한 난수 seed")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MISSION_OUTPUT),
        help="Phase 1 Mission EDA JSON 경로 (eda_summary.json 은 거부한다)",
    )
    parser.add_argument(
        "--wav-header-cache",
        default=str(DEFAULT_WAV_HEADER_CACHE),
        help="WAV header 캐시 CSV 경로",
    )
    parser.add_argument(
        "--json-max-train",
        type=int,
        default=0,
        help="Training에서 처리할 최대 행 수 (0이면 전수). 개발 smoke test 전용",
    )
    parser.add_argument(
        "--json-max-val",
        type=int,
        default=0,
        help="Validation에서 처리할 최대 행 수 (0이면 전수). 개발 smoke test 전용",
    )
    parser.add_argument(
        "--rebuild-wav-cache",
        action="store_true",
        help="file_size/mtime_ns가 같아도 WAV header를 전부 다시 읽는다",
    )
    return parser.parse_args()


def sample_rows(rows: list[ManifestRow], n: int, rng: random.Random) -> list[ManifestRow]:
    """n이 0이거나 전체 개수보다 크면 전수 조사, 아니면 무작위 표본."""

    if n <= 0 or n >= len(rows):
        return rows
    return rng.sample(rows, n)


def read_record(data_root: Path, row: ManifestRow) -> dict | None:
    """JSON 하나를 읽어서 이번 EDA에 필요한 필드만 뽑아온다.

    실패(깨진 파일 등)는 조용히 넘기지 않고 None을 반환해서,
    호출하는 쪽에서 "실패 건수"를 따로 세도록 한다.
    """

    path = data_root / row.json_path
    try:
        with path.open("r", encoding="utf-8") as stream:
            doc = json.load(stream)
    except Exception:
        return None
    return doc


def summarize_durations(values_ms: list[float]) -> dict:
    """밀리초(ms) 리스트를 초(s) 단위 요약 통계로 바꾼다.

    왜 평균(mean)만 보지 않고 중앙값(median)·p90도 같이 보는가?
    음성 길이 같은 데이터는 아주 긴 이상치(outlier) 몇 개가 평균을
    확 끌어올릴 수 있다. 중앙값은 이상치에 덜 흔들리고, p90은
    "짧은 쪽부터 90%가 이 값 이하"라는 뜻이라 padding/crop 길이를
    정할 때 실질적인 기준이 된다.
    """

    if not values_ms:
        return {}
    seconds = sorted(v / 1000.0 for v in values_ms)
    n = len(seconds)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return seconds[idx]

    return {
        "n": n,
        "min_s": round(seconds[0], 2),
        "median_s": round(statistics.median(seconds), 2),
        "mean_s": round(statistics.fmean(seconds), 2),
        "p90_s": round(pct(0.90), 2),
        "max_s": round(seconds[-1], 2),
    }


def summarize_values(values: list[float], unit: str) -> dict:
    """개수·초·문자 수를 같은 형식으로 요약한다. unit 은 반드시 명시한다."""

    if not values:
        return {
            "n": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "unit": unit,
        }

    ordered = sorted(float(item) for item in values)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return ordered[idx]

    return {
        "n": n,
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": ordered[-1],
        "unit": unit,
    }


def empty_split_block() -> dict:
    """Phase 2/3 집계 전까지 비워 두는 split 블록."""

    return {
        "common": {},
        "mission1": {},
        "mission2": {},
        "mission3": {},
    }


def limit_rows(rows: list[ManifestRow], n: int) -> list[ManifestRow]:
    """n이 0 이하면 전수, 아니면 앞에서 n개. smoke test용으로 순서를 고정한다."""

    if n <= 0 or n >= len(rows):
        return rows
    return rows[:n]


def scope_meta(total: int, used: int, limit: int) -> dict:
    limited = limit > 0 and limit < total
    return {
        "mode": "limited" if limited else "full",
        "n_manifest": total,
        "n_used": used,
        "json_max": None if limit <= 0 else limit,
    }


def assert_not_legacy_output(path: Path) -> None:
    if path.resolve() == LEGACY_EDA_PATH.resolve():
        raise ValueError(
            "Refusing to overwrite the preserved legacy EDA file: "
            f"{LEGACY_EDA_PATH}. Pass a different --output path."
        )


def cache_row_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["split"], row["stem"])


def load_wav_header_cache(cache_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not cache_path.is_file():
        return {}
    with cache_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != WAV_HEADER_FIELDS:
            return {}
        loaded: dict[tuple[str, str], dict[str, str]] = {}
        for item in reader:
            loaded[cache_row_key(item)] = {field: item.get(field, "") for field in WAV_HEADER_FIELDS}
        return loaded


def write_wav_header_cache(
    cache_path: Path, rows: dict[tuple[str, str], dict[str, str]]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    with cache_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=WAV_HEADER_FIELDS)
        writer.writeheader()
        for item in ordered:
            writer.writerow({field: item.get(field, "") for field in WAV_HEADER_FIELDS})


def _blank_wav_audio_fields() -> dict[str, str]:
    return {
        "channels": "",
        "sample_rate": "",
        "sample_width": "",
        "frame_count": "",
        "duration_s": "",
    }


def stat_wav_file(wav_path: Path) -> tuple[str, str, str]:
    """Return (file_size, mtime_ns, error). error is empty on success."""

    try:
        stat = wav_path.stat()
    except OSError as exc:
        return "", "", f"{type(exc).__name__}: {exc}"
    return str(stat.st_size), str(stat.st_mtime_ns), ""


def inspect_wav_header(wav_path: Path) -> dict[str, str]:
    """WAV header only. Does not decode waveform samples."""

    file_size, mtime_ns, stat_error = stat_wav_file(wav_path)
    if stat_error:
        return {
            "file_size": file_size,
            "mtime_ns": mtime_ns,
            **_blank_wav_audio_fields(),
            "error": stat_error,
        }

    result = {
        "file_size": file_size,
        "mtime_ns": mtime_ns,
        **_blank_wav_audio_fields(),
        "error": "",
    }
    try:
        info = read_wav_info(wav_path)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["channels"] = str(info.channels)
    result["sample_rate"] = str(info.sample_rate)
    result["sample_width"] = str(info.sample_width)
    result["frame_count"] = str(info.frame_count)
    result["duration_s"] = f"{info.frame_count / info.sample_rate:.10g}"
    return result


def wav_cache_is_fresh(cached: dict[str, str], file_size: str, mtime_ns: str) -> bool:
    return (
        bool(cached)
        and file_size != ""
        and mtime_ns != ""
        and cached.get("file_size") == file_size
        and cached.get("mtime_ns") == mtime_ns
    )


def read_or_build_wav_header_cache(
    rows: list[ManifestRow],
    data_root: Path,
    cache_path: Path,
    *,
    rebuild: bool = False,
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, int]]:
    """Read WAV headers with a size/mtime cache. Never decodes waveforms.

    Existing cache rows for files not in ``rows`` are kept so a smoke test
    cannot wipe a previously built full cache.
    """

    merged = load_wav_header_cache(cache_path)
    stats = {
        "n_requested": len(rows),
        "n_reused": 0,
        "n_computed": 0,
        "n_error": 0,
    }

    for index, row in enumerate(rows, start=1):
        key = (row.split, row.stem)
        wav_path = data_root / row.wav_path
        file_size, mtime_ns, stat_error = stat_wav_file(wav_path)
        cached = merged.get(key, {})
        if (
            not rebuild
            and not stat_error
            and wav_cache_is_fresh(cached, file_size, mtime_ns)
        ):
            cached = dict(cached)
            cached["wav_path"] = row.wav_path
            merged[key] = cached
            stats["n_reused"] += 1
            if cached.get("error"):
                stats["n_error"] += 1
        else:
            current = inspect_wav_header(wav_path)
            record = {
                "split": row.split,
                "stem": row.stem,
                "wav_path": row.wav_path,
                **current,
            }
            merged[key] = record
            stats["n_computed"] += 1
            if record.get("error"):
                stats["n_error"] += 1
        if index % 2000 == 0 or index == len(rows):
            print(f"  wav header {index:,}/{len(rows):,}")

    write_wav_header_cache(cache_path, merged)
    return merged, stats


def accumulate_common(*_args: object, **_kwargs: object) -> dict:
    """Phase 2 skeleton: WAV/JSON duration stats. Not computed in Phase 1."""

    return {}


def accumulate_mission1(*_args: object, **_kwargs: object) -> dict:
    """Phase 2 skeleton: gender and reporter-speech stats. Not computed in Phase 1."""

    return {}


def accumulate_mission2(*_args: object, **_kwargs: object) -> dict:
    """Phase 2 skeleton: startAt/endAt/speaker and WAV-derived stats only."""

    return {}


def accumulate_mission3(*_args: object, **_kwargs: object) -> dict:
    """Phase 3 skeleton: transcript text and symptom targets only."""

    return {}


def run_eda(rows: list[ManifestRow], data_root: Path, split_label: str) -> dict:
    gender_counter: Counter[str] = Counter()
    call_durations_ms: list[float] = []
    utterance_durations_ms: list[float] = []
    utterances_per_call: list[int] = []
    speaker_counter: Counter[int] = Counter()
    symptom_counter: Counter[str] = Counter()
    unknown_symptom_counter: Counter[str] = Counter()
    symptoms_per_call: list[int] = []
    media_type_counter: Counter[str] = Counter()
    disaster_medium_counter: Counter[str] = Counter()
    urgency_counter: Counter[str] = Counter()
    sentiment_counter: Counter[str] = Counter()
    triage_counter: Counter[str] = Counter()
    failed = 0

    for row in rows:
        doc = read_record(data_root, row)
        if doc is None:
            failed += 1
            continue

        # --- Mission 1: gender ---
        gender = doc.get("gender")
        if isinstance(gender, str):
            gender_counter[gender] += 1

        # --- 통화(call) 전체 길이 (top-level startAt/endAt, 단위: ms) ---
        start_at = doc.get("startAt")
        end_at = doc.get("endAt")
        if isinstance(start_at, (int, float)) and isinstance(end_at, (int, float)):
            call_durations_ms.append(end_at - start_at)

        # --- Mission 2: 발화 조각(utterance) 단위 ---
        utterances = doc.get("utterances")
        if isinstance(utterances, list):
            utterances_per_call.append(len(utterances))
            for utt in utterances:
                if not isinstance(utt, dict):
                    continue
                u_start = utt.get("startAt")
                u_end = utt.get("endAt")
                if isinstance(u_start, (int, float)) and isinstance(u_end, (int, float)):
                    utterance_durations_ms.append(u_end - u_start)
                speaker = utt.get("speaker")
                if isinstance(speaker, int):
                    speaker_counter[speaker] += 1

        # --- Mission 3: symptom (9개 평가 대상 + 그 외) ---
        symptoms = doc.get("symptom")
        if isinstance(symptoms, list):
            known = [s for s in symptoms if s in SYMPTOM_SET]
            unknown = [s for s in symptoms if s not in SYMPTOM_SET]
            symptom_counter.update(known)
            unknown_symptom_counter.update(unknown)
            symptoms_per_call.append(len(known))

        # --- 참고용 메타데이터 (미션 입력으로는 못 쓰지만 상황 파악용) ---
        if isinstance(doc.get("mediaType"), str):
            media_type_counter[doc["mediaType"]] += 1
        if isinstance(doc.get("disasterMedium"), str):
            disaster_medium_counter[doc["disasterMedium"]] += 1
        if isinstance(doc.get("urgencyLevel"), str):
            urgency_counter[doc["urgencyLevel"]] += 1
        if isinstance(doc.get("sentiment"), str):
            sentiment_counter[doc["sentiment"]] += 1
        if isinstance(doc.get("triage"), str):
            triage_counter[doc["triage"]] += 1

    n_ok = len(rows) - failed
    gender_total = sum(gender_counter.values()) or 1

    return {
        "split": split_label,
        "sampled": len(rows),
        "read_failed": failed,
        "gender_distribution": {
            k: {"count": v, "pct": round(100 * v / gender_total, 1)}
            for k, v in gender_counter.most_common()
        },
        "call_duration_seconds": summarize_durations(call_durations_ms),
        "utterance_duration_seconds": summarize_durations(utterance_durations_ms),
        "utterances_per_call": summarize_durations(
            [x * 1000 for x in utterances_per_call]
        )
        if utterances_per_call
        else {},
        # speaker: Mission 2 정의상 0 = 119대원, 1 = 신고자
        "speaker_balance": {
            ("119대원(0)" if k == 0 else "신고자(1)" if k == 1 else str(k)): v
            for k, v in speaker_counter.most_common()
        },
        "symptom_frequency_top9": {
            label: symptom_counter.get(label, 0) for label in SYMPTOM_CLASSES
        },
        "symptoms_per_call": summarize_durations(
            [x * 1000 for x in symptoms_per_call]
        )
        if symptoms_per_call
        else {},
        "zero_symptom_call_pct": round(
            100
            * sum(1 for x in symptoms_per_call if x == 0)
            / (len(symptoms_per_call) or 1),
            1,
        ),
        "unknown_symptom_labels_seen": dict(unknown_symptom_counter.most_common(10)),
        "context_media_type": dict(media_type_counter.most_common()),
        "context_disaster_medium": dict(disaster_medium_counter.most_common(10)),
        "context_urgency_level": dict(urgency_counter.most_common()),
        "context_sentiment": dict(sentiment_counter.most_common(10)),
        "context_triage": dict(triage_counter.most_common()),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    cache_path = Path(args.wav_header_cache)
    assert_not_legacy_output(output_path)

    # validate_paths=False: 32,840개 파일 존재 여부를 다시 확인하지 않는다.
    # (data_audit 스크립트가 이미 한 번 확인했으므로 여기서는 속도를 우선한다.)
    all_rows = load_manifest(args.manifest, validate_paths=False)
    train_rows = [r for r in all_rows if r.split == "Training"]
    val_rows = [r for r in all_rows if r.split == "Validation"]
    train_used = limit_rows(train_rows, args.json_max_train)
    val_used = limit_rows(val_rows, args.json_max_val)

    print("Building WAV header cache (header only, no waveform decode)...")
    _headers, wav_stats = read_or_build_wav_header_cache(
        [*train_used, *val_used],
        data_root,
        cache_path,
        rebuild=args.rebuild_wav_cache,
    )

    result = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(Path(args.manifest)),
            "data_root": str(data_root),
            "legacy_eda_path": str(LEGACY_EDA_PATH),
            "legacy_eda_recomputed": False,
            "training_scope": scope_meta(
                len(train_rows), len(train_used), args.json_max_train
            ),
            "validation_scope": scope_meta(
                len(val_rows), len(val_used), args.json_max_val
            ),
            "wav_header_cache_path": str(cache_path),
            "wav_header": {**wav_stats, "rebuild": bool(args.rebuild_wav_cache)},
            "waveform_decoded": False,
            "phase": 1,
        },
        "training": empty_split_block(),
        "validation": empty_split_block(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[저장 완료] {output_path}")
    print(f"[WAV cache] {cache_path}")


if __name__ == "__main__":
    main()
