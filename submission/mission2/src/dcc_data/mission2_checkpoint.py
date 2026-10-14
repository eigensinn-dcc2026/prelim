"""CNN 체크포인트의 학습 설정 기록과 검사: 가중치 옆에 "무엇으로 학습했는지"를 적어 두고, 추론할 때 확인한다.

[규정 준수의 안전장치] SpeakerCNNLarge 가중치는 어떤 mel 설정으로 학습했든, 심지어 모양만 같은 다른 모델이었든 오류 없이 불러와진다.
가중치만으로는 알 수 없으니, 잘못된 체크포인트도 아무 오류 없이 돌면서 예측만 나빠질 수 있다.

그래서 scripts/train_mission2_cnn.py 는 ``build_checkpoint_metadata(...)`` 를 ``checkpoint["metadata"]`` 에 넣고,
추론(scripts/mission2_inference.py, scripts/mission2_final_inference.py)은 기록된 설정이 추론 설정과 다르거나 아무것도 기록돼 있지 않으면
실행을 거부한다. 특히 Training 만으로, 제공 라벨을 그대로 써서 학습했는지(label_source)를 확인한다.

이 모듈은 일부러 torch 를 불러오지 않는다(가벼운 검사만 한다).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .guards import TRAINING
from .mission2_features import HOP_LENGTH, N_FFT
from .mission2_spectrogram import ALLOWED_LABEL_SOURCES, N_MELS

METADATA_VERSION = 1

# 추론이 만드는 유일한 CNN (train_mission2_cnn.py 의 ``--model large``)
INFERENCE_MODEL_NAME = "large"

# 앙상블 체크포인트(scripts/pack_mission2_l2_ensemble.py)가 멤버 가중치들을 담는 키. 단일 모델 체크포인트의
# "best_model_state_dict" 대신 쓰이며, 추론은 이 키가 있으면 여러 모델의 확률을 평균한다.
ENSEMBLE_STATE_DICTS_KEY = "ensemble_model_state_dicts"

# 체크포인트에 반드시 기록돼 있어야 하는 학습 설정 항목
REQUIRED_METADATA_KEYS = (
    "metadata_version",
    "model_name",
    "model_class",
    "index_split",
    "label_source",
    "n_mels",
    "hop_length",
    "n_fft",
    "sample_rate",
    "split_seed",
    "test_size",
    "torch_seed",
)


class CheckpointConfigError(ValueError):
    """체크포인트에 기록된 학습 설정이 추론 설정과 다르거나 없을 때.

    발화 하나의 문제가 아니라 설정 오류라서, 추론은 이 오류를 다수 클래스로 덮지 않고 실행을 멈춘다.
    """


def checkpoint_sha256(path: str | Path) -> str:
    """파일의 sha256 지문."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_metadata(
    *,
    model_name: str,
    model_class: str,
    index_provenance: dict,
    split_seed: int,
    test_size: float,
    torch_seed: int | None,
) -> dict:
    """train_mission2_cnn.py 가 ``checkpoint["metadata"]`` 로 저장하는 학습 설정.

    ``index_provenance`` 는 학습에 쓴 스펙트로그램 목록에 대해 dcc_data.mission2_spectrogram.assert_training_spectrogram_index 가
    돌려준 값이다(Training 전용, 제공 라벨, mel 설정). 단순한 파이썬 값만 넣어서 ``torch.load(..., weights_only=True)`` 로도 열린다.
    """

    return {
        "metadata_version": METADATA_VERSION,
        "model_name": model_name,
        "model_class": model_class,
        "index_split": index_provenance["split"],
        "label_source": index_provenance["label_source"],
        "n_mels": index_provenance["n_mels"],
        "hop_length": index_provenance["hop_length"],
        "n_fft": index_provenance["n_fft"],
        "sample_rate": index_provenance["sample_rate"],
        "split_seed": split_seed,
        "test_size": test_size,
        "torch_seed": torch_seed,
    }


# 설정 기록이 생기기 전에 학습된 체크포인트를 파일 sha256 으로 등록해 두는 곳.
#
# 일부러 비워 둔다. 예전에 여기 등록돼 있던 파일(m2_cnn_large.pt, 공식 Validation 90.89%)은 ``text`` 를 읽고 Training 6통화의
# speaker 라벨을 고친 목록으로 학습한 것이었다. 운영 측 판단(2026-10-01)에 따라 학습은 제공 라벨을 그대로 써야 하므로 그 파일은
# 제출하면 안 되고 더 이상 받지 않는다. 제공 라벨로 학습했다고 *확실히 아는* 파일만 등록한다. 아니면 설정을 기록하는
# scripts/train_mission2_cnn.py 로 다시 학습한다. (최종 제출 CNN 은 설정이 기록된 체크포인트라 여기와 무관하다.)
LEGACY_L2_CHECKPOINTS: dict[str, dict] = {}


def _require_complete(metadata: dict, where: str) -> None:
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise CheckpointConfigError(f"checkpoint metadata ({where}) is incomplete, missing: {missing}")


def resolve_l2_checkpoint_metadata(checkpoint: dict, ckpt_path: str | Path) -> tuple[dict, str]:
    """불러온 CNN 체크포인트의 (학습 설정, 그 출처)를 돌려준다. 없으면 CheckpointConfigError.

    체크포인트 안에 기록된 설정을 쓴다. 기록이 없는 파일은 sha256 이 LEGACY_L2_CHECKPOINTS 에 있을 때만 받는다(지금은 비어 있음).
    """

    embedded = checkpoint.get("metadata")
    if embedded is not None:
        _require_complete(embedded, "embedded in the checkpoint")
        return dict(embedded), "embedded in the checkpoint"

    digest = checkpoint_sha256(ckpt_path)
    legacy = LEGACY_L2_CHECKPOINTS.get(digest)
    if legacy is None:
        raise CheckpointConfigError(
            f"{ckpt_path} records no training config and its sha256 ({digest}) is not a registered legacy "
            "checkpoint, so nothing says which model or mel settings it was trained with (a SpeakerCNNLarge-"
            "shaped state_dict loads into the wrong variant without any error). Retrain with the current "
            "scripts/train_mission2_cnn.py, which embeds the config. (dcc_data.mission2_checkpoint."
            "LEGACY_L2_CHECKPOINTS is for files known to have been trained on the provided labels as they are; "
            "the earlier m2_cnn_large.pt was trained on corrected labels and is deliberately not in it.)"
        )
    _require_complete(legacy["metadata"], "legacy registry")
    return dict(legacy["metadata"]), f"legacy registry: {legacy['description']}"


def build_ensemble_metadata(member_metadata: list[dict]) -> dict:
    """같은 구조의 CNN 체크포인트 여러 개의 설정을 하나의 앙상블 설정으로 합친다.

    멤버들은 ``torch_seed`` 외의 모든 항목이 같아야 한다. seed 앙상블은 seed 만 바꾸는 것이고, 나머지(mel 설정, 라벨 출처, 분할)가
    다르면 함께 평균 내는 것이 공정하지 않다. 합친 설정의 ``torch_seed`` 는 None 으로 두고, 멤버별 seed 와 개수를 따로 기록한다.
    """

    if len(member_metadata) < 2:
        raise ValueError(f"an ensemble needs at least 2 member checkpoints, got {len(member_metadata)}")
    for metadata in member_metadata:
        _require_complete(metadata, "ensemble member")

    merged = dict(member_metadata[0])
    for key in REQUIRED_METADATA_KEYS:
        if key == "torch_seed":
            continue
        values = {metadata[key] for metadata in member_metadata}
        if len(values) != 1:
            raise CheckpointConfigError(
                f"ensemble members disagree on {key!r}: {sorted(map(str, values))} -- they must be the same "
                "architecture/training setup, varied only by torch_seed, to be averaged together"
            )
    # 같은 seed 가 두 번이면 같은 실행(또는 같은 파일의 복사본)이라 앙상블이 아니다
    seeds = [metadata["torch_seed"] for metadata in member_metadata]
    if None in seeds or len(set(seeds)) != len(seeds):
        raise CheckpointConfigError(
            f"ensemble members must each record a different torch_seed, got {seeds} -- the same seed twice is the same "
            "run (or a copy of one file), and averaging it with itself is not an ensemble"
        )
    merged["torch_seed"] = None
    merged["ensemble_member_torch_seeds"] = seeds
    merged["ensemble_size"] = len(member_metadata)
    return merged


def l2_member_state_dicts(checkpoint: dict, metadata: dict) -> list[dict]:
    """추론이 돌릴 가중치(state_dict) 목록: 단일 모델이면 하나, 앙상블이면 멤버마다 하나.

    저장된 가중치 수가 설정에 적힌 수(``ensemble_size``, 없으면 1)와 같아야 한다. 멤버가 빠진 앙상블, 앙상블이라고 적힌 단일 모델
    (또는 반대)처럼 설정과 가중치가 안 맞는 파일은 안에 든 것만으로 평균 내지 않고 거부한다.
    """

    if ENSEMBLE_STATE_DICTS_KEY in checkpoint:
        state_dicts = list(checkpoint[ENSEMBLE_STATE_DICTS_KEY])
    elif "best_model_state_dict" in checkpoint:
        state_dicts = [checkpoint["best_model_state_dict"]]
    else:
        raise CheckpointConfigError(
            f"checkpoint has neither {ENSEMBLE_STATE_DICTS_KEY!r} nor 'best_model_state_dict', so there are no weights to run"
        )
    declared = metadata.get("ensemble_size", 1)
    seeds = metadata.get("ensemble_member_torch_seeds")
    if seeds is not None and len(set(seeds)) != len(seeds):
        raise CheckpointConfigError(
            f"checkpoint records the same torch_seed more than once ({seeds}): its members are copies of one run, not an ensemble"
        )
    if declared != len(state_dicts) or (seeds is not None and len(seeds) != declared):
        raise CheckpointConfigError(
            f"checkpoint metadata declares {declared} model(s) (seeds recorded: {seeds}) but the file holds "
            f"{len(state_dicts)} set(s) of weights -- refusing to run a checkpoint whose weights and recorded config disagree"
        )
    return state_dicts


def require_resumable_checkpoint(checkpoint: dict, expected_metadata: dict, ckpt_path: str | Path) -> None:
    """이어서 학습(--resume)할 체크포인트가 지금과 똑같은 설정에서 나왔음이 증명될 때만 허락한다.

    ``--resume`` 은 옛 가중치를 불러 학습을 이어간 뒤, *지금* 실행의 설정을 ``metadata`` 로 적어 저장한다. 옛 파일을 확인하지 않으면
    다른 라벨(또는 다른 mel 설정)로 학습한 모델이 오늘의 목록으로 학습한 것처럼 기록되어 나오고, 추론은 그 기록을 믿는다.
    그래서 이어갈 파일은 설정이 빠짐없이 기록돼 있어야 하고 모든 항목(``torch_seed`` 포함)이 지금 실행과 같아야 한다.
    설정이 기록되지 않은 파일과, 옵티마이저 상태가 없는 앙상블 파일도 거부한다.
    """

    if ENSEMBLE_STATE_DICTS_KEY in checkpoint:
        raise CheckpointConfigError(
            f"{ckpt_path} is a packed ensemble (scripts/pack_mission2_l2_ensemble.py): it keeps only each member's "
            "best weights, no optimiser state, so it cannot be resumed -- resume one of the member checkpoints instead"
        )
    recorded = checkpoint.get("metadata")
    if recorded is None or "last_model_state_dict" not in checkpoint:
        raise CheckpointConfigError(
            f"cannot resume {ckpt_path}: it records no training config (or predates resume support), so nothing "
            "shows which labels and settings its weights were trained with, and the checkpoint saved after "
            "resuming would claim this run's. Start a new run with the current scripts/train_mission2_cnn.py instead"
        )
    _require_complete(recorded, f"in the checkpoint being resumed, {ckpt_path}")
    differences = [
        f"{key}: the checkpoint recorded {recorded[key]!r}, this run uses {expected_metadata[key]!r}"
        for key in REQUIRED_METADATA_KEYS
        if recorded[key] != expected_metadata[key]
    ]
    if differences:
        raise CheckpointConfigError(
            f"cannot resume {ckpt_path}: it was trained with a different setup -- " + "; ".join(differences)
        )


def require_l2_inference_config(metadata: dict) -> None:
    """체크포인트가 추론 설정에 맞지 않으면 CheckpointConfigError 로 멈춘다.

    추론은 SpeakerCNNLarge 와 현재 기본값의 log-mel 스펙트로그램을 쓴다. 그리고 체크포인트는 Training 만으로,
    제공된 speaker 라벨을 그대로 써서 학습된 것이어야 한다(대회 규정).
    """

    expected = {
        "model_name": INFERENCE_MODEL_NAME,
        "index_split": TRAINING,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
    }
    mismatches = [
        f"{key}: checkpoint recorded {metadata[key]!r}, inference uses {value!r}"
        for key, value in expected.items()
        if metadata[key] != value
    ]
    if metadata["label_source"] not in ALLOWED_LABEL_SOURCES:
        mismatches.append(
            f"label_source: checkpoint recorded {metadata['label_source']!r}, but Mission 2 training must use "
            f"the provided speaker labels as they are (one of {', '.join(ALLOWED_LABEL_SOURCES)})"
        )
    if mismatches:
        raise CheckpointConfigError(
            "checkpoint was not trained for this inference setup -- " + "; ".join(mismatches)
        )
