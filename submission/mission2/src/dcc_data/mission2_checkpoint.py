"""Mission 2 L2 체크포인트가 무엇으로 학습됐는지를 가중치 옆에 기록해 두고, 추론 때 확인한다.

SpeakerCNNLarge의 state_dict는 어떤 mel 설정으로 학습했든, 그리고 모듈은 같지만 forward 경로만
다른 변형이든(모델 교체 실험의 masked-pooling 모델은 SpeakerCNNLarge와 파라미터가 완전히 같다)
SpeakerCNNLarge 모양의 모든 모델에 로드된다. 가중치 안에는 어느 쪽인지 알려 주는 것이 없으므로,
잘못된 체크포인트도 오류 없이 실행되고 예측만 더 나빠진다.

그래서 scripts/train_mission2_cnn.py는 ``build_checkpoint_metadata(...)``를
``checkpoint["metadata"]`` 아래에 넣어 두고, scripts/mission2_inference.py는 기록된 설정이
추론에 쓸 설정과 다르거나 아무것도 기록하지 않은 체크포인트(아래에 sha256으로 등록된 기존
legacy 체크포인트가 아닌 경우)는 실행을 거부한다.

이 모듈은 일부러 torch를 import하지 않는다. L1만 쓰는 추론 경로가 torch 비용을 치르면 안 되기 때문이다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .guards import TRAINING
from .mission2_features import HOP_LENGTH, N_FFT
from .mission2_spectrogram import ALLOWED_LABEL_SOURCES, N_MELS

METADATA_VERSION = 1

# scripts/mission2_inference.py가 만드는 유일한 L2 모델 (train_mission2_cnn.py의 ``--model large``).
INFERENCE_MODEL_NAME = "large"

# 앙상블 체크포인트(scripts/pack_mission2_l2_ensemble.py)가 멤버 state_dict를 저장하는 키.
# 단일 모델 체크포인트의 "best_model_state_dict" 자리를 대신한다. scripts/mission2_inference.py는
# 이 키가 있는지 보고, 모델 하나만 돌릴지 여러 모델을 소프트 보팅할지 정한다.
ENSEMBLE_STATE_DICTS_KEY = "ensemble_model_state_dicts"

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
    """체크포인트에 기록된 학습 설정이 추론이 쓸 설정과 다르다 (또는 기록이 없다).

    발화 하나의 문제가 아니라 설정 오류다. scripts/mission2_inference.py는 이를 다수 클래스
    대체값으로 바꾸지 않는다.
    """


def checkpoint_sha256(path: str | Path) -> str:
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
    """train_mission2_cnn.py가 ``checkpoint["metadata"]``로 저장하는 dict.

    ``index_provenance``는 이 실행이 학습에 쓴 인덱스에 대해
    dcc_data.mission2_spectrogram.assert_training_spectrogram_index가 돌려준 값이다.
    파이썬 기본 값만 담으므로 체크포인트를 ``torch.load(..., weights_only=True)``로도 읽을 수 있다.
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


# metadata가 생기기 전에 학습한 체크포인트를 파일의 sha256으로 등록하는 곳. 당시 적용되던 설정을
# (모듈 기본값을 읽어 오지 않고) 그대로 풀어서 적어 둔다. 그래야 나중에 기본값을 바꿨을 때
# 추론이 옛 파일을 조용히 다른 설정으로 취급하지 않고 거부한다.
#
# 일부러 비어 있다. 예전에 여기 등록돼 있던 체크포인트 하나(m2_cnn_large.pt, M2-VAL-003의 공식
# Validation 90.89% 결과, sha256 04c0cc04e2c9f4033db4f28164b4c3c888b6642588728bc3ea1b3b73ce406189)는
# Training 통화 6개의 화자 라벨을 ``text``를 읽고 고친 인덱스로 학습한 것이었다. 주최 측이
# (2026-10-01) 학습은 제공된 라벨을 그대로 써야 한다고 판정했으므로 그 파일은 제출하면 안 되고,
# 더는 받아들이지 않는다. 제공된 라벨 그대로 학습했다는 것을 (추측이 아니라) 확실히 알 때만
# 여기에 파일을 등록한다. 그렇지 않으면 ``label_source``를 체크포인트에 직접 기록하는
# scripts/train_mission2_cnn.py로 다시 학습한다.
LEGACY_L2_CHECKPOINTS: dict[str, dict] = {}


def _require_complete(metadata: dict, where: str) -> None:
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise CheckpointConfigError(f"checkpoint metadata ({where}) is incomplete, missing: {missing}")


def resolve_l2_checkpoint_metadata(checkpoint: dict, ckpt_path: str | Path) -> tuple[dict, str]:
    """로드한 L2 체크포인트의 (metadata, 그 출처)를 돌려준다. 없으면 CheckpointConfigError.

    체크포인트에 내장된 metadata가 우선한다. 내장된 것이 없는 체크포인트는 sha256이
    LEGACY_L2_CHECKPOINTS에 등록돼 있을 때만 받아들인다.
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
    """구조가 같은 L2 체크포인트 여럿의 metadata를 앙상블 체크포인트 하나의 metadata로 합친다.

    멤버들은 ``torch_seed``를 뺀 ``REQUIRED_METADATA_KEYS``의 모든 필드가 같아야 한다.
    시드 앙상블에서 달라야 하는 것은 바로 이 ``torch_seed`` 하나뿐이고, 나머지(mel 설정, 라벨 출처,
    split)는 같아야 멤버들을 공정하게 평균 낼 수 있다. 합친 dict는 ``torch_seed``를
    ``None``(하나의 시드가 해당되지 않음)으로 바꾸고, 대신 모든 멤버의 시드와 개수를 기록한다.
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
    """scripts/mission2_inference.py가 실행하는 state_dict들. 단일 모델이면 하나, 묶은 앙상블이면 멤버마다 하나.

    저장된 가중치의 개수는 metadata가 말하는 개수(``ensemble_size``, 기록이 없으면 1)와 같아야 한다.
    그래서 metadata와 가중치가 어긋난 파일(멤버가 빠진 앙상블, 앙상블로 표시된 단일 모델, 또는 그 반대)은
    안에 있는 것을 대충 평균 내지 않고 거부한다.
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
    """같은 설정에서 나온 것이 증명되지 않는 체크포인트는 이어서 학습하지 않고 거부한다.

    ``--resume``은 예전 가중치를 불러와 계속 학습한 뒤, ``metadata``를 *현재* 실행 기준으로 다시 만들어
    체크포인트를 저장한다. 예전 파일을 확인하지 않으면, 다른 라벨(또는 다른 mel 설정)로 학습한 모델이
    마치 오늘의 인덱스로 학습한 것처럼 도장이 찍혀 나온다. scripts/mission2_inference.py는 바로 그
    도장을 믿는다. 그래서 이어서 학습할 파일은 완전한 설정을 기록하고 있어야 하고, 그 모든 필드가
    이번 실행이 찍으려는 값과 같아야 한다 (``torch_seed`` 포함: 다른 시드로 이어 학습하면
    서로 다른 두 셔플/초기화 이력이 한 라벨 아래 섞인다). 아무것도 기록하지 않은 체크포인트(설정을
    저장하기 전에 쓴 모든 파일. 가중치를 무엇으로 학습했든)는 일치함을 보일 수 없으므로 거부하며,
    이어서 학습할 optimizer 상태가 없는 묶음 앙상블 파일도 마찬가지다.
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
    """체크포인트가 scripts/mission2_inference.py가 실행하는 설정에 맞지 않으면 CheckpointConfigError를 낸다.

    추론은 SpeakerCNNLarge와 log-mel 스펙트로그램을 현재 모듈 기본값으로 만들고, 체크포인트는
    Training만으로, 제공된 그대로의 화자 라벨로 학습한 것이어야 한다.
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
