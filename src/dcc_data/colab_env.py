"""Locate the official data root and the run-output root the same way on every machine.

Teammates only need the Drive shortcuts described in ``docs/team_colab_setup.md``;
no path has to be edited in a notebook or script.

Data root precedence: explicit argument -> ``DCC_DATA_ROOT`` (``M1_DATA_ROOT`` legacy)
-> repository ``data/raw_original`` -> Colab Drive auto-discovery.
Run root precedence: explicit argument -> ``DCC_RUN_ROOT`` -> Colab ``MyDrive/dcc_runs``
-> repository ``outputs``.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_NAME = "대학부 데이터"
RUN_DIR_NAME = "dcc_runs"
DRIVE_MOUNT = Path("/content/drive")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def is_data_root(path: Path) -> bool:
    """A data root holds the official ``Training`` and ``Validation`` folders directly."""
    return (path / "Training").is_dir() and (path / "Validation").is_dir()


def _unwrap(path: Path) -> Path | None:
    # Drive downloads often nest the folder once more (대학부 데이터/대학부 데이터).
    for candidate in (path, path / DATA_DIR_NAME):
        if candidate.is_dir() and is_data_root(candidate):
            return candidate
    return None


def mount_drive_if_colab() -> bool:
    """Mount Drive on Colab (no-op elsewhere). Returns True if MyDrive is available."""
    if (DRIVE_MOUNT / "MyDrive").is_dir():
        return True
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return False
    drive.mount(str(DRIVE_MOUNT))
    return (DRIVE_MOUNT / "MyDrive").is_dir()


def _drive_candidates() -> list[Path]:
    # Shallow lookups only: walking a whole Drive mount is very slow.
    base = DRIVE_MOUNT
    found = [base / "MyDrive" / DATA_DIR_NAME]
    found += sorted(base.glob(f".shortcut-targets-by-id/*/{DATA_DIR_NAME}"))
    found += sorted(base.glob(f"Shareddrives/*/{DATA_DIR_NAME}"))
    found += sorted(base.glob(f"Shared drives/*/{DATA_DIR_NAME}"))
    return found


def find_data_root(explicit: str | os.PathLike | None = None) -> Path:
    """Return the folder that contains ``Training/`` and ``Validation/``."""
    tried: list[str] = []

    def check(label: str, raw) -> Path | None:
        path = Path(raw).expanduser()
        tried.append(f"{label}: {path}")
        return _unwrap(path)

    if explicit:
        # An explicit choice must be honoured or fail loudly, never silently replaced.
        hit = check("argument", explicit)
        if hit:
            return hit
        raise FileNotFoundError(f"Training/Validation not found under explicit data root.\n  " + "\n  ".join(tried))
    for name in ("DCC_DATA_ROOT", "M1_DATA_ROOT"):
        if os.environ.get(name):
            hit = check(f"${name}", os.environ[name])
            if hit:
                return hit
            raise FileNotFoundError(f"${name} is set but has no Training/Validation.\n  " + "\n  ".join(tried))
    hit = check("repo default", _REPO_ROOT / "data" / "raw_original")
    if hit:
        return hit
    if mount_drive_if_colab():
        for candidate in _drive_candidates():
            hit = check("drive", candidate)
            if hit:
                return hit
    raise FileNotFoundError(
        f"Official data folder '{DATA_DIR_NAME}' (with Training/ and Validation/) was not found.\n  "
        + "\n  ".join(tried)
        + f"\nColab: add a Drive shortcut named '{DATA_DIR_NAME}' under My Drive "
        "(see docs/team_colab_setup.md). Local: set DCC_DATA_ROOT."
    )


def find_run_root(explicit: str | os.PathLike | None = None, create: bool = True) -> Path:
    """Return the folder that holds checkpoints and logs (Drive on Colab, ``outputs`` locally)."""
    if explicit:
        root = Path(explicit).expanduser()
    elif os.environ.get("DCC_RUN_ROOT"):
        root = Path(os.environ["DCC_RUN_ROOT"]).expanduser()
    elif mount_drive_if_colab():
        root = DRIVE_MOUNT / "MyDrive" / RUN_DIR_NAME
    else:
        root = _REPO_ROOT / "outputs"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root
