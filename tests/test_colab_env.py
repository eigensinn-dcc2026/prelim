"""Synthetic folders only; no competition data is touched."""
from __future__ import annotations

import pytest

from dcc_data import colab_env


def make_root(path):
    (path / "Training").mkdir(parents=True)
    (path / "Validation").mkdir()
    return path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ("DCC_DATA_ROOT", "M1_DATA_ROOT", "DCC_RUN_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(colab_env, "_REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(colab_env, "DRIVE_MOUNT", tmp_path / "drive")
    monkeypatch.setattr(colab_env, "mount_drive_if_colab", lambda: (tmp_path / "drive" / "MyDrive").is_dir())


def test_explicit_beats_env(monkeypatch, tmp_path):
    explicit = make_root(tmp_path / "explicit")
    monkeypatch.setenv("DCC_DATA_ROOT", str(make_root(tmp_path / "env")))
    assert colab_env.find_data_root(explicit) == explicit


def test_env_and_legacy_env(monkeypatch, tmp_path):
    root = make_root(tmp_path / "한글 data")
    monkeypatch.setenv("M1_DATA_ROOT", str(root))
    assert colab_env.find_data_root() == root


def test_bad_env_fails_instead_of_falling_back(monkeypatch, tmp_path):
    make_root(tmp_path / "repo" / "data" / "raw_original")
    monkeypatch.setenv("DCC_DATA_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="DCC_DATA_ROOT"):
        colab_env.find_data_root()


def test_nested_wrapper_is_unwrapped(tmp_path):
    inner = make_root(tmp_path / "x" / colab_env.DATA_DIR_NAME)
    assert colab_env.find_data_root(tmp_path / "x") == inner


@pytest.mark.parametrize("where", [
    "MyDrive/대학부 데이터",
    ".shortcut-targets-by-id/abc123/대학부 데이터",
    "Shareddrives/team/대학부 데이터",
])
def test_drive_discovery(where, tmp_path):
    (tmp_path / "drive" / "MyDrive").mkdir(parents=True)
    root = make_root(tmp_path / "drive" / where)
    assert colab_env.find_data_root() == root


def test_not_found_message_guides_setup(tmp_path):
    (tmp_path / "drive" / "MyDrive").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="team_colab_setup"):
        colab_env.find_data_root()


def test_run_root(monkeypatch, tmp_path):
    assert colab_env.find_run_root() == tmp_path / "repo" / "outputs"
    (tmp_path / "drive" / "MyDrive").mkdir(parents=True)
    assert colab_env.find_run_root() == tmp_path / "drive" / "MyDrive" / "dcc_runs"
    monkeypatch.setenv("DCC_RUN_ROOT", str(tmp_path / "custom"))
    assert colab_env.find_run_root() == tmp_path / "custom"
