"""Mission 3 module (owner: DCC M3). Not wired yet.

Must implement ``predict(audio_dir, label_dir, ckpt_path) -> pd.DataFrame``
with columns ``registry.MISSION_COLUMNS[3]``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def predict(audio_dir: Path, label_dir: Path, ckpt_path: Path) -> pd.DataFrame:
    raise NotImplementedError("Mission 3 inference module is not wired yet (owner: M3)")