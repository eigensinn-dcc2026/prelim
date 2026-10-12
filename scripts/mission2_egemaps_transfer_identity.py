"""Run manually on Windows after stopping extraction, before copying data to Drive."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo / "src"))
    from dcc_data.mission2_egemaps_colab import write_local_transfer_identity

    manifest_path = repo / "tmp/mission2_eda/utterance_manifest_v1.parquet"
    metadata_path = repo / "tmp/mission2_eda/utterance_manifest_meta_v1.json"
    chain_path = (repo / "data/processed/mission2_stage4_clean"
                  / "stage4_clean_v1_090422f783f0d911/chain.json")
    manifest = pd.read_parquet(manifest_path)
    training = manifest.loc[manifest["split"].astype(str) == "Training"].copy()
    result = write_local_transfer_identity(
        training, manifest_path, metadata_path, args.raw_root, chain_path, args.output,
    )
    print("LOCAL_TRANSFER_IDENTITY_WRITTEN", result)


if __name__ == "__main__":
    main()
