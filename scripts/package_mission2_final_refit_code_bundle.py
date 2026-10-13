#!/usr/bin/env python3
"""Build a deterministic, source-only Colab bundle for Mission 2 final refit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "dcc_data"
TEST_PATH = REPOSITORY_ROOT / "tests" / "test_mission2_final_fulltrain_refit.py"
SCRIPT_PATH = Path(__file__).resolve()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_bundle(output_path: str | Path) -> tuple[Path, str, dict]:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite code bundle: {output}")
    if not PACKAGE_ROOT.is_dir() or not TEST_PATH.is_file():
        raise FileNotFoundError("Mission 2 source package or final-refit contract test is missing")

    files = {
        **{path.relative_to(REPOSITORY_ROOT).as_posix(): path.read_bytes()
           for path in sorted(PACKAGE_ROOT.glob("*.py"))},
        TEST_PATH.relative_to(REPOSITORY_ROOT).as_posix(): TEST_PATH.read_bytes(),
        SCRIPT_PATH.relative_to(REPOSITORY_ROOT).as_posix(): SCRIPT_PATH.read_bytes(),
    }
    manifest = {
        "schema_version": "mission2_final_refit_code_bundle_v1",
        "contains_raw_data": False,
        "files": {name: _sha256(content) for name, content in files.items()},
    }
    files["bundle_manifest.json"] = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8")
    with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    digest = _sha256(output.read_bytes())
    sidecar = output.with_suffix(output.suffix + ".sha256")
    with sidecar.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{digest}  {output.name}\n")
    return output, digest, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPOSITORY_ROOT / "tmp" / "mission2_final_refit_code_bundle.zip",
    )
    args = parser.parse_args()
    output, digest, manifest = build_bundle(args.output)
    print(f"Bundle: {output}")
    print(f"SHA-256: {digest}")
    print(f"Source files: {len(manifest['files'])}")
    print("Raw data included: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
