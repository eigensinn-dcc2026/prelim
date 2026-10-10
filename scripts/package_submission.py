"""Create a local reviewable DRAFT zip: three mission folders and one PPTX."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission"
REPORT = ROOT / "reports/submission"
OUTPUT = ROOT / "outputs/deliverables/DCC2026_eigensinn_submission_DRAFT.zip"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    checks = json.loads((ROOT / "outputs/submission_check/checks.json").read_text(encoding="utf-8"))
    ppt = json.loads((ROOT / "outputs/submission_deck/validation.json").read_text(encoding="utf-8"))
    assert ppt["packageIntegrity"]["status"] == "pass"
    assert ppt["nativeChartValidation"]["passed"] and ppt["firstPartyImport"]["passed"]
    files = []
    for mission in (1, 2, 3):
        directory = SOURCE / f"mission{mission}"
        for name in ("inference.py", f"mission{mission}_model.py", "requirements.txt", "README.md"):
            assert (directory / name).is_file(), name
        for path in directory.rglob("*"):
            relative = path.relative_to(SOURCE)
            if path.is_file() and not any(part.startswith(".") or part in ("__pycache__", "outputs") for part in relative.parts):
                files.append(path)
    files.append(SOURCE / "DCC2026_예선_결과정리.pptx")
    assert digest(files[-1]) == ppt["finalSha256"]
    inventory = [dict(path=p.relative_to(SOURCE).as_posix(), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(files)]
    assert len({p["path"].split("/")[0] for p in inventory}) == 4
    for mission in (1, 3):
        notebook = json.loads((SOURCE / f"mission{mission}/training.ipynb").read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4 and notebook["cells"]
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and cell["outputs"] == []
                ast.parse("".join(cell["source"]))
    total = sum(p["bytes"] for p in inventory)
    if shutil.disk_usage(ROOT).free < total + 1024**3:
        raise RuntimeError("Insufficient free disk space for the submission archive")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(SOURCE).as_posix())
            if path.stat().st_size > 10_000_000:
                print(f"Packed {path.relative_to(SOURCE)}", flush=True)
    with zipfile.ZipFile(temporary) as archive:
        assert len(archive.namelist()) == len(inventory)
        assert archive.testzip() is None
        for name in archive.namelist():
            assert not name.startswith("/") and ".." not in Path(name).parts
    temporary.replace(OUTPUT)
    REPORT.mkdir(parents=True, exist_ok=True)
    report = dict(status="draft_missing_m2_artifacts", ready_for_final_submission=False,
                  archive_name=OUTPUT.name, archive_bytes=OUTPUT.stat().st_size, archive_sha256=digest(OUTPUT),
                  unpacked_bytes=total, files=inventory, checks=checks,
                  presentation=dict(slides=19, sha256=ppt["finalSha256"], structural_checks="passed",
                                    native_chart_workbooks="passed", visual_review="performed",
                                    layout_warnings="four intentional index/title overlaps on contents slide 3",
                                    native_powerpoint_open_verified=False),
                  training_records=dict(m1_m3="original training code and real lab completion records preserved as cited markdown; no fabricated code outputs",
                                        m2="reference notebook only; selected run's executed notebook unavailable"),
                  missing=["M2-RERUN-001 selected 3-seed CNN checkpoint", "M2-RERUN-001 executed training notebook/log"],
                  fresh_environment_install_verified=False, archive_crc_verified=True,
                  source_data_included=False, test_weights_included=False, uploaded=False)
    (REPORT / "submission_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DRAFT archive verified: {OUTPUT}, {report['archive_bytes']} bytes, {len(inventory)} files", flush=True)


if __name__ == "__main__":
    main()
