"""Recover rounded aggregate evidence from a saved Colab notebook, never weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def output_text(cell):
    pieces = []
    for output in cell.get("outputs", []):
        value = output.get("text", output.get("data", {}).get("text/plain", ""))
        pieces.append("".join(value) if isinstance(value, list) else value)
    return "".join(pieces)


def recover(notebook_path, frozen_path, out):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    outputs = [output_text(cell) for cell in notebook["cells"]]
    training = next(text for text in outputs if "Completed evidence: macro=" in text)
    final = re.search(r'\{"status": "fine_pilots_failed"[^\n]+\}', training)
    if not final:
        raise ValueError("A completed failed full-data fine pilot is required")
    summary = json.loads(final.group())
    summary.update(smoke=False, source=frozen["source"], seed=42,
                   official_validation_used=False, adoption_eligible=False,
                   provenance=dict(kind="saved_colab_stdout", decimal_places=6,
                                   notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                                   missing=["fine per-label metrics", "fine holdout predictions", "fine checkpoint files"]))
    summary["pilot"] = {key: frozen["pilot"][key] for key in ("roberta003", "existing_ensemble")}
    for method in ("control", "memory", "evidence"):
        matches = re.findall(rf"Completed {method}: macro=(\d+\.\d{{6}}), micro=(\d+\.\d{{6}})", training)
        if len(matches) != 1:
            raise ValueError(f"Ambiguous or missing completed score: {method}")
        macro, micro = map(float, matches[0])
        summary["pilot"][method + "_frozen"] = frozen["pilot"][method]
        summary["pilot"][method + "_fine"] = dict(macro_f1=macro, micro_f1=micro, rows=5792)
    export = next(text for text in outputs if "Exported experimental evidence:" in text)
    start = export.index('{\n  "experimental": true')
    manifest, _ = json.JSONDecoder().raw_decode(export[start:])
    if manifest["qualified_for_submission"] or len(manifest["checkpoints"]) != 3:
        raise ValueError("Unexpected experimental export manifest")
    manifest["provenance"] = dict(kind="saved_colab_stdout", files_recovered=False)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (("fine_stdout_summary.json", summary), ("fine_export_manifest_from_stdout.json", manifest)):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "fine_saved_stdout.txt").write_text(training, encoding="utf-8")
    (out / "fine_export_saved_stdout.txt").write_text(export, encoding="utf-8")
    print("Recovered aggregate scores and export log; checkpoint files remain unavailable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("reports/mission3/context/frozen_metrics.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/mission3/context"))
    args = parser.parse_args()
    recover(args.notebook, args.frozen, args.out)
