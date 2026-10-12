"""Explicit single-candidate entrypoint for the same runner used by Colab."""

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dcc_data import mission2_model_family_screen as screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--shared-archive-root", type=Path, help="Directory containing the 12 shared Training ZIPs")
    parser.add_argument("--raw-root", type=Path, help="Temporary local staging directory for shared Training")
    parser.add_argument("--transfer-identity", type=Path, help="Existing transfer-time raw SHA256 proof")
    parser.add_argument("--candidate", choices=screen.CANDIDATES, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], required=True)
    parser.add_argument("--run-id", help="Existing run to append another candidate; never overwrites one")
    parser.add_argument("--reference-artifact", type=Path, help="Strict screen-schema reference directory")
    parser.add_argument("--check-environment", action="store_true", help="Synthetic backend probe only; no competition cache load")
    args = parser.parse_args()
    shared = [args.shared_archive_root, args.raw_root, args.transfer_identity]
    if any(shared) and not all(shared):
        parser.error("Shared source requires --shared-archive-root, --raw-root and --transfer-identity together")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id must be a single safe directory name")
    readiness = screen.check_candidate_environment(args.candidate, args.device)
    print(readiness)
    if args.check_environment or readiness["status"] not in {"READY_CPU", "READY_GPU"}:
        return
    repo_root = screen.resolve_repo_root(args.repo_root)
    context = (screen.load_shared_screen(repo_root, screen.shared_training_archive_paths(args.shared_archive_root),
                                         args.raw_root, args.transfer_identity)
               if all(shared) else screen.load_screen(repo_root))
    run_root = repo_root / "data/processed/mission2_model_family_screen" / args.run_id if args.run_id else None
    if run_root is None or not run_root.exists():
        run_root = screen.create_run(repo_root, context, run_id=args.run_id)
    if args.reference_artifact:
        reference = screen.load_reference_artifact(args.reference_artifact, context, run_root)
    print("Run directory:", run_root)
    if args.reference_artifact and args.candidate == "mlp_reference":
        print(reference)
    else:
        print(screen.run_candidate(args.candidate, context, run_root, device=args.device))
    # Read-only preview; comparison.csv is finalized in the Notebook once the
    # selected candidates have finished, preserving existing artifacts.
    print(screen.build_comparison(run_root, context).to_string(index=False))


if __name__ == "__main__":
    main()
