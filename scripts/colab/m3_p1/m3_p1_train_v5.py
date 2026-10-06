# M3 P1: KcBERT front-256, 9 sigmoid, unweighted BCE, Training(Seoul) 80:20 multilabel strat seed=42
# Based on notebooks/mission3_jieun_baseline.ipynb (main 4bfd2c4). Official Validation is NOT used for training/selection.
# v4 (2026-10-06): path handling only via colab_env-compatible lookup; hyperparameters identical to v3
# v5 (2026-10-06): P0 internal-Val split via scripts/mission3_baseline 최종 수정본.py; hyperparameters identical to v4/v3
import csv, importlib.util, json, os, subprocess, sys, tarfile, tempfile, time, random, hashlib, shutil, unicodedata, datetime, platform, traceback, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# DATA_ROOT, OUT, REPO, COMMIT are set by resolve_paths() (not hardcoded).
WORK = Path("/content/m3_p1")
MODEL_NAME = "beomi/kcbert-base"
TARGET_SYMPTOMS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]
SEED = 42; MAX_LENGTH = 256; EPOCHS = 3; LR = 2e-5; DROPOUT = 0.1
VAL_RATIO = 0.2; TH = 0.5; TIME_CAP_S = 6 * 3600; P0_MACRO_F1 = 0.5676
T0 = time.time()

# Embedded copy of the precedence in src/dcc_data/colab_env.py (PR #47, m2/retrain-notebooks).
# ImportError uses this fallback. The v5 launcher zip (6cc1181) ships the module; older zips do not.
DATA_DIR_NAME = "대학부 데이터"
RUN_DIR_NAME = "dcc_runs"
DRIVE_MOUNT = Path("/content/drive")

def _is_data_root(path):
    return (path / "Training").is_dir() and (path / "Validation").is_dir()

def _unwrap_labeled(path):
    # One level of nesting only: 대학부 데이터/대학부 데이터. A data root must contain
    # Training/ and Validation/ directly. The outer folder wins when it already is one.
    path = Path(path)
    if path.is_dir() and _is_data_root(path):
        return "", path
    nested = path / DATA_DIR_NAME
    if nested.is_dir() and _is_data_root(nested):
        return " unwrapped", nested
    return None

def _script_repo_root():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "dcc_data").is_dir():
            return parent
    return None

def _mount_drive_if_colab(drive_mount):
    if (drive_mount / "MyDrive").is_dir():
        return True
    # Only the real Colab mount should try google.colab. A test drive_mount must not.
    if Path(drive_mount) != DRIVE_MOUNT:
        return False
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return False
    drive.mount(str(drive_mount))
    return (drive_mount / "MyDrive").is_dir()

def _drive_candidates(drive_mount):
    # Shallow lookups only: walking a whole Drive mount is very slow. Same order as colab_env.
    base = Path(drive_mount)
    found = [base / "MyDrive" / DATA_DIR_NAME]
    found += sorted(base.glob(f".shortcut-targets-by-id/*/{DATA_DIR_NAME}"))
    found += sorted(base.glob(f"Shareddrives/*/{DATA_DIR_NAME}"))
    found += sorted(base.glob(f"Shared drives/*/{DATA_DIR_NAME}"))
    return found

def _drive_source_label(drive_mount, candidate):
    try:
        rel = candidate.parent.relative_to(drive_mount)
    except ValueError:
        return "drive"
    head = rel.parts[0] if rel.parts else ""
    if head == "MyDrive":
        return "drive MyDrive"
    if head == ".shortcut-targets-by-id":
        return "drive shortcut"
    if head == "Shareddrives":
        return "drive Shareddrives"
    if head == "Shared drives":
        return "drive Shared drives"
    return "drive"

def locate_data_root(explicit=None, repo_root=None, drive_mount=None):
    """Return (source, path). Same precedence as dcc_data.colab_env.find_data_root.

    explicit arg -> $DCC_DATA_ROOT ($M1_DATA_ROOT legacy) -> repo data/raw_original ->
    MyDrive -> shortcut-targets-by-id -> Shareddrives -> 'Shared drives'.
    Explicit or env set but wrong raises; it does not fall through.
    """
    drive_mount = Path(drive_mount) if drive_mount is not None else DRIVE_MOUNT
    if repo_root is None:
        repo_root = _script_repo_root()
    tried = []

    def check(label, raw):
        path = Path(raw).expanduser()
        tried.append(f"{label}: {path}")
        hit = _unwrap_labeled(path)
        if not hit:
            return None
        suffix, resolved = hit
        return label + suffix, resolved

    if explicit:
        hit = check("argument", explicit)
        if hit:
            return hit
        raise FileNotFoundError("Training/Validation not found under explicit data root.\n  " + "\n  ".join(tried))
    for name in ("DCC_DATA_ROOT", "M1_DATA_ROOT"):
        if os.environ.get(name):
            hit = check(f"${name}", os.environ[name])
            if hit:
                return hit
            raise FileNotFoundError(f"${name} is set but has no Training/Validation.\n  " + "\n  ".join(tried))
    if repo_root is not None:
        hit = check("repo data/raw_original", Path(repo_root) / "data" / "raw_original")
        if hit:
            return hit
    if _mount_drive_if_colab(drive_mount):
        for candidate in _drive_candidates(drive_mount):
            hit = check(_drive_source_label(drive_mount, candidate), candidate)
            if hit:
                return hit
    raise FileNotFoundError(
        f"Official data folder '{DATA_DIR_NAME}' (with Training/ and Validation/) was not found.\n  "
        + "\n  ".join(tried)
        + f"\nColab: add a Drive shortcut named '{DATA_DIR_NAME}' under My Drive "
        "(see docs/team_colab_setup.md). Local: set DCC_DATA_ROOT."
    )

def locate_run_root(explicit=None, create=True, repo_root=None, drive_mount=None):
    """Same precedence as dcc_data.colab_env.find_run_root."""
    drive_mount = Path(drive_mount) if drive_mount is not None else DRIVE_MOUNT
    if repo_root is None:
        repo_root = _script_repo_root()
    if explicit:
        root = Path(explicit).expanduser()
    elif os.environ.get("DCC_RUN_ROOT"):
        root = Path(os.environ["DCC_RUN_ROOT"]).expanduser()
    elif _mount_drive_if_colab(drive_mount):
        root = drive_mount / "MyDrive" / RUN_DIR_NAME
    elif repo_root is not None:
        root = Path(repo_root) / "outputs"
    else:
        root = Path("outputs")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root

def _label_imported_data_root(path, repo_root, drive_mount=DRIVE_MOUNT):
    path = Path(path)
    for name in ("DCC_DATA_ROOT", "M1_DATA_ROOT"):
        raw = os.environ.get(name)
        if not raw:
            continue
        hit = _unwrap_labeled(Path(raw).expanduser())
        if hit and hit[1] == path:
            return f"${name}{hit[0]}"
        return f"${name}"
    if repo_root is not None:
        hit = _unwrap_labeled(Path(repo_root) / "data" / "raw_original")
        if hit and hit[1] == path:
            return "repo data/raw_original" + hit[0]
    if (drive_mount / "MyDrive").is_dir():
        for candidate in _drive_candidates(drive_mount):
            hit = _unwrap_labeled(candidate)
            if hit and hit[1] == path:
                return _drive_source_label(drive_mount, candidate) + hit[0]
    return "colab_env"

def _prepend_sys_path(path):
    if path is None:
        return
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

def _require_writable(out):
    probe = Path(out) / ".write_probe"
    try:
        Path(out).mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"OUT is not writable: {out} ({exc}). "
            "Set DCC_RUN_ROOT to a writable folder in your own Drive."
        ) from exc

def resolve_paths():
    """Set DATA_ROOT, OUT, REPO, COMMIT. Hyperparameters are not touched."""
    global DATA_ROOT, OUT, REPO, COMMIT
    REPO = Path(os.environ["M3P1_REPO"]) if os.environ.get("M3P1_REPO") else Path("/content/dcc2026-prelim-main")
    COMMIT = os.environ["M3P1_COMMIT"] if os.environ.get("M3P1_COMMIT") else "6cc1181"
    script_repo = _script_repo_root()
    # Repo checkout src first, then the frozen zip's src. colab_env is absent from 4bfd2c4.
    _prepend_sys_path(REPO / "src")
    _prepend_sys_path(script_repo)
    _prepend_sys_path(script_repo / "src" if script_repo is not None else None)
    try:
        from dcc_data.colab_env import find_data_root, find_run_root
    except ImportError:
        find_data_root = find_run_root = None
    if os.environ.get("M3P1_OUT"):
        OUT = Path(os.environ["M3P1_OUT"])
    elif find_run_root is not None:
        OUT = Path(find_run_root()) / "m3_p1"
    else:
        run_repo = script_repo if script_repo is not None else (REPO if REPO.is_dir() else None)
        OUT = locate_run_root(repo_root=run_repo) / "m3_p1"
    _require_writable(OUT)
    WORK.mkdir(parents=True, exist_ok=True)
    if find_data_root is not None:
        DATA_ROOT = Path(find_data_root())
        try:
            import dcc_data.colab_env as colab_env
            lib_repo = Path(colab_env.__file__).resolve().parents[2]
        except Exception:
            lib_repo = script_repo
        source = _label_imported_data_root(DATA_ROOT, lib_repo)
    else:
        data_repo = script_repo if script_repo is not None else (REPO if REPO.is_dir() else None)
        source, DATA_ROOT = locate_data_root(repo_root=data_repo)
        DATA_ROOT = Path(DATA_ROOT)
    status(f"DATA_ROOT source={source} path={DATA_ROOT}")
    status(f"OUT path={OUT}")


P0_EXPECTED_FILES = 29200
P0_EXPECTED_VAL = 5792
P0_EXPECTED_TRAIN = 23408  # 29200 - 5792
P0_MIN1_MACRO_F1 = 0.5680  # P0 on this split with the at-least-one-symptom rule
P0_SCRIPT_NAME = "mission3_baseline 최종 수정본.py"
LABEL_DIR_NAME = "TL_서울_구급"

def load_p0_module(repo):
    """Load P0's split helpers from the extracted repo. The module's main() is under __name__ guard."""
    path = Path(repo) / "scripts" / P0_SCRIPT_NAME
    if not path.is_file():
        raise SystemExit(f"P0 module not found: {path}")
    spec = importlib.util.spec_from_file_location("m3_p0_baseline", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load P0 module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def find_tl_seoul_dir(data_root):
    """The single Training label folder named TL_서울_구급. No other region is included."""
    name = "TL_서울_구급"
    hits = []
    root = Path(data_root)
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for p in sorted(root.glob(f"{depth}/{name}")):
            if p.is_dir() and nfc(p.name) == name:
                hits.append(p.resolve())
    uniq = []
    seen = set()
    for p in hits:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    training = [p for p in uniq if ("training" in nfc(str(p)).lower() or "학습" in nfc(str(p)))]
    chosen = training or uniq
    if len(chosen) != 1:
        raise SystemExit(f"expected exactly one {name} directory, found {len(chosen)}: {[str(p) for p in chosen]}")
    return chosen[0]

def nfc(s): return unicodedata.normalize("NFC", s)

def save_val_pred(path, names, probs, labels):
    """Internal-Val probabilities and 0/1 labels. Rows sorted by NFC file name to match val_files.txt."""
    names = [nfc(str(name)) for name in names]
    order = sorted(range(len(names)), key=lambda i: names[i])
    gt_cols = [f"{s}_true" for s in TARGET_SYMPTOMS]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label_file_name", *TARGET_SYMPTOMS, *gt_cols])
        for i in order:
            writer.writerow([names[i], *[f"{float(p):.6f}" for p in probs[i]], *[int(v) for v in labels[i]]])

def save_epoch_val_pred(path, names, probs, labels):
    """Write one val_pred_ep{N}.csv. A failed or short write exits non-zero."""
    try:
        save_val_pred(path, names, probs, labels)
        with open(path, encoding="utf-8") as handle:
            pred_rows = sum(1 for _ in handle) - 1
    except Exception as exc:
        print(f"{Path(path).name} save failed: {exc}", flush=True)
        raise SystemExit(1) from exc
    if pred_rows != len(names) or not Path(path).is_file():
        print(f"{Path(path).name} rows={pred_rows} expected={len(names)}", flush=True)
        raise SystemExit(1)

def write_val_files(path, names):
    """Write sorted NFC names. A failed or incomplete write exits non-zero."""
    names = sorted(nfc(str(name)) for name in names)
    try:
        Path(path).write_text("\n".join(names) + "\n", encoding="utf-8")
        written = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"val_files.txt save failed: {exc}", flush=True)
        raise SystemExit(1) from exc
    if written != names:
        print(f"val_files.txt verify failed lines={len(written)} expected={len(names)}", flush=True)
        raise SystemExit(1)

def assert_split_sizes(n_train, n_val, n_index):
    if n_val != P0_EXPECTED_VAL or n_index != P0_EXPECTED_VAL or n_train != P0_EXPECTED_TRAIN:
        print(
            f"split size train={n_train} expected_train={P0_EXPECTED_TRAIN} "
            f"val={n_val} index={n_index} expected_val={P0_EXPECTED_VAL}",
            flush=True,
        )
        raise SystemExit(1)

def count_json(path):
    return sum(1 for item in Path(path).rglob("*.json") if item.is_file())

def local_label_dir():
    return Path(os.environ.get("M3P1_LOCAL_LABELS", f"/content/m3_labels/{LABEL_DIR_NAME}"))

def find_label_archives(data_root):
    """Shallow search for one archive whose file name contains TL_서울_구급."""
    root = Path(data_root)
    hits = []
    seen = set()
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for path in root.glob(depth):
            if not path.is_file():
                continue
            name = nfc(path.name)
            lower = name.lower()
            if LABEL_DIR_NAME not in name:
                continue
            if not lower.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                hits.append(resolved)
    return hits

def extract_archive(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    lower = archive.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(dest)
        return
    with tarfile.open(archive) as handle:
        handle.extractall(dest)

def extracted_label_dir(dest):
    hits = [path for path in dest.rglob(LABEL_DIR_NAME) if path.is_dir() and nfc(path.name) == LABEL_DIR_NAME]
    if len(hits) == 1:
        return hits[0]
    if not hits and any(dest.glob("*.json")):
        return dest
    raise SystemExit(f"archive did not contain a single {LABEL_DIR_NAME} directory: {[str(path) for path in hits]}")

def copy_tree(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["rsync", "-a", f"{str(src).rstrip('/')}/", f"{str(dst).rstrip('/')}/"])
    else:
        subprocess.check_call(["cp", "-a", str(src), str(dst)])

def stage_tl_seoul(data_root):
    """Copy TL_서울_구급 onto local disk and return that directory.

    An existing local copy with the expected JSON count is reused. Otherwise an archive
    under DATA_ROOT is preferred, and the folder is copied when no archive is present.
    The copy stays on local disk and is not written back to Drive.
    """
    local = local_label_dir()
    if local.is_dir() and count_json(local) == P0_EXPECTED_FILES:
        status(f"LOCAL_LABELS skip re-copy path={local} json={P0_EXPECTED_FILES}")
        return local
    archives = find_label_archives(data_root)
    if len(archives) > 1:
        raise SystemExit("multiple TL_서울_구급 archives, refusing to guess: " + ", ".join(str(path) for path in archives))
    try:
        if len(archives) == 1:
            archive = archives[0]
            status(f"LOCAL_LABELS extract archive={archive} -> {local}")
            parent = "/content" if Path("/content").is_dir() else None
            tmp = Path(tempfile.mkdtemp(prefix="m3p1_labels_", dir=parent))
            try:
                extract_archive(archive, tmp)
                copy_tree(extracted_label_dir(tmp), local)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            source = find_tl_seoul_dir(data_root)
            status(f"LOCAL_LABELS copy folder={source} -> {local}")
            copy_tree(source, local)
    except subprocess.CalledProcessError as exc:
        print(f"TL_서울_구급 local copy failed: {exc}", flush=True)
        raise SystemExit(1) from exc
    n_local = count_json(local)
    print(f"TL_서울_구급 local json count={n_local} expected={P0_EXPECTED_FILES} path={local}", flush=True)
    status(f"LOCAL_LABELS path={local} json={n_local}")
    if n_local != P0_EXPECTED_FILES:
        raise SystemExit(1)
    return local

def multilabel_macro_f1(pred, labels):
    pred = np.asarray(pred); labels = np.asarray(labels)
    tp = ((pred == 1) & (labels == 1)).sum(0)
    fp = ((pred == 1) & (labels == 0)).sum(0)
    fn = ((pred == 0) & (labels == 1)).sum(0)
    pr = tp / (tp + fp + 1e-8); rc = tp / (tp + fn + 1e-8)
    f1 = 2 * pr * rc / (pr + rc + 1e-8)
    return float(f1.mean()), f1

def min1_predictions(probs, threshold=0.5):
    """Force the argmax label on when no probability is >= threshold."""
    probs = np.asarray(probs)
    pred = (probs >= threshold).astype(np.int32)
    none = pred.sum(axis=1) == 0
    if np.any(none):
        rows = np.flatnonzero(none)
        pred[rows, probs[none].argmax(axis=1)] = 1
    return pred

def write_metrics(hist, best_ep, best_f1, adoption, min1):
    payload = {
        "threshold": TH,
        "threshold_rule": "probability > 0.5",
        "min1_rule": "if no label >= 0.5, set the argmax label to 1",
        "internal_val_macro_f1_per_epoch": [
            {"epoch": int(row["epoch"]), "internal_macro_f1": float(row["internal_macro_f1"])} for row in hist
        ],
        "best_epoch_ref": None if best_ep is None else {"epoch": int(best_ep), "internal_macro_f1": float(best_f1)},
        "p0_min1_macro_f1": P0_MIN1_MACRO_F1,
        "n_internal_val": P0_EXPECTED_VAL,
        "n_train": P0_EXPECTED_TRAIN,
    }
    if adoption is not None:
        payload["adoption_macro_f1_ep3"] = float(adoption)
    if min1 is not None:
        payload["ep3_macro_f1_min1"] = float(min1)
    path = OUT / "metrics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    status(f"METRICS epochs={len(hist)} file={path.name}")
def kst(): return (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S KST")
def status(msg):
    line = f"{kst()} {msg}"
    print(line, flush=True)
    with open(OUT / "STATUS.txt", "a", encoding="utf-8") as f: f.write(line + "\n")
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def find_label_dirs(split_kw, prefix):
    found = set()
    for depth in ("*/", "*/*/", "*/*/*/", "*/*/*/*/"):
        for p in DATA_ROOT.glob(depth + "*"):
            if not p.is_dir(): continue
            if not nfc(p.name).upper().startswith(prefix): continue
            if split_kw not in nfc(p.as_posix()).lower(): continue
            found.add(p)
    return sorted(found)

def main():
    resolve_paths()
    status(f"START m3_p1 model={MODEL_NAME} max_len={MAX_LENGTH} commit={COMMIT}")
    # Count and the P0 split run before torch import so a bad file count exits before training.
    p0 = load_p0_module(REPO)
    if list(p0.TARGET_SYMPTOMS) != list(TARGET_SYMPTOMS):
        raise SystemExit(f"P0 TARGET_SYMPTOMS mismatch: {list(p0.TARGET_SYMPTOMS)}")
    label_dir = stage_tl_seoul(DATA_ROOT)
    tl = [label_dir]
    files = sorted(label_dir.rglob("*.json"))
    status(f"TL_서울_구급 dir={label_dir} json files={len(files)}")
    if len(files) != P0_EXPECTED_FILES:
        print(f"TL_서울_구급 json count={len(files)} expected={P0_EXPECTED_FILES}", flush=True)
        status(f"FILE_COUNT {len(files)} != {P0_EXPECTED_FILES}")
        raise SystemExit(1)
    df = p0.load_training_dataframe(label_dir)
    n_loaded = int(len(df))
    n_json = int(df.attrs.get("json_file_count", -1))
    n_fail = len(df.attrs.get("load_failures", []))
    if n_loaded != P0_EXPECTED_FILES or n_json != P0_EXPECTED_FILES or n_fail:
        print(f"loaded rows={n_loaded} json_file_count={n_json} failures={n_fail} expected={P0_EXPECTED_FILES}", flush=True)
        status(f"LOADED n={n_loaded} json_file_count={n_json} failures={n_fail}")
        raise SystemExit(1)
    # df["text"] is P0 transcript_from_json (whitespace cleanup). v3/v4 joined raw strips and used "증상 없음" when empty.
    texts = [str(t) for t in df["text"].tolist()]
    file_names = df["label_file_name"].tolist()
    mlb = p0.MultiLabelBinarizer(classes=list(p0.TARGET_SYMPTOMS))
    mlb.fit([list(p0.TARGET_SYMPTOMS)])
    random.seed(SEED); np.random.seed(SEED)
    train_df, valid_df = p0._make_split(df, 0.2, 42, mlb)
    tr_idx = np.asarray(train_df.index, dtype=np.int64)
    va_idx = np.asarray(valid_df.index, dtype=np.int64)
    Y = np.asarray(mlb.transform(df["symptoms"]), dtype=np.int8)
    val_names = sorted(nfc(str(name)) for name in valid_df["label_file_name"].tolist())
    try:
        assert_split_sizes(len(tr_idx), len(val_names), len(va_idx))
    except SystemExit:
        status(f"SPLIT_SIZE train={len(tr_idx)} val={len(val_names)}")
        raise
    write_val_files(OUT / "val_files.txt", val_names)
    (OUT / "val_file_count.txt").write_text(f"{len(val_names)}\n", encoding="utf-8")
    status(f"LOADED n={n_loaded} label_counts={dict(zip(TARGET_SYMPTOMS, Y.sum(0).tolist()))} load_s={time.time()-T0:.0f}")
    status(f"SPLIT train={len(tr_idx)} internal_val={len(va_idx)} val_files={len(val_names)} (P0 _make_split seed=42 test_size=0.2)")

    import torch, torch.nn as nn
    import transformers, pandas as pd
    from transformers import AutoModel, AutoTokenizer
    import iterstrat

    def load(p):
        with open(p, encoding="utf-8") as f: j = json.load(f)
        text = " ".join(u.get("text", "").strip() for u in j.get("utterances", []) if u.get("text", "").strip())
        sy = set(nfc(s) for s in j.get("symptom", []) if isinstance(s, str))
        return (text or "증상 없음"), [int(s in sy) for s in TARGET_SYMPTOMS], nfc(p.name)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tok(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]; Yt = torch.tensor(Y, dtype=torch.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    status(f"TOKENIZED device={dev} gpu={gpu}")

    class Clf(nn.Module):
        def __init__(s):
            super().__init__(); s.bert = AutoModel.from_pretrained(MODEL_NAME)
            s.dropout = nn.Dropout(DROPOUT); s.classifier = nn.Linear(s.bert.config.hidden_size, len(TARGET_SYMPTOMS))
        def forward(s, i, a): return s.classifier(s.dropout(s.bert(input_ids=i, attention_mask=a).last_hidden_state[:, 0]))

    use_amp = dev.type == "cuda"
    crit = nn.BCEWithLogitsLoss()
    # OOM probe with batch 16
    batch, accum = 16, 1
    try:
        torch.manual_seed(SEED); m = Clf().to(dev); m.train()
        with torch.autocast(device_type=dev.type, enabled=use_amp):
            l = crit(m(ids[:16].to(dev), am[:16].to(dev)), Yt[:16].to(dev))
        l.backward(); del m, l; torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        batch, accum = 8, 2; torch.cuda.empty_cache()
    status(f"BATCH batch={batch} accum={accum} effective={batch*accum} amp={use_amp}")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = Clf().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    g = torch.Generator().manual_seed(SEED)

    @torch.no_grad()
    def evaluate(idx):
        model.eval(); P = []; S = []
        for s in range(0, len(idx), 64):
            b = idx[s:s+64]
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                lo = model(ids[b].to(dev), am[b].to(dev))
            prob = torch.sigmoid(lo.float()).cpu().numpy()
            S.append(prob)
            P.append((prob > TH).astype(np.int32))
        P = np.vstack(P); S = np.vstack(S); L = Y[idx]
        tp = ((P == 1) & (L == 1)).sum(0); fp = ((P == 1) & (L == 0)).sum(0); fn = ((P == 0) & (L == 1)).sum(0)
        pr = tp / (tp + fp + 1e-8); rc = tp / (tp + fn + 1e-8); f1 = 2 * pr * rc / (pr + rc + 1e-8)
        return float(f1.mean()), f1, P, S

    best, best_ep, hist, capped = -1.0, None, [], False
    adoption_f1, min1_f1 = None, None
    ckpt = WORK / "mission3_kcbert_front256_best.pt"
    tr_idx_t = torch.as_tensor(tr_idx)
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = tr_idx_t[torch.randperm(len(tr_idx_t), generator=g)]
        tot, nb, et = 0.0, 0, time.time(); opt.zero_grad(set_to_none=True)
        nsteps = (len(perm) + batch - 1) // batch
        for k in range(nsteps):
            b = perm[k*batch:(k+1)*batch]
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = crit(model(ids[b].to(dev), am[b].to(dev)), Yt[b].to(dev))
            if not torch.isfinite(loss): status(f"WARN non-finite loss ep={ep} step={k}")
            (scaler.scale(loss / accum) if use_amp else loss / accum).backward()
            if (k + 1) % accum == 0 or k + 1 == nsteps:
                if use_amp: scaler.step(opt); scaler.update()
                else: opt.step()
                opt.zero_grad(set_to_none=True)
            tot += loss.item(); nb += 1
            if k % 200 == 0:
                print(f"{kst()} ep{ep} step {k}/{nsteps} loss={tot/nb:.4f}", flush=True)
            if time.time() - T0 > TIME_CAP_S - 900: capped = True; break
        mf1, lf1, _, probs = evaluate(va_idx)
        ep_csv = OUT / f"val_pred_ep{ep}.csv"
        save_epoch_val_pred(ep_csv, [file_names[int(i)] for i in va_idx], probs, Y[va_idx])
        status(f"VAL_PRED ep={ep} rows={len(va_idx)} file={ep_csv.name}")
        if ep == 3:
            adoption_f1 = mf1
            min1_f1, _ = multilabel_macro_f1(min1_predictions(probs), Y[va_idx])
            print(f"{kst()} ADOPTION ep3 internal_val_macro_f1={adoption_f1:.4f} threshold=0.5", flush=True)
            print(
                f"{kst()} EP3_MIN1 macro_f1={min1_f1:.4f} rule=argmax_if_no_label_ge_0.5 p0_min1={P0_MIN1_MACRO_F1:.4f}",
                flush=True,
            )
            status(f"ADOPTION ep3 internal_val_macro_f1={adoption_f1:.4f} threshold=0.5")
            status(f"EP3_MIN1 macro_f1={min1_f1:.4f}")
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "internal_macro_f1": mf1,
               "per_symptom_f1": dict(zip(TARGET_SYMPTOMS, map(float, lf1))), "epoch_s": time.time() - et, "steps": nb}
        if ep == 3:
            rec["adoption_macro_f1_ep3"] = mf1
            rec["ep3_macro_f1_min1"] = min1_f1
        hist.append(rec)
        print(f"{kst()} EPOCH {ep}/{EPOCHS} internal_val_macro_f1={mf1:.4f} threshold=0.5 loss={rec['train_loss']:.4f}", flush=True)
        status(f"EPOCH {ep}/{EPOCHS} loss={rec['train_loss']:.4f} internal_val_macro_f1={mf1:.4f} threshold=0.5 epoch_s={rec['epoch_s']:.0f}")
        try:  # v3: every-epoch ckpt to Drive (team rule 2026-10-06)
            ep_ck = WORK / f"mission3_kcbert_front256_ep{ep}.pt"
            torch.save({"model_state_dict": model.state_dict(), "mission": 3, "epoch": ep, "internal_macro_f1": mf1,
                        "model_name": MODEL_NAME, "max_length": MAX_LENGTH, "threshold": TH,
                        "target_symptoms": TARGET_SYMPTOMS, "seed": SEED, "commit": COMMIT}, ep_ck)
            shutil.copy2(ep_ck, OUT / ep_ck.name); status(f"EPOCH_CKPT ep={ep} copied to Drive sha256={sha256(ep_ck)}")
            (OUT / "epoch_history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
            ep_ck.unlink()
        except Exception as e: status(f"WARN epoch ckpt save failed {e!r}")
        if mf1 >= best:
            best, best_ep = mf1, ep
            torch.save({"model_state_dict": model.state_dict(), "mission": 3, "epoch": ep, "best_f1": best,
                        "model_name": MODEL_NAME, "max_length": MAX_LENGTH, "threshold": TH,
                        "target_symptoms": TARGET_SYMPTOMS, "seed": SEED, "commit": COMMIT}, ckpt)
            status(f"BEST epoch={ep} macro_f1={best:.4f} (reference only; adoption is ep3)")
            try:
                shutil.copy2(ckpt, OUT / ckpt.name); status(f"BEST epoch={ep} copied to Drive")
            except Exception as e: status(f"WARN drive copy failed {e!r}")
        try:
            write_metrics(hist, best_ep, best, adoption_f1, min1_f1)
        except Exception as exc:
            print(f"metrics.json save failed: {exc}", flush=True)
            status(f"METRICS_FAIL {exc!r}")
            raise SystemExit(1) from exc
        if capped: status("TIME_CAP reached, stopping"); break

    if adoption_f1 is None or not (OUT / "val_pred_ep3.csv").is_file():
        print("missing ep3 adoption score or val_pred_ep3.csv", flush=True)
        status("ADOPTION_MISSING ep3")
        raise SystemExit(1)
    shutil.copy2(ckpt, OUT / ckpt.name); ck_sha = sha256(ckpt)
    (OUT / (ckpt.name + ".sha256")).write_text(f"{ck_sha}  {ckpt.name}\n")
    status(f"CKPT saved to Drive sha256={ck_sha}")

    best_rec = [h for h in hist if h["epoch"] == best_ep][0]
    vers = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "iterstrat": getattr(iterstrat, "__version__", "unknown"),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": gpu}
    report = {"candidate": "M3-P1-kcbert-front256", "commit": COMMIT, "model": MODEL_NAME, "truncation": "front",
              "max_length": MAX_LENGTH, "head": "CLS->dropout->linear(9), sigmoid", "loss": "BCEWithLogitsLoss (unweighted)",
              "data": "TL_서울_구급 only; P0 _make_split 80:20 seed=42 (scripts/mission3_baseline 최종 수정본.py); official Validation not used for training/selection",
              "train_label_dirs": [nfc(str(p)) for p in tl], "n_total": int(n_loaded), "n_train": int(len(tr_idx)), "n_internal_val": int(len(va_idx)),
              "epochs": EPOCHS, "lr": LR, "dropout": DROPOUT, "batch_size": batch, "grad_accum": accum, "effective_batch": batch * accum,
              "amp": use_amp, "threshold": TH, "seed": SEED, "best_epoch": best_ep, "internal_macro_f1": best,
              "adoption_macro_f1_ep3": adoption_f1, "best_epoch_ref": {"epoch": best_ep, "internal_macro_f1": best},
              "ep3_macro_f1_min1": min1_f1, "p0_min1_macro_f1": P0_MIN1_MACRO_F1,
              "per_symptom_f1": best_rec["per_symptom_f1"], "history": hist, "time_capped": capped,
              "p0_macro_f1": P0_MACRO_F1, "delta_vs_p0": best - P0_MACRO_F1,
              "ckpt_drive": str(OUT / ckpt.name), "ckpt_sha256": ck_sha,
              "elapsed_s": time.time() - T0, "packages": vers, "finished_kst": kst()}
    for d in (OUT / "reports" / "mission3", REPO / "reports" / "mission3"):
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / "m3_reeval_kcbert_front256.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e: print("report write fail", d, e)
    status(f"REPORT_DONE internal_macro_f1={best:.4f} best_epoch={best_ep} eff_batch={batch*accum} elapsed_s={time.time()-T0:.0f}")

    # Optional: official Validation inference + scoring (evaluation only, no selection)
    try:
        vl = find_label_dirs("validation", "VL")
        vfiles = sorted({f for d in vl for f in d.rglob("*.json")}, key=lambda x: nfc(x.name))
        with ThreadPoolExecutor(16) as ex: vrows = list(ex.map(load, vfiles))
        venc = tok([r[0] for r in vrows], padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        sd = torch.load(ckpt, map_location=dev, weights_only=False); model.load_state_dict(sd["model_state_dict"]); model.eval()
        P = []
        with torch.no_grad():
            for s in range(0, len(vrows), 64):
                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    lo = model(venc["input_ids"][s:s+64].to(dev), venc["attention_mask"][s:s+64].to(dev))
                P.append((torch.sigmoid(lo.float()) > TH).int().cpu().numpy())
        P = np.vstack(P); L = np.asarray([r[1] for r in vrows])
        tp = ((P == 1) & (L == 1)).sum(0); fp = ((P == 1) & (L == 0)).sum(0); fn = ((P == 0) & (L == 1)).sum(0)
        pr = tp / (tp + fp + 1e-8); rc = tp / (tp + fn + 1e-8); f1 = 2 * pr * rc / (pr + rc + 1e-8)
        pd.DataFrame({"label file name": [r[2] for r in vrows],
                      "symptom": [str([s for s, v in zip(TARGET_SYMPTOMS, p) if v]) for p in P]}).to_csv(OUT / "mission3_val.csv", index=False, encoding="utf-8-sig")
        vrep = {"n_val": len(vrows), "val_label_dirs": [nfc(str(p)) for p in vl], "official_val_macro_f1": float(f1.mean()),
                "per_symptom_f1": dict(zip(TARGET_SYMPTOMS, map(float, f1))), "note": "evaluation only; not used for training/selection"}
        (OUT / "reports" / "mission3" / "m3_p1_official_val_eval.json").write_text(json.dumps(vrep, ensure_ascii=False, indent=2), encoding="utf-8")
        status(f"VAL_EVAL_DONE n={len(vrows)} official_val_macro_f1={f1.mean():.4f}")
    except Exception as e:
        status(f"VAL_EVAL_FAILED {e!r}")
    status(f"ALL_DONE adoption_macro_f1_ep3={adoption_f1:.4f} ep3_macro_f1_min1={min1_f1:.4f} best_epoch_ref={best:.4f} ckpt_sha256={ck_sha}")

if __name__ == "__main__":
    try: main()
    except BaseException as e:
        status(f"FAILED {e!r}"); traceback.print_exc(); raise
