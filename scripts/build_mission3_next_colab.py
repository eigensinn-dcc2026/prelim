"""Build a source-only Colab kit; reuse the previously verified private input TAR."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/m3_next"
FILES=[
    "src/dcc_data/__init__.py","src/dcc_data/manifest.py","src/dcc_data/guards.py","src/dcc_inference/__init__.py",
    "src/dcc_data/mission3_context_experiment.py","src/dcc_data/mission3_context_release.py",
    "src/dcc_inference/mission3_bert.py","src/dcc_inference/mission3_context.py",
    "src/dcc_data/mission3_evidence_data.py","src/dcc_data/mission3_next_experiment.py",
    "src/dcc_inference/mission3_evidence.py","scripts/train_mission3_next.py",
    "scripts/evaluate_mission3_context.py","scripts/package_mission3_context.py",
    "submission/mission3_evidence/inference.py","submission/mission3_evidence/requirements.txt",
    "submission/mission3_evidence/README.md"]


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={name:(ROOT/name).read_text(encoding="utf-8") for name in FILES}
    manifest={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()}
    with zipfile.ZipFile(OUT/"next_source.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for name,text in sources.items():archive.writestr(name,text)
        archive.writestr("next_source_manifest.json",json.dumps(manifest,indent=2))
    cells=[]
    def md(text):cells.append(dict(cell_type="markdown",metadata={},source=text.splitlines(True)))
    def code(text):
        ast.parse(text)
        cells.append(dict(cell_type="code",metadata={},source=text.splitlines(True),execution_count=None,outputs=[]))
    md("# M3 다음 실험: 손실·입력 범위·근거 상태\n\n"
       "기존 실패 실험은 보존합니다. 이 노트북만 위에서 아래로 실행하세요. 기존 Colab 노트북의 모두 실행은 사용하지 않습니다. "
       "입력 파일 `input.tar.gz`를 Colab 세션 저장소에 업로드해야 합니다. source는 아래 셀에 포함되어 있습니다. "
       "Training 23,408 / 내부 검증 5,792, 임계값 0.5를 유지합니다. 공식 Validation은 자동으로 읽지 않습니다.\n\n"
       "GPU 전체 실행은 누적 6시간으로 제한됩니다. 예산이 끝나면 결과를 내려받고 같은 파일·설정으로 재개하세요. "
       "규칙으로 생성한 근거 상태는 약한 감독 신호이며 수동 주석으로 표시하지 않습니다.")
    code("import os,sys,subprocess\nfrom pathlib import Path\nimport json,hashlib,tarfile,zipfile\n"
         "os.environ['USE_TF']='0'\n"
         "subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6','scikit-learn','matplotlib','numpy','pandas'],check=True)\n"
         "import torch\nassert torch.cuda.is_available(), 'GPU 런타임이 필요합니다.'\n"
         "print('Runtime:',sys.version,torch.__version__,torch.cuda.get_device_name(0))\n")
    code("# Source-only files; contains no Training transcript or labels.\n"
         "ROOT=Path('/content/m3_next')\nROOT.mkdir(exist_ok=True)\n"
         "SOURCES="+repr(sources)+"\nMANIFEST="+repr(manifest)+"\n"
         "for name,text in SOURCES.items():\n    target=ROOT/name\n    target.parent.mkdir(parents=True,exist_ok=True)\n    target.write_text(text,encoding='utf-8')\n"
         "    assert hashlib.sha256(target.read_bytes()).hexdigest()==MANIFEST[name]\n"
         "print('Source files verified:',len(MANIFEST))\n")
    code("INPUT=Path('/content/input.tar.gz')\nassert INPUT.is_file(), 'input.tar.gz를 세션 저장소에 업로드하세요.'\n"
         "if not (ROOT/'bundle.json').exists():\n    with tarfile.open(INPUT) as archive:\n        archive.extractall(ROOT,filter='data')\n"
         "# The old input archive also contains v1 source; restore the fixed v2 source afterwards.\n"
         "for name,text in SOURCES.items():\n    (ROOT/name).parent.mkdir(parents=True,exist_ok=True)\n    (ROOT/name).write_text(text,encoding='utf-8')\n"
         "meta=json.loads((ROOT/'bundle.json').read_text())\n"
         "assert (meta['train_rows'],meta['internal_holdout_rows'],meta['training_rows'])==(23408,5792,29200)\n"
         "assert not meta['official_validation_included']\n"
         "def sha(path):\n    digest=hashlib.sha256()\n    with path.open('rb') as stream:\n        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)\n    return digest.hexdigest()\n"
         "assert sha(ROOT/'baseline.pt')==meta['checkpoint_sha256']\nprint('Private inputs verified:',meta)\n")
    code("def run_next(phase,smoke=0):\n"
         "    output=ROOT/('smoke' if smoke else 'run')\n    output.mkdir(exist_ok=True)\n"
         "    cmd=[sys.executable,'-u',str(ROOT/'scripts/train_mission3_next.py'),'--train-label-dir',str(ROOT/'training'),'--base-ckpt',str(ROOT/'baseline.pt'),'--reference-probs-dir',str(ROOT/'reference'),'--out',str(output),'--device','cuda','--phase',phase,'--smoke-rows',str(smoke),'--max-hours','0.25' if smoke else '6']\n"
         "    with (output/'execution.log').open('a',encoding='utf-8') as log:\n"
         "        process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',env=dict(os.environ,USE_TF='0'))\n"
         "        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()\n"
         "        if process.wait():raise RuntimeError('실험 실패: execution.log 확인')\n"
         "    return json.loads((output/'metrics.json').read_text())\n"
         "smoke_result=run_next('all',16)\nassert smoke_result['smoke'] and not smoke_result['gate']['passed']\nprint(smoke_result['status'])\n")
    code("# Stage 1: same checkpoint, one encoder epoch, three loss functions.\n"
         "loss_result=run_next('loss')\n"
         "print(json.dumps(loss_result.get('loss_comparison',{}),ensure_ascii=False,indent=2))\n")
    code("# Stage 2 resumes Stage 1; all input pilots, then only a successful winner.\n"
         "result=run_next('all')\nprint('Status:',result['status'],'Selected:',result['selected_method'])\n"
         "print('Internal gate:',json.dumps(result['gate'],ensure_ascii=False,indent=2))\n"
         "print('Official Validation remains unused.')\n")
    md("## 보존 후 런타임 해제\n\n결과 ZIP과 가중치를 각각 다운로드합니다. ZIP 무결성·파일 크기·SHA-256을 로컬에서 확인하기 전까지 런타임을 삭제하지 마세요. "
       "`experimental.pt`도 다운로드 대상으로 포함됩니다. Drive에는 자동 전송하지 않습니다. "
       "전체 Training 후보가 생기면 `final_candidate.pt`를 먼저 확보합니다.")
    code("from google.colab import files\n"
         "run=ROOT/'run'\nweights=sorted(run.rglob('experimental.pt'))+sorted(run.glob('final_candidate.pt'))\n"
         "checkpoint_manifest={str(path.relative_to(ROOT)):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in weights}\n"
         "(run/'checkpoint_manifest.json').write_text(json.dumps(checkpoint_manifest,indent=2))\n"
         "archive_path=Path('/content/m3_next_results.zip')\n"
         "with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:\n"
         "    for path in sorted(run.rglob('*')):\n"
         "        if path.is_file() and path.suffix in ('.json','.md','.log','.npz','.png','.csv'):\n"
         "            archive.write(path,path.relative_to(ROOT).as_posix())\n"
         "with zipfile.ZipFile(archive_path) as archive:assert archive.testzip() is None\n"
         "print('Results:',archive_path.stat().st_size,sha(archive_path))\n"
         "print('Checkpoint manifest:',json.dumps(checkpoint_manifest,indent=2))\nfiles.download(str(archive_path))\n")
    code("# Download actual weights separately; they are NOT inside the metrics ZIP.\n"
         "for path in weights:\n    print('Downloading weights:',path.name,path.stat().st_size,sha(path))\n    files.download(str(path))\n"
         "print('Verify local files against checkpoint_manifest.json before deleting the runtime.')\n")
    md("## 조건부 공식 평가와 패키지\n\n`final_candidate_ready`일 때만 전체 Training 후보를 로컬에 내려받습니다. "
       "기존 `scripts/evaluate_mission3_context.py`로 공식 평가를 실행하며 공유 `official_once.json`을 유지합니다. "
       "공식 Macro F1 ≥ 0.6540일 때만 `scripts/package_mission3_context.py`로 새 디렉터리에 패키지를 만듭니다. "
       "새 모델을 위해 공식 평가 기록을 초기화하거나 공식 점수로 설정을 조정하지 않습니다.")
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python"),colab=dict(name="M3_next_experiment.ipynb")),cells=cells)
    for i,cell in enumerate(cells):cell['id']=f'm3-next-{i:02d}'
    path=ROOT/"notebooks/M3_next_experiment.ipynb"
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (OUT/"next_source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(path)
    print(OUT/"next_source.zip")


if __name__=="__main__":build()
