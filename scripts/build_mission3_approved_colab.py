"""Create a source-only, disabled-by-default Colab notebook and ZIP."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "src/dcc_data/__init__.py", "src/dcc_data/manifest.py", "src/dcc_data/guards.py",
    "src/dcc_inference/__init__.py", "src/dcc_inference/mission3_bert.py",
    "src/dcc_inference/mission3_context.py", "src/dcc_data/mission3_context_experiment.py",
    "src/dcc_data/mission3_approved_core.py", "scripts/train_mission3_approved.py",
    "scripts/inference_mission3_approved.py", "scripts/summarize_mission3_approved.py",
    "tests/test_mission3_approved.py", "tests/conftest.py",
    "reports/mission3/approved_experiments/README.md"]


def build():
    output = ROOT/"outputs/m3_approved"
    output.mkdir(parents=True, exist_ok=True)
    contents = {name:(ROOT/name).read_bytes() for name in FILES}
    manifest = {name:hashlib.sha256(value).hexdigest() for name,value in contents.items()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,value in contents.items(): archive.writestr(name,value)
        archive.writestr("source_manifest.json",json.dumps(manifest,indent=2))
    archive_bytes=buffer.getvalue()
    (output/"approved_source.zip").write_bytes(archive_bytes)
    encoded=base64.b64encode(archive_bytes).decode()
    cells=[]
    def md(text): cells.append(dict(cell_type="markdown",metadata={},source=text.splitlines(True)))
    def code(text):
        ast.parse(text)
        cells.append(dict(cell_type="code",metadata={},source=text.splitlines(True),execution_count=None,outputs=[]))
    md("# M3 승인 실험 — 실행 보류 상태\n\n계정: **daulkim2001@gmail.com**. 2026-10-12 사용자의 최신 지시: GPU 실행 보류, 코드와 로컬 검증만 준비.\n\n"
       "이 노트북은 기본 Run All로 GPU 학습·다운로드·데이터 업로드를 시작하지 않습니다. "
       "향후 GPU 실행을 다시 승인한 뒤 런타임을 선택하고 RUN_GPU=True로 바꾸세요. 추가 단위 구매는 하지 않습니다. "
       "여기에는 소스만 포함되며 비공개 전사·라벨·가중치는 없습니다.\n\n"
       "목표: ① microbatch/global soft-F1 대조 ② 사람 근거 감독 ③ 누수 없는 OOF 증류 ④ 선택적 512토큰 재검토. "
       "공식 Validation을 읽거나 제출본을 교체하는 단계는 없습니다.")
    code("RUN_GPU = False\nRUN_PREFLIGHT = False\nPHASE = 'loss'  # loss / teachers / student / annotations / evidence / cascade\nSEED = 42\nMAX_HOURS = 3\nHUMAN_ANNOTATION_RULES_CONFIRMED = False\nANNOTATIONS = None\n"
         "print('GPU 실행:', RUN_GPU, '| 데이터 점검:', RUN_PREFLIGHT, '| 단계:', PHASE)\n")
    code("import base64,hashlib,io,json,os,sys,zipfile,tarfile,subprocess\nfrom pathlib import Path\n"
         "ROOT=Path('/content/m3_approved')\nROOT.mkdir(parents=True,exist_ok=True)\n"
         "SOURCE_B64="+repr(encoded)+"\n"
         "blob=base64.b64decode(SOURCE_B64)\nassert hashlib.sha256(blob).hexdigest()=="+repr(hashlib.sha256(archive_bytes).hexdigest())+"\n"
         "with zipfile.ZipFile(io.BytesIO(blob)) as archive:\n"
         "    for name in archive.namelist():\n        target=(ROOT/name).resolve()\n        assert target.is_relative_to(ROOT.resolve())\n    archive.extractall(ROOT)\n"
         "for name,digest in json.loads((ROOT/'source_manifest.json').read_text()).items():\n"
         "    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest\n"
         "print('Source-only package verified:',len(json.loads((ROOT/'source_manifest.json').read_text())))\n")
    code("if RUN_GPU or RUN_PREFLIGHT:\n"
         "    os.environ['USE_TF']='0'\n"
         "    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6','scikit-learn','iterative-stratification','numpy','pytest'],check=True)\n"
         "    subprocess.run([sys.executable,'-m','pytest','tests/test_mission3_approved.py','-q'],cwd=ROOT,check=True)\n"
         "else:\n    print('패키지 설치와 테스트 실행 보류')\n")
    md("## 입력 준비\n\n향후 실행할 때 기존 `input.tar.gz`를 Colab 세션 저장소에 업로드합니다. "
       "내용은 Training 전사·9개 라벨, 기존 체크포인트, 내부 검증 확률입니다. 외부 API에 전사를 보내지 않습니다. "
       "아래 업로드 셀은 RUN_GPU 또는 RUN_PREFLIGHT를 명시적으로 켠 경우에만 작동합니다.")
    code("if RUN_GPU or RUN_PREFLIGHT:\n"
         "    INPUT=Path('/content/input.tar.gz')\n"
         "    if not INPUT.exists():\n        from google.colab import files\n        files.upload()\n"
         "    assert INPUT.exists(),'input.tar.gz가 필요합니다'\n"
         "    bundle=ROOT/'input'\n    bundle.mkdir(exist_ok=True)\n"
         "    if not (bundle/'bundle.json').exists():\n"
         "        with tarfile.open(INPUT) as archive:\n"
         "            members=[m for m in archive.getmembers() if m.name in ('baseline.pt','bundle.json') or m.name.startswith(('training/','reference/'))]\n"
         "            archive.extractall(bundle,members=members,filter='data')\n"
         "    command=[sys.executable,'scripts/train_mission3_approved.py','preflight','--bundle',str(bundle),'--out',str(ROOT/'run')]\n"
         "    subprocess.run(command,cwd=ROOT,check=True)\n"
         "else:\n    print('비공개 데이터 업로드·읽기 보류')\n")
    md("## 실행 단계\n\n- **loss**: 16·64 유효 배치 각각 micro8 vs global, 동일 체크포인트 1 epoch 연속 학습.\n"
       "- **teachers → student**: Training 23,408건 안에서 새로운 3-fold 교사 학습. 29,200건 전체 OOF를 재사용하지 않습니다. "
       "공개 사전학습 모델에서 시작하며 교사당 3 epochs. 학생 hard-label 대조군과 증류를 비교합니다. 비용이 큰 단계입니다.\n"
       "- **annotations**: GPU 없이 Training 300통화의 검토 양식만 생성. 자동 규칙·AI 결과를 사람 주석으로 취급하지 않습니다.\n"
       "- **evidence**: 사람이 검토한 JSONL과 추가 주석 규정 확인이 있어야 실행됩니다. 동일 구조 무감독 대조군 포함.\n"
       "- **cascade**: 256·512 각각 동일 1 epoch 연속 학습. 내부 검증에서 5-fold 교차 적합으로 재검토 기준 선택.\n\n"
       "단계별 실행 후 다음 PHASE를 선택합니다. 3시간은 각 단계가 아니라 동일 run 폴더 전체 누적 한도입니다. "
       "seed42 pilot이 유망하면 SEED=43/44로 반복하세요. 자동으로 채택하지 않습니다.")
    code("if RUN_GPU:\n"
         "    import torch\n    assert torch.cuda.is_available(),'향후 승인 후 Colab GPU 런타임을 선택하세요'\n"
         "    command=[sys.executable,'-u','scripts/train_mission3_approved.py',PHASE,'--bundle',str(ROOT/'input'),'--out',str(ROOT/'run'),'--seed',str(SEED),'--max-hours',str(MAX_HOURS),'--execute']\n"
         "    if PHASE=='evidence':\n"
         "        assert HUMAN_ANNOTATION_RULES_CONFIRMED and ANNOTATIONS,'사람 주석 및 규정 확인이 필요합니다'\n"
         "        command+=['--annotations',str(ANNOTATIONS),'--human-annotation-rules-confirmed']\n"
         "    log=ROOT/'run'/'execution.log'\n    log.parent.mkdir(exist_ok=True)\n"
         "    with log.open('a',encoding='utf-8') as stream:\n"
         "        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)\n"
         "        for line in process.stdout:\n            print(line,end='',flush=True)\n            stream.write(line)\n            stream.flush()\n"
         "        assert process.wait()==0,'단계 중단: execution.log와 budget.json 확인'\n"
         "elif RUN_PREFLIGHT and PHASE=='annotations':\n"
         "    subprocess.run([sys.executable,'scripts/train_mission3_approved.py','annotations','--bundle',str(ROOT/'input'),'--out',str(ROOT/'run')],cwd=ROOT,check=True)\n"
         "else:\n    print('학습을 시작하지 않았습니다.')\n")
    code("if (ROOT/'run').exists():\n"
         "    subprocess.run([sys.executable,'scripts/summarize_mission3_approved.py','--run',str(ROOT/'run')],cwd=ROOT,check=True)\n"
         "else:\n    print('아직 실제 학습 결과 없음')\n")
    md("## 보존과 CPU 속도 측정\n\n학습 뒤 모델과 결과를 모두 확보하기 전에는 런타임을 삭제하지 않습니다. "
       "다음 셀은 별도로 DOWNLOAD_RESULTS=True를 지정해야 내려받습니다. resume.pt는 optimizer를 포함하므로 크며, "
       "다음 세션에서 학습을 재개할 때는 별도로 보존해야 합니다.\n\n"
       "로컬 속도 측정은 `scripts/inference_mission3_approved.py --ckpt-path ... --label-dir TrainingSubset --output predictions.csv --limit 200 --threads 4`로 합니다. "
       "교사 전체는 --ckpt-path 대신 --teacher-dir run/teachers로 측정합니다. 같은 PC·동일 표본·동일 스레드로 반복 측정하고 "
       "total_seconds_including_load를 비교합니다. 단일 모델 warm-up과 교사 해시 검사 포함 여부가 달라 장기 상주 추론 성능으로 일반화하지 않습니다.")
    code("DOWNLOAD_RESULTS=False\n"
         "if DOWNLOAD_RESULTS:\n"
         "    from google.colab import files\n    result=ROOT/'run'\n    assert result.exists()\n"
         "    manifest={}\n"
         "    for path in result.rglob('model.pt'):\n"
         "        digest=hashlib.sha256()\n        with path.open('rb') as stream:\n            for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)\n"
         "        manifest[str(path.relative_to(result))]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}\n"
         "    (result/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2))\n"
         "    archive_path=ROOT/'results.zip'\n"
         "    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:\n"
         "        for path in result.rglob('*'):\n            if path.is_file() and path.suffix in ('.json','.npz','.log','.md','.joblib'):\n                archive.write(path,path.relative_to(result))\n"
         "    files.download(str(archive_path))\n"
         "    for name in manifest: files.download(str(result/name))\n"
         "    print('체크포인트 다운로드 후 로컬에서 크기와 SHA256을 대조하세요. GPU 세션은 자동 삭제하지 않습니다.')\n")
    for index,cell in enumerate(cells): cell["id"]=f"approved-{index:02d}"
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,
                  metadata=dict(kernelspec=dict(name="python3",display_name="Python 3",language="python"),
                                language_info=dict(name="python"),colab=dict(name="M3_approved_experiments_20261012.ipynb")))
    path=ROOT/"notebooks/M3_approved_experiments_20261012.ipynb"
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")
    (output/"source_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(path)
    print(output/"approved_source.zip")


if __name__ == "__main__": build()
