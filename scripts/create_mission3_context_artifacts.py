"""Create a reproducible experiment notebook and PPT-ready method figures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def create_notebook(results):
    results=Path(results)
    notebook=nbf.v4.new_notebook()
    notebook.metadata.update(kernelspec=dict(name="python3",language="python",display_name="Python 3"),
                             language_info=dict(name="python",version="3.11"))
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    notebook.cells=[
        md("# M3: 유사 사례 대조와 질문·답변 근거\n\n"
           "## tl;dr\n\n전체 실험 완료 전에는 성능 향상을 주장하지 않습니다. 이 노트북은 고정된 설정, 실행 로그, "
           "내부 검증 및 채택 결정을 보여 주는 실행 기록입니다. `smoke=true` 결과는 구현 점검용입니다."),
        md("## Context & Methods\n\nA: 증상별 양성 8·음성 8의 표현과 유사도 차이, 전체 이웃 32건의 라벨 일치도를 읽습니다. "
           "B: 현재 발화와 뒤의 두 발화 묶음을 증상별 attention으로 읽습니다. 통화 전체 표현만 받는 보정 모델을 함께 학습합니다.\n\n"
           "### Key Assumptions\n\n기존 23,408/5,792 분할, ROBERTA-003 seed 42, 앞 256토큰, 임계값 0.5를 유지합니다. "
           "검색 저장소는 Training만 포함하며 자기 통화·같은 통화·동일 정규화 전사본을 제외합니다. "
           "B의 답변 제거 실험은 명시적인 뒤 발화 입력을 제거하며, 전체 인코더의 문맥 정보는 남아 있습니다.\n\n"
           "근거: [검색 결합 분류](https://arxiv.org/abs/2305.13058), "
           "[label attention](https://www.ijcai.org/proceedings/2020/461)."),
        md("### 1. 실행 경로와 설정\n\n명령은 아래 설정을 사용합니다. `RUN_TRAINING=False`는 저장된 실행 결과를 읽는 모드입니다."),
        code("from pathlib import Path\nimport json, sys, subprocess\nimport numpy as np\nimport pandas as pd\n"
             "ROOT = Path.cwd()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
             f"RESULTS = ROOT / {str(results).replace(chr(92),'/')!r}\n"
             "TRAIN_LABEL_DIR = ROOT / 'data/raw_merged/Training/2.라벨링데이터/TL_서울_구급'\n"
             "BASE_CKPT = ROOT / 'checkpoints/mission3/m3_roberta003_seed42.pt'\n"
             "REFERENCE_DIR = ROOT / 'reports/mission3/ensemble/probs'\n"
             "RUN_TRAINING = False\nPHASE = 'pilot'\nSMOKE_ROWS = 0\nMAX_HOURS = 10\n"
             "sys.path.insert(0, str(ROOT/'src'))\nfrom dcc_data.mission3_context_experiment import RECIPE\n"
             "print(json.dumps(RECIPE, indent=2))"),
        md("## Data\n\n기준 예측 파일 이름으로 기존 holdout을 복원하며, 세 대조군의 파일 집합과 정답 일치를 확인합니다. "
           "공식 Validation은 최종 후보 전용 평가 명령에서만 읽습니다.\n\n### 2. 학습 실행 및 로그"),
        code("if RUN_TRAINING:\n"
             "    command = [sys.executable, '-u', str(ROOT/'scripts/train_mission3_context.py'),\n"
             "        '--train-label-dir', str(TRAIN_LABEL_DIR), '--base-ckpt', str(BASE_CKPT),\n"
             "        '--reference-probs-dir', str(REFERENCE_DIR), '--out', str(RESULTS),\n"
             "        '--phase', PHASE, '--smoke-rows', str(SMOKE_ROWS), '--max-hours', str(MAX_HOURS)]\n"
             "    RESULTS.mkdir(parents=True, exist_ok=True)\n"
             "    with (RESULTS/'execution.log').open('a', encoding='utf-8') as log:\n"
             "        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')\n"
             "        for line in process.stdout:\n            print(line, end=''); log.write(line); log.flush()\n"
             "        if process.wait(): raise RuntimeError('Training failed; inspect execution.log')\n"
             "else:\n    print('Saved-run inspection mode; training is not launched.')"),
        md("## Results\n\n### 3. 실제 실행 상태와 비교"),
        code("summary_path = RESULTS/'metrics.json'\nsummary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}\n"
             "print('Status:', summary.get('status', 'not_run'))\nprint('Smoke only:', summary.get('smoke', 'unknown'))\n"
             "table = pd.DataFrame([{ 'model': name, 'macro_f1': item['macro_f1'], 'micro_f1': item['micro_f1'], 'calls': item['rows']}\n"
             "                      for name, item in summary.get('pilot', {}).items()])\n"
             "display(table.round(4))"),
        md("### 4. 비교 그림\n\n모든 막대는 0에서 시작합니다. 축소 실행의 점수는 대회 성능으로 해석할 수 없습니다."),
        code("import matplotlib.pyplot as plt\n"
             "if not table.empty:\n"
             "    fig, ax = plt.subplots(figsize=(9,4))\n"
             "    bars = ax.barh(table['model'], table['macro_f1'], color='#326a91')\n"
             "    ax.bar_label(bars, fmt='%.4f', padding=4)\n"
             "    ax.set(xlim=(0,1), xlabel='Macro F1 (threshold 0.5)', title='Internal holdout' + (' — SMOKE ONLY' if summary.get('smoke') else ''))\n"
             "    fig.tight_layout(); plt.show()\n"
             "else:\n    print('No measured results are available yet.')"),
        md("### 5. 반복·구성 요소 제거·채택 판단\n\npaired bootstrap은 통화 단위로 같은 통화를 함께 재표집합니다. "
           "평균 +0.003, 최소 2개 seed 개선, 95% CI 하한 > 0, Micro F1 하락 ≤ 0.005를 모두 확인합니다."),
        code("repeat_table = []\n"
             "for method, stages in summary.get('repeat', {}).items():\n"
             "    for stage, records in stages.items():\n"
             "        for seed, result in zip([42,43,44], records):\n"
             "            repeat_table.append(dict(method=method,stage=stage,seed=seed,macro_f1=result['macro_f1'],micro_f1=result['micro_f1']))\n"
             "display(pd.DataFrame(repeat_table).round(4))\n"
             "print(json.dumps(summary.get('gate', {'status':'not_available'}), ensure_ascii=False, indent=2))"),
        md("## Takeaways\n\n공식 점수 0.6540 이상일 때만 별도 패키지를 채택합니다. 목표 0.67과 실제 결과를 구분합니다. "
           "현재 후보 교체는 이 노트북을 열거나 축소 실행하는 것으로 발생하지 않습니다."),
        code("print('Measured decision:', summary.get('status','not_run'))\n"
             "print('Selected method:', summary.get('selected_method'))\n"
             "print('Reports:', RESULTS/'report.md')\n"
             "print('Official evaluation is a separate, once-only command after full Training.')"),
    ]
    path=ROOT/"notebooks/M3_context_experiment.ipynb"
    nbf.validate(notebook);nbf.write(notebook,path)
    return path


def method_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    directory=ROOT/"reports/mission3/context"
    directory.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,1,figsize=(13,7))
    def box(ax,x,y,width,text,color='#eaf1f6'):
        ax.add_patch(FancyBboxPatch((x,y),width,.55,boxstyle='round,pad=0.02',facecolor=color,edgecolor='#526573'))
        ax.text(x+width/2,y+.275,text,ha='center',va='center',fontsize=11,color='#18242c')
    def arrow(ax,a,b):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#526573',lw=1.5))
    for ax,title in zip(axes,['A: Positive / negative call contrast','B: Symptom attention over adjacent utterances']):
        ax.set(xlim=(0,12.7),ylim=(0,2.5));ax.axis('off');ax.set_title(title,loc='left',fontsize=15,pad=12)
        box(ax,.05,.9,2.,'RoBERTa-003\nfirst 256 tokens')
        box(ax,8.7,.9,1.7,'Learned gate\n+ base logits')
        box(ax,11.,.9,1.55,'9 symptoms\nsigmoid >= 0.5')
        arrow(ax,(10.42,1.175),(10.95,1.175))
        ax.text(.05,.05,'Design schematic; improvement requires measured internal and official validation.',fontsize=10,color='#526573')
    ax=axes[0]
    box(ax,2.8,1.35,2.5,'Training-only memory\nexclude same call / transcript')
    box(ax,5.85,1.35,2.2,'Per symptom: +8 / -8\nTop 32 label agreement')
    box(ax,5.85,.4,2.2,'CLS + prototypes +\nsimilarity gap (128d)')
    arrow(ax,(2.07,1.175),(2.75,1.6));arrow(ax,(5.32,1.625),(5.8,1.625))
    arrow(ax,(6.95,1.32),(6.95,.99));arrow(ax,(8.1,.675),(8.65,1.175))
    ax=axes[1]
    box(ax,2.8,1.35,2.5,'Offset-restored boundaries\nno speaker IDs / extra tokens')
    box(ax,5.85,1.35,2.2,'Utterance t + t+1 + t+2\n128d windows')
    box(ax,5.85,.4,2.2,'9 label queries\nattention + CLS')
    arrow(ax,(2.07,1.175),(2.75,1.6));arrow(ax,(5.32,1.625),(5.8,1.625))
    arrow(ax,(6.95,1.32),(6.95,.99));arrow(ax,(8.1,.675),(8.65,1.175))
    fig.tight_layout(h_pad=2)
    fig.savefig(directory/"methods.png",dpi=180)
    fig.savefig(directory/"methods.svg")
    svg=directory/"methods.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",default="outputs/m3_context")
    args=parser.parse_args()
    print(create_notebook(args.results));method_figures()
