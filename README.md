# DCC 2026 예선 준비

DCC(데이터 크리에이터 캠프) 2026 데이터 대회 예선 준비를 위한 저장소입니다.

## 폴더 구조

```
dcc2026-prelim/
├── notebooks/     # Colab에서 사용할 .ipynb 파일 모음
├── src/           # 재사용 가능한 함수/모듈 (.py) 모음
├── data/          # 데이터 저장용 (git에는 올리지 않음, .gitignore 처리)
├── checkpoints/   # 모델 체크포인트 저장용 (git에는 올리지 않음, .gitignore 처리)
├── requirements.txt
└── .gitignore
```

- **notebooks/**: 탐색적 분석, 실험, 제출용 코드를 담는 Colab 노트북(.ipynb)을 저장합니다.
- **src/**: 노트북에서 반복적으로 쓰는 전처리, 모델링, 유틸 함수를 `.py` 모듈로 분리해서 저장합니다. 노트북에서 `import`해서 재사용합니다.
- **data/**: 대회 데이터, 전처리 결과 등을 저장합니다. 용량이 크고 대회 규정상 공유가 제한될 수 있어 git 추적 대상에서 제외했습니다.
- **checkpoints/**: 학습 중간/최종 모델 가중치를 저장합니다. 마찬가지로 용량 문제로 git에서 제외했습니다.

## Colab에서 사용하는 방법

1. Colab 노트북 셀에서 저장소를 clone합니다.

```bash
git clone <저장소 URL>
```

2. 작업 디렉터리를 저장소 안으로 이동합니다.

```bash
%cd dcc2026-prelim
```

3. 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

4. `data/`는 git에 포함되어 있지 않으므로, Google Drive 마운트나 별도 업로드/다운로드로 데이터를 채워 넣어야 합니다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

5. 작업 후 변경 사항을 커밋/푸시하려면 Colab에서 git 사용자 정보를 설정합니다.

```bash
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git add .
git commit -m "메시지"
git push
```
d6735153a38d166c56def9e3cff8e88ee6448b4d
