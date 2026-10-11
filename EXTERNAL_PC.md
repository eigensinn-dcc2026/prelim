# 외부 PC에서 작업 재개

`wip/library-handoff`는 팀 `main`과 외부 작업용 추가 문서를 함께 보관하는 브랜치입니다.
2026-10-11에 `main`의 `1e425778b456ccdde06d6bc59604f6021baa0065`까지 반영했습니다.
기존 M2 spectrogram 캐시 SHA256 설정, 전략 문서, M1 학습 노트북과 M3 추가 실험 기록도 유지합니다.

## 코드 받기

Git과 Python 3.11을 설치한 뒤 실행합니다. 공개 저장소의 코드 다운로드에는 GitHub 로그인이 필요하지 않습니다.

```powershell
git clone --branch wip/library-handoff https://github.com/eigensinn-dcc2026/prelim.git
cd prelim
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

이미 이 브랜치를 받았다면 수정 파일을 확인한 뒤 업데이트합니다.

```powershell
git status
git pull --ff-only origin wip/library-handoff
```

Codex에서 이 폴더를 열고 다음 내용을 전달하면 됩니다.

> EXTERNAL_PC.md와 README.md, docs/experiment_log.md, docs/decisions.md를 읽고 DCC 예선 작업을 이어가줘. 현재 브랜치와 남은 작업을 먼저 확인해줘.

새 PC에서 커밋하려면 본인의 `git config user.name`, `git config user.email`을 설정합니다.
GitHub에 변경을 올릴 때는 쓰기 권한이 있는 본인 계정으로 인증합니다.

## 학습·추론에 필요한 별도 파일

GitHub 공개 전환은 데이터·모델·미커밋 파일을 추가로 업로드하지 않습니다.
이 브랜치에는 원본 데이터, 모델 가중치, 기존 학습의 optimizer 상태와 모든 로컬 캐시가 포함되지 않습니다.

- 원본 데이터와 Colab 경로 설정: `docs/team_colab_setup.md`.
- 실험과 모델 결정 기록: `docs/experiment_log.md`, `docs/decisions.md`.
- 원래 실험의 학습 재개: 공식 데이터와 해당 실험의 Drive 체크포인트·로그를 연결합니다.
- 로컬 추론: 각 실행 스크립트/제출 폴더 README가 요구하는 체크포인트를 별도로 준비합니다.

현재 PC의 독립 제출 후보 패키지와 M1·M3 가중치를 그대로 복원하려면
[개인 백업 릴리스](https://github.com/sooyahh/dcc-lab/releases/tag/untagged-abd5745ea843a0d7547b)를 사용합니다.
이 백업은 본인 GitHub 계정 로그인이 필요한 비공개 초안 릴리스이며, 2026-10-11 저장 시점의 파일입니다.
팀 `main`에 병합된 제출 코드와 별도 백업의 제출 후보는 구성·모델이 다를 수 있으므로
실행할 패키지의 README에 맞는 가중치를 사용합니다.
백업에 기록된 M2-RERUN-001 최종 가중치와 실제 재학습 출력 노트북은 미확보 상태입니다.

GPU가 없는 PC에서는 백업의 `resume_colab.ipynb`로 Colab GPU와 기존 공식 Drive를 연결할 수 있습니다.
코드 수정만 한다면 이 공개 브랜치를 clone하는 것으로 충분합니다.

Windows에서 M2 eGeMAPS 실험의 `opensmile==2.6.0`을 사용할 때는 가상환경 설치 경로를 영문으로 둡니다.
한글이 포함된 설치 경로에서는 openSMILE 설정 파일 경로 처리 중 `UnicodeEncodeError`가 발생할 수 있습니다.
이 경로 문제는 Colab의 `/content` 환경에서는 적용되지 않습니다.

## PPT

Node.js 설치 후 저장소의 `ppt/README.md`에 따라 `ppt/build_deck.js`를 수정하고 재생성합니다.

```powershell
cd ppt
npm ci
node build_deck.js DCC2026_예선_결과정리.pptx
```

`m2/l2-spec-sha256`는 M2 캐시 해시를 채웠던 과거 작업 브랜치입니다.
그 수정은 이 브랜치에 포함되어 있으므로 외부 PC에서는 `wip/library-handoff`를 사용합니다.
