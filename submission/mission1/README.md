# Mission 1 — 신고자 성별 분류

채택 모델은 M1-XVEC-001입니다. 신고자 발화를 시간순으로 이어 붙인 파형의 52차원 음향 통계와 앞 20초의 정규화된 512차원 WavLM 임베딩을 결합해 LR로 분류합니다. 입력 JSON은 `startAt`, `endAt`, `speaker`만 읽습니다.

```bash
pip install -r requirements.txt
python inference.py --audio_dir WAV_FOLDER --label_dir JSON_FOLDER --ckpt_path m1_xvec001_combined.ckpt --output outputs/mission1.csv
```

체크포인트 옆의 `xvector-model/`을 반드시 함께 제출합니다. WavLM config, feature extractor, safetensors를 포함해 추론 중 다운로드가 없습니다. 출력은 UTF-8 CSV이며 `audio file name,gender` 두 컬럼, gender는 `F` 또는 `M`입니다. 파일 이름으로 정렬하며 중복이나 누락된 WAV/JSON 짝은 오류로 처리합니다.

- 공식 Validation: 3,640통화, Accuracy **97.472527%**(3,548개 정답).
- 내부 검증: Training 통화별 층화 80:20, seed 42, Accuracy 97.380137%.
- 공식 평가: Colab Tesla T4, encoder batch size 1, 평가 1,212.628초.
- 파라미터 수: WavLM encoder 101,112,153개, LR 계수·절편 565개(학습형 파라미터 합계 101,112,718개). StandardScaler 통계는 별도입니다.
- LR 가중치 SHA256: `4fffb8aa97c2f3083b05820ca45b84ecddbc60c5162e02961546f37d133dd9d5`.
- WavLM 가중치 SHA256: `94c3defe08248d81c7b2bd0a058ea9985269cefed13076434669c47fade41182`.
- Encoder revision: `1bfd64eca136543feb28c5ffaf05381c6af33121`.

`training.ipynb`에는 원본 학습 코드와 실제 GPU 완료 기록을 함께 보존했습니다. 이 노트북 자체에서 재실행한 로그는 아닙니다. 재학습은 별도 GPU 환경과 Training TL/TS zip, 환경변수 DCC_RUN_ROOT/LAB_WORK/LAB_OUT 및 이 폴더를 가리키는 PYTHONPATH가 필요합니다. 전체 Training으로 추가 재학습하지 않았으며 공식 평가한 고정 head를 그대로 제출합니다.
