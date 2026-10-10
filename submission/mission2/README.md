# Mission 2 — 발화 화자 분류

**현재 제출 준비 미완료: 선택한 M2-RERUN-001 가중치가 없습니다.** 실행 코드와 규격은 준비했으며 아래 체크포인트를 확보하고 해시와 공식 점수를 대조한 뒤 제출해야 합니다.

선택 후보는 원본 음성과 제공된 라벨로 재학습한 SpeakerCNNLarge 3-seed soft-vote 앙상블입니다. 기록된 공식 Accuracy는 102,419/111,947 = **91.488829%**입니다. 이전 CNN의 91.55%와 지연우님의 내부 검증 87.3760%는 이 가중치의 점수로 사용하지 않습니다.

```bash
pip install -r requirements.txt
python inference.py --audio_dir WAV_FOLDER --label_dir JSON_FOLDER --ckpt_path m2_rerun001_ensemble.pt --output outputs/mission2.csv
```

출력 컬럼은 `audio file name,startAt,endAt,speaker`입니다. 시간은 JSON 값을 그대로 보존하며 speaker는 정수 0/1입니다. 예측은 각 발화의 오디오와 startAt/endAt만 사용합니다. 전사문, 정답 speaker, 발화 순서 특징은 사용하지 않습니다. JSON 순서는 출력 행 정렬에만 사용합니다.

`mission2_model.py`, `mission2_pipeline.py`, `dcc_data/`는 저장소 외부에서 실행되도록 의존 모듈을 포함합니다. 추론 로직과 mel/CNN 설정 검증은 기존 검증된 코드 그대로입니다. 필요 시 `pack_ensemble.py`로 세 seed 파일을 공식 명령에 맞는 단일 `.pt`로 묶을 수 있습니다.

```bash
python pack_ensemble.py --ckpt-path SEED1.pt --ckpt-path SEED2.pt --ckpt-path SEED3.pt --output m2_rerun001_ensemble.pt
```

찾아야 할 원래 위치: Drive `m2_cache/m2_rerun_from_raw/rerun_2026-10-06b/`. 기록된 SHA256은 앞 `19ea482a`, 끝 `3246`만 확인되어 완전한 무결성 확인에 부족합니다. 가중치 확보 후 실제 전체 SHA256, 원본 라벨 설정, 3-seed 동일 구조, 공식 Validation 재현을 확인해야 합니다.

`training_reference.ipynb`는 기존 L2 재학습 절차 참고본입니다. **M2-RERUN-001의 실행 로그가 포함된 최종 노트북으로 확인된 파일이 아닙니다.** 최종 가중치와 해당 재학습 실행 로그를 함께 보완해야 합니다. 임의의 대체 모델을 최종 모델로 넣지 않았습니다.
