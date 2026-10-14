# M1-XVEC-002(c) 규정 점검 — 2026-10-11

모델 구조와 확인한 학습·평가 코드에서 입력 제한, 외부 데이터 추가, Validation 학습 위반은 발견하지 않았다. 다중 창 평균은 M1의 통화 단위 음성 분류와 양립한다. 다만 제출 구현의 판정 경계값과 패키징은 보완해야 하며, 가중치 실물을 확인한 최종 제출 적합성 판정은 아니다.

## 근거와 범위

- 원본 `공식전달사항/0820_DCC 대학부_예선_출제문제.pdf`(저장소 밖 보관본)의 5·6·7·11·12쪽 텍스트를 직접 확인하고 7쪽 렌더도 대조했다.
- 외부 사전학습 모델·Validation 활용·Decision Threshold 추가 공지는 `dcc2026-prelim/competition_rule/DCC_2026_대학부_예선_문제명세_공식대조본.md`의 기록을 사용했다. 이번 점검에서 원본 Q&A 게시물/공지 메일을 새로 조회하지 않았다.
- 학습 코드: 로컬 Git의 `origin/sooyah/lab-m1-xvec-002`, 커밋 `bb9921b0a2f12e7d8d2cc0f2053f53221cf41a91`, `scripts/lab/m1-xvec-002.py`.
- 공식 평가 코드: `dcc2026-prelim/scripts/lab/m1-xvec-002-offval.py`.
- 공통 입력 처리·LR 구성: `dcc2026-prelim/src/dcc_data/mission1.py`, `mission1_train.py`, `mission2_features.py`.
- 현재 제출 폴더는 여전히 XVEC-001이다. XVEC-002의 Drive 가중치는 다운로드하거나 실행하지 않았다. 따라서 소스 검토만으로 실제 best.ckpt의 학습 출처·해시를 확정하지 않는다.

## 조항별 확인

| 항목 | 판정 및 증거 |
|---|---|
| M1 허용 입력 | 신고자 음성 구간을 startAt/endAt/speaker로 선택. 특징은 WAV에서 계산한 52D와 WavLM 512D. text/symptom/address/gender를 예측 특징으로 사용하지 않음 |
| gender·call_id 접근 | 학습 gender는 정답/층화 분할에만 사용. call_id는 중복·분할·행 식별에만 사용. 공식 평가 gender는 예측을 확정한 뒤 채점할 때 읽음 |
| 여러 신고자 발화 및 창 평균 | M1은 통화 단위 신고자 성별 분류. PDF 7쪽은 긴 음성 sliding window를 안내. M2의 한 발화 독립 입력 제한을 M1에 적용할 근거 없음 |
| 전처리 통일 | 신고자 음성을 시간순 연결, 앞 최대 80초를 20초씩 최대 4창으로 분할. 이후 창의 3초 미만 꼬리는 제외. 창마다 정규화 → 길이 가중 평균 → 재정규화. Training/Validation에 같은 방식 적용하며 정답별 분기 없음 |
| 사전학습 모델 | 공개 microsoft/wavlm-base-plus-sv, 고정 revision 1bfd64eca136543feb28c5ffaf05381c6af33121. 로컬 특징 추출, 외부 상용 추론 API 호출 없음. 공개 사전학습 모델 사용은 추가 Q&A 기록상 허용 |
| 학습 데이터 | 서울 Training TL/TS만 사용, 29,200통화 중 23,360통화로 LR+StandardScaler fit. Training 내부 holdout 5,840통화. call_id 중복/분할 겹침 검사 |
| 공식 Validation | 학습된 Pipeline을 load 후 predict만 수행. fit/보정/재학습 없음. 3,640통화의 정답은 채점용으로만 사용 |
| 전체 Training 재학습 | 제출 전 반드시 100%로 재학습해야 한다는 조항은 확인되지 않음. 기존 80% 학습 가중치라는 이유 자체가 규정 위반은 아님 |

## 보완 1: 정확히 0.5인 판정

공식 공지 대조본은 prediction >= 0.5를 클래스 1로 정한다. 학습 평가 코드와 공식 평가 코드는 Pipeline.predict를 사용한다. scikit-learn 이진 LogisticRegression.predict는 동률에서 클래스 0을 반환한다. 이번 로컬 재현에서 계수·절편이 0인 LR은 predict_proba=[[0.5,0.5]], predict=[0]이었고, 명시적 >=0.5 판정은 [1]이었다.

제출에서는 클래스 순서를 확인한 뒤 `predict_proba`의 클래스 1(M) 열을 `>= 0.5`로 비교해 F/M을 출력하도록 구현한다. 이 변경은 threshold tuning이 아니라 공지의 경계값 구현이다. 실제 XVEC-002 가중치에서 동률이 발생하는지와 기존 98.13% 점수가 달라지는지는 확인하지 못했다. 실제 가중치 확보 후 확인해야 한다.

## 보완 2: 평가 스크립트를 그대로 제출하지 않기

공식 평가 스크립트는 --out/LAB_OUT 등 실험 환경을 요구하고, 정답을 읽어 채점하며 call_id,stem,prediction,target CSV를 만든다. 모델 캐시가 없으면 Hugging Face 다운로드, sklearn 버전 폴더가 없으면 pip 설치를 수행한다. 이는 평가 작업 코드로서는 역할에 맞지만 제출용 오프라인 inference.py가 아니다.

별도 제출 코드에 동일 특징 추출과 LR을 옮겨 공식 --audio_dir/--label_dir/--ckpt_path/--output 명령 및 audio file name,gender CSV를 제공해야 한다. LR best.ckpt, WavLM config/processor/weights, 고정 requirements, 학습 기록과 PPT를 포함하고 빈 모델 캐시·네트워크 차단 상태에서 실행 검증한다. 기존 XVEC-001 제출 코드는 앞 20초 특징과 고정 head 해시를 사용하므로 가중치만 바꾸면 안 된다.

## 결론

M1-XVEC-002(c)의 다중 창 평균 + 52D + LR 방식은 확인한 규정과 양립한다. 제출 코드의 정확히 0.5 판정과 독립 오프라인 패키지를 보완하고, 실제 가중치·학습 기록·코드 대응 및 해시를 확인한 뒤 최종 제출 준비 완료로 표시한다.
