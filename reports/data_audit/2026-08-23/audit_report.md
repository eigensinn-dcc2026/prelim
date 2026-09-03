# DCC 데이터 원격·로컬 무결성 감사 보고서

## 감사 범위

- Snapshot 시각(UTC): `2026-08-23T13:03:25+00:00`
- Google Drive: `https://drive.google.com/drive/folders/1-OTkaH2s5m5qnXXfBoAnXcUKRJ-h6bFI?usp=sharing`
- 원격 조회 방식: `gdown.download_folder(..., skip_download=True)`
- 원격 파일 본문 다운로드: 없음
- 로컬 원본 파일 변경: 없음
- 주의: 이 원격 manifest는 현재 Drive 상태이며 2026-08-23 당시의 역사적 snapshot을 보장하지 않는다.

## 핵심 결과

- 원격 파일: 18,280개
- 로컬 파일: 65,680개
- matched: 18,280개
- remote-only: 0개
- local-only: 47,400개
- 0-byte: 0개
- JSON parse 성공/실패: 32,840 / 0
- WAV header 성공/실패: 32,840 / 0
- SHA-256 미생성: 0개

### WAV 포맷 스펙 (전수 집계, 2026-08-27)

위 "WAV header 성공/실패" 검사는 파일을 열 수 있었는지만 집계했고, 실제 채널·샘플레이트·비트뎁스 값은 여태 요약되지 않았다.
이번에 `local_manifest.csv`에 이미 파일별로 기록되어 있던 `wav_channels`, `wav_sample_rate` 값을 전수 집계했다.

- 전체 32,840개 WAV 전량에서 **1채널(mono), 8,000 Hz**로 통일됨 (예외 0건)
- 비트 뎁스(bit depth)는 `wave` 표준 라이브러리로 8개 파일을 표본 확인한 결과 **16-bit(2 byte) PCM**로 확인됨 (전수는 아님)
- 참고: `jangsoo/eda` 브랜치의 `scripts/eda_dcc_dataset.py` Phase 1이 파일별 `channels`/`sample_rate`/`sample_width`/`frame_count`/`duration_s`를 캐싱하는 `wav_header_cache.csv`를 새로 만들고 있으므로, 이 캐시를 전수로 돌리면 비트 뎁스도 전수 확인으로 승격할 수 있다.

이 스펙은 Mission 1·2 전처리에서 `sample_rate` 통일(resampling) 단계를 생략해도 된다는 뜻이므로, baseline 구현 속도를 높이는 데 바로 쓸 수 있다.

### 원격 split·지역·형식별 수량

| split | region | extension | count |
|---|---|---|---|
| Training | 서울 | .json | 5500 |
| Training | 서울 | .wav | 5500 |
| Validation | 서울 | .json | 3640 |
| Validation | 서울 | .wav | 3640 |

### 로컬 split·지역·형식별 수량

| split | region | extension | count |
|---|---|---|---|
| Training | 서울 | .json | 29200 |
| Training | 서울 | .wav | 29200 |
| Validation | 서울 | .json | 3640 |
| Validation | 서울 | .wav | 3640 |

### Local-only 분포

| split | region | extension | count |
|---|---|---|---|
| Training | 서울 | .json | 23700 |
| Training | 서울 | .wav | 23700 |

### Local-only 다운로드 조각별 분포

| download_chunk | split | region | extension | count |
|---|---|---|---|---|
| 대학부 데이터-20260823T080055Z-1-001 | Training | 서울 | .json | 4575 |
| 대학부 데이터-20260823T080055Z-1-001 | Training | 서울 | .wav | 1909 |
| 대학부 데이터-20260823T080055Z-1-002 | Training | 서울 | .json | 3310 |
| 대학부 데이터-20260823T080055Z-1-002 | Training | 서울 | .wav | 1871 |
| 대학부 데이터-20260823T080055Z-1-003 | Training | 서울 | .json | 2857 |
| 대학부 데이터-20260823T080055Z-1-003 | Training | 서울 | .wav | 1755 |
| 대학부 데이터-20260823T080055Z-1-004 | Training | 서울 | .json | 2583 |
| 대학부 데이터-20260823T080055Z-1-004 | Training | 서울 | .wav | 1703 |
| 대학부 데이터-20260823T080055Z-1-005 | Training | 서울 | .json | 2349 |
| 대학부 데이터-20260823T080055Z-1-005 | Training | 서울 | .wav | 2432 |
| 대학부 데이터-20260823T080055Z-1-006 | Training | 서울 | .json | 1988 |
| 대학부 데이터-20260823T080055Z-1-006 | Training | 서울 | .wav | 2556 |
| 대학부 데이터-20260823T080055Z-1-007 | Training | 서울 | .json | 1707 |
| 대학부 데이터-20260823T080055Z-1-007 | Training | 서울 | .wav | 2505 |
| 대학부 데이터-20260823T080055Z-1-008 | Training | 서울 | .json | 1433 |
| 대학부 데이터-20260823T080055Z-1-008 | Training | 서울 | .wav | 2378 |
| 대학부 데이터-20260823T080055Z-1-009 | Training | 서울 | .json | 1190 |
| 대학부 데이터-20260823T080055Z-1-009 | Training | 서울 | .wav | 2300 |
| 대학부 데이터-20260823T080055Z-1-010 | Training | 서울 | .json | 465 |
| 대학부 데이터-20260823T080055Z-1-010 | Training | 서울 | .wav | 1856 |
| 대학부 데이터-20260823T080055Z-1-011 | Training | 서울 | .json | 738 |
| 대학부 데이터-20260823T080055Z-1-011 | Training | 서울 | .wav | 2162 |
| 대학부 데이터-20260823T080055Z-1-012 | Training | 서울 | .json | 505 |
| 대학부 데이터-20260823T080055Z-1-012 | Training | 서울 | .wav | 273 |

> WAV와 대응 JSON이 서로 다른 다운로드 조각에 존재할 수 있으므로 위 조각별 수량은 파일 단위 집계다.

## Pairing

```json
{
  "matched": 32840
}
```

## 중복 그룹

```json
{}
```

내용 중복은 삭제 근거로 사용하지 않는다.

## 무결성 이슈

```json
{}
```

WAV 표준 라이브러리가 지원하지 않는 codec은 손상으로 자동 판정하지 않는다.

## 공식 배포와 학습 허용 상태

- `official_distribution_status`와 `training_eligibility_status`는 독립적으로 기록했다.
- 현재 원격 일치 파일은 `current_remote_candidate`이며 주최측 확인 전까지 공식 확정 상태가 아니다.
- 로컬 전용 파일은 `local_only_unverified`이며 삭제·격리·이동 대상이라고 판정하지 않는다.
- Validation은 학습 불가로 표시한다.
- 서울 Training도 공식 배포 범위가 확인되기 전까지 `pending_official_distribution`이다.

## 구조 변경 게이트

- [ ] 공식 Training 범위에 대한 주최측 확인 또는 공식 manifest 확보
- [x] 공식 배포 상태와 학습 허용 상태를 별도 컬럼으로 기록
- [x] split·지역·하위 경로·다운로드 조각별 집계
- [x] WAV-JSON exact stem pairing 감사
- [x] 파일명·상대 경로 중복 감사
- [x] SHA-256 내용 중복 감사
- [x] 0-byte 검사
- [x] JSON 인코딩·parse 검사
- [x] WAV header 검사
- [x] 현재 원격 manifest의 모든 파일이 로컬에 존재
- [x] local-only의 경로·지역·조각별 분포 기록
- [x] 로컬 SHA-256 및 provenance manifest 저장

**현재 구조 변경 게이트: 통과하지 않음.** 주최측 확인 또는 공식 manifest가 필요하다.

## 주최측 문의 문안

```text
안녕하세요.

현재 제공된 Google Drive 공유 폴더를 확인하면 다음과 같습니다.

- Training: WAV 5,500개, JSON 5,500개
- Validation: WAV 3,640개, JSON 3,640개

그러나 2026-08-23에 동일 공유 Drive의 ‘대학부 데이터’ 전체 폴더를
다운로드한 로컬 사본에는 다음 파일이 있습니다.

- Training: WAV 29,200개, JSON 29,200개
- Validation: WAV 3,640개, JSON 3,640개

현재 원격 파일 중 로컬에 없는 파일은 확인되지 않았으며,
Training에서만 로컬에 WAV/JSON 23,700쌍이 추가로 존재합니다.
로컬 데이터는 Google Drive 전체 폴더 다운로드 과정에서
‘대학부 데이터-20260823T080055Z-1-001’부터 ‘-012’까지의
여러 다운로드 조각으로 나뉘어 저장됐습니다.

아래 내용을 확인 부탁드립니다.

1. 공식적으로 배포된 Training 데이터의 전체 수량은 몇 쌍인가요?
2. 2026-08-23 전후 공유 Drive의 Training 폴더 구성 변경이 있었나요?
3. 29,200쌍이 당시 공식 배포본이었다면 전체를 공식 배포 원본으로
   보존해야 하는지 확인 부탁드립니다.
4. 29,200쌍 안에 서울 외 지역 데이터가 포함되어 있다면 해당 데이터도
   공식 배포 원본에는 포함되지만 실제 모델 학습에는 사용할 수 없는
   데이터로 이해하면 될까요?
5. 실제 모델 Training에 사용할 수 있는 범위는 문제 안내대로
   서울 Training 데이터만인지 다시 확인 부탁드립니다.
6. 가능하다면 공식 파일 manifest 또는 파일명·크기·checksum 목록을
   제공해 주실 수 있을까요?

감사합니다.
```

## 답변별 후속 처리

```text
주최측 답변 또는 공식 manifest 확보
├─ Case A: 29,200쌍 전체가 공식 배포본
│  ├─ 전체 원본 보존
│  ├─ 서울 Training만 학습 적격 판정
│  └─ 서울 외 데이터는 보존하되 학습 경로 차단
├─ Case B: 공식 Training이 5,500쌍
│  ├─ 현재 원격 일치 항목을 official candidate로 유지
│  └─ local-only 23,700쌍은 이동·삭제 없이 unverified 유지
├─ Case C: 공식 manifest/checksum 제공
│  ├─ 공식 목록을 최우선 기준으로 재감사
│  └─ 누락·불일치 항목을 덮어쓰지 않고 별도 보고
└─ Case D: 수량 확답 없음
   ├─ 001~012 원본과 양쪽 snapshot manifest 유지
   └─ 불명확한 데이터는 학습 경로에서 차단
```
