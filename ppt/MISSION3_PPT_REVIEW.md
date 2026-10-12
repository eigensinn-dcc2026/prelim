# Mission 3 PPT 변경 검토 — 2026-10-12

> **상태: Draft / 병합 금지.** 제출까지의 시간을 고려해 우선 `ppt/build_deck.js`의 Mission 3 부분만 최신 결과로 갱신한 검토용 변경입니다.
> `ppt/DCC2026_예선_결과정리.pptx`는 **아직 이 브랜치의 코드로 재생성·커밋하지 않았습니다.**
> `ppt/README.md`의 '제작 코드와 PPTX 함께 제출' 조건을 충족하기 전에는 Ready for Review / merge하지 않습니다.

## 변경 범위

- **18p**: `utterances[].text`만 사용, 발화 입력 순서 유지, 잘못된 시간순 정렬 설명 제거. 최종 RoBERTa-large 3-seed + TF-IDF 구조와 0.55:0.45 결합 비율 표시.
- **19p**: 공식 Validation Macro F1 `0.6534 = 65.34%`, Micro F1 0.6690, 3,640건 중 빈 예측 149건. A100 GPU 109초, CPU 203건 표본 462초(모델 로딩 포함), 모델 약 4.05GB. 오래된 단일모델 증상별 F1 그래프 자리는 기록에 기반한 실험 단계별 점수 그래프로 교체(학습 조건이 달라 인과적 개선 효과로 해석 불가).
- **20p**: 과거 기각된 세 모델 앙상블(0.6510)과 최종 채택된 RoBERTa 3-seed+TF-IDF(0.6534)를 구별. 공식 Validation은 최종 평가 전용.
- **21p**: **Mission 3 행만** 최종 모델 `RoBERTa×3 + TF-IDF`, 공식 Val `65.34%`(Macro F1), 상태 `최종 확정`으로 변경. Mission 1/2의 행은 그대로 둠.
- **22p**: Mission 3 향후 계획만 오심·두통 오류 분석 / 최종 앙상블 추론 효율 개선으로 변경.
- **기타**: Mission 1/2의 원래 코드 블록, 공식 표지·마무리 및 템플릿 에셋 **미변경**.

## 근거

- `submission/mission3/README.md` (M3-FINAL-002)
- `submission/mission3/m3_final_train.executed.ipynb` (실제 학습·공식 Val 로그)
- `docs/decisions.md` (2026-10-11 최종 모델 채택 기록)
- `docs/submission_format.md` (전사 텍스트 전용, threshold 및 제출 형식)

## 병합 전 꼭 남은 작업

- [ ] **이 `ppt/build_deck.js`로 PPTX 재생성**하고 `ppt/DCC2026_예선_결과정리.pptx`를 같은 PR에 커밋. 원래의 23장 PPT를 임의로 다른 소스에서 만든 파일로 교체하지 말 것.
- [ ] 재생성한 PPTX와 기존 23장 제공본을 비교. 현재 main의 M1/M2 페이지 및 수치는 제공받은 23장 검토본과 다르므로 담당자 내용 유실·버전 역행 여부 확인.
- [ ] 열려 있는 PPT PR #64의 Mission 2 작업 통합 방침 확인.
- [ ] 18~22p 글자 넘침·차트·수치 및 전체 페이지 수 육안 검토.
- [ ] 자동 판정 `merge-gate/verdict`가 `병합-가능`인지 확인.

## 재생성 명령 (프로젝트 표준)

```bash
cd ppt
npm install
node build_deck.js DCC2026_예선_결과정리.pptx
```

**주의:** 기존 사용자 제공 23장 PPT를 바탕으로 별도로 복구한 제작 소스 검토본은 ChatGPT 대화에 보관한 *별도 후보*이며, 이 PR의 `build_deck.js`에서 만든 PPTX가 아닙니다. 두 결과물을 혼동하거나 무검토로 덮어쓰지 마세요.
