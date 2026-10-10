import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob, PresentationFile} from 'file:///C:/Users/sooyah/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/sooyah/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'outputs/submission_deck');
const skill = 'C:/Users/sooyah/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const p = await PresentationFile.importPptx(await FileBlob.load(path.join(root, '../wt-ppt-latest/ppt/DCC2026_예선_결과정리.pptx')));
const records = (await fs.readFile(path.join(out,'source-inspect.ndjson'),'utf8')).trim().split('\n').map(x=>JSON.parse(x));
const changes = new Map([
  ['성별은 목소리 높낮이 차이가 커서 — 단순하고 해석 가능한 모델로 충분히 높은 정확도를 노림','WavLM 화자 임베딩과 음향 통계를 결합해 공식 성별 분류 성능을 개선'],
  ['음높이·음색 등\n52차원 통계','52D 음향 통계\n+ 512D WavLM'],
  ['성별 차이는 음높이 같은 저수준 음향 특징에 이미 크게 드러남\nGBM과 정확도 차이가 0.19%p로 작아 더 단순하고 해석 쉬운 LR 채택\nx-vector+52D LR은 내부 97.38%, 공식 평가 전이므로 LR 95.11%와 구분','신고자 구간 전체의 음향 통계와 앞 20초 화자 임베딩을 결합\n동일 고정 LR head로 내부 97.38%, 공식 97.47% 확인\n기존 52D LR 공식 95.11% 대비 +2.36%p로 채택'],
  ['공식 Validation LR 95.11%, x-vector 결합 97.38%는 내부 검증 결과','WavLM+52D LR 공식 97.47%, 기존 52D LR 95.11% 대비 +2.36%p'],
  ['95.11%','97.47%'],
  ['95% 신뢰구간\n[94.40, 95.82]','3,548 / 3,640 정답\n오분류 92건'],
  ['오류가 몰린 곳  178건 오분류, 길이·발화 수와 단조 관계는 불명확\n과적합 점검  내부 94.8~95.0% vs 공식 95.1% — 차이 작음','오류 감소  기존 178건에서 92건으로 감소\n분할 검증  내부 97.38%, 공식 97.47%\n성별 recall  여성 97.96%, 남성 96.91%'],
  ['52D + StandardScaler + LR\n공식 Val 3,640통화, seed 42','564D + StandardScaler + LR\nT4 batch 1, 공식 평가 1,212.6초'],
  ['근거: reports/mission1/ m1_val_eval.json, m1_val_bootstrap_ci.json, M1_error_analysis_report.md','근거: reports/mission1/m1_xvec001_official_validation.json (10/10 GPU 평가)'],
  ['이전 91.55%는 참고용\n원본부터 재학습하고\n에폭별 로그 보존\n최종 후보 비교 대기','이전 91.55%는 참고용\n원본 재학습 후보 유지\n최종 가중치 확보 필요\n공식 점수 재현 대기'],
  ['공식 Val은 학습에 쓰지 않음. 현 제출본 KCBERT-005와 교체 후보 RoBERTa를 구분','공식 Val은 학습에 쓰지 않음. seed 42 RoBERTa 단일 모델 채택'],
  ['RoBERTa 공식 Macro F1 0.6433, 현 제출본 KcBERT 0.6246 대비 +0.0187','RoBERTa 단일 모델 채택, 공식 Macro F1 0.6433 (+0.0187)'],
  ['공식 Val 3,640건, 교체 후보','공식 Val 3,640건, 채택'],
  ['3모델 분석 0.6510\n단독 대비 +0.0077\n채택 기준 +0.01 미달\n최종 채택 미정','3모델 분석 0.6510\n단독 대비 +0.0077\n채택 기준 +0.01 미달\n앙상블 기각'],
  ['RoBERTa seed 42, .pt 약 1.35GB\nCPU 공식 Val 2,098초(약 0.58초/통화)','336,665,609 파라미터, batch 32\nCPU 재검증 5,477.6초, M1 동시 실행'],
  ['진행 중 (결과 미확정): Qwen LoRA #116, 보조 손실 개선 #127','제출 모델: RoBERTa-003 seed 42. 추가 실험은 이번 제출에 반영하지 않음'],
  ['남은 제출 작업: 후보 확정, 체크포인트·학습 로그 동봉, 새 환경에서 inference.py 재실행','제출 폴더에 코드·가중치·실제 학습 기록 동봉, 별도 새 설치 환경 확인 필요'],
  ['10/14  예선 결과물 제출, 교육 진도 확인\n모델  M1 x-vector 공식 평가, M2 비교, M3 후보 확정\n제출물  학습 출력·가중치·코드·requirements·PPT\n10/28  본선 진출팀 발표 (공식 일정)','10/14  예선 결과물 제출, 교육 진도 확인\nM1·M3  선택 완료, 미션별 제출 폴더 준비\nM2  선택 CNN 가중치·실행 학습 로그 확보 필요\n10/28  본선 진출팀 발표 (공식 일정)'],
]);
for (const record of records) {
  for (const e of record.elements ?? []) {
    if (e.kind === 'textbox') {
      for (const [before, after] of changes) {
        if (e.text?.includes(before)) p.resolve(e.id).text.replace(before, after);
      }
    }
  }
}
const chart = p.resolve('ch/10325wja');
chart.categories = ['52D LR','WavLM+52D LR'];
chart.series.getItemAt(0).values = [95.10989010989,97.472527472527];
chart.yAxis = {...chart.yAxis,min:90,max:100,majorUnit:2};
const m3chart = p.slides.getItem(14).charts.items[0];
m3chart.categories = ['TF-IDF','KcBERT-005','KcBERT-012','RoBERTa-003'];
p.resolve('tb/z65wfe58').getCell(4,3).text.replace('최종 채택 미정','기각 (+0.01 미달)');
const summary = p.resolve('tb/fmd032ls');
summary.getCell(1,2).text.replace('음향 특징 + LR','WavLM+52D LR');
summary.getCell(1,4).text.replace('95.11%','97.47%');
summary.getCell(1,5).text.replace('1차 확정','채택');
summary.getCell(2,5).text.replace('최종 비교 대기','가중치 확보 필요');
summary.getCell(3,5).text.replace('제출본 교체 후보','채택');
for (const n of [9,10,12,14,15,16,17,18]) {
  const slide = p.slides.getItem(n-1);
  slide.speakerNotes.text = (records[n-1].notes?.text ?? '') + '\n최종 선택 반영: 2026-10-10. 로컬 근거: reports/mission1/m1_xvec001_official_validation.json, reports/mission2/m2_final_candidate_comparison.json, reports/mission3/official_val/roberta003_submission_check.json. M2 최종 가중치 미확보.';
}
const candidate = path.join(out,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(path.join(out,'after-inspect.ndjson'),(await p.inspect({select:'$.slides[*]',include:['id','index','elements.id','elements.kind','elements.text','notes.text']})).ndjson);
for (let i=0;i<p.slides.items.length;i++) {
  const blob = await p.export({slide:p.slides.getItem(i),format:'png',scale:1});
  await fs.writeFile(path.join(out,`after-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
// Restore the original embedded Excel parts with repair_submission_chart_workbooks.py,
// then validate/export with finalize_submission_deck.mjs.
console.log('Updated 19-slide candidate and slide previews. Workbook restoration/finalization next.');
