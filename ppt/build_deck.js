// DCC 2026 대학부 예선 결과 정리 PPT — EDA 2장, Mission 2 지연우 담당 10장 및 Mission 3 별첨 5장 포함, 총 38장
// 디자인 규칙은 ppt/design_guide.md, 공식 템플릿 이미지는 ppt/figures/tpl_*.png
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const JSZip = require("jszip");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "DCC 2026 대학부 예선 결과 정리 — eigensinn";

// ---------- 디자인 토큰 (가이드 1.1 + 공식 템플릿 브랜드 컬러에 맞춤) ----------
// blue(네이비)·sky(버튼 블루)는 D+AICC 대학부 공식 템플릿에서 실측(2026.10.02)한 색상으로 교체
const C = {
  blue: "2F4E91", sky: "009DDA", light: "EAF3FA",
  mint: "8BC34A", mintLight: "E8F5E9",
  red: "E53935", pink: "FFCDD2",
  bgGray: "F8F9FA", text: "212121", sub: "616161", line: "E0E0E0",
  white: "FFFFFF", bigNum: "DCE3EC",
};
// 공식 템플릿 자산 (D+AICC PPT 대학부 템플릿.pptx에서 추출)
const TPL = {
  cover: "figures/tpl_cover.png",     // 표지 (슬라이드1) — 수정 없이 그대로 사용
  header: "figures/tpl_header.png",   // 본문 상단 띠 (슬라이드2에서 크롭) — 모든 본문 장에 적용
  closing: "figures/tpl_closing.png", // 마무리 (슬라이드4) — 수정 없이 그대로 사용
};
const HB = 0.704; // 공식 헤더 띠 높이 (13.333 x 7.5 캔버스 기준, 원본 비율 그대로)
const F = "맑은 고딕";
const W = 13.333, M = 0.5, CW = W - 2 * M;
const TEAM = "eigensinn";
const IC = "figures/icons/"; // ppt/tools/make_icons.py 로 생성

// ---------- Mission 2 교체 지점 (원고의 ⟪M2 교체⟫) ----------
// M2 제출 모델을 바꾸면 이 블록만 고친다. 12·13·14·15·30·32장이 모두 여기서 읽는다.
// 현재 값: CNN 3-seed 앙상블 + WavLM-large 2개 확률 가중 평균 (submission/mission2/README.md, experiment_log.md M2-BLEND-001)
const M2 = {
  // 12장 문제·제약·최종 모델
  core12: "발화 하나만 보고 신고자/대원을 가린다. CNN 3개 + WavLM-large 2개 확률 가중 평균, 공식 Validation 92.90%.",
  // 구조도 (flow 함수 입력): 앞 단계 → 앙상블 멤버 → 판정
  flow: {
    pre: [["발화 구간", "startAt~endAt만"], ["그 발화만 사용", "text·순서·다른 발화 X"]], preW: 1.95,
    branches: [["CNN 3개 평균", "log-mel · 300프레임", "× 0.4"], ["WavLM-large 2개", "16 kHz · 3초 창", "× 0.3 + 0.3"]], brW: 2.6, brGap: 0.1,
    post: [["가중 평균", "≥ 0.5 → 신고자\n< 0.5 → 대원"]], postW: 2.0,
  },
  why: [
    ["음향 차이", "대원 헤드셋·정해진 말투\n신고자 휴대전화·잡음"],
    ["단계적 검증", "통계 특징(L1) → CNN(L2)\n→ WavLM 결합"],
    ["두 종류 결합", "CNN 처음부터 학습 +\n공개 사전학습 WavLM 미세조정"],
  ],
  // 13장 성능 개선 과정 (막대 차트: 공식 Validation Accuracy)
  core13: "86.44% → 91.49% → 92.90%. CNN의 이득은 겹친 구간에 몰렸고, 사전학습 WavLM을 더해 +1.41%p를 더 얻었다.",
  chart: { labels: ["L1 LightGBM", "L2 CNN\n3-seed 앙상블", "CNN 3개\n+ WavLM 2개"], values: [86.44, 91.49, 92.90] },
  gainTitle: "■ CNN의 이득이 몰린 곳",
  gains: [["약 +7%p", "겹친 구간"], ["약 +2%p", "겹치지 않은 구간"]],
  gainNote: "통계 특징(L1) 대비 CNN 단일 모델의 개선 폭 (공식 Validation)",
  finding: "제출 방식(한 발화씩 추론)으로 바꾸자 정확도가 떨어졌다. 원인은 학습 때의 배치 패딩과 추론 방식 차이였고, 모든 발화를 자기 자신만으로 같은 길이까지 채우도록 맞춰 회복했다.",
  // 14장 다른 접근 — 마지막 행이 최종 채택한 WavLM-large 미세조정 + CNN 결합
  core14: "CNN 하나로 끝내지 않고 여러 방향을 검증했다. 효과가 없거나 규칙에 어긋난 것은 이유와 함께 기록했다.",
  lastRow: ["WavLM-large 미세조정 2개 + CNN 결합", "채택", "내부 +1.27%p (공식 Val 92.90%)"],
  // 15장 최종 결과와 보완할 점
  core15: "공식 Validation 92.90%. 제출 폴더 그대로 GPU에서 검증했고, 코드 정리·속도 개선 전후 판정이 모두 같았다.",
  acc: "92.90%",
  recalls: [["88.36%", "대원 recall"], ["97.10%", "신고자 recall"]],
  repro: "제출 폴더 그대로 다시 실행해도 같은 결과\n(92.90%, 111,947발화 판정 차이 0건)",
  improve: "대원 recall이 더 낮다.\n\"네\", \"여보세요\" 같은 짧은 응답은 두 화자가 비슷하게 들리고, 규칙상 통화 맥락을 쓸 수 없다.",
  // 30장 성능 요약 표, 32장 향후 계획
  summary: { task: "M2 신고자/대원", model: "CNN × 3 + WavLM-large × 2", val: "92.90%", gain: "+6.46%p" },
  future: "짧은 응답(\"네\"·\"예\") 보강 — 오답의 약 78%가 모든 모델 공통",
  evidence: "submission/mission2/README.md, docs/experiment_log.md M2-BLEND-001·M2-WLML-001·002·M2-RERUN-001·M2-RETRAIN-001",
  // 슬라이드 노트·캡션의 근거 (13·15장)
  evidence13: "근거: docs/experiment_log.md M2-RETRAIN-001, M2-RERUN-001, M2-BLEND-001, M2-COMPARE-002, M2-PAD-001·002",
  notes13: "근거: L1 86.44%(M2-RETRAIN-001, 제공 라벨 재학습 L1) / CNN 단일 90.98%(M2-ENS-003 ts1) / 3-seed 앙상블 91.49%(M2-RERUN-001) / CNN 3개 + WavLM-large 2개 92.90%(M2-BLEND-001, submission/mission2/README.md). 겹침 구간: M2-COMPARE-002 — 겹침 L1 81.83%→L2 88.88%(+7.05%p), 안 겹침 90.01%→92.44%(+2.43%p), L2 단일 모델 기준. 패딩: M2-PAD-001(발화별 300프레임 채우기), M2-VAL-005, M2-PAD-002.",
  notes15: "근거: submission/mission2/README.md '결과와 계산 효율성' 표(92.90% = 103,994/111,947, 대원 recall 88.36%, 신고자 recall 97.10%). 검증: submission/mission2/verify_mission2_final_on_gpu.ipynb(제출 폴더 그대로 Colab A100, requirements만·인터넷 차단, 코드 정리·속도 개선 전후 판정 동일), PR #216·#222. 비율(CNN 0.4 / WavLM 0.3·0.3)은 Training 내부 20,000발화로 고름(공식 Val로 고르지 않음).",
};

const shadow = () => ({ type: "outer", blur: 5, offset: 1, angle: 90, color: "000000", opacity: 0.1 });

// ---------- 공통 요소 ----------
function txt(slide, text, o) {
  slide.addText(text, Object.assign({ isTextBox: true, fontFace: F, fontSize: 14, color: C.text, margin: 0, valign: "top" }, o));
}
// 공식 템플릿 상단 띠(HB=0.704in) + 그 아래 섹션 라벨·구분선·소제목.
// 헤더 블록 전체 높이는 1.43in — 핵심 메시지는 1.6, 본문은 2.45부터.
function header(slide, sec, secName, sub, page) {
  slide.addImage({ path: TPL.header, x: 0, y: 0, w: W, h: HB, sizing: { type: "cover", w: W, h: HB } });
  txt(slide, `${sec}  ${secName}`, { x: M, y: HB + 0.04, w: 8, h: 0.26, fontSize: 12.5, bold: true, color: C.blue });
  txt(slide, `${TEAM}  |  ${pres._slides.length}`, { x: W - M - 3, y: HB + 0.04, w: 3, h: 0.26, fontSize: 10, color: C.sub, align: "right" });
  slide.addShape(pres.shapes.LINE, { x: M, y: HB + 0.32, w: CW, h: 0, line: { color: C.line, width: 1 } });
  txt(slide, sub, { x: M, y: HB + 0.37, w: 11, h: 0.36, fontSize: 19, bold: true });
}
// 원고의 "핵심 메시지" 한 줄 — 서면 심사라 이 줄만 읽어도 결론이 보이게
function coreMsg(slide, runs, y = 1.6) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y, w: CW, h: 0.6, fill: { color: C.light }, rectRadius: 0.1, line: { color: C.light } });
  txt(slide, runs, { x: M + 0.2, y, w: CW - 0.4, h: 0.6, fontSize: 15, bold: true, color: C.blue, align: "center", valign: "middle" });
}
function caption(slide, t, y = 7.0) {
  txt(slide, t, { x: M, y, w: CW, h: 0.3, fontSize: 10, color: C.sub });
}
// 스타일 B 카드: 흰 배경 + 연회색 테두리 + 미세 그림자
function card(slide, x, y, w, h, fill = C.white) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: 0.1, line: { color: C.line, width: 1 }, shadow: shadow() });
}
// 제목바가 달린 프로세스 박스 (가이드 3.5)
function stepBox(slide, x, y, w, h, title, body, barColor = C.blue, bodySize = 12, icon = null) {
  card(slide, x, y, w, h);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.5, fill: { color: barColor }, rectRadius: 0.1, line: { color: barColor } });
  slide.addShape(pres.shapes.RECTANGLE, { x, y: 0.3 + y, w, h: 0.2, fill: { color: barColor }, line: { color: barColor } });
  txt(slide, title, { x, y, w, h: 0.5, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle" });
  if (icon) {
    slide.addImage({ path: icon, x: x + w / 2 - 0.32, y: y + 0.68, w: 0.64, h: 0.64 });
    txt(slide, body, { x: x + 0.12, y: y + 1.4, w: w - 0.24, h: h - 1.5, fontSize: bodySize, color: C.text, align: "center", valign: "top", paraSpaceAfter: 4 });
    return;
  }
  txt(slide, body, { x: x + 0.12, y: y + 0.58, w: w - 0.24, h: h - 0.66, fontSize: bodySize, color: C.text, align: "center", valign: "middle", paraSpaceAfter: 4 });
}
function arrow(slide, x, y, w = 0.28, h = 0.4, color = C.sky) {
  slide.addShape(pres.shapes.CHEVRON, { x, y, w, h, fill: { color }, line: { color } });
}
// 가로로 늘어선 단계 박스 + 사이 화살표
function steps(slide, items, y, h, o = {}) {
  const n = items.length, bw = o.bw || 2.1, gap = (CW - n * bw) / (n - 1);
  items.forEach(([t, b, col], i) => {
    const x = M + i * (bw + gap);
    stepBox(slide, x, y, bw, h, t, b, col || (i === n - 1 ? C.sky : C.blue), o.bodySize, o.icons ? o.icons[i] : null);
    if (i < n - 1) arrow(slide, x + bw + (gap - 0.24) / 2, y + h / 2 - 0.19, 0.24, 0.38);
  });
}
// 실제 EDA 그림 삽입 (흰 배경 PNG를 카드 안에 맞춤)
function fig(slide, path, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.white }, line: { color: C.line, width: 1 } });
  slide.addImage({ path, x: x + 0.08, y: y + 0.08, w: w - 0.16, h: h - 0.16, sizing: { type: "contain", w: w - 0.16, h: h - 0.16 } });
}
function insight(slide, x, y, w, h, runs, fill = C.light, size = 13) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: 0.1, line: { color: fill } });
  txt(slide, "💡", { x: x + 0.15, y, w: 0.4, h, fontSize: 18, valign: "middle" });
  txt(slide, runs, { x: x + 0.6, y, w: w - 0.75, h, fontSize: size, valign: "middle" });
}
function stat(slide, x, y, w, num, label, color = C.blue, size = 40) {
  txt(slide, num, { x, y, w, h: 0.75, fontSize: size, bold: true, color, align: "center", valign: "middle" });
  txt(slide, label, { x, y: y + 0.78, w, h: 0.4, fontSize: 12, color: C.sub, align: "center" });
}
// 카드 제목 (■ …)
function title(slide, t, x, y, w, size = 15) {
  txt(slide, t, { x, y, w, h: 0.36, fontSize: size, bold: true, color: C.blue });
}
// 글머리표 목록 — 문자열 또는 run 배열
function bullets(slide, items, x, y, w, h, size = 13) {
  const runs = [];
  items.forEach((item, i) => {
    const parts = Array.isArray(item) ? item : [{ text: item }];
    parts.forEach((r, k) => {
      const o = Object.assign({}, r.options);
      if (k === 0) o.bullet = true;
      if (k === parts.length - 1 && i < items.length - 1) o.breakLine = true;
      runs.push({ text: r.text, options: o });
    });
  });
  txt(slide, runs, { x, y, w, h, fontSize: size, paraSpaceAfter: 8 });
}
function table(slide, rows, x, y, colW, o = {}) {
  const head = rows[0].map(t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.blue }, align: "center" } }));
  const body = rows.slice(1).map(r => r.map(c => (typeof c === "string" ? { text: c } : c)));
  slide.addTable([head, ...body], Object.assign({
    x, y, colW, fontFace: F, fontSize: 12, color: C.text, valign: "middle", margin: 0.08,
    border: { type: "solid", pt: 1, color: C.line }, rowH: 0.42,
  }, o));
}
const strong = (t, color = C.blue) => ({ text: t, options: { bold: true, color } });
// 막대 차트 공통 설정 (PowerPoint에서 편집되는 내장 차트). 콤보 차트는 파일이 안 열리는 문제가 있어 쓰지 않음.
function barChart(slide, labels, values, o) {
  slide.addChart(pres.charts.BAR, [{ name: o.name || "Accuracy", labels, values }], Object.assign({
    barDir: "col", varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisLabelColor: C.sub, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 11,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, titleFontSize: 12, titleColor: C.text, titleFontFace: F,
  }, o));
}
// 숫자 요약 박스. 신규 M3 데이터 관찰 슬라이드의 상단 4개 지표에 사용.
function metricBox(slide, x, y, w, num, label, note = "", accent = false) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h: 0.78, fill: { color: accent ? C.blue : C.white }, rectRadius: 0.08,
    line: { color: accent ? C.blue : C.sky, width: 1.2 },
  });
  txt(slide, num, { x, y: y + 0.06, w, h: 0.34, fontSize: 22, bold: true, color: accent ? C.white : C.blue, align: "center", valign: "middle" });
  txt(slide, label, { x: x + 0.05, y: y + 0.4, w: w - 0.1, h: note ? 0.2 : 0.28, fontSize: 9.5, bold: true, color: accent ? C.white : C.text, align: "center", valign: "middle" });
  if (note) txt(slide, note, { x: x + 0.05, y: y + 0.59, w: w - 0.1, h: 0.14, fontSize: 7.8, color: accent ? C.white : C.sub, align: "center", valign: "middle" });
}
// 구조도 노드: 연한 파랑 상자(마지막 판정 상자만 진한 파랑), 제목 + 작은 설명
function node(slide, x, y, w, h, t, b, dark = false) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: dark ? C.blue : C.light }, rectRadius: 0.08, line: { color: dark ? C.blue : C.sky, width: 1 } });
  const runs = [{ text: t, options: { bold: true, fontSize: 12.5, color: dark ? C.white : C.blue, breakLine: !!b } }];
  if (b) runs.push({ text: b, options: { fontSize: 10.5, color: dark ? C.white : C.text } });
  txt(slide, runs, { x: x + 0.06, y, w: w - 0.12, h, align: "center", valign: "middle" });
}
// 화살표 연결선 (왼쪽 → 오른쪽)
function link(slide, x1, y1, x2, y2) {
  slide.addShape(pres.shapes.LINE, { x: x1, y: Math.min(y1, y2), w: x2 - x1, h: Math.abs(y2 - y1), flipV: y2 < y1, line: { color: C.sky, width: 1.5, endArrowType: "triangle" } });
}
// 구조도: 앞 단계(pre, 일렬) → 갈래(branches, 세로로 쌓음) → 뒤 단계(post, 일렬). 갈래의 세 번째 값은 연결선 라벨(가중치 등).
function flow(slide, x, y, w, h, f) {
  const pre = f.pre || [], post = f.post || [], br = f.branches;
  const total = pre.length * f.preW + f.brW + post.length * f.postW;
  const gap = (w - total) / (pre.length + post.length);
  const mid = y + h / 2;
  let cx = x;
  pre.forEach(([t, b], i) => {
    node(slide, cx, y, f.preW, h, t, b);
    if (i > 0) link(slide, cx - gap, mid, cx, mid);
    cx += f.preW + gap;
  });
  const bh = (h - (br.length - 1) * f.brGap) / br.length, bx = cx;
  const px = cx + f.brW + gap;
  br.forEach(([t, b, label], i) => {
    const by = y + i * (bh + f.brGap), bm = by + bh / 2;
    node(slide, bx, by, f.brW, bh, t, b);
    if (pre.length) link(slide, bx - gap, mid, bx, bm);
    link(slide, bx + f.brW, bm, px, mid);
    // 라벨은 선이 시작하는 높이(bm)의 바깥쪽(위 갈래는 위, 아래 갈래는 아래)에 둬서 선과 겹치지 않게
    if (label) txt(slide, label, { x: bx + f.brW + 0.04, y: bm < mid ? bm - 0.34 : bm + 0.08, w: gap - 0.08, h: 0.26, fontSize: 11, bold: true, color: C.blue, align: "left" });
  });
  cx = px;
  post.forEach(([t, b], i) => {
    node(slide, cx, y, f.postW, h, t, b, i === post.length - 1);
    if (i > 0) link(slide, cx - gap, mid, cx, mid);
    cx += f.postW + gap;
  });
}

// =====================================================================
// 1. 표지 — D+AICC PPT 대학부 공식 템플릿 그대로 사용 (수정 없음)
// =====================================================================
{
  const s = pres.addSlide();
  s.addImage({ path: TPL.cover, x: 0, y: 0, w: W, h: 7.5, sizing: { type: "cover", w: W, h: 7.5 } });
  s.addNotes("공식 템플릿(D+AICC PPT 대학부 템플릿.pptx) 슬라이드1 그대로. 이 장에는 글자를 추가하지 않음 — 프로젝트·팀 정보는 다음 장(2p)에서 제시.");
}

// =====================================================================
// 2. 프로젝트 타이틀 — 공식 표지엔 제목·팀명 넣을 자리가 없어 별도 장으로 분리
// =====================================================================
{
  const s = pres.addSlide();
  s.addImage({ path: TPL.header, x: 0, y: 0, w: W, h: HB, sizing: { type: "cover", w: W, h: HB } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: W / 2 - 1.1, y: 1.0, w: 2.2, h: 0.4, fill: { color: C.sky }, rectRadius: 0.08, line: { color: C.sky } });
  txt(s, "대학부 예선", { x: W / 2 - 1.1, y: 1.0, w: 2.2, h: 0.4, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle" });
  txt(s, "2026 데이터+AI 크리에이터 캠프  |  결과 정리", { x: M, y: 1.5, w: CW, h: 0.4, fontSize: 13, color: C.sub, align: "center" });
  txt(s, "전화대화 음성 및 전사 데이터를 이용한", { x: M, y: 2.05, w: CW, h: 0.6, fontSize: 28, bold: true, align: "center" });
  txt(s, "응급상황 인식", { x: M, y: 2.6, w: CW, h: 0.9, fontSize: 44, bold: true, color: C.blue, align: "center" });
  { const hh = 2.05, hw = hh * 1884 / 712; s.addImage({ path: "figures/canva/hero_call_flow_crop.png", x: (W - hw) / 2, y: 4.05, w: hw, h: hh }); }
  txt(s, "Mission 1~3 수행 결과 정리", { x: M, y: 3.5, w: CW, h: 0.5, fontSize: 20, color: C.sub, align: "center" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: W / 2 - 3.2, y: 6.25, w: 6.4, h: 0.7, fill: { color: C.white }, rectRadius: 0.1, line: { color: C.line, width: 1 }, shadow: shadow() });
  txt(s, [{ text: "팀 eigensinn", options: { bold: true, color: C.blue } }, { text: "   김장수 · 김광희 · 지연우 · 이수빈 · 조지은", options: { color: C.text } }],
    { x: W / 2 - 3.2, y: 6.25, w: 6.4, h: 0.7, fontSize: 15, align: "center", valign: "middle" });
  txt(s, "제출일 2026. 10. 14", { x: W - M - 4, y: 7.05, w: 4, h: 0.3, fontSize: 12, color: C.sub, align: "right" });
  s.addNotes("프로젝트 타이틀. 공식 표지 바로 뒤에 배치해 제목·팀명을 보강.");
}

// =====================================================================
// 3. 목차 (가이드 2.2 — 4등분, 큰 회색 번호 위에 섹션명) — 공식 헤더 띠 적용
// =====================================================================
{
  const s = pres.addSlide();
  s.addImage({ path: TPL.header, x: 0, y: 0, w: W, h: HB, sizing: { type: "cover", w: W, h: HB } });
  txt(s, "목차", { x: M, y: HB + 0.2, w: 4, h: 0.6, fontSize: 28, bold: true, color: C.blue });
  const secs = [
    ["01", "준비 과정", ["사전 학습 로드맵", "팀 구성과 협업 체계"]],
    ["02", "데이터 분석", ["EDA 핵심 발견"]],
    ["03", "미션별 수행", ["Mission 1  신고자 성별 분류", "Mission 2  신고자/대원 분류\n   (+ 실험 흐름·공식 검증·오류 분석)", "Mission 3  환자 증상 인식"]],
    ["04", "종합 및 시사점", ["성능 요약과 교훈", "사회안전 시사점", "추후 활동 계획"]],
  ];
  const cw = (CW - 3 * 0.3) / 4;
  secs.forEach(([n, name, items], i) => {
    const x = M + i * (cw + 0.3);
    txt(s, n, { x, y: 2.1, w: cw, h: 1.5, fontSize: 96, bold: true, color: C.bigNum });
    s.addImage({ path: IC + ["sec_prep", "sec_data", "sec_mission", "sec_wrap"][i] + ".png", x: x + cw - 0.95, y: 2.2, w: 0.7, h: 0.7 });
    txt(s, name, { x: x + 0.1, y: 3.15, w: cw - 0.1, h: 0.55, fontSize: 22, bold: true });
    s.addShape(pres.shapes.LINE, { x: x + 0.1, y: 3.85, w: cw - 0.3, h: 0, line: { color: C.line, width: 1 } });
    txt(s, items.map((t, k) => ({ text: `${k + 1}. ${t}`, options: { breakLine: k < items.length - 1 } })),
      { x: x + 0.1, y: 4.05, w: cw - 0.1, h: 2.2, fontSize: 14, color: C.sub, paraSpaceAfter: 10 });
  });
  s.addNotes("EDA 2장, M2 지연우 담당 10장 및 M3 별첨 5장을 포함한 총 38장 구성.");
}

// =====================================================================
// 4. 01-(1) 사전 학습 로드맵 (step0 저장소)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "01", "준비 과정", "(1) 발대식 전 사전 학습 로드맵", 4);
  coreMsg(s, "처음 보는 데이터로도  QA → Baseline → 검증 → 통제 실험 → 제출  흐름을 끝까지 돌 수 있도록 대회 전에 미리 연습");
  const items = [
    ["W1  기초", "Python·pandas\n타이타닉 EDA"],
    ["W2  Baseline", "첫 모델 학습\n실험 로그"],
    ["W3  데이터 QA", "CIFAR-10\n누락·중복 점검"],
    ["W4  통제 실험", "baseline 감사\n변수 하나씩"],
    ["W5  데이터 정제", "이상치 점수\n기준 사전 고정"],
    ["W6  제출 연습", "제출 패키지\n채점 스크립트"],
  ].map(([t, b], i) => [t, b, i === 0 ? C.sky : C.blue]);
  steps(s, items, 2.55, 2.1, { bw: 1.78, bodySize: 11.5, icons: ["w1", "w2", "w3", "w4", "w5", "w6"].map(n => IC + n + ".png") });
  insight(s, M, 5.0, CW, 1.3, [
    { text: "예선에 그대로 가져온 습관  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "점검이 모델보다 먼저  ·  실험마다 ID·로그  ·  변수는 하나씩  ·  기준은 결과 보기 전에 고정" },
  ]);
  caption(s, "근거: GitHub eigensinn-dcc2026/step0 (2026.07.14 ~ 08.23), 팀원별 notebooks/ 및 outputs/ 실습 산출물");
  s.addNotes("근거: GitHub eigensinn-dcc2026/step0 README의 W1~W6 로드맵, 팀원별 notebooks/·outputs/.");
}

// =====================================================================
// 5. 01-(2) 팀 구성과 협업 체계
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "01", "준비 과정", "(2) 팀 구성과 협업 체계", 5);
  coreMsg(s, "5명이 한 저장소에서 충돌 없이 일하도록 규칙·자동 검사·기록 체계를 먼저 만들고 미션에 들어갔다.");
  txt(s, "■ 역할 분담", { x: M, y: 2.45, w: 5, h: 0.4, fontSize: 18, bold: true });
  table(s, [
    ["팀원", "예선 담당"],
    ["김장수", "Mission 1 · 2 · 3"],
    ["김광희", "Mission 2"],
    ["지연우", "Mission 2"],
    ["이수빈", "Mission 3"],
    ["조지은", "Mission 3"],
  ], M, 2.95, [1.8, 3.7], { rowH: 0.6, fontSize: 13 });

  const rx = 6.6, rw = W - M - rx;
  txt(s, "■ GitHub 작업 흐름", { x: rx, y: 2.45, w: rw, h: 0.4, fontSize: 18, bold: true });
  const flow = ["main에서\n브랜치 생성", "실험·코드\n작성", "PR 작성\n(템플릿)", "자동 검사\n(CI·병합 판정)", "팀원 리뷰\n후 병합"];
  const fw = 1.05, fg = (rw - 5 * fw) / 4;
  flow.forEach((t, i) => {
    const x = rx + i * (fw + fg);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.95, w: fw, h: 1.0, fill: { color: i === 3 ? C.blue : C.light }, rectRadius: 0.1, line: { color: i === 3 ? C.blue : C.light } });
    txt(s, t, { x, y: 2.95, w: fw, h: 1.0, fontSize: 11, bold: true, color: i === 3 ? C.white : C.blue, align: "center", valign: "middle" });
    if (i < 4) arrow(s, x + fw + (fg - 0.14) / 2, 3.3, 0.14, 0.3);
  });
  txt(s, "■ 기록 체계", { x: rx, y: 4.3, w: rw, h: 0.4, fontSize: 18, bold: true });
  const bw = (rw - 0.3) / 2;
  [["docs/experiment_log.md", "실험 기록", "무엇을 했고 결과는?"],
   ["docs/decisions.md", "결정 기록", "왜 채택·기각했나? 기각도 기록"]].forEach(([f, t, q], i) => {
    const x = rx + i * (bw + 0.3);
    card(s, x, 4.8, bw, 1.75);
    txt(s, f, { x: x + 0.2, y: 4.95, w: bw - 0.4, h: 0.3, fontSize: 11, color: C.sub });
    txt(s, t, { x: x + 0.2, y: 5.3, w: bw - 0.4, h: 0.45, fontSize: 18, bold: true, color: C.blue });
    txt(s, q, { x: x + 0.2, y: 5.8, w: bw - 0.4, h: 0.6, fontSize: 13 });
  });
  caption(s, "근거: GitHub eigensinn-dcc2026/prelim — README.md(역할), WORKFLOW.md, .github/workflows (pr-ci, pr-verdict), docs/");
  s.addNotes("근거: README.md 역할표, WORKFLOW.md(브랜치·PR 흐름), .github/workflows/(pr-ci, pr-verdict), docs/experiment_log.md, docs/decisions.md.");
}

// =====================================================================
// 6~7. 02 EDA 핵심 발견 — 원본 그래프를 유지한 채 음성·증상으로 분리
// =====================================================================
{
  const s = pres.addSlide();
  const message = "세 미션 모두 \"쉬운 단서 하나\"로는 풀리지 않는다. 이 발견이 미션별 모델 선택의 출발점이다.";
  const evidence = "근거: reports/eda/eda_report.md (M2 겹침: 7.2절 Training), reports/eda/figures/";
  const notes = "근거: reports/eda/eda_report.md, reports/eda/figures/(gender_distribution, speaker_duration_buckets, symptom_prevalence), docs/experiment_log.md COM-EDA-001.";
  header(s, "02", "데이터 분석", "EDA 핵심 발견 (1/2) — M1·M2", 6);
  coreMsg(s, message);
  const cards = [
    ["figures/gender_distribution.png", "M1 성별",
      "성별 비율은 균형. 길이·음량 같은 요약 특징으로는 성별이 거의 안 갈림",
      "음색을 담는 화자 임베딩(x-vector)"],
    ["overlap", "M2 화자",
      "통화의 98%에 겹침이 있지만 시간은 중앙값 3.8%. 길이만으로는 구분이 어려움",
      "소리 모양 전체를 보는 CNN"],
    ["figures/symptom_prevalence.png", "M3 증상",
      "세 통화 중 하나는 증상이 2개 이상. 증상별 빈도 차이가 큼",
      "증상 9개를 각각 0/1로 푸는 다중 레이블"],
  ];
  const cw = (CW - 0.3) / 2;
  cards.slice(0, 2).forEach(([path, t, found, design], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.4, cw, 4.6);
    s.addImage({ path: IC + ["m1", "m2"][i] + ".png", x: x + cw - 0.6, y: 2.46, w: 0.3, h: 0.3 });
    title(s, t, x + 0.2, 2.47, cw - 1.1, 16);
    if (path === "overlap") {
      s.addShape(pres.shapes.RECTANGLE, { x: x + 0.2, y: 2.86, w: cw - 0.4, h: 3.29, fill: { color: C.white }, line: { color: C.line, width: 1 } });
      barChart(s, ["겹침이 있는\n통화 비율", "통화당 겹침 시간\n(중앙값)"], [98.15, 3.82], {
        name: "%", x: x + 0.25, y: 2.9, w: cw - 0.5, h: 3.21, chartColors: [C.sky, C.red], title: "Training 통화 기준 (%)", titleFontSize: 14,
        dataLabelFormatCode: "0.00", dataLabelFontSize: 18, catAxisLabelFontSize: 14, valAxisMinVal: 0, valAxisMaxVal: 115, valAxisHidden: true, valGridLine: { style: "none" },
      });
    } else {
      // 내부 글씨를 30% 키운 PNG를 배치하며, 원본 1264×784 비율을 보존한다.
      const imageW = 5.3, imageH = imageW * 784 / 1264;
      fig(s, path, x + (cw - imageW - 0.16) / 2, 2.78, imageW + 0.16, imageH + 0.16);
    }
    txt(s, [
      { text: "발견  ", options: { bold: true } },
      { text: found, options: { breakLine: true } },
      { text: "설계  ", options: { bold: true, color: C.blue } },
      { text: design },
    ], { x: x + 0.2, y: 6.2, w: cw - 0.4, h: 0.77, fontSize: 14, paraSpaceAfter: 2 });
  });
  caption(s, evidence, 7.12);
  s.addNotes(notes);

  const symptomSlide = pres.addSlide();
  header(symptomSlide, "02", "데이터 분석", "EDA 핵심 발견 (2/2) — M3", 7);
  coreMsg(symptomSlide, message);
  const [path, t, found, design] = cards[2];
  // 내부 글씨를 30% 키운 PNG의 증상 9개와 범례를 원본 1421×1024 비율로 유지한다.
  const imageW = 6.05, imageH = imageW * 1024 / 1421;
  fig(symptomSlide, path, M, 2.43, imageW + 0.16, imageH + 0.16);
  const textX = 7.05, textW = W - M - textX;
  card(symptomSlide, textX, 2.43, textW, 4.5);
  s.addNotes("EDA를 2장으로 나누고 기존 그래프·문구·수치를 유지함. M1·M3 PNG의 제목·축·범례·수치 글씨를 원래 그래프 대비 30% 확대함.");
  symptomSlide.addImage({ path: IC + "m3.png", x: textX + textW - 0.75, y: 2.62, w: 0.45, h: 0.45 });
  title(symptomSlide, t, textX + 0.25, 2.68, textW - 1.1, 18);
  txt(symptomSlide, [
    { text: "발견  ", options: { bold: true, breakLine: true } },
    { text: found },
  ], { x: textX + 0.25, y: 3.45, w: textW - 0.5, h: 1.35, fontSize: 16, paraSpaceAfter: 6 });
  txt(symptomSlide, [
    { text: "설계  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: design },
  ], { x: textX + 0.25, y: 5.15, w: textW - 0.5, h: 1.2, fontSize: 16, paraSpaceAfter: 6 });
  caption(symptomSlide, evidence);
  symptomSlide.addNotes(notes);
}

// ======================================================================
// Mission 1 — 신고자 성별 분류 (8~11장)
// ======================================================================

// 8. 문제와 최종 모델
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1  신고자 성별 분류 — 문제와 최종 모델", 8);
  coreMsg(s, "음향 특징 + 화자 임베딩 → 로지스틱 회귀. 공식 Validation Accuracy 98.10%.");
  const lw = 9.05;
  txt(s, [strong("■ 문제   "), { text: "신고자 발화 음성 → 성별(F/M)   ·   지표 Accuracy" }], { x: M, y: 2.45, w: lw, h: 0.4, fontSize: 15 });
  flow(s, M, 2.95, lw, 1.7, {
    pre: [["신고자 구간", "신고자 발화만\n잘라내기"]], preW: 1.5,
    branches: [["음향 특징 52D", "통계 특징"], ["WavLM x-vector 512D", "20초 창 최대 4개 평균"]], brW: 2.55, brGap: 0.14,
    post: [["결합·표준화", "52D + 512D\n= 564D"], ["로지스틱 회귀", "F / M"]], postW: 1.65,
  });
  title(s, "■ 왜 이 구조인가", M, 4.85, lw, 16);
  bullets(s, [
    "EDA에서 요약 통계가 성별을 못 가름 → 음색을 담는 사전학습 화자 임베딩 추가",
    "분류기는 단순한 로지스틱 회귀로 충분했다 (더 복잡한 분류기는 이득이 기준에 못 미침 → 10장)",
    "최종 모델은 학습 데이터 전체로 재학습했다",
  ], M, 5.3, lw, 1.5, 13);
  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, 2.45, rw, 4.35, C.light);
  txt(s, "98.10%", { x: rx, y: 3.6, w: rw, h: 1.0, fontSize: 44, bold: true, color: C.blue, align: "center", valign: "middle" });
  txt(s, "Accuracy\n공식 Validation", { x: rx, y: 4.65, w: rw, h: 0.7, fontSize: 13, color: C.sub, align: "center" });
  caption(s, "근거: submission/mission1/README.md, docs/experiment_log.md M1-REFIT-OFFVAL-001");
  s.addNotes("근거: submission/mission1/README.md(구성·상태), docs/experiment_log.md M1-XVEC-002-OFFVAL(다중 구간 평균 x-vector), M1-REFIT-OFFVAL-001(전체 재학습 제출 모델 98.10%).");
}

// 9. 성능 개선 과정
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1  신고자 성별 분류 — 성능 개선 과정", 9);
  coreMsg(s, "95.11% → 98.10%. 개선의 대부분은 화자 임베딩(x-vector)에서 나왔다.");
  card(s, M, 2.45, 7.3, 4.35);
  barChart(s, ["① 음향 특징 52개\n+ LR (baseline)", "② + x-vector\n(앞 20초)", "③ + 여러 구간\n평균 x-vector", "④ ③을 전체 데이터로\n재학습 (제출)"],
    [95.11, 97.47, 98.13, 98.10], {
      x: M + 0.15, y: 2.55, w: 7.0, h: 4.15, chartColors: [C.line, C.sky, C.sky, C.blue],
      title: "공식 Validation Accuracy (%)", valAxisMinVal: 94, valAxisMaxVal: 99, catAxisLabelFontSize: 10.5,
    });
  const rx = M + 7.6, rw = W - M - rx;
  card(s, rx, 2.45, rw, 2.2);
  title(s, "■ 해석", rx + 0.2, 2.57, rw - 0.4);
  bullets(s, [
    "음향 특징 단독보다 x-vector 단독이 훨씬 높았다 → 개선의 대부분이 x-vector.",
    "긴 통화는 앞부분만 쓰면 나머지를 버리게 되어, 여러 구간을 평균했다.",
  ], rx + 0.2, 2.98, rw - 0.4, 1.6, 12);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 4.85, w: rw, h: 1.95, fill: { color: C.pink }, rectRadius: 0.1, line: { color: C.pink } });
  txt(s, [
    { text: "주의  ", options: { bold: true, color: C.red, breakLine: true } },
    { text: "다중 구간 평균은 내부 홀드아웃에서는 사전 기준에 못 미쳐 일단 기각했고, 이후 공식 Validation에서 별도로 고정한 기준을 넘어 제출 후보가 됐다. 두 판정과 이유를 모두 기록했다." },
  ], { x: rx + 0.2, y: 4.85, w: rw - 0.4, h: 1.95, fontSize: 12, valign: "middle", paraSpaceAfter: 4 });
  caption(s, "근거: docs/experiment_log.md M1-XVEC-001-OFFVAL, M1-XVEC-002(2차), M1-XVEC-002-OFFVAL, M1-REFIT-OFFVAL-001");
  s.addNotes("근거: ① 95.11% 기존 52D LR(M1-XVEC-001-OFFVAL 비교값, docs/decisions.md 2026-10-05) / ② 97.47% M1-XVEC-001-OFFVAL / ③ 98.13% M1-XVEC-002-OFFVAL(사전 고정 기준 97.77% 충족) / ④ 98.10% M1-REFIT-OFFVAL-001. 주의 박스: M1-XVEC-002(2차) 내부 +0.34%p로 기준 +0.4%p 미달 → 기각.");
}

// 10. 채택하지 않은 시도
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1  신고자 성별 분류 — 채택하지 않은 시도", 10);
  coreMsg(s, "점수가 올라도 미리 정한 기준을 못 넘으면 채택하지 않았다.");
  table(s, [
    ["시도", "판정", "이유"],
    ["LightGBM", strong("✓ LR 채택"), "점수 차이가 작고 더 단순·해석 쉬움"],
    ["RBF-SVM 분류기", strong("✗ 기각", C.red), "이득이 기준 미달, 추론이 훨씬 느림"],
    ["ECAPA 임베딩 추가·교체", strong("✗ 기각", C.red), "이득이 기준 미달"],
    ["x-vector LR + ECAPA LR 확률 평균", strong("✗ 기각", C.red), "공식 Val 98.30%로 소폭 높았지만 미리 정한 기준 미달, 차이가 우연과 구분되지 않음"],
  ], M, 2.45, [3.5, 1.4, 7.43], { rowH: 0.62, fontSize: 13 });
  insight(s, M, 5.75, CW, 1.05, [
    { text: "공식 Val에서 더 높았던 모델을 왜 버렸나?  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "몇 통화 차이는 우연과 구분되지 않는다. 결과를 본 뒤 기준을 낮추면 Validation에 과적합된다." },
  ], C.light, 13.5);
  caption(s, "근거: docs/decisions.md 2026-10-05, docs/experiment_log.md M1-HEAD-001, M1-ECAPA-002, M1-FUSE-001-OFFVAL");
  s.addNotes("근거: LightGBM vs LR — docs/decisions.md 2026-10-05 / RBF-SVM — M1-HEAD-001 / ECAPA — M1-ECAPA-001(3차)·M1-ECAPA-002 / 확률 평균 — M1-FUSE-001, M1-FUSE-001-OFFVAL(98.30%, 사전 고정 기준 98.434% 미달, McNemar p=0.345).");
}

// 11. 최종 결과와 후속 과제
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1  신고자 성별 분류 — 최종 결과와 후속 과제", 11);
  coreMsg(s, "공식 Validation 98.10%. 남녀 recall이 고르다.");
  const lw = 5.6;
  card(s, M, 2.45, lw, 4.35);
  stat(s, M, 2.75, lw, "98.10%", "Accuracy (공식 Validation)", C.blue, 48);
  s.addShape(pres.shapes.LINE, { x: M + 0.4, y: 4.25, w: lw - 0.8, h: 0, line: { color: C.line, width: 1 } });
  const hw = lw / 2;
  stat(s, M, 4.55, hw, "98.26%", "여성 recall", C.sky, 30);
  stat(s, M + hw, 4.55, hw, "97.92%", "남성 recall", C.sky, 30);
  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, 2.45, rw, 4.35, C.bgGray);
  title(s, "■ 후속 과제", rx + 0.25, 2.62, rw - 0.5, 18);
  bullets(s, [
    "여러 seed로 결과 재현 확인",
    "오분류 통화의 화자 단위 정성 분석",
    "피치(F0) 특징 검증 (미검증 가설)",
  ], rx + 0.25, 3.25, rw - 0.5, 3.3, 17);
  caption(s, "근거: submission/mission1/README.md 상태(2026-10-12), docs/experiment_log.md M1-REFIT-OFFVAL-001");
  s.addNotes("근거: submission/mission1/README.md — 공식 Validation 3,640건 98.10%, 여성 recall 98.26%, 남성 recall 97.92%. docs/experiment_log.md M1-REFIT-OFFVAL-001(단일 seed).");
}

// ======================================================================
// Mission 2 — 신고자/119대원 분류 (12~15장, 수치·모델명은 상단 M2 상수)
// ======================================================================

// 12. 문제·제약·최종 모델
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 문제·제약·최종 모델", 12);
  coreMsg(s, M2.core12);
  const lw = 8.9;
  txt(s, [strong("■ 문제   "), { text: "한 발화 조각 → 신고자(1) / 대원(0)   ·   지표 Accuracy" }], { x: M, y: 2.5, w: lw, h: 0.4, fontSize: 15 });
  txt(s, [strong("■ 제약   ", C.red), { text: "text · 발화 순서 · 다른 발화 정보 사용 불가" }], { x: M, y: 2.95, w: lw, h: 0.4, fontSize: 15 });
  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, 2.45, rw, 0.95, C.light);
  txt(s, [{ text: M2.acc, options: { fontSize: 28, bold: true, color: C.blue } }, { text: "   공식 Validation", options: { fontSize: 12, color: C.sub } }],
    { x: rx, y: 2.45, w: rw, h: 0.95, align: "center", valign: "middle" });
  flow(s, M, 3.55, CW, 1.15, M2.flow);
  // 합성 도식(실제 통화 아님): ppt/tools/make_m2_synthetic.py
  const fw = 6.6, fh = fw * 482 / 1658;
  s.addShape(pres.shapes.RECTANGLE, { x: M, y: 4.9, w: fw + 0.16, h: fh + 0.16, fill: { color: C.white }, line: { color: C.line, width: 1 } });
  s.addImage({ path: "figures/anim/m2_waveform_to_mel.gif", x: M + 0.08, y: 4.98, w: fw, h: fh });
  const wx = M + fw + 0.4, ww = W - M - wx, wh = (fh + 0.16 - 2 * 0.12) / 3;
  M2.why.forEach(([k, v], i) => {
    const y = 4.9 + i * (wh + 0.12);
    card(s, wx, y, ww, wh, i === 0 ? C.light : C.white);
    txt(s, k, { x: wx + 0.15, y, w: 1.45, h: wh, fontSize: 13, bold: true, color: C.blue, valign: "middle" });
    txt(s, v, { x: wx + 1.6, y, w: ww - 1.7, h: wh, fontSize: 11.5, valign: "middle" });
  });
  caption(s, "※ 왼쪽 도식은 합성 신호(실제 통화 아님).  근거: " + M2.evidence);
  s.addNotes("근거: " + M2.evidence + ". 제약은 출제문제·주최측 Q&A(2026.09.15), 추론 방식은 submission/mission2/README.md '추론' 절.");
}

// 13. 성능 개선 과정
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 성능 개선 과정", 13);
  coreMsg(s, M2.core13);
  const lw = 6.2;
  card(s, M, 2.45, lw, 4.35);
  barChart(s, M2.chart.labels, M2.chart.values, {
    x: M + 0.15, y: 2.55, w: lw - 0.3, h: 4.15, chartColors: [C.line, C.sky, C.blue],
    title: "공식 Validation Accuracy (%)", valAxisMinVal: 84, valAxisMaxVal: 94,
  });
  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, 2.45, rw, 2.15);
  title(s, M2.gainTitle, rx + 0.2, 2.56, rw - 0.4);
  const iw = 2.6, ih = iw * 370 / 900;
  s.addImage({ path: "figures/anim/speakers_separation.gif", x: rx + 0.2, y: 3.0, w: iw, h: ih });
  M2.gains.forEach(([n, l], i) => {
    const y = 3.0 + i * 0.58;
    txt(s, n, { x: rx + 3.0, y, w: 1.5, h: 0.5, fontSize: 24, bold: true, color: i === 0 ? C.blue : C.sky, valign: "middle" });
    txt(s, l, { x: rx + 4.45, y, w: rw - 4.55, h: 0.5, fontSize: 11.5, color: C.sub, valign: "middle" });
  });
  txt(s, M2.gainNote, { x: rx + 0.2, y: 4.2, w: rw - 0.4, h: 0.3, fontSize: 10, color: C.sub, align: "center" });
  insight(s, rx, 4.75, rw, 2.05, [
    { text: "발견  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: M2.finding },
  ], C.mintLight, 12);
  caption(s, M2.evidence13);
  s.addNotes(M2.notes13);
}

// 14. 다른 접근과 기각·보류 이유
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 다른 접근과 기각·보류 이유", 14);
  coreMsg(s, M2.core14);
  const red = t => strong("✗ " + t, C.red);
  const [lastTry, lastVerdict, lastWhy] = M2.lastRow;
  table(s, [
    ["시도", "판정", "이유"],
    ["같은 통화의 다른 발화와 비교하는 특징", red("제거"), "규칙 위반 소지 (주최 측 답변)"],
    [{ text: "text로 고친 라벨로 학습" }, strong("제공 라벨로 재학습"), "점수는 거의 그대로"],
    ["통계 모델과 CNN 확률 평균", red("기각"), "단독 CNN보다 낮음"],
    ["더 큰 CNN / 패딩 제외 풀링 / 길이 입력 / Mixup", red("기각"), "사전 기준 미달 또는 재현 안 됨"],
    ["WavLM 고정 임베딩 · 기본 미세조정", red("기각"), "이득 없음, 추론 비용 큼"],
    [strong(lastTry, C.text), strong(lastVerdict, C.sub), lastWhy],
  ], M, 2.45, [5.2, 2.4, 4.73], { rowH: 0.6, fontSize: 13 });
  caption(s, "비교는 Training 내부 통화 단위 홀드아웃 기준 (공식 Val 아님).", 6.75);
  s.addNotes("근거: 통화 단위 특징 M2-EXP-008 / 제공 라벨 재학습 M2-RETRAIN-001 / 확률 평균 M2-ENS-001 / 더 큰 CNN M2-CNN-008, 패딩 제외 풀링 M2-CNN-009, 길이 입력 M2-CNN-004·006, Mixup M2-MIX-001 / WavLM 고정 임베딩 M2-SSL-001, 기본 미세조정 M2-SSL-002 / WavLM-large 미세조정 2개 + CNN 결합 M2-WLML-001·002, M2-BLEND-001(내부 20,000발화 CNN 단독 91.94% → 93.21%). 모두 docs/experiment_log.md·docs/decisions.md.");
}

// 15. 최종 결과와 보완할 점
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 최종 결과와 보완할 점", 15);
  coreMsg(s, M2.core15);
  const lw = 5.6;
  card(s, M, 2.45, lw, 4.35);
  stat(s, M, 2.75, lw, M2.acc, "Accuracy (공식 Validation)", C.blue, 48);
  s.addShape(pres.shapes.LINE, { x: M + 0.4, y: 4.25, w: lw - 0.8, h: 0, line: { color: C.line, width: 1 } });
  M2.recalls.forEach(([n, l], i) => {
    const y = 4.45 + i * 1.05, bx = M + 0.4, bw = lw - 0.8, pct = parseFloat(n) / 100, col = i === 0 ? C.red : C.sky;
    txt(s, l, { x: bx, y, w: 2.0, h: 0.35, fontSize: 13, bold: true, color: C.sub, valign: "middle" });
    txt(s, n, { x: bx + bw - 1.8, y, w: 1.8, h: 0.35, fontSize: 20, bold: true, color: col, align: "right", valign: "middle" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: y + 0.42, w: bw, h: 0.22, fill: { color: C.line }, rectRadius: 0.1, line: { color: C.line } });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: y + 0.42, w: bw * pct, h: 0.22, fill: { color: col }, rectRadius: 0.1, line: { color: col } });
  });
  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, 2.45, rw, 1.6);
  title(s, "■ 재현성", rx + 0.25, 2.6, rw - 0.5, 16);
  txt(s, M2.repro, { x: rx + 0.25, y: 3.05, w: rw - 0.5, h: 0.9, fontSize: 14 });
  card(s, rx, 4.2, rw, 2.6, C.bgGray);
  title(s, "■ 보완할 점", rx + 0.25, 4.35, rw - 0.5, 16);
  txt(s, M2.improve, { x: rx + 0.25, y: 4.8, w: rw - 0.5, h: 1.9, fontSize: 14 });
  caption(s, "근거: " + M2.evidence);
  s.addNotes(M2.notes15);
}

// ======================================================================
// ---------- Mission 2 추가: 지연우 음향 특징 실험 (10장) ----------
// 초기 비교 / 내부 비교 / 고정 후보의 최종 평가는 각 평가 조건 안에서만 설명한다.
{
  const leftW = 6.0, rightX = M + leftW + 0.3, rightW = W - M - rightX;
  const page = (name, message) => {
    const s = pres.addSlide();
    header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — " + name, pres._slides.length);
    coreMsg(s, message);
    return s;
  };
  const panels = s => {
    card(s, M, 2.45, leftW, 4.35);
    card(s, rightX, 2.45, rightW, 4.35);
  };
  const comparison = (s, a, b, aName, bName, gain, label = "Accuracy (%)") => {
    title(s, "■ " + label, rightX + 0.25, 2.65, rightW - 0.5, 17);
    const half = (rightW - 0.5) / 2;
    stat(s, rightX + 0.15, 3.35, half, a, aName, C.sky, 33);
    stat(s, rightX + 0.35 + half, 3.35, half, b, bName, C.blue, 33);
    txt(s, gain, { x: rightX + 0.3, y: 4.9, w: rightW - 0.6, h: 0.55, fontSize: 29, bold: true, color: C.blue, align: "center" });
  };

  // 1. Baseline: 초기 수치는 화면에서 평가 집합 명칭을 생략하고 실제 출처는 노트에 보존.
  {
    const s = page("MFCC Baseline과 오류 관찰", "MFCC로 첫 기준선을 만들고, 짧은 발화와 겹침 구간에서 보완할 정보를 찾았습니다.");
    panels(s);
    title(s, "■ 한 발화를 40개 숫자로 요약", M + 0.25, 2.65, leftW - 0.5, 17);
    const items = [["발화 음성", "현재 발화만"], ["MFCC 40D", "20계수의 평균·표준편차"], ["역할 분류", "Logistic Regression"]];
    items.forEach(([t, b], i) => {
      stepBox(s, M + 0.2 + i * 1.9, 3.3, 1.75, 1.75, t, b, i === 2 ? C.sky : C.blue, 12);
      if (i < 2) arrow(s, M + 1.96 + i * 1.9, 4.0, 0.13, 0.3);
    });
    txt(s, "음색과 주파수 모양을 요약한 간단한 기준선\ntext·발화 순서·다른 발화 정보는 사용하지 않음", { x: M + 0.3, y: 5.5, w: leftW - 0.6, h: 0.9, fontSize: 14, paraSpaceAfter: 6 });
    stat(s, rightX + 0.2, 2.95, rightW - 0.4, "80.9%", "Baseline Accuracy", C.blue, 44);
    title(s, "■ 오류에서 발견한 점", rightX + 0.3, 4.6, rightW - 0.6, 17);
    bullets(s, ["짧은 발화에서 정확도가 낮음", "목소리가 겹친 구간에서도 정확도가 낮음"], rightX + 0.3, 5.15, rightW - 0.6, 1.1, 15);
    s.addNotes("초기 E00: Official Validation 111,947행, Accuracy 0.809106147, CM [[41190,12655],[8715,49387]]. short는 Training 분위수로 정의한 구간이며 이후 0.5–1초 구간과 같지 않다. 짧은 발화와 overlap의 낮은 정확도는 관찰이며 원인 확정이 아니다. 화면에서 평가 집합 이름만 생략했으며 Internal Validation으로 재분류하지 않는다.");
  }

  // 2. 초기 feature 비교: E00/E05의 같은 평가 집합 결과만 연결.
  {
    const s = page("Rolloff·ZCR 특징 보완", "Rolloff와 ZCR을 추가한 E05에서 정확도 상승을 확인해 후속 모델 비교에 사용했습니다.");
    panels(s);
    title(s, "■ 가설: MFCC에 없는 주파수 단서 보완", M + 0.25, 2.65, leftW - 0.5, 17);
    txt(s, "에너지의 주파수 분포와 파형 변화 정보를 더하면\n역할 구분에 도움이 될까?", { x: M + 0.3, y: 3.2, w: leftW - 0.6, h: 0.85, fontSize: 16 });
    stepBox(s, M + 0.3, 4.3, 2.5, 1.35, "Rolloff", "에너지가 퍼지는 주파수 경계", C.blue, 13);
    stepBox(s, M + 3.15, 4.3, 2.5, 1.35, "ZCR · 영교차율", "파형의 부호 변화 빈도", C.sky, 13);
    txt(s, "MFCC 40D + Rolloff·ZCR 2D = E05 42D", { x: M + 0.3, y: 6.0, w: leftW - 0.6, h: 0.35, fontSize: 15, bold: true, color: C.blue, align: "center" });
    comparison(s, "80.9%", "81.3%", "E00 · MFCC40", "E05 · 42D", "+0.4%p");
    txt(s, "같은 평가 조건에서 확인한 초기 특징 조합", { x: rightX + 0.3, y: 5.95, w: rightW - 0.6, h: 0.4, fontSize: 13, color: C.sub, align: "center" });
    s.addNotes("E00/E05는 초기 Official Validation 결과: 0.809106147 / 0.8131169214, 차이 약 +0.4011%p. 개별 E02–E04는 Markdown 기록만 남아 있어 본문 비교에서 제외. 두 정보를 추가한 조합의 관측 결과이며 각 특징의 인과 효과나 통계적 유의성을 주장하지 않는다. 이후 내부 검증 MLP 결과와 Delta를 계산하지 않는다.");
  }

  // 3. 동일 42D 조건의 모델 비교: 초기 E05 Official 수치와 연결하지 않는다.
  {
    const s = page("선형 모델에서 MLP로 확장", "같은 42D 입력에서 MLP가 4.22%p 높아, 후속 음향 특징 비교의 기준 모델로 고정했습니다.");
    panels(s);
    title(s, "■ 가설: 특징 사이의 비선형 관계 학습", M + 0.25, 2.65, leftW - 0.5, 17);
    node(s, M + 1.1, 3.2, 3.8, 0.85, "동일한 E05 42D 입력", "같은 통화 단위 내부 분할·전처리");
    node(s, M + 0.3, 4.55, 2.5, 0.9, "Logistic Regression", "선형 비교 기준");
    node(s, M + 3.2, 4.55, 2.5, 0.9, "MLP", "비선형 관계 학습", true);
    s.addShape(pres.shapes.LINE, { x: M + 3.0, y: 4.05, w: 0, h: 0.2, line: { color: C.sky, width: 1.5 } });
    s.addShape(pres.shapes.LINE, { x: M + 1.55, y: 4.25, w: 2.9, h: 0, line: { color: C.sky, width: 1.5 } });
    link(s, M + 1.55, 4.25, M + 1.55, 4.55);
    link(s, M + 4.45, 4.25, M + 4.45, 4.55);
    txt(s, "입력 특징은 유지하고 분류 방법을 비교", { x: M + 0.3, y: 6.05, w: leftW - 0.6, h: 0.35, fontSize: 15, color: C.blue, bold: true, align: "center" });
    comparison(s, "81.73%", "85.95%", "Logistic Regression", "MLP", "+4.22%p", "내부 검증 Accuracy");
    txt(s, "이 MLP를 고정한 뒤 음향 특징을 추가 비교", { x: rightX + 0.3, y: 5.95, w: rightW - 0.6, h: 0.45, fontSize: 14, color: C.sub, align: "center" });
    s.addNotes("M00 LR 0.8173319625, M04-100 MLP 0.8595319610: 같은 E05 42D, fixed GroupSplit, seed 42, internal-train-only scaler. 차이 4.21999985%p. MLP (64,)의 ReLU 비선형 상호작용은 실험 가설이며 개선 원인을 증명한 것은 아니다. max_iter=100에서도 convergence warning이 있어 수렴 완료를 주장하지 않는다. 81.31% 초기 Official 결과와 직접 비교하지 않는다.");
  }

  // 4. 당시 고정 MLP의 AF02 직접 비교.
  {
    const s = page("Pitch·Voicing의 효과", "음높이·유성성 특징은 정확도와 두 역할의 Recall을 함께 개선해 채택했습니다.");
    panels(s);
    title(s, "■ 가설: 음높이와 발성 정보를 보완", M + 0.25, 2.65, leftW - 0.5, 17);
    txt(s, "기존 입력에 없는 F0 변화와 주기적 발성 정보가\n역할 분류를 보완할까?", { x: M + 0.3, y: 3.2, w: leftW - 0.6, h: 0.85, fontSize: 16 });
    stepBox(s, M + 0.3, 4.3, 2.5, 1.4, "Pitch · 음높이", "F0 중앙값·변동폭·기울기", C.blue, 13);
    stepBox(s, M + 3.15, 4.3, 2.5, 1.4, "Voicing · 유성성", "유성 비율·주기성", C.sky, 13);
    txt(s, "E05 42D + AF02 5D = 47D", { x: M + 0.3, y: 6.05, w: leftW - 0.6, h: 0.35, fontSize: 15, bold: true, color: C.blue, align: "center" });
    comparison(s, "85.95%", "86.63%", "42D · MLP", "+ Pitch/Voicing", "+0.67%p", "당시 내부 검증 Accuracy");
    txt(s, "Recall 변화: 대원 +0.71%p / 신고자 +0.64%p", { x: rightX + 0.25, y: 5.95, w: rightW - 0.5, h: 0.5, fontSize: 14, bold: true, color: C.blue, align: "center" });
    s.addNotes("historical cache 내부 비교: M04-100 42D 0.8595319610 → AF02 47D 0.8662664219, +0.67344609%p. Recall0 0.815608→0.822662, Recall1 0.900043→0.906484. clean chain의 별도 결과로 재해석하지 않는다. 짧거나 noisy/unvoiced한 구간의 pitch 추출 위험도 확인한 실험이다.");
  }

  // 5. 과거 조합 점수 대신 재현된 clean B4 52D/59D 비교 사용.
  {
    const s = page("활동성·엔트로피의 효과", "활동성과 엔트로피를 함께 추가해 전체 정확도를 높이고, 역할별 변화도 확인했습니다.");
    panels(s);
    title(s, "■ 가설: 평균이 놓치는 발화 변화 표현", M + 0.25, 2.65, leftW - 0.5, 17);
    txt(s, "음성 활동 패턴과 스펙트럼 분포 변화가\n짧거나 잘린 발화의 정보를 보완할까?", { x: M + 0.3, y: 3.2, w: leftW - 0.6, h: 0.85, fontSize: 16 });
    stepBox(s, M + 0.3, 4.3, 2.5, 1.4, "AF10 · 활동성", "음성 활동 비율·경계 패턴", C.blue, 13);
    stepBox(s, M + 3.15, 4.3, 2.5, 1.4, "AF11 · 엔트로피", "스펙트럼의 퍼짐·변동성", C.sky, 13);
    txt(s, "clean 52D + 활동성 4D + 엔트로피 3D = 59D", { x: M + 0.25, y: 6.05, w: leftW - 0.5, h: 0.35, fontSize: 14, bold: true, color: C.blue, align: "center" });
    comparison(s, "86.96%", "87.38%", "clean 52D", "clean 59D", "+0.42%p", "clean 내부 검증 Accuracy");
    txt(s, "Recall 변화: 대원 +1.23%p / 신고자 −0.33%p", { x: rightX + 0.25, y: 5.95, w: rightW - 0.5, h: 0.5, fontSize: 14, color: C.sub, align: "center" });
    s.addNotes("clean B4 same-run: 52D 0.869594, CM [[71215,14136],[9062,83478]]; 59D 0.873760, CM [[72265,13086],[9371,83169]], 순 정답 +741. 대원 Recall 0.834378→0.846680, 신고자 0.902075→0.898736. 따라서 두 역할 모두 개선이라고 하지 않는다. 52D prefix는 MFCC40/E05/AF02/AF04/AF09이며 CPP 단독 증분은 partially verified. old-cache 59D 87.46% 및 과거 조합 점수와 연결하지 않는다.");
  }

  // 6. F01/F02의 가설과 joint 149D 결과. 단독 개선량은 주장하지 않음.
  {
    const s = page("MFCC 분포·대역 대비와 149D 입력", "분포 요약과 대역 대비를 더한 149D 입력을 세 모델에 적용해 앙상블 후보로 검증했습니다.");
    panels(s);
    title(s, "■ 두 가지 보완 가설", M + 0.25, 2.65, leftW - 0.5, 17);
    title(s, "F01 · MFCC Robust Stats · 80D", M + 0.3, 3.25, leftW - 0.6, 16);
    txt(s, "평균·표준편차에 분포 정보를 더하면 도움이 될까?\n20계수의 q10·중앙값·q90·IQR을 추가", { x: M + 0.3, y: 3.8, w: leftW - 0.6, h: 0.9, fontSize: 14, paraSpaceAfter: 5 });
    title(s, "F02 · Spectral Contrast · 10D", M + 0.3, 5.0, leftW - 0.6, 16);
    txt(s, "대역 안의 강한 성분과 약한 성분의 대비가 보완할까?\n0–4 kHz 5대역의 contrast 평균·표준편차", { x: M + 0.3, y: 5.55, w: leftW - 0.6, h: 0.9, fontSize: 14, paraSpaceAfter: 5 });
    title(s, "■ 결합 입력과 내부 검증 결과", rightX + 0.25, 2.65, rightW - 0.5, 17);
    const bw = (rightW - 0.8) / 3;
    [["clean 59D", "기존 음향 특징"], ["F01 80D", "MFCC 분포"], ["F02 10D", "대역 대비"]].forEach(([t, b], i) => {
      node(s, rightX + 0.2 + i * (bw + 0.2), 3.25, bw, 0.95, t, b, i === 2);
      if (i < 2) txt(s, "+", { x: rightX + 0.2 + bw + i * (bw + 0.2), y: 3.55, w: 0.2, h: 0.3, align: "center", fontSize: 14, bold: true });
    });
    txt(s, "총 149D", { x: rightX + 0.3, y: 4.35, w: rightW - 0.6, h: 0.35, fontSize: 18, bold: true, color: C.blue, align: "center" });
    table(s, [["모델", "Accuracy"], ["TabM", "88.64%"], ["Tabular ResNet", "88.55%"], ["GANDALF", "88.58%"]], rightX + 0.3, 4.95, [(rightW - 0.6) * 0.6, (rightW - 0.6) * 0.4], { rowH: 0.36, fontSize: 12 });
    s.addNotes("F01는 20 MFCC 계수별 q10/median/q90/IQR의 80D. F02는 8kHz 발화의 0–250, 250–500, 500–1000, 1000–2000, 2000–4000Hz 5대역 spectral contrast mean/std 10D. Stage14 TabM 0.8864304546, ResNet 0.8854916775, GANDALF 0.8858458269. 별도 학습의 59D 결과와 차이를 F01/F02 각각의 독립 개선량으로 귀속하지 않는다. 개념 Diagram은 실제 분포/파형 그래프가 아니다.");
  }

  // 7. 성공한 특징 설명 바로 다음: 제외/보류 실험을 한 장으로 압축.
  {
    const s = page("제외·보류한 실험", "특징 수를 늘리는 대신, 비교 결과를 확인해 필요한 정보를 선별했습니다.");
    table(s, [
      ["실험 방향", "확인한 결과", "판단"],
      ["Duration", "초기 비교에서 정확도 변화가 거의 없음", "제외"],
      ["Energy · Spectral Flux", "각각 당시 42D 기준보다 정확도가 낮음", "제외"],
      ["Delta-MFCC", "재검토의 전체 순이득이 작고 역할별 Recall 변화가 반대", "보류"],
      ["eGeMAPS로 대체", "동일 clean 내부 비교: 59D 87.38% / eGeMAPS 86.40%", "기존 입력 유지"],
    ], M, 2.65, [2.7, 7.0, CW - 9.7], { rowH: 0.66, fontSize: 13.5 });
    insight(s, M, 6.05, CW, 0.65, "효과가 작거나 균형이 나빠진 방향은 제외·보류하고, 확인된 특징을 후속 비교에 사용했습니다.", C.light, 14);
    s.addNotes("Duration는 초기 Official 비교. Energy/Flux는 당시 42D M04-100 내부 비교. Delta-MFCC 재검토는 old-cache 59→61D +0.018%p, net +32, Recall0 +1.47%p / Recall1 −1.32%p로 HOLD. eGeMAPS는 같은 clean 계약에서 0.8639897465, clean59D 0.8737597742. 서로 다른 실험의 결과를 한 순위나 합산 효과로 읽지 않는다. Formants는 미실행, Stage12 pretrained는 성능 미평가이므로 실패 성능 사례로 넣지 않는다.");
  }

  // 8. 별도 stage의 clean 59D 저장 비교 기준: 모든 후보의 탐색표는 제거.
  {
    const s = page("모델 비교와 후보 선정", "TabM·ResNet을 비교 기준으로 유지하고, 신규 계열 중 가장 높은 GANDALF를 결합 후보에 포함했습니다.");
    const gap = 0.3, bw = (CW - 2 * gap) / 3;
    const models = [["TabM", "88.40%", "59D 동결 비교 기준"], ["Tabular ResNet", "88.38%", "59D 동결 비교 기준"], ["GANDALF", "88.17%", "신규 계열 중 가장 높음"]];
    models.forEach(([name, accuracy, role], i) => {
      const x = M + i * (bw + gap);
      card(s, x, 2.8, bw, 2.7);
      title(s, name, x + 0.2, 3.05, bw - 0.4, 18);
      stat(s, x + 0.1, 3.75, bw - 0.2, accuracy, role, i === 2 ? C.sky : C.blue, 32);
    });
    txt(s, "clean 59D · 같은 고정 내부 분할 · Training에서만 전처리 fit", { x: M, y: 2.3, w: CW, h: 0.3, fontSize: 12, color: C.sub, align: "center" });
    insight(s, M, 5.85, CW, 0.85, "GANDALF가 두 기준 모델을 넘지는 못했지만, 다른 모델 계열의 결합 후보로 포함해 검증했습니다.", C.light, 14);
    s.addNotes("TabM 0.8839795155460366, Tabular ResNet 0.8837546587517076는 Stage10 dual frozen references. GANDALF 약 0.8817은 Stage13 신규 6-family first-screen 중 최고. 동일 clean59D와 고정 내부 split을 사용했으나 다른 stage의 학습 모델이다. 세 모델만 같은 run에서 학습했다고 하지 않는다. 실제 오류 상보성은 이후 Stage14 149D 결과로 평가했으며 family가 다르다는 사실만으로 상보성을 보장하지 않는다.");
  }

  // 9. 앙상블 구성 + 고정 후보 최종 평가 통합. 초기 Official 사용 문장은 화면에서 제거.
  {
    const s = page("149D E_TRG 구성과 최종 평가", "세 모델을 동일 가중치로 결합하고, 고정한 연우 최종 모델에서 정확도 88.49%를 확인했습니다.");
    panels(s);
    title(s, "■ E_TRG · 세 모델 확률 평균", M + 0.25, 2.65, leftW - 0.5, 17);
    flow(s, M + 0.2, 3.3, leftW - 0.4, 2.2, {
      pre: [["149D", "음향 특징"]], preW: 1.1,
      branches: [["TabM", "", "1/3"], ["ResNet", "", "1/3"], ["GANDALF", "", "1/3"]], brW: 1.75, brGap: 0.14,
      post: [["확률 평균", "≥ 0.5 → 신고자\n< 0.5 → 대원"]], postW: 1.65,
    });
    txt(s, "추가 튜닝 없이 전체 Training으로 재학습\n149D 입력·동일 가중치·threshold 0.5 고정", { x: M + 0.3, y: 5.95, w: leftW - 0.6, h: 0.6, fontSize: 13, color: C.sub, align: "center" });
    stat(s, rightX + 0.2, 2.85, rightW - 0.4, "88.49%", "최종 평가 · Official Validation", C.blue, 42);
    const half = (rightW - 0.4) / 2;
    stat(s, rightX + 0.1, 4.5, half, "84.2%", "119대원 Recall", C.blue, 26);
    stat(s, rightX + 0.3 + half, 4.5, half, "92.5%", "신고자 Recall", C.sky, 26);
    txt(s, "내부 검증: 88.85%\n평가 집합이 달라 두 수치의 차이를 개선량으로 계산하지 않음", { x: rightX + 0.3, y: 6.0, w: rightW - 0.6, h: 0.6, fontSize: 12, color: C.sub, align: "center" });
    s.addNotes("E_TRG는 연우 개인 최종 실험 모델이며 팀 제출 모델 개발자로 오해시키지 않는다. Stage14 Internal 177,891행 Accuracy 0.8884822729; Stage15 full-Training refit의 Official Validation 111,947행/3,640통화 Accuracy 0.884918756, Recall0 0.84178661, Recall1 0.92489071. FINAL_VERIFIED, 세 reload EXACT_PASS, alignment/metric recomputation PASS. 이 refit에서는 Official을 scaler fit, gradient/loss, early stopping, checkpoint 선택에 쓰지 않았고 가중치/threshold 튜닝도 없다. 과거 E-series에서 같은 Official을 사용한 이력이 있어 독립된 최초 1회 holdout이라고 주장하지 않는다. 과거 사용 문장은 화면에서 생략하고 근거 기록에 보존한다. 학습 artifact는 Drive run mission2_final_fulltrain_official_validation_20261012T151108Z_a1606771에 있다.");
  }

  // 10. 실제 marginal slice만 시각화하며 잔여 오류를 원인으로 단정하지 않음.
  {
    const s = page("잔여 오류와 실험 결론", "음향 특징의 보완 효과를 비교하고, 짧은 발화와 겹침 구간의 잔여 오류를 확인했습니다.");
    panels(s);
    title(s, "■ 최종 평가의 구간별 오류율", M + 0.25, 2.65, leftW - 0.5, 17);
    barChart(s, ["전체", "0.5–1초", "Overlap"], [11.5, 15.7, 15.3], {
      x: M + 0.2, y: 3.2, w: leftW - 0.4, h: 2.9, name: "오류율", title: "오류율 (%)",
      chartColors: [C.line, C.sky, C.blue], valAxisMinVal: 0, valAxisMaxVal: 20, valAxisMajorUnit: 5, dataLabelFormatCode: "0.0",
    });
    txt(s, "각 구간은 서로 겹칠 수 있으며 오류율을 합산하지 않음", { x: M + 0.2, y: 6.3, w: leftW - 0.4, h: 0.28, fontSize: 11, color: C.sub, align: "center" });
    title(s, "■ 음향 특징 실험에서 얻은 결론", rightX + 0.25, 2.65, rightW - 0.5, 17);
    bullets(s, [
      "가설에 따라 음향 정보를 추가하고 비교",
      "효과를 확인한 특징으로 입력 구성",
      "모델 결합 후 고정 후보를 최종 평가",
    ], rightX + 0.3, 3.25, rightW - 0.6, 2.4, 16);
    insight(s, rightX + 0.2, 5.8, rightW - 0.4, 0.8, "짧은 발화·겹침 구간은 추가 분석이 필요한 영역", C.light, 13);
    s.addNotes("FINAL_VERIFIED Official 저장 예측: 전체 111,947행 중 12,883오류(11.5081%); 0.5–1초 27,392행 중 4,299오류(15.6944%); overlap 48,863행 중 7,457오류(15.2610%). 각 marginal slice는 겹칠 수 있어 동시에 짧고 겹친 구간의 결과로 해석하지 않는다. 저장 예측의 기술통계이며 음성 청취/원인 확정/새 학습을 수행한 결과가 아니다. 개인 분석의 의미를 설명하며 팀 제출 CNN에 특정 특징이 반영됐다는 기여 관계를 주장하지 않는다.");
  }
}


// Mission 3 — 환자 증상 인식 (26~29장)
// ======================================================================

// 26. 문제와 최종 모델 (main 유지 + 수빈 님 PR #200의 파이프라인·근거 문구)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 문제 정의와 접근", 26);
  coreMsg(s, "TF-IDF 기준선부터 RoBERTa 문맥 모델을 검증하고, 최종 3-seed + TF-IDF 앙상블 채택");
  const prob = [["입력", "119 통화 전사문\n(utterances[].text만)"], ["출력", "증상 9종 중 0~9개\n(다중 레이블)"], ["평가", "Macro F1\n(증상별 F1 평균)"]];
  const symNames = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"];
  const lw = 4.1;
  card(s, M, 2.45, lw, 4.35);
  title(s, "■ 문제 정의", M + 0.25, 2.6, lw - 0.5, 18);
  prob.forEach(([k, v], i) => {
    const y = 3.05 + i * 0.62;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 0.25, y, w: 0.95, h: 0.36, fill: { color: C.blue }, rectRadius: 0.08, line: { color: C.blue } });
    txt(s, k, { x: M + 0.25, y, w: 0.95, h: 0.36, fontSize: 12, bold: true, color: C.white, align: "center", valign: "middle" });
    txt(s, v, { x: M + 1.35, y: y - 0.04, w: lw - 1.55, h: 0.55, fontSize: 11.5 });
  });
  symNames.forEach((nm, i) => {
    const row = i < 5 ? 0 : 1, col = row ? i - 5 : i, cell = (lw - 0.5) / 5, off = row ? cell / 2 : 0;
    const cx = M + 0.25 + off + col * cell + cell / 2, y = 4.95 + row * 0.95;
    s.addImage({ path: "figures/canva/sym_" + i + ".png", x: cx - 0.26, y, w: 0.52, h: 0.52 });
    txt(s, nm, { x: cx - cell / 2, y: y + 0.56, w: cell, h: 0.22, fontSize: 9.5, color: C.sub, align: "center" });
  });
  const rx = M + lw + 0.4, rw = W - M - rx;
  txt(s, "■ 모델 구성", { x: rx, y: 2.45, w: rw, h: 0.4, fontSize: 18, bold: true });
  flow(s, rx, 2.95, rw, 1.75, {
    pre: [["입력 정제", "발화 text만\n원본 순서 유지"]], preW: 1.55,
    branches: [["RoBERTa-large", "seed 42·43·44 평균", "55%"], ["TF-IDF + 로지스틱 회귀", "초기 기준선", "45%"]], brW: 2.5, brGap: 0.15,
    post: [["가중 평균", "증상별 ≥ 0.5\n판정"]], postW: 1.75,
  });
  txt(s, "■ 왜 이 방법인가", { x: rx, y: 4.95, w: rw, h: 0.4, fontSize: 18, bold: true });
  bullets(s, [
    "오류 단서: 키워드 탐색에서 전사에 직접 근거가 드러나지 않는 양성 라벨을 발견",
    "모델 비교: 키워드 표현과 문맥 표현이 다른 오류를 낼 가능성을 확인",
    "검증 원칙: 결합 비율은 Training 내부에서 확정하고, 공식 Val은 채점만 수행",
  ], rx, 5.4, rw, 1.4, 13);
  caption(s, "근거: submission/mission3/README.md, docs/decisions.md M3-FINAL-002");
  s.addNotes("근거: submission/mission3/README.md '모델' 절, docs/decisions.md M3-FINAL-002. 파이프라인·'왜' 문구는 이수빈 님 PR #200(subin/ppt-m3-storytelling-20261013)에서 가져옴.");
}

// =====================================================================
// 27. 최종 결과
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 최종 결과", 27);
  coreMsg(s, "최종 3-seed RoBERTa + TF-IDF 앙상블: 공식 Validation Macro F1 65.34%");
  card(s, M, 2.45, 7.3, 4.35);
  txt(s, "65.34%", { x: M + 0.2, y: 2.65, w: 2.6, h: 0.75, fontSize: 38, color: C.blue, bold: true, align: "center", valign: "middle" });
  txt(s, "Macro F1 · 공식 Val 3,640건", { x: M + 0.2, y: 3.43, w: 2.6, h: 0.5, fontSize: 11.5, color: C.sub, align: "center" });
  insight(s, M + 0.2, 4.28, 2.6, 2.3, [
    { text: "Micro F1  66.90%", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "빈 예측 149건", options: { breakLine: true } },
    { text: "앙상블 가중치\nRoBERTa 평균 55%\n+ TF-IDF 45%", options: { fontSize: 10.5 } },
  ]);
  // 증상별 F1 (최종 앙상블, 공식 Val) — 낮은 3개만 빨강. 근거: dcc-lab results/M3-FINAL-002/metrics.json
  const sym = [["열상",88.28],["복통",82.71],["고열",70.01],["호흡곤란",67.95],["어지러움",65.70],["구토",60.88],["전신쇠약",59.39],["두통",53.15],["오심",40.00]];
  barChart(s, sym.map(r => r[0]), sym.map(r => r[1]), {
    name: "F1 (%)", barDir: "bar", catAxisOrientation: "maxMin", dataLabelFormatCode: '0.00"%"', x: M + 2.95, y: 2.6, w: 4.2, h: 4.1,
    chartColors: sym.map((r, i) => (i >= 6 ? C.red : C.sky)), showTitle: true, title: "증상별 F1 (%) · 최종 앙상블 공식 Val", titleFontSize: 11,
    dataLabelFontSize: 11, dataLabelFontBold: false, valAxisMinVal: 0, valAxisMaxVal: 100, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 12,
  });
  const rx = M + 7.6, rw = W - M - rx;
  card(s, rx, 2.45, rw, 2.75);
  title(s, "■ 개선 흐름 (모두 공식 Val)", rx + 0.2, 2.56, rw - 0.4);
  [["TF-IDF", "56.89%"], ["KcBERT", "62.46%"], ["RoBERTa-large", "64.33%"], ["앙상블", "65.34%"]].forEach(([m, v], i) => {
    const y = 3.0 + i * 0.52, last = i === 3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.25, y, w: rw - 0.5, h: 0.42, fill: { color: last ? C.blue : C.light }, rectRadius: 0.08, line: { color: last ? C.blue : C.light } });
    txt(s, m, { x: rx + 0.4, y, w: rw - 1.9, h: 0.42, fontSize: 13, bold: last, color: last ? C.white : C.text, valign: "middle" });
    txt(s, v, { x: rx + rw - 1.6, y, w: 1.2, h: 0.42, fontSize: 13, bold: true, color: last ? C.white : C.blue, align: "right", valign: "middle" });
  });
  card(s, rx, 5.35, rw, 1.45, C.bgGray);
  title(s, "■ 가장 약한 증상 (빨강)", rx + 0.2, 5.47, rw - 0.4);
  txt(s, "오심 40.00% · 두통 53.15% · 전신쇠약 59.39%", { x: rx + 0.2, y: 5.92, w: rw - 0.4, h: 0.6, fontSize: 13.2, bold: true, color: C.red, align: "center", valign: "middle" });
  caption(s, "근거: submission/mission3/README.md, m3_final_train.executed.ipynb. 모델별 점수 차이가 특정 원인만의 효과임을 뜻하지 않음.");
  s.addNotes("근거: submission/mission3/README.md 결과 표(seed 42·43·44 64.79%/64.41%/64.45%, 앙상블 65.34%, Micro 66.90%, 빈 예측 149). 개선 흐름: M3-OFFVAL-001(TF-IDF 56.89%, KcBERT 62.46%), M3-OFFVAL-002(RoBERTa-large 64.33%), M3-FINAL-002(앙상블 65.34%). 증상별 약점: M3-FINAL-002(오심·두통·전신쇠약 F1이 가장 낮음).");
}

// 28. 환자 증상 인식 — 모델 선정 (네이티브 편집 가능 차트와 카드)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "환자 증상 인식 - 모델 선정", 28);
  coreMsg(s, "서로 다른 오류 특성을 지닌 문맥 모델과 어휘 모델을 결합해 최종 제출 모델 선정");
  barChart(s, ["TF-IDF", "KcBERT", "RoBERTa-large", "최종 앙상블"],
    [56.89, 62.46, 64.33, 65.34], {
      name: "Macro F1 (%) · 공식 Validation", barDir: "bar", catAxisOrientation: "maxMin",
      x: M + 0.18, y: 2.25, w: CW - 0.36, h: 1.83,
      chartColors: [C.sky, C.blue, C.blue, C.mint], showTitle: true,
      title: "모델별 공식 Validation Macro F1 (%)", titleFontSize: 11.5,
      valAxisMinVal: 0, valAxisMaxVal: 100, valAxisHidden: true,
      valGridLine: { style: "none" }, dataLabelFormatCode: '0.00"%"',
      dataLabelFontSize: 12, catAxisLabelFontSize: 11.5,
    });
  const y = 4.18, h = 2.34, gap = 0.30, w = (CW-gap)/2, rx = M+w+gap;
  card(s, M, y, w, h, C.light);
  title(s, "REJECT · 유사한 문맥 모델 결합", M+0.18, y+0.13, w-0.36, 15);
  txt(s, "KcBERT + RoBERTa", { x:M+0.18, y:y+0.53, w:w-0.36, h:0.27, fontSize:12.6, bold:true, color:C.blue });
  bullets(s, [
    "두 모델 모두 Transformer 기반 문맥 표현을 사용",
    "전사 근거가 부족한 증상에서 오류가 반복",
    "두 모델이 함께 틀리는 사례가 많아 결합 효과가 제한적",
    "내부 Macro F1 개선폭 +0.28%p"
  ], M+0.20, y+0.89, w-0.40, 1.34, 11.4);
  card(s, rx, y, w, h, C.mintLight);
  title(s, "ADOPT · 서로 다른 표현 방식 결합", rx+0.18, y+0.13, w-0.36, 15);
  txt(s, "RoBERTa + TF-IDF", { x:rx+0.18, y:y+0.53, w:w-0.36, h:0.27, fontSize:12.6, bold:true, color:C.blue });
  bullets(s, [
    "RoBERTa: 문맥과 부정 표현을 반영",
    "TF-IDF: 직접적인 증상 단어와 어휘 패턴을 포착",
    "서로 다른 오류 특성을 확률 결합으로 보완",
    "RoBERTa 55% + TF-IDF 45%",
    "최종 공식 Validation Macro F1 65.34%"
  ], rx+0.20, y+0.86, w-0.40, 1.43, 10.8);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x:M, y:6.62, w:CW, h:0.39, fill:{ color:C.light }, rectRadius:0.06, line:{ color:C.sky, width:0.8 } });
  txt(s, "비슷한 모델을 추가하기보다 서로 다른 정보를 학습하는 모델을 결합했다", { x:M+0.16, y:6.64, w:CW-0.32, h:0.33, fontSize:14, bold:true, color:C.blue, align:"center", valign:"middle" });
  caption(s, "근거: docs/decisions.md M3-ENS-001·M3-FINAL-002. 내부 홀드아웃과 공식 Validation 차이를 직접 인과 효과로 해석하지 않음.", 7.14);
  s.addNotes("근거: docs/decisions.md M3-ENS-001(Training 내부 80:20: 신경망 결합 64.55%, RoBERTa 64.27%, +0.28%p로 사전 기준 +1.00%p 미달), M3-FINAL-002(최종 RoBERTa 3-seed 55%와 TF-IDF 45% 확률 결합, 공식 Validation Macro F1 65.34%). 상단의 공식 Validation 비교는 서로 다른 학습 조건이므로 인과 효과가 아니며 Training 내부 홀드아웃과 직접 비교하지 않음.");
}

// 29. MISSION 3 환자 증상 인식 — 오류분석 (왼쪽만 사용, 오른쪽 비워 둠)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "MISSION 3 환자 증상 인식 - 오류분석", 29);
  const lw = 6.18;
  card(s, M, 1.60, lw, 2.52);
  title(s, "오류 분석 - 전사에 직접 표현이 없는 양성 라벨", M+0.16, 1.72, lw-0.32, 13.7);
  s.addChart(pres.charts.DOUGHNUT, [{ name:"양성 라벨", labels:["직접 표현 없음", "직접 표현 있음"], values:[52,48] }], {
    x:M+0.14, y:2.12, w:2.55, h:1.73, holeSize:63, chartColors:[C.red,C.line],
    showLegend:false, showPercent:false, showValue:false, showLabel:false,
    dataBorder:{ pt:1, color:"FFFFFF" }
  });
  txt(s, "약 52%", { x:M+0.14, y:2.68, w:2.55, h:0.42, fontSize:19, color:C.red, bold:true, align:"center", valign:"middle" });
  txt(s, "직접 표현 없음 52% · 있음 48%", { x:M+0.18, y:3.76, w:2.6, h:0.24, fontSize:9.4, color:C.sub, align:"center" });
  txt(s, "92%", { x:M+2.95, y:2.26, w:1.28, h:0.56, fontSize:27, bold:true, color:C.red, align:"center" });
  txt(s, "두 모델이 놓친 양성 중\n직접 표현 없는 라벨", { x:M+2.81, y:2.87, w:1.56, h:0.53, fontSize:9.7, color:C.sub, align:"center" });
  txt(s, "약 90%", { x:M+4.45, y:2.26, w:1.47, h:0.56, fontSize:24, bold:true, color:C.sky, align:"center" });
  txt(s, "표현 있는 라벨 중\n맞힌 비율", { x:M+4.40, y:2.87, w:1.55, h:0.53, fontSize:9.7, color:C.sub, align:"center" });
  txt(s, "※ 과거 TF-IDF·KcBERT 오류 분석이며 최종 RoBERTa 오류 통계가 아님", { x:M+2.86, y:3.57, w:3.02, h:0.4, fontSize:9.3, bold:true, color:C.blue, align:"center" });
  card(s, M, 4.24, lw, 2.48);
  title(s, "저성능 3개 증상의 직접 표현 미검출률", M+0.17, 4.36, lw-0.34, 14);
  txt(s, "Training 내부 홀드아웃 5,792건", { x:M+0.17, y:4.77, w:lw-0.34, h:0.26, fontSize:10, color:C.sub });
  [["오심",88.6],["전신쇠약",75.7],["두통",63.7]].forEach(([name,value],i)=>{
    const y=5.16+i*0.45;
    txt(s,name,{ x:M+0.22,y,w:1.03,h:0.27,fontSize:11.5,bold:true,color:C.text });
    s.addShape(pres.shapes.RECTANGLE,{ x:M+1.42,y:y+0.03,w:3.6,h:0.20,fill:{color:C.bgGray},line:{color:C.line,width:0.5} });
    s.addShape(pres.shapes.RECTANGLE,{ x:M+1.42,y:y+0.03,w:3.6*value/100,h:0.20,fill:{color:i===0?C.red:C.sky},line:{color:i===0?C.red:C.sky} });
    txt(s,value.toFixed(1)+"%",{ x:M+5.07,y:y-0.01,w:0.75,h:0.31,fontSize:11.2,bold:true,color:C.blue,align:"right" });
  });
  txt(s, "※ 직접 표현 미검출은 증상 부재나 오라벨을 의미하지 않으며 정규식은 구어체 변형을 모두 포착하지 못함", { x:M+0.17, y:6.75, w:lw-0.34, h:0.34, fontSize:9.1, color:C.sub });
  caption(s, "근거: docs/experiment_log.md M3-ERR-001; reports/mission3/error_analysis/m3_error_analysis.json · README.md",7.15);
  s.addNotes("과거 TF-IDF·KcBERT 오류 분석: 전사에 직접 표현이 없는 양성 라벨 약 52%, 두 모델이 놓친 양성 중 직접 표현 없는 사례 92%, 직접 표현이 있는 라벨 중 맞힌 비율 약 90%. 이 수치는 최종 RoBERTa 오류 통계가 아님. Training 내부 홀드아웃 5,792건에서 직접 표현 미검출률: 오심 88.6%, 전신쇠약 75.7%, 두통 63.7%. 정규식 키워드 탐색은 구어체 변형을 모두 포착하지 못하므로 직접 표현 없음은 증상 부재 또는 오라벨의 증거가 아님. 근거: docs/experiment_log.md M3-ERR-001; reports/mission3/error_analysis/m3_error_analysis.json; reports/mission3/error_analysis/README.md.");
}

// =====================================================================
// 30. 04-(1) 성능 요약과 교훈 (기존 종합 성과 장)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(1) 성능 요약과 교훈", 30);
  coreMsg(s, "공식 Validation은 튜닝에 쓰지 않고, 결과를 보기 전에 고정한 기준만 확인했다.");
  const v = t => strong(t);
  const tiles = [
    ["M1 신고자 성별", "음향 특징 + x-vector + LR", "98.10%", "+2.99%p"],
    [M2.summary.task, M2.summary.model, M2.summary.val, M2.summary.gain],
    ["M3 증상 9종", "RoBERTa-large × 3 + TF-IDF", "65.34%", "+8.45%p"],
  ];
  const tw = (CW - 2 * 0.3) / 3;
  tiles.forEach(([t, m, val, gain], i) => {
    const x = M + i * (tw + 0.3);
    card(s, x, 2.45, tw, 2.2);
    txt(s, t, { x: x + 0.25, y: 2.55, w: tw - 0.5, h: 0.4, fontSize: 15, bold: true, color: C.blue });
    txt(s, val, { x: x + 0.25, y: 2.95, w: tw * 0.6, h: 0.9, fontSize: 38, bold: true, color: C.blue, valign: "middle" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + tw - 1.55, y: 3.15, w: 1.3, h: 0.5, fill: { color: C.mintLight }, rectRadius: 0.1, line: { color: C.mint } });
    txt(s, "▲ " + gain, { x: x + tw - 1.55, y: 3.15, w: 1.3, h: 0.5, fontSize: 13, bold: true, color: "2E7D32", align: "center", valign: "middle" });
    txt(s, m, { x: x + 0.25, y: 3.95, w: tw - 0.5, h: 0.3, fontSize: 11.5, color: C.sub });
    txt(s, i === 2 ? "Macro F1 · 공식 Val" : "Accuracy · 공식 Val", { x: x + 0.25, y: 4.25, w: tw - 0.5, h: 0.3, fontSize: 10.5, color: C.sub });
  });
  txt(s, "baseline: M1 95.11%, M2 86.44%, M3 56.89%", { x: M, y: 4.7, w: CW, h: 0.25, fontSize: 11, color: C.sub, align: "right" });
  const lessons = [
    ["기준은 결과를 보기 전에\n고정한다","결과를 본 뒤 기준을 낮추면 Validation에 과적합된다"],
    ["기각도 기록한다", "효과 없던 실험과 그 이유를 decisions.md에 남겨 같은 시도를 반복하지 않음"],
    ["규칙은 설계 단계에서 지킨다", "규칙 위반 소지가 있는 특징은 성능이 좋아도 제거 (M2 통화 단위 정보)"],
  ];
  const cw = (CW - 2 * 0.3) / 3;
  lessons.forEach(([t, d], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 4.95, cw, 1.9);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 5.12, w: 0.5, h: 0.5, fill: { color: C.blue }, line: { color: C.blue } });
    txt(s, String(i + 1), { x: x + 0.25, y: 5.12, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.white, align: "center", valign: "middle" });
    txt(s, t, { x: x + 0.9, y: 5.07, w: cw - 1.05, h: 0.6, fontSize: 15, bold: true, color: C.blue, valign: "middle" });
    txt(s, d, { x: x + 0.25, y: 5.8, w: cw - 0.5, h: 0.95, fontSize: 12 });
  });
  s.addNotes("근거: submission/mission1/README.md(98.10%), " + M2.evidence + ", submission/mission3/README.md(65.34%). baseline: M1 95.11%(52D LR, docs/decisions.md 2026-10-05), M2 86.44%(L1 LightGBM, M2-RETRAIN-001), M3 56.89%(TF-IDF, M3-OFFVAL-001).");
}

// =====================================================================
// 31. 04-(2) 활용 흐름과 기대효과 — 편집 가능한 연결 도형, 현장 적용 원칙 명시
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(2) 활용 흐름과 기대효과", 31);
  coreMsg(s, "세 미션을 연결해 신고 전화의 화자와 증상 정보를 접수 화면에 구조화한다.");
  // 연결 과정: GIF 대신 개별 도형·텍스트로 구성 (PowerPoint에서 직접 편집 가능).
  // 다섯 박스를 동일 크기로 배치하며, 사이에는 하늘색 오른쪽 화살표를 둔다.
  {
    const labels = ["신고 전화", "M2\n화자 구분", "M1\n성별 분류", "M3\n증상 인식", "상황실"];
    const bw = 2.15, bh = 0.77, gap = 0.38, fy = 2.58;
    const fx = (W - (labels.length * bw + (labels.length - 1) * gap)) / 2;
    labels.forEach((label, i) => {
      const x = fx + i * (bw + gap);
      const last = i === labels.length - 1;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y: fy, w: bw, h: bh, rectRadius: 0.09,
        fill: { color: last ? C.blue : C.light },
        line: { color: last ? C.blue : C.sky, width: 1.2 },
      });
      txt(s, label, {
        x: x + 0.06, y: fy, w: bw - 0.12, h: bh,
        fontFace: F, fontSize: 15, bold: true,
        color: last ? C.white : C.blue,
        align: "center", valign: "middle", breakLine: false,
      });
      if (!last) arrow(s, x + bw + (gap - 0.22) / 2, fy + (bh - 0.28) / 2, 0.22, 0.28, C.sky);
    });
  }
  const gap = 0.34, cw = (CW - gap) / 2, y = 3.95, h = 2.30;
  const blocks = [
    { x: M, heading: "■ 기대효과", items: [
      ["접수 과정 지원", "통화 정보를 자동으로 구조화해 수기 기록 부담 완화"],
      ["정보 누락 완화", "통화 중 언급된 증상 후보를 화면에 표시"],
      ["판단 보조", "접수요원이 확인할 정보를 한 화면에 정리"]
    ] },
    { x: M + cw + gap, heading: "■ 현장 적용 원칙", items: [
      ["근거 확인", "모델 결과와 통화 원문을 함께 표시"],
      ["불확실성 구분", "신뢰도가 낮은 결과는 별도로 표시"],
      ["사람의 수정", "접수요원이 결과를 수정하거나 제외"],
      ["정보 최소화", "개인정보는 필요한 범위에서만 처리"]
    ] }
  ];
  blocks.forEach(({x, heading, items}) => {
    card(s, x, y, cw, h);
    title(s, heading, x + 0.22, y + 0.13, cw - 0.44, 17);
    const itemY = y + 0.61, itemStep = items.length === 4 ? 0.40 : 0.52;
    items.forEach(([label, body], i) => {
      const yy = itemY + i * itemStep;
      txt(s, [strong(label), { text: " — " + body }], {
        x: x + 0.23, y: yy, w: cw - 0.46, h: items.length === 4 ? 0.38 : 0.46,
        fontSize: items.length === 4 ? 11.1 : 11.7, breakLine: false, valign: "middle"
      });
    });
  });
  insight(s, M, 6.39, CW, 0.55, [
    strong("활용 범위  "),
    { text: "모든 출력은 접수요원의 판단을 돕는 보조 정보이며, 사람이 통화 원문을 확인한 뒤 최종 결정한다." },
  ], C.bgGray, 12.2);
  s.addNotes("기대효과는 실제로 검증된 성과가 아니라 제안된 활용 가능성이다. 실제 접수 시간 단축과 현장 효과는 별도의 모의 접수 시나리오로 검증해야 한다. 연결 과정은 GIF 대신 PowerPoint에서 편집 가능한 박스와 화살표 도형으로 구성했다.");
}

// =====================================================================
// 32. 04-(3) 사회안전 시사점과 향후 계획 — 신규 삽입 (별첨은 33쪽부터)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(3) 사회안전 시사점과 향후 계획", 32);
  coreMsg(s, "통화 정보를 구조화해 접수와 현장 인계의 정보 누락을 줄이고, 향후 미션별 오류 분석과 모델 보완을 진행한다.");
  const y = 2.38, h = 3.88, gap = 0.28, leftW = 4.88, rightX = M + leftW + gap, rightW = CW - leftW - gap;
  card(s, M, y, leftW, h);
  card(s, rightX, y, rightW, h);
  title(s, "■ 사회안전 시사점", M + 0.22, y + 0.12, leftW - 0.44, 17);
  const implications = [
    ["초기 대응의 정보 공백 완화", "긴박하거나 소음이 있는 통화의 증상 정보 재확인\n짧거나 불완전한 표현도 증상 후보로 표시"],
    ["현장 인계의 일관성 향상", "신고자·증상 정보를 같은 형식으로 기록\n확인된 정보를 출동팀 전달 자료로 활용"],
    ["반복 오류를 활용한 개선", "비식별 사례에서 반복되는 실패 조건 확인\n교육과 모델 개선의 근거로 활용"]
  ];
  implications.forEach(([heading, detail], i) => {
    const yy = y + 0.64 + i * 1.02;
    s.addShape(pres.shapes.OVAL, { x: M + 0.24, y: yy + 0.02, w: 0.34, h: 0.34, fill: { color: C.blue }, line: { color: C.blue } });
    txt(s, String(i + 1), { x: M + 0.24, y: yy + 0.02, w: 0.34, h: 0.34, fontSize: 12, bold: true, color: C.white, align: "center", valign: "middle" });
    txt(s, heading, { x: M + 0.71, y: yy, w: leftW - 0.96, h: 0.39, fontSize: 13, bold: true, color: C.blue, valign: "middle" });
    txt(s, detail, { x: M + 0.71, y: yy + 0.42, w: leftW - 0.97, h: 0.55, fontSize: 10.7, color: C.text, breakLine: false });
  });
  title(s, "■ 향후 활동 계획", rightX + 0.22, y + 0.12, rightW - 0.44, 17);
  // 미션별로 실제 명시된 후속 계획만 반영한 3단계 구성.
  const plans = [
    ["오류 원인 분석", ["M1  오분류 통화 정성 분석", "M2  짧은 응답 발화의 오류 사례 확인", "M3  근거 없는 라벨 구간 오류 분석"]],
    ["미션별 모델 보완", ["M1  피치(F0) 특징 검증", "M2  짧은 응답 발화 보강", "M3  오심·두통 성능 개선"]],
    ["실시간 활용 시나리오 검토", ["통화 도중 세 미션의 처리 순서 검토", "세 미션 결과를 접수 화면에 제공하는 방식 검토", "사람이 최종 확인하는 활용 범위 명시"]]
  ];
  plans.forEach(([heading, lines], i) => {
    const px = rightX + 0.21, py = y + 0.60 + i * 1.10, ph = 1.01;
    const last = i === plans.length - 1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: px, y: py, w: rightW - 0.42, h: ph, fill: { color: last ? C.mintLight : C.bgGray }, rectRadius: 0.07, line: { color: last ? C.mint : C.line, width: 0.7 } });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: px + 0.12, y: py + 0.25, w: 0.43, h: 0.50, fill: { color: last ? C.mint : C.blue }, rectRadius: 0.05, line: { color: last ? C.mint : C.blue } });
    txt(s, String(i + 1), { x: px + 0.12, y: py + 0.25, w: 0.43, h: 0.50, fontSize: 15, bold: true, color: C.white, align: "center", valign: "middle" });
    txt(s, heading, { x: px + 0.68, y: py + 0.19, w: 2.42, h: 0.60, fontSize: 12.5, bold: true, color: last ? "2E7D32" : C.blue, valign: "middle" });
    txt(s, lines.join("\n"), { x: px + 3.18, y: py + 0.14, w: rightW - 3.48, h: 0.73, fontSize: 10.6, color: C.text, valign: "middle", breakLine: false });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.39, w: CW, h: 0.60, fill: { color: C.light }, rectRadius: 0.07, line: { color: C.light } });
  txt(s, [strong("향후 정리 내용  "), { text: "미션별 오류 분석 결과 · 특성 및 모델 검증 결과 · 실시간 처리 시나리오" }],
    { x: M + 0.23, y: 6.47, w: CW - 0.46, h: 0.22, fontSize: 12.0, valign: "middle" });
  txt(s, "※ 실제 접수 시간과 현장 적용 효과는 현재 결과만으로 확인할 수 없으며, 별도의 평가가 필요하다.",
    { x: M + 0.23, y: 6.72, w: CW - 0.46, h: 0.18, fontSize: 9.5, color: C.sub });
  s.addNotes("사회안전 시사점과 향후 활동 계획. 미션별 오류 분석·모델 보완 및 통화 도중 실시간 처리 시나리오를 검토하는 계획이다. 현장 적용 효과는 별도 평가가 필요하다.");
}

// =====================================================================
// =====================================================================
// 33. 별첨 1 — Mission 3 데이터 특성과 모델 설계 근거 (기존 p17 + p21 통합)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "별첨", "Mission 3", "별첨 1. Mission 3 데이터 특성과 모델 설계 근거", 33);
  coreMsg(s, "데이터의 불균형과 입력 길이를 고려해 학습 손실과 최종 앙상블 방식을 설계했다.");

  // 상단: 데이터 관찰(좌)과 손실 설계(우) — 수식은 모두 편집 가능한 텍스트.
  const topY = 2.36, topH = 2.82, gap = 0.25, leftW = 6.0;
  const rightX = M + leftW + gap, rightW = CW - leftW - gap;
  card(s, M, topY, leftW, topH);
  card(s, rightX, topY, rightW, topH);
  title(s, "■ 데이터 관찰 → 설계 결정", M + 0.2, topY + 0.11, leftW - 0.4, 15);
  const observations = [
    ["라벨 불균형", "9개 증상 양성 비율 9.9~23.2%", "pos_weight BCE + Soft macro-F1"],
    ["입력 길이", "40.7%가 256토큰 초과 · 512토큰 +0.74%p", "기준 +1.00%p 미달·약 4배 시간 → 256 유지"],
    ["입력 정보", "전사 text만 원본 시간순 연결", "화자 태그·허용되지 않은 라벨 제외"],
  ];
  observations.forEach(([name, observed, decision], i) => {
    const y = topY + 0.53 + i * 0.73;
    txt(s, name, { x: M + 0.23, y, w: 1.15, h: 0.25, fontSize: 11.3, bold: true, color: C.blue });
    txt(s, observed, { x: M + 1.43, y, w: 4.32, h: 0.25, fontSize: 10.8, color: C.text });
    txt(s, "→ " + decision, { x: M + 1.43, y: y + 0.29, w: 4.32, h: 0.3, fontSize: 10.6, bold: true, color: C.blue });
    if (i < 2) s.addShape(pres.shapes.LINE, { x: M + 0.22, y: y + 0.65, w: leftW - 0.44, h: 0, line: { color: C.line, width: 0.6 } });
  });

  title(s, "■ 학습 손실 설계", rightX + 0.2, topY + 0.11, rightW - 0.4, 15);
  txt(s, "증상별 가중치", { x: rightX + 0.2, y: topY + 0.52, w: rightW - 0.4, h: 0.23, fontSize: 10.8, bold: true, color: C.sub });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rightX + 0.18, y: topY + 0.78, w: rightW - 0.36, h: 0.45, fill: { color: C.light }, rectRadius: 0.06, line: { color: C.line, width: 0.6 } });
  txt(s, "wⱼ = clip(0.5 × √((N − Pⱼ) / max(Pⱼ, 1)), 1, 6)", { x: rightX + 0.28, y: topY + 0.78, w: rightW - 0.56, h: 0.45, fontSize: 12.7, bold: true, color: C.blue, valign: "middle", align: "center" });
  txt(s, "N: Training 통화 수 · Pⱼ: 증상 j 양성 통화 수", { x: rightX + 0.25, y: topY + 1.28, w: rightW - 0.5, h: 0.23, fontSize: 10.1, color: C.sub });
  txt(s, "희소 증상의 양성 오류에 더 큰 가중치", { x: rightX + 0.25, y: topY + 1.54, w: rightW - 0.5, h: 0.23, fontSize: 10.1, color: C.text });
  txt(s, "최종 학습 손실", { x: rightX + 0.2, y: topY + 1.81, w: rightW - 0.4, h: 0.23, fontSize: 10.8, bold: true, color: C.sub });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rightX + 0.18, y: topY + 2.07, w: rightW - 0.36, h: 0.42, fill: { color: C.bgGray }, rectRadius: 0.06, line: { color: C.line, width: 0.6 } });
  txt(s, "L = BCEWithLogits(pos_weight = wⱼ) + 1 − meanⱼ(SoftF1ⱼ)", { x: rightX + 0.27, y: topY + 2.07, w: rightW - 0.54, h: 0.42, fontSize: 11.6, bold: true, color: C.blue, valign: "middle", align: "center" });
  txt(s, "실효 batch 16 = micro batch 8 × 2 · Soft-F1은 micro batch별 계산", { x: rightX + 0.2, y: topY + 2.54, w: rightW - 0.4, h: 0.18, fontSize: 9.3, color: C.sub });

  // 하단: 3단계 최종 앙상블과 판정 — 가중치·임계값 고정.
  const bottomY = 5.31, bottomH = 1.55;
  card(s, M, bottomY, CW, bottomH);
  title(s, "■ 최종 앙상블과 판정", M + 0.21, bottomY + 0.08, CW - 0.42, 15);
  const boxY = bottomY + 0.44, boxH = 0.77, bw = 3.75, arrowW = 0.25;
  const steps = [
    ["STEP 1 · RoBERTa 3-seed 평균", "pR = (p42 + p43 + p44) / 3"],
    ["STEP 2 · 최종 앙상블 확률", "pfinal = 0.55 × pR + 0.45 × pTF-IDF"],
    ["STEP 3 · 최종 판정", "ŷⱼ = 1 if pfinal,ⱼ ≥ 0.5"],
  ];
  steps.forEach(([lab, eq], i) => {
    const x = M + 0.2 + i * (bw + arrowW + 0.1);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: boxY, w: bw, h: boxH, fill: { color: i === 1 ? C.light : C.white }, rectRadius: 0.06, line: { color: i === 1 ? C.sky : C.line, width: 1 } });
    txt(s, lab, { x: x + 0.12, y: boxY + 0.11, w: bw - 0.24, h: 0.2, fontSize: 10.2, color: C.sub, bold: true, align: "center" });
    txt(s, eq, { x: x + 0.07, y: boxY + 0.37, w: bw - 0.14, h: 0.29, fontSize: i === 1 ? 13.6 : 14, color: C.blue, bold: true, align: "center" });
    if (i < 2) txt(s, "→", { x: x + bw + 0.015, y: boxY + 0.25, w: arrowW, h: 0.3, fontSize: 16, bold: true, color: C.sky, align: "center" });
  });
  txt(s, "모든 증상에 동일한 threshold 0.5 적용 · 증상별 threshold 조정 및 별도 후처리 없음", { x: M + 0.2, y: bottomY + 1.27, w: CW - 0.4, h: 0.2, fontSize: 10.1, bold: true, color: C.blue, align: "center" });
  caption(s, "※ 입력 길이와 512토큰 실험은 Training 내부 80:20 홀드아웃 기준이며, 공식 Validation은 모델 선택에 사용하지 않음.", 7.02);
  s.addNotes("기존 p17 + p21 통합. 데이터: Training 29,200통화, 평가 전용 Validation 3,640통화, 9개 증상, Training 통화당 평균 증상 수 1.43. 라벨별 양성 비율(Training): 열상 23.2, 어지러움 21.7, 전신쇠약 18.2, 고열 17.8, 구토 15.3, 호흡곤란 13.4, 복통 11.9, 오심 11.4, 두통 9.9(%). 입력 길이와 512토큰 실험은 Training 내부 80:20 홀드아웃: 40.7% 256토큰 초과, Macro F1 +0.74%p, 사전 기준 +1.00%p 미달, 학습시간 약 4배. 전사 utterances[].text만 원본 순서로 연결하며 화자 태그와 허용되지 않은 라벨은 사용하지 않음. Soft macro-F1 상세: pᵢⱼ = sigmoid(zᵢⱼ), SoftF1ⱼ = 2Σᵢ(pᵢⱼyᵢⱼ) / [Σᵢpᵢⱼ + Σᵢyᵢⱼ + 10⁻⁸]. 가중치 wⱼ = clip(0.5 × √((N − Pⱼ) / max(Pⱼ,1)),1,6). 실효 batch 16 = micro 8 × 2, soft-F1 micro batch별 계산. 최종 RoBERTa seeds 42·43·44 확률 평균 가중치 55%와 TF-IDF 확률 가중치 45%를 결합, 모든 증상 threshold 0.5 적용(별도 후처리 없음). 근거: reports/eda/eda_report.md; docs/experiment_log.md M3-ROBERTA-007; submission/mission3/README.md; submission/mission3/train/train_final.py; submission/mission3/train/build_ensemble.py; submission/mission3/mission3_model.py.");
}

// =====================================================================
// 34. 별첨 2 — M3 입력 길이 확대 실험
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "별첨", "Mission 3", "별첨 2. M3 입력 길이 확대 실험", 34);
  coreMsg(s, "512토큰은 성능이 개선됐지만 사전 채택 기준과 계산비용 조건을 충족하지 못함");

  const epochs = ["Epoch 1", "Epoch 2", "Epoch 3"];
  s.addChart(pres.charts.LINE, [
    { name: "512토큰 모델", labels: epochs, values: [64.39, 65.04, 65.15] },
    { name: "256토큰 기준 성능", labels: epochs, values: [64.41, 64.41, 64.41] },
    { name: "사전 채택 기준", labels: epochs, values: [65.41, 65.41, 65.41] },
  ], {
    x: M, y: 2.45, w: 7.0, h: 3.75, showTitle: true,
    title: "512토큰 모델의 epoch별 내부 홀드아웃 Macro F1 (%)", titleFontFace: F, titleFontSize: 11.5, titleColor: C.text,
    chartColors: [C.blue, "9E9E9E", C.red], showLegend: true, legendPos: "b", legendFontFace: F, legendFontSize: 9,
    showValue: false, lineSize: 2.2, lineDataSymbol: "circle", lineDataSymbolSize: 5,
    catAxisLabelFontFace: F, catAxisLabelFontSize: 10.5, catAxisLabelColor: C.text,
    valAxisLabelFontFace: F, valAxisLabelFontSize: 9.5, valAxisLabelColor: C.sub,
    valAxisMinVal: 63, valAxisMaxVal: 66, valAxisMajorUnit: 0.5, valLabelFormatCode: '0.0"%"',
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" },
    chartArea: { border: { color: C.white, pt: 0 }, fill: { color: C.white, transparency: 100 } },
    plotArea: { border: { color: C.white, pt: 0 }, fill: { color: C.white, transparency: 100 } },
  });
  // 512토큰 계열만 필요한 값 레이블을 별도 편집 가능한 텍스트로 표시한다.
  [["64.39%", 1.38, 4.34], ["65.04%", 3.35, 3.75], ["65.15%", 5.34, 3.63]].forEach(([v, x, y]) => {
    txt(s, v, { x, y, w: 0.64, h: 0.22, fontSize: 9.5, bold: true, color: C.blue, align: "center" });
  });

  const expRows = [
    ["항목", "결과"],
    ["256토큰 초과 통화", "2,359건 · 40.7%"],
    ["512토큰 초과 통화", "2.9%"],
    ["256토큰 기준 Macro F1", "64.41%"],
    ["512토큰 최종 Macro F1", "65.15%"],
    ["개선폭", "+0.74%p"],
    ["사전 채택 기준", "+1.00%p"],
    ["학습시간", "77.5분"],
    ["계산비용", "약 4배"],
    ["결정", "512토큰 확대안 기각"],
  ];
  table(s, expRows, 7.75, 2.48, [3.0, 2.55], {
    w: 5.55, rowH: [0.32, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.33], fontSize: 9.3, margin: 0.04,
    fill: C.white, border: { type: "solid", pt: 0.7, color: C.line },
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.75, y: 6.2, w: 5.08, h: 0.68, fill: { color: C.light }, rectRadius: 0.06, line: { color: C.sky, width: 1 } });
  txt(s, "성능은 개선됐지만 사전 기준에 미달했고 계산비용이 증가해 앞 256토큰을 유지", { x: 7.95, y: 6.2, w: 4.68, h: 0.68, fontSize: 11.5, bold: true, color: C.blue, align: "center", valign: "middle" });
  caption(s, "※ Training 내부 80:20 홀드아웃, seed 42 기준. 공식 Validation은 사용하지 않음.", 7.0);
  s.addNotes("근거: docs/experiment_log.md M3-ROBERTA-007, docs/decisions.md의 입력 512토큰 기각 결정");
}

// =====================================================================
// 35. 별첨 3 — Mission 3 낮은 성능 3개 라벨의 오류 구조
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "별첨", "Mission 3", "별첨 3. Mission 3 낮은 성능 3개 라벨의 오류 구조", 35);
  coreMsg(s, "직접 표현이 없는 양성이 많았고, 두통과 오심은 표현 유무에 따라 recall 차이가 컸다");

  const labels = ["전신쇠약", "두통", "오심"];
  s.addChart(pres.charts.BAR, [
    { name: "직접 표현 있음 recall", labels, values: [0.907, 0.820, 0.684] },
    { name: "직접 표현 없음 recall", labels, values: [0.549, 0.195, 0.356] },
  ], {
    x: M, y: 2.46, w: 6.35, h: 3.7, barDir: "bar", barGrouping: "clustered", catAxisOrientation: "maxMin",
    chartColors: [C.blue, "EF9A9A"], showTitle: true, title: "직접 표현 유무별 recall · Training 내부 홀드아웃",
    titleFontFace: F, titleFontSize: 11.5, titleColor: C.text, showLegend: true, legendPos: "b", legendFontFace: F, legendFontSize: 9.5,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0%", dataLabelFontFace: F, dataLabelFontSize: 10,
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0%",
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, catAxisLabelFontFace: F, catAxisLabelFontSize: 11,
    valAxisLabelFontFace: F, valAxisLabelFontSize: 9, showCatName: false,
  });

  const headerCell = (text, options = {}) => ({ text, options: Object.assign({ bold: true, color: C.white, fill: { color: C.blue }, align: "center", valign: "middle" }, options) });
  const diagFill = "3F629F";
  const errRows = [
    [
      headerCell("증상", { rowspan: 2 }),
      headerCell("최종 앙상블\n공식 Validation F1\n(3,640건)", { rowspan: 2 }),
      headerCell("Training 내부 홀드아웃 오류 진단 (5,792건)", { colspan: 3, fill: { color: diagFill } }),
    ],
    [
      headerCell("내부 홀드아웃\n양성 수", { fill: { color: diagFill } }),
      headerCell("직접 표현 없음", { fill: { color: diagFill } }),
      headerCell("내부 TF-IDF\nTP / FP / FN", { fill: { color: diagFill } }),
    ],
    ["전신쇠약", "59.39%", "1,062", "804건 · 75.7%", "675 / 846 / 387"],
    ["두통", "53.15%", "581", "370건 · 63.7%", "245 / 372 / 336"],
    ["오심", "40.00%", "668", "592건 · 88.6%", "263 / 754 / 405"],
  ];
  s.addTable(errRows, {
    x: 7.08, y: 2.52, w: 5.75, colW: [0.72, 1.12, 0.96, 1.35, 1.6], rowH: [0.48, 0.55, 0.56, 0.56, 0.56],
    fontFace: F, fontSize: 9.2, color: C.text, valign: "middle", align: "center", margin: 0.05,
    border: { type: "solid", pt: 0.7, color: C.line },
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.08, y: 5.45, w: 5.75, h: 0.72, fill: { color: C.light }, rectRadius: 0.06, line: { color: C.sky, width: 1 } });
  txt(s, "오심은 양성의 88.6%에서 직접 표현을 찾지 못했고, 두통은 표현이 없을 때 recall이 19.5%까지 낮아졌다.", { x: 7.28, y: 5.5, w: 5.35, h: 0.35, fontSize: 11.2, bold: true, color: C.blue, align: "center", valign: "middle" });
  txt(s, "통화에서 관찰할 수 없는 정보가 성능을 제한했을 가능성", { x: 7.28, y: 5.86, w: 5.35, h: 0.22, fontSize: 9.5, color: C.sub, align: "center" });
  txt(s, [
    { text: "※ 최종 F1은 최종 RoBERTa 3-seed + TF-IDF 앙상블의 공식 Validation 결과", options: { breakLine: true } },
    { text: "※ 오류 진단은 Training 80:20 홀드아웃 5,792건의 TF-IDF 모델 기준", options: { breakLine: true } },
    { text: "※ 직접 표현 여부는 증상별 정규식 탐색 결과이며, 구어체 변형을 모두 포착하지 못하므로 표현 비율은 하한값", options: { breakLine: true } },
    { text: "※ 직접 표현이 없다는 결과는 오라벨을 의미하지 않음" },
  ], { x: M, y: 6.34, w: CW, h: 0.74, fontSize: 8.8, color: C.sub, paraSpaceAfter: 1.5 });
  s.addNotes("근거: 최종 라벨별 F1은 기존 16장의 sym 데이터와 docs/experiment_log.md M3-FINAL-002. 표현 비율과 근거별 recall: reports/mission3/error_analysis/m3_error_analysis.json. 오류 분석 설명: reports/mission3/error_analysis/README.md. precision, recall, F1, support: reports/mission3/m3_reeval_tfidf_lr/per_label_metrics.csv. 예측 양성 수와 TP/FP/FN 계산 근거: reports/mission3/p0_rerun/min1_per_label.csv");
}

// =====================================================================
// =====================================================================
// 36. 별첨 4 — Mission 3 채택 실험 로그 — 네이티브 편집 가능 표
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "별첨", "Mission 3", "별첨 4. Mission 3 채택 실험 로그", 36);
  coreMsg(s, "가중 BCE와 soft macro-F1, RoBERTa 확장이 단계적인 성능 개선을 만들었다.");
  const rows = [
    ["실험 ID", "핵심 변경", "평가 기준", "Macro F1", "판단"],
    ["M3-REEVAL-P0", "TF-IDF + One-vs-Rest LR", "내부 80:20", "56.76%", "기준선"],
    ["M3-KCBERT-002", "KcBERT, 앞 256토큰, 비가중 BCE", "내부 80:20", "58.33%", "채택"],
    ["M3-KCBERT-003", "완화 가중 BCE", "내부 80:20", "59.90%", "채택"],
    ["M3-KCBERT-005", "강화 가중 BCE", "내부 80:20", "60.92%", "채택"],
    ["M3-KCBERT-012", "강화 가중 BCE + soft macro-F1", "내부 80:20", "62.65%", "채택"],
    ["M3-ROBERTA-003", "KLUE RoBERTa-large + 동일 손실", "내부 80:20", "64.27%", "최종 구조 채택"],
    ["M3-FINAL-002", "RoBERTa 3-seed + TF-IDF", "공식 Validation", "65.34%", "최종 제출 모델"],
  ];
  const nativeRows = rows.map((r, i) => r.map((v, j) => {
    const fill = i === 0 ? C.blue : i === 7 ? C.mintLight : i === 1 ? C.bgGray : i === 6 ? C.light : (i % 2 === 0 ? C.white : C.light);
    return { text: v, options: { fill: { color: fill }, color: i === 0 ? C.white : C.text,
      bold: i === 0 || (i === 7 && (j === 3 || j === 4)),
      align: (j === 3 || j === 4) ? "center" : "left", valign: "middle" } };
  }));
  s.addTable(nativeRows, { x: M, y: 2.43, w: CW, colW: [2.05, 4.18, 2.0, 1.3, 2.8],
    rowH: [0.53, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.55], fontFace: F,
    fontSize: 11, color: C.text, margin: 0.065, valign: "middle",
    border: { type: "solid", pt: 0.65, color: C.line } });
  card(s, M, 6.66, 6.06, 0.38, C.light);
  txt(s, [{ text: "내부 검증  ", options: { bold: true, color: C.blue } },
    { text: "56.76% → 64.27%  (+7.51%p)", options: { bold: true } }],
    { x: M + 0.1, y: 6.68, w: 5.85, h: 0.3, fontSize: 12.5, valign: "middle", align: "center" });
  card(s, 6.77, 6.66, 6.06, 0.38, C.mintLight);
  txt(s, [{ text: "공식 Validation  ", options: { bold: true, color: C.blue } },
    { text: "Macro F1  65.34%", options: { bold: true } }],
    { x: 6.87, y: 6.68, w: 5.85, h: 0.3, fontSize: 12.5, valign: "middle", align: "center" });
  caption(s, "※ 56.76%~64.27%는 Training 내부 80:20 홀드아웃, 65.34%는 전체 Training 학습 후 공식 Validation 결과.", 7.12);
  s.addNotes("근거: docs/experiment_log.md, docs/decisions.md — M3-REEVAL-P0, M3-KCBERT-002·003·005·012, M3-ROBERTA-003, M3-FINAL-002. 내부 홀드아웃과 공식 Validation 수치를 직접 비교하지 않음.");
}

// 37. 별첨 5 — Mission 3 주요 기각 실험 (네이티브 편집 가능 표)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "별첨", "Mission 3", "별첨 5. Mission 3 주요 기각 실험", 37);
  coreMsg(s, "사전에 정한 개선 기준 +1.00%p를 넘지 못한 변형은 최종 모델에 사용하지 않았다.");
  const rows = [
    ["실험 방향", "대표 실험", "대표 결과", "비교 기준", "기각 이유"],
    ["긴 입력 처리", "M3-ROBERTA-007", "65.15%", "RoBERTa seed 42 64.41%", "+0.74%p, 계산 비용 약 4배"],
    ["학습 기간 확대", "M3-KCBERT-006", "58.83%", "KcBERT-005 60.92%", "3 epochs보다 성능 하락"],
    ["정규화·도메인 적응", "M3-KCBERT-011·013", "61.32% / 61.18%", "KcBERT-005 60.92%", "개선폭이 seed 편차 수준"],
    ["손실 함수 변형", "M3-KCBERT-014·015", "62.53% / 62.49%", "KcBERT-012 62.65%", "soft-F1 기준선보다 낮음"],
    ["다른 인코더", "M3-E5-001·KCELECTRA-001·QWEN-001", "최고 63.84%", "RoBERTa seed 42 64.41%", "RoBERTa-large를 넘지 못함"],
    ["보조 증상·출력 구조", "M3-ROBERTA-004·STRUCT-001", "64.74% / 64.31%", "각 기준선", "최대 개선폭 +0.33%p"],
    ["신경망 앙상블", "M3-ENS-001", "64.55%", "RoBERTa 64.27%", "+0.28%p, 모델 간 오류 중복"],
  ];
  const nativeRows = rows.map((r, i) => r.map((v, j) => ({ text: v, options: {
    fill: { color: i === 0 ? C.blue : (i % 2 ? C.white : C.bgGray) },
    color: i === 0 ? C.white : C.text, bold: i === 0 || (i > 0 && j === 0),
    align: j === 2 ? "center" : "left", valign: "middle"
  } })));
  s.addTable(nativeRows, { x: M, y: 2.48, w: CW,
    colW: [2.05, 3.12, 1.43, 2.35, 3.38],
    rowH: [0.46, 0.48, 0.48, 0.50, 0.48, 0.62, 0.54, 0.48],
    fontFace: F, fontSize: 10.1, color: C.text, margin: 0.055, valign: "middle",
    border: { type: "solid", pt: 0.65, color: C.line } });
  card(s, M, 6.82, CW, 0.34, C.light);
  txt(s, "모델 복잡화보다 클래스 불균형과 Macro F1을 반영한 손실 설계가 더 효과적이었다.",
    { x: M + 0.16, y: 6.84, w: CW - 0.32, h: 0.28, fontSize: 12.2, bold: true,
      color: C.blue, align: "center", valign: "middle" });
  caption(s, "근거: docs/experiment_log.md, docs/decisions.md — M3-ROBERTA-007 등 주요 기각 실험", 7.17);
  s.addNotes("근거: docs/experiment_log.md, docs/decisions.md — M3-ROBERTA-007, M3-KCBERT-006·011·013·014·015, M3-E5-001, M3-KCELECTRA-001, M3-QWEN-001, M3-ROBERTA-004, M3-STRUCT-001, M3-ENS-001. 미리 정한 +1.00%p 개선 기준에 못 미치거나 비교 모델보다 낮은 주요 변형만 정리함.");
}

// 38. 마무리 — 기존 마무리 장 공식 템플릿 그대로 사용 + 팀 정보 유지
// =====================================================================
{
  const s = pres.addSlide();
  s.addImage({ path: TPL.closing, x: 0, y: 0, w: W, h: 7.5, sizing: { type: "cover", w: W, h: 7.5 } });
  txt(s, "팀 eigensinn  |  김장수 · 김광희 · 지연우 · 이수빈 · 조지은", { x: M, y: 2.2, w: CW, h: 0.4, fontSize: 16, bold: true, color: C.blue, align: "center" });
  txt(s, "코드 · 실험 기록: GitHub eigensinn-dcc2026/prelim  (docs/experiment_log.md, docs/decisions.md)", { x: M, y: 2.58, w: CW, h: 0.3, fontSize: 11, color: C.sub, align: "center" });
  s.addNotes("공식 템플릿(슬라이드4) 그대로. 흰 여백(약 2.1~2.9in)에만 팀 정보를 얹음 — 배너 그래픽은 건드리지 않음.");
}

// PptxGenJS 4.0은 선 계열별 점선 옵션을 직접 노출하지 않는다. 생성 후 해당 네이티브 차트 XML의
// 두 기준선에만 점선을 지정해 PowerPoint 편집 가능성을 유지한다.
async function styleLengthExperimentLines(fileName) {
  const zip = await JSZip.loadAsync(await fs.promises.readFile(fileName));
  const setDash = (seriesXml, dash) => {
    if (/<a:prstDash\b/.test(seriesXml)) return seriesXml.replace(/<a:prstDash\b[^>]*\/>/, `<a:prstDash val="${dash}"/>`);
    return seriesXml.replace(/<\/a:ln>/, `<a:prstDash val="${dash}"/></a:ln>`);
  };
  let patched = false;
  for (const [name, entry] of Object.entries(zip.files)) {
    if (!/^ppt\/charts\/chart\d+\.xml$/.test(name)) continue;
    let xml = await entry.async("string");
    if (!xml.includes("512토큰 모델") || !xml.includes("사전 채택 기준")) continue;
    xml = xml.replace(/<c:ser>[\s\S]*?<\/c:ser>/g, block => {
      if (block.includes("256토큰 기준 성능") || block.includes("사전 채택 기준")) return setDash(block, "dash");
      return setDash(block, "solid");
    });
    zip.file(name, xml);
    patched = true;
  }
  if (!patched) throw new Error("M3 입력 길이 차트의 계열을 찾지 못했습니다.");
  // PptxGenJS가 실제로 만들지 않은 slideMaster2..N의 Content-Type 선언을 남기는 문제를 정리한다.
  const contentTypesName = "[Content_Types].xml";
  let contentTypes = await zip.file(contentTypesName).async("string");
  contentTypes = contentTypes.replace(/<Override\s+PartName="\/ppt\/slideMasters\/slideMaster(\d+)\.xml"[^>]*\/>/g, (tag, number) =>
    zip.file(`ppt/slideMasters/slideMaster${number}.xml`) ? tag : ""
  );
  zip.file(contentTypesName, contentTypes);
  await fs.promises.writeFile(fileName, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

// 빌드 이전 자동 검증: 표지·타이틀·목차(1~3), 본문 및 별첨(4~37), 마무리(38).
const EXPECTED_SLIDE_COUNT = 38;
if (pres._slides.length !== EXPECTED_SLIDE_COUNT) {
  throw new Error(`슬라이드 수 오류: ${pres._slides.length}장 (예상 ${EXPECTED_SLIDE_COUNT}장)`);
}
const headerPageNumbers = [];
for (const slide of pres._slides) {
  // header()가 생성한 우측 상단 'eigensinn | N' 텍스트만 검사.
  const textShapes = (slide._slideObjects || []).filter(o => o._type === 'text');
  for (const obj of textShapes) {
    const t = obj.text;
    if (typeof t === 'string') {
      const match = t.match(/^eigensinn\s*\|\s*(\d+)$/);
      if (match) headerPageNumbers.push(Number(match[1]));
    }
  }
}
// 내부 객체 구조가 버전에 따라 달라질 수 있으므로 비어 있으면 출력하지 않는다.
if (headerPageNumbers.length && (
  headerPageNumbers.length !== 34 ||
  headerPageNumbers.some((n, i) => n !== i + 4)
)) {
  throw new Error(`헤더 페이지 번호 중복/누락: ${headerPageNumbers.join(', ')}`);
}
console.log(`슬라이드 수 확인: ${pres._slides.length}장`);

const outputFile = process.argv[2] || "DCC2026_예선_결과정리.pptx";
pres.writeFile({ fileName: outputFile }).then(async f => {
  await styleLengthExperimentLines(outputFile);
  console.log("wrote", f);
});
