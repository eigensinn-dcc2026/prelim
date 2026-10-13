// DCC 2026 대학부 예선 결과 정리 PPT — 장별 원고 v2(팀장 검토본)를 옮김. main의 Mission 2 상세 14장 + Mission 3 별첨 3장 통합, 총 37장
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
// M2 제출 모델을 바꾸면 이 블록만 고친다. 11·12·13·14·18·19장이 모두 여기서 읽는다.
// 현재 값: CNN 3-seed 앙상블 (submission/mission2/README.md, experiment_log.md M2-RERUN-001)
const M2 = {
  // 11장 문제·제약·최종 모델
  core12: "발화 하나만 보고 신고자/대원을 가린다. log-mel + CNN 3개 평균, 공식 Validation 91.49%.",
  // 구조도 (flow 함수 입력): 앞 단계 → 앙상블 멤버 → 판정
  flow: {
    pre: [["발화 구간", "startAt~endAt만"], ["log-mel", "40 mel · 8 kHz"], ["길이 맞춤", "300프레임\n그 발화만 사용"]], preW: 1.95,
    branches: [["SpeakerCNNLarge · seed 1"], ["SpeakerCNNLarge · seed 2"], ["SpeakerCNNLarge · seed 3"]], brW: 2.35, brGap: 0.1,
    post: [["확률 평균", "≥ 0.5 → 신고자\n< 0.5 → 대원"]], postW: 2.0,
  },
  why: [
    ["음향 차이", "대원 헤드셋·정해진 말투\n신고자 휴대전화·잡음"],
    ["단계적 검증", "통계 특징(L1) → CNN(L2)"],
    ["처음부터 학습", "사전학습·외부 데이터 없음"],
  ],
  // 12장 성능 개선 과정 (막대 차트: 공식 Validation Accuracy)
  core13: "86.44% → 90.98% → 91.49%. CNN의 이득은 목소리가 겹친 구간에 몰려 있었다.",
  chart: { labels: ["L1 LightGBM", "L2 CNN 단일", "L2 CNN\n3-seed 앙상블"], values: [86.44, 90.98, 91.49] },
  gainTitle: "■ CNN의 이득이 몰린 곳",
  gains: [["약 +7%p", "겹친 구간"], ["약 +2%p", "겹치지 않은 구간"]],
  gainNote: "통계 특징(L1) 대비 CNN 단일 모델의 개선 폭 (공식 Validation)",
  finding: "제출 방식(한 발화씩 추론)으로 바꾸자 정확도가 떨어졌다. 원인은 학습 때의 배치 패딩과 추론 방식 차이였고, 모든 발화를 자기 자신만으로 같은 길이까지 채우도록 맞춰 회복했다.",
  // 13장 다른 접근 — 마지막 행이 보류 중인 WavLM 미세조정
  core14: "CNN 하나로 끝내지 않고 여러 방향을 검증했다. 효과가 없거나 규칙에 어긋난 것은 이유와 함께 기록했다.",
  lastRow: ["WavLM 미세조정 (3초 창·겹침 증강)", "보류", "공식 평가 진행 중"],
  // 14장 최종 결과와 보완할 점
  core15: "공식 Validation 91.49%. 원본 데이터부터 다시 학습해도 같은 수준이 재현됐다.",
  acc: "91.49%",
  recalls: [["87.15%", "대원 recall"], ["95.51%", "신고자 recall"]],
  repro: "원본부터 다시 학습해도 같은 수준(91.49%)",
  improve: "대원 recall이 더 낮다.\n\"네\", \"여보세요\" 같은 짧은 응답은 두 화자가 비슷하게 들리고, 규칙상 통화 맥락을 쓸 수 없다.",
  // 18장 성능 요약 표, 19장 향후 계획
  summary: { task: "M2 신고자/대원", model: "log-mel + CNN × 3", val: "91.49%", gain: "+5.05%p" },
  future: "사전학습 음성 모델(WavLM) 앙상블의 공식 검증, 짧은 응답 발화 보강",
  evidence: "submission/mission2/README.md, docs/experiment_log.md M2-RERUN-001·M2-ENS-003·M2-RETRAIN-001",
  // 슬라이드 노트·캡션의 근거 (12·14장)
  evidence13: "근거: docs/experiment_log.md M2-RETRAIN-001, M2-ENS-003, M2-RERUN-001, M2-COMPARE-002, M2-PAD-001·002",
  notes13: "근거: L1 86.44%(M2-RETRAIN-001, 제공 라벨 재학습 L1) / CNN 단일 90.98%(M2-ENS-003 ts1) / 3-seed 앙상블 91.49%(M2-RERUN-001, submission/mission2/README.md). 겹침 구간: M2-COMPARE-002 — 겹침 L1 81.83%→L2 88.88%(+7.05%p), 안 겹침 90.01%→92.44%(+2.43%p), L2 단일 모델 기준. 패딩: M2-PAD-001(발화별 300프레임 채우기), M2-VAL-005, M2-PAD-002.",
  notes15: "근거: submission/mission2/README.md '결과와 계산 효율성' 표(91.49%, 대원 recall 87.15%, 신고자 recall 95.51%). 재현성: M2-RERUN-001(원본부터 재학습 91.49%) vs M2-ENS-003(이전 앙상블 91.55%).",
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
  txt(slide, `${TEAM}  |  ${page}`, { x: W - M - 3, y: HB + 0.04, w: 3, h: 0.26, fontSize: 10, color: C.sub, align: "right" });
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
  s.addNotes("main의 M2 상세 14장과 M3 별첨 3장을 통합한 총 37장 구성.");
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
// 6. 02 EDA 핵심 발견 (3단 그리드)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "02", "데이터 분석", "EDA 핵심 발견", 6);
  coreMsg(s, "세 미션 모두 \"쉬운 단서 하나\"로는 풀리지 않는다. 이 발견이 미션별 모델 선택의 출발점이다.");
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
  const cw = (CW - 2 * 0.3) / 3;
  cards.forEach(([path, t, found, design], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.45, cw, 4.4);
    s.addImage({ path: IC + ["m1", "m2", "m3"][i] + ".png", x: x + cw - 0.75, y: 2.52, w: 0.45, h: 0.45 });
    title(s, t, x + 0.2, 2.56, cw - 1.1, 15);
    if (path === "overlap") {
      s.addShape(pres.shapes.RECTANGLE, { x: x + 0.2, y: 2.98, w: cw - 0.4, h: 2.6, fill: { color: C.white }, line: { color: C.line, width: 1 } });
      barChart(s, ["겹침이 있는\n통화 비율", "통화당 겹침 시간\n(중앙값)"], [98.15, 3.82], {
        name: "%", x: x + 0.25, y: 3.02, w: cw - 0.5, h: 2.52, chartColors: [C.sky, C.red], title: "Training 통화 기준 (%)", titleFontSize: 11,
        dataLabelFormatCode: "0.00", dataLabelFontSize: 14, catAxisLabelFontSize: 11.5, valAxisMinVal: 0, valAxisMaxVal: 115, valAxisHidden: true, valGridLine: { style: "none" },
      });
    } else fig(s, path, x + 0.2, 2.98, cw - 0.4, 2.6);
    txt(s, [
      { text: "발견  ", options: { bold: true } },
      { text: found, options: { breakLine: true } },
      { text: "설계  ", options: { bold: true, color: C.blue } },
      { text: design },
    ], { x: x + 0.2, y: 5.68, w: cw - 0.4, h: 1.15, fontSize: 11.5, paraSpaceAfter: 4 });
  });
  caption(s, "근거: reports/eda/eda_report.md (M2 겹침: 7.2절 Training), reports/eda/figures/");
  s.addNotes("근거: reports/eda/eda_report.md, reports/eda/figures/(gender_distribution, speaker_duration_buckets, symptom_prevalence), docs/experiment_log.md COM-EDA-001.");
}

// ======================================================================
// Mission 1 — 신고자 성별 분류 (7~10장)
// ======================================================================

// 7. 문제와 최종 모델
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1  신고자 성별 분류 — 문제와 최종 모델", 7);
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
    "분류기는 단순한 로지스틱 회귀로 충분했다 (더 복잡한 분류기는 이득이 기준에 못 미침 → 9장)",
    "최종 모델은 학습 데이터 전체로 재학습했다",
  ], M, 5.3, lw, 1.5, 13);
  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, 2.45, rw, 4.35, C.light);
  txt(s, "98.10%", { x: rx, y: 3.6, w: rw, h: 1.0, fontSize: 44, bold: true, color: C.blue, align: "center", valign: "middle" });
  txt(s, "Accuracy\n공식 Validation", { x: rx, y: 4.65, w: rw, h: 0.7, fontSize: 13, color: C.sub, align: "center" });
  caption(s, "근거: submission/mission1/README.md, docs/experiment_log.md M1-REFIT-OFFVAL-001");
  s.addNotes("근거: submission/mission1/README.md(구성·상태), docs/experiment_log.md M1-XVEC-002-OFFVAL(다중 구간 평균 x-vector), M1-REFIT-OFFVAL-001(전체 재학습 제출 모델 98.10%).");
}

// 8. 성능 개선 과정
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1  신고자 성별 분류 — 성능 개선 과정", 8);
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

// 9. 채택하지 않은 시도
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1  신고자 성별 분류 — 채택하지 않은 시도", 9);
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

// 10. 최종 결과와 후속 과제
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1  신고자 성별 분류 — 최종 결과와 후속 과제", 10);
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
// Mission 2 — 신고자/119대원 분류 (11~14장, 수치·모델명은 상단 M2 상수)
// ======================================================================

// 11. 문제·제약·최종 모델
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 문제·제약·최종 모델", 11);
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

// 12. 성능 개선 과정
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 성능 개선 과정", 12);
  coreMsg(s, M2.core13);
  const lw = 6.2;
  card(s, M, 2.45, lw, 4.35);
  barChart(s, M2.chart.labels, M2.chart.values, {
    x: M + 0.15, y: 2.55, w: lw - 0.3, h: 4.15, chartColors: [C.line, C.sky, C.blue],
    title: "공식 Validation Accuracy (%)", valAxisMinVal: 84, valAxisMaxVal: 93,
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

// 13. 다른 접근과 기각·보류 이유
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 다른 접근과 기각·보류 이유", 13);
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
  s.addNotes("근거: 통화 단위 특징 M2-EXP-008 / 제공 라벨 재학습 M2-RETRAIN-001 / 확률 평균 M2-ENS-001 / 더 큰 CNN M2-CNN-008, 패딩 제외 풀링 M2-CNN-009, 길이 입력 M2-CNN-004·006, Mixup M2-MIX-001 / WavLM 고정 임베딩 M2-SSL-001, 기본 미세조정 M2-SSL-002 / WavLM 미세조정(3초 창·겹침 증강) M2-SSL-003. 모두 docs/experiment_log.md·docs/decisions.md.");
}

// 14. 최종 결과와 보완할 점
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 최종 결과와 보완할 점", 14);
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
// ---------- Mission 2 추가: 지연우 전체 실험 흐름 ----------
// These inserted pages summarize the log from the first baseline through the adopted ensemble.
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 처음 기준선: MFCC로 발화 요약", 15);
  coreMsg(s, "먼저 발화 하나에서 목소리 특징을 뽑고, 신고자와 119대원을 구분하는 기준을 만들었습니다");

  const gap = 0.35, cardW = (CW - gap) / 2;
  const leftX = M, rightX = M + cardW + gap;
  card(s, leftX, 2.5, cardW, 3.9);
  card(s, rightX, 2.5, cardW, 3.9);
  txt(s, "무엇을 숫자로 바꿨나", { x: leftX + 0.25, y: 2.75, w: cardW - 0.5, h: 0.38, fontSize: 18, bold: true, color: C.blue });
  txt(s, [
    { text: "전화 음성 한 발화를 입력으로 사용", options: { bold: true, breakLine: true } },
    { text: "MFCC는 소리의 음색·주파수 모양을 간단한 숫자로 요약합니다.", options: { breakLine: true } },
    { text: "20개 계수의 평균과 표준편차를 붙여 발화마다 40개 숫자를 만들었습니다." },
  ], { x: leftX + 0.28, y: 3.35, w: cardW - 0.56, h: 2.5, fontSize: 15, paraSpaceAfter: 7 });

  txt(s, "첫 기준선 결과", { x: rightX + 0.25, y: 2.75, w: cardW - 0.5, h: 0.38, fontSize: 18, bold: true, color: C.blue });
  stat(s, rightX + 0.2, 3.25, cardW - 0.4, "80.91%", "초기 공식 검증 정확도", C.blue, 36);
  txt(s, "이 점수를 이후의 간단한 음향 특징이 얼마나 보완하는지 확인하는 출발점으로 삼았습니다.", { x: rightX + 0.3, y: 4.75, w: cardW - 0.6, h: 0.9, fontSize: 14, color: C.text, align: "center", valign: "middle" });
  caption(s, "근거: docs/mission2_experiment_log.md §1, §2 (초기 Official Validation 결과)");
  s.addNotes("처음부터 복잡한 모델을 쓰기보다, MFCC라는 음성의 주파수 요약값 40개와 로지스틱 회귀로 기준선을 만들었습니다. 초기 정확도는 공식 검증에서 약 80.91%였습니다. 후속 실험은 Training 내부 검증으로 평가해 조건이 다릅니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 간단한 음향 정보도 보탬이 될까?", 16);
  coreMsg(s, "발화 길이와 주파수 특징을 하나씩 더해 보고, 함께 쓸 만한 조합을 골랐습니다");
  table(s, [
    ["추가한 정보", "왜 확인했나", "공식 검증 정확도와 판단"],
    ["발화 길이", "짧고 긴 발화의 차이를 보완", "80.91% → 80.91%, 사실상 변화 없음 · 제외"],
    ["고주파 경계(rolloff)", "에너지가 어느 주파수까지 퍼지는지 확인", "81.15% (+0.24%p vs E00) · 단일 추가 중 최대 개선"],
    ["영교차율(ZCR)", "마찰음·무성음 단서를 보완", "81.01% (+0.10%p) · 단독 이득은 작아 조합 확인"],
    ["스펙트럼 중심(centroid)", "소리의 무게중심 주파수 확인", "81.05% (+0.14%p); 두 Recall은 상승했지만 HCE 2,086→2,102 · 미승격"],
    [{ text: "rolloff + ZCR", options: { bold: true, color: C.blue } }, "두 정보가 서로 보완되는지 확인", { text: "81.31% (+0.40%p vs E00) · 둘 다 든 E05 42D를 다음 모델 비교에 사용", options: { bold: true, color: C.blue } }],
  ], M + 0.08, 2.42, [2.65, 4.1, 5.15], { rowH: 0.58, fontSize: 10.7 });
  txt(s, "E05는 두 특징이 함께 든 후속 모델 비교용 표현입니다. 최종 모델 채택을 뜻하지 않습니다. E02–E04는 저장 Notebook 출력 없이 Markdown 기록만 남아 있습니다.", { x: M + 0.1, y: 6.0, w: CW - 0.2, h: 0.55, fontSize: 10.8, color: C.sub, align: "center" });
  caption(s, "근거: docs/mission2_experiment_log.md §2, §9.3", 7.02);
  s.addNotes("평균적인 음색 정보 외에 발화 길이, 고주파 경계, 영교차율, 스펙트럼 중심을 시험했습니다. 발화 길이는 거의 도움이 되지 않았고, rolloff와 ZCR를 함께 넣었을 때 초기 기준선보다 높은 81.31%가 나와 다음 분류기 비교에 사용했습니다. 이 초기 표는 공식 검증 결과입니다. 일부 개별 수치는 Markdown에만 남아 있다는 기록 한계도 같이 밝혀 둡니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 같은 특징도 모델에 따라 달라질까?", 17);
  coreMsg(s, "42개 입력은 고정하고 분류 방법을 바꾸자, 작은 모델보다 MLP가 높은 정확도를 보였습니다");
  table(s, [
    ["분류 방법", "내부 검증 정확도", "관찰"],
    ["로지스틱 회귀", "81.73%", "비교 기준"],
    ["HistGradientBoosting", "84.19%", "나무 기반 후보 중 가장 높음"],
    ["ExtraTrees", "82.00%", "로지스틱보다 +0.27%p, HistGradientBoosting보다 −2.19%p"],
    ["QDA", "79.73%", "정확도와 화자 균형이 낮아 제외"],
    [{ text: "MLP 신경망", options: { bold: true, color: C.blue } }, { text: "85.95%", options: { bold: true, color: C.blue } }, { text: "가장 높은 후보로 유지; 수렴 경고는 남음", options: { bold: true, color: C.blue } }],
  ], M + 0.2, 2.5, [3.7, 2.4, 5.8], { rowH: 0.57, fontSize: 11.5 });
  txt(s, "이 단계부터 Training 내부의 통화 단위 고정 분할로 평가했습니다.", { x: M + 0.25, y: 6.12, w: CW - 0.5, h: 0.4, fontSize: 12, color: C.sub, align: "center" });
  caption(s, "근거: docs/mission2_experiment_log.md §3 (42D 입력, 내부 검증)", 7.02);
  s.addNotes("입력 특징을 42개로 고정한 채 로지스틱 회귀, 나무 모델, QDA, MLP를 비교했습니다. 같은 내부 검증 조건에서 MLP가 약 85.95%로 가장 높았습니다. 다만 MLP는 정해 둔 반복 횟수 안에 완전히 수렴하지 않았다는 경고가 있었으므로, 이 점수만 보고 모델이 충분히 학습됐다고 단정하지 않습니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 어떤 음향 단서가 유용했나?", 18);
  coreMsg(s, "목소리 높낮이·유성성 요약은 도움을 줬지만, 모든 음향 특징이 성능을 높이지는 않았습니다");
  table(s, [
    ["살펴본 소리 단서", "왜 더했나", "내부 검증 결과"],
    ["에너지 변화", "발화 안에서 소리 크기 변화 확인", "85.71% (42D 기준 −0.24%p) · 제외"],
    [{ text: "Pitch / Voicing (음높이·유성성)", options: { bold: true, color: C.blue } }, "높낮이와 주기적 발성 단서 반영", { text: "86.63% (+0.67%p); Recall 0 +0.71%p, Recall 1 +0.64%p · 47D 기준에 포함", options: { bold: true, color: C.blue } }],
    ["Spectral Flux (프레임 간 변화)", "연속 프레임 사이의 소리 변화 확인", "85.38% (42D 기준 −0.57%p) · 제외"],
    ["CPP (조화성 단서)", "목소리 주기성이 뚜렷한 정도 확인", "AF02+CPP 누적 49D에서 86.86%; 47D 직접 비교 output이 없어 CPP 단독 증분은 미확인"],
    ["Delta-MFCC (시간 변화)", "프레임 순서에 따른 음색 변화 확인", "최초 단독 점수·혼동행렬 미보존; 뒤의 old-cache 재검토는 17-F에 표시"],
    ["Formants (성도 공명)", "성도 공명 위치 단서 검토", "실험 전에 위험·중복 우려로 보류, 성능 결과 없음"],
  ], M + 0.06, 2.35, [2.7, 4.5, 4.85], { rowH: 0.55, fontSize: 10.9 });
  caption(s, "근거: docs/mission2_experiment_log.md §4.1–4.3", 7.02);
  s.addNotes("각 특징군은 서로 다른 질문을 확인했습니다. 에너지 변화와 프레임 간 flux는 이 비교에서 기준보다 낮았습니다. 반면 F0 높낮이와 유성성 요약은 정확도를 약 0.67%포인트 높여 다음 기준에 포함했습니다. CPP 결과는 기록에 남아 있지만 당시 단독 승격 비교 일부가 없어 근거가 완전하지 않습니다. Delta-MFCC의 초기 단독 점수는 남아 있지 않고, 포먼트는 실제 학습 전에 보류됐습니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 특징 사이의 관계와 재현성", 19);
  coreMsg(s, "주파수끼리의 관계도 시험했지만, 과거 점수 하나만으로 특징을 채택하지 않았습니다");
  const gap = 0.35, cardW = (CW - gap) / 2;
  const leftX = M, rightX = M + cardW + gap;
  card(s, leftX, 2.48, cardW, 3.95);
  card(s, rightX, 2.48, cardW, 3.95);
  txt(s, "관계 특징 비교", { x: leftX + 0.25, y: 2.73, w: cardW - 0.5, h: 0.38, fontSize: 18, bold: true, color: C.blue });
  table(s, [
    ["무엇을 확인했나", "결과"],
    ["MFCC cross-correlation (계수 간 상관)", "86.91% · 소폭 상승, 이후 추가 시 제외"],
    ["Pitch candidates (복수 음높이 후보)", "86.84% · 기준보다 낮음"],
    ["Band-envelope coherence (대역 동조)", "초기 86.90%; 재학습 86.95%"],
    ["MFCC reduction (계수 축소)", "세 조합 모두 86.32–86.87%로 기존 표현을 대체하지 못함"],
  ], leftX + 0.15, 3.2, [2.55, 2.85], { rowH: 0.5, fontSize: 10.3 });

  txt(s, "과거 점수 재현 여부", { x: rightX + 0.25, y: 2.73, w: cardW - 0.5, h: 0.38, fontSize: 18, bold: true, color: C.blue });
  txt(s, [
    { text: "대역 간 포락선 특징의 최초 86.90%는 같은 과거 캐시로 재현되지 않았습니다.", options: { breakLine: true } },
    { text: "검증된 과거 캐시 재학습은 86.95%였고, 이후 clean chain은 86.96%였습니다.", options: { breakLine: true } },
    { text: "실행 조건이 달라 점수 차이는 직접 비교하지 않았습니다.", options: { bold: true } },
  ], { x: rightX + 0.28, y: 3.35, w: cardW - 0.56, h: 2.3, fontSize: 14, paraSpaceAfter: 8 });
  caption(s, "근거: docs/mission2_experiment_log.md §4.4, §9.1", 7.02);
  s.addNotes("MFCC 계수끼리의 상관, 여러 음높이 후보의 경쟁, 주파수 대역 간 포락선 결합을 확인했습니다. 과거 결과가 재현되는지도 살펴봤습니다. 포락선의 첫 기록 86.90%는 재현되지 않았고, 과거 캐시 재학습과 clean 결과는 별도로 기록했습니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 발화 안의 시간 변화도 더해 보기", 20);
  coreMsg(s, "경계·엔트로피·시간대별 음색 변화는 일부 점수를 높였지만, 조합별 이득은 균형을 따져 보류했습니다");
  table(s, [
    ["검토한 정보", "확인하려던 점", "요약 결과와 결정"],
    ["발화 경계·활동 패턴", "짧거나 잘린 발화의 시작·끝 단서", "52D 대비 +630 정답, 가장 높은 단일 추가 결과지만 자동 승격 보류"],
    ["스펙트럼 엔트로피 변화", "소리의 퍼짐·변동성", "+216 정답; 화자별 균형 차이로 보류"],
    ["시간대별 MFCC 대비", "발화 앞·중간·뒤의 차이", "+103 정답; 보류"],
    ["특징 조합", "단일 특징의 상보성 확인", "과거 캐시에서 3개 조합은 87.21–87.60%; 가장 높은 조합도 화자·취약 구간 저하로 자동 채택 안 함"],
    ["Delta-MFCC 재검토", "짧은 겹침 발화의 오류 보완", "old-cache 59→61D +0.018%p (+32건); Recall 0 +1.47%p / 1 −1.32%p. 취약 구간 개선에도 순이득이 작고 균형 교환이 있어 HOLD"],
  ], M + 0.04, 2.38, [2.45, 4.1, 5.05], { rowH: 0.59, fontSize: 9.8 });
  txt(s, "과거 캐시의 59D 점수(87.46%)와 clean chain의 59D(87.38%)는 같은 결과가 아닙니다.", { x: M + 0.2, y: 6.28, w: CW - 0.4, h: 0.28, fontSize: 11.5, bold: true, color: C.sub, align: "center" });
  caption(s, "근거: docs/mission2_experiment_log.md §4.5–4.7", 7.02);
  s.addNotes("에너지 경계, 스펙트럼 엔트로피, 발화 위치별 MFCC 차이를 추가로 확인했습니다. 단일 특징과 여러 조합 모두 일부 정확도 향상이 있었지만, 가장 높은 조합에서도 특정 화자나 취약 구간이 나빠졌습니다. AF05도 취약 구간에서는 나아졌지만 전체 정답 순증이 32건에 그치고 화자별 Recall이 반대 방향으로 움직여 채택하지 않았습니다. 이 수치는 과거 캐시 기준이며 이후 clean 기준과 섞지 않습니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — clean 기준에서 표준 음향 특징 비교", 21);
  coreMsg(s, "더 많은 표준 음향 통계를 써도 clean 59개 특징보다 정확도가 높아지지는 않았습니다");
  const gap = 0.35, cardW = (CW - gap) / 2;
  const leftX = M, rightX = M + cardW + gap;
  card(s, leftX, 2.52, cardW, 3.55);
  card(s, rightX, 2.52, cardW, 3.55);
  stat(s, leftX + 0.2, 2.85, cardW - 0.4, "87.38%", "clean handcrafted 59D", C.blue, 34);
  txt(s, "현재 비교 기준\nMFCC + 음높이·유성성 + CPP + 대역·시간 변화 특징", { x: leftX + 0.3, y: 4.28, w: cardW - 0.6, h: 1.0, fontSize: 14, color: C.text, align: "center", valign: "middle" });
  stat(s, rightX + 0.2, 2.85, cardW - 0.4, "86.40%", "eGeMAPSv02 표준 88D", C.red, 34);
  txt(s, "같은 내부 분할과 학습 조건\n오류 1,738건 증가 · 화자 0 재현율 약 2.30%p 하락", { x: rightX + 0.3, y: 4.28, w: cardW - 0.6, h: 1.0, fontSize: 14, color: C.text, align: "center", valign: "middle" });
  insight(s, M, 6.25, CW, 0.5, [
    { text: "판단  ", options: { bold: true, color: C.blue } },
    { text: "eGeMAPS를 단독 입력으로 바꾸는 방향은 채택하지 않았습니다. 일부 구간의 장점은 별도 가설로 남겼습니다." },
  ]);
  caption(s, "근거: docs/mission2_experiment_log.md §4.8–4.9 (같은 clean 내부 검증 비교)", 7.02);
  s.addNotes("여기서는 이전 실험과 평가 조건을 섞지 않기 위해, 같은 clean Training 내부 분할에서 handcrafted 59D와 표준 eGeMAPS 88D를 비교했습니다. eGeMAPS는 더 많은 숫자를 제공했지만 정확도가 87.38%에서 86.40%로 내려갔고 오류가 1,738건 늘었습니다. 그래서 단독 대체 표현으로는 채택하지 않았습니다. 특정 짧은 발화 구간의 결과가 나았던 점은 전체 우위와 다르므로 별도 가능성으로만 기록했습니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — clean 59개 입력에서 더 강한 모델 탐색", 22);
  coreMsg(s, "모델 계열을 바꾸자 TabM과 ResNet이 약 88.4%로 가장 높은 내부 기준이 됐습니다");
  table(s, [
    ["비교 단계", "무엇을 시험했나", "내부 검증 결과와 결정"],
    ["기본 분류기 비교", "부스팅 모델과 더 큰 MLP", "LightGBM·XGBoost·CatBoost는 85.36–85.97%; 강한 MLP는 87.60%"],
    ["수치 임베딩 MLP", "각 숫자를 학습 표현으로 변환", "87.49%; 학습-검증 간격과 높은 확신 오류가 커 기준으로 채택 안 함"],
    [{ text: "TabM", options: { bold: true, color: C.blue } }, "여러 MLP 예측을 함께 학습", { text: "88.40% · 비교 기준 A", options: { bold: true, color: C.blue } }],
    ["FT-Transformer", "특징 간 관계를 attention으로 학습", "88.22% · 결과는 보존, 기준 두 모델에는 미포함"],
    [{ text: "Tabular ResNet", options: { bold: true, color: C.blue } }, "층 사이 정보를 연결해 학습", { text: "88.38% · 비교 기준 B, 화자 0 재현율이 상대적으로 높음", options: { bold: true, color: C.blue } }],
  ], M + 0.05, 2.35, [2.3, 4.4, 4.9], { rowH: 0.57, fontSize: 10.7 });
  txt(s, "TabM과 ResNet 모두 최종 모델로 정한 것은 아니며, 이후 표현 비교에 쓰는 두 기준으로 보존했습니다.", { x: M + 0.2, y: 6.2, w: CW - 0.4, h: 0.33, fontSize: 11.5, bold: true, color: C.sub, align: "center" });
  caption(s, "근거: docs/mission2_experiment_log.md §4.10–4.12", 7.02);
  s.addNotes("먼저 clean 59D를 유지하고 여러 분류기를 비교했습니다. 부스팅 계열은 강한 MLP보다 낮았고, 더 복잡한 숫자 임베딩 MLP는 검증 간격과 높은 확신 오류가 커 채택하지 않았습니다. 이어서 TabM, FT-Transformer, Tabular ResNet을 비교했고 TabM과 ResNet의 정확도가 약 88.4%로 가장 높고 서로 다른 화자 강점도 보여 후속 비교 기준으로 남겼습니다. 학습 recipe도 함께 달라졌으므로 차이를 모델 구조 하나의 효과라고 단정하지 않습니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 점수 대신 실행 가능성을 본 경로", 23);
  coreMsg(s, "일부 표현 실험은 성능을 겨룬 것이 아니라 중복과 계산 시간을 따져 멈췄습니다");
  const gap = 0.35, cardW = (CW - gap) / 2;
  const leftX = M, rightX = M + cardW + gap;
  card(s, leftX, 2.48, cardW, 3.88);
  card(s, rightX, 2.48, cardW, 3.88);
  txt(s, "Time-Frequency Models (시간·주파수)", { x: leftX + 0.25, y: 2.75, w: cardW - 0.5, h: 0.38, fontSize: 16, bold: true, color: C.blue });
  txt(s, [
    { text: "목적  ", options: { bold: true, color: C.blue } },
    { text: "음성의 시간 변화와 주파수 패턴을 직접 보는 계열 검토", options: { breakLine: true } },
    { text: "결정  ", options: { bold: true, color: C.blue } },
    { text: "팀원이 같은 계열을 탐색 중이라 중복을 피하려고 개인 실험은 건너뜀", options: { breakLine: true } },
    { text: "상태  ", options: { bold: true, color: C.blue } },
    { text: "성능 실패가 아니라 팀 역할 분담에 따른 미실행" },
  ], { x: leftX + 0.3, y: 3.38, w: cardW - 0.6, h: 2.35, fontSize: 14, paraSpaceAfter: 8 });

  txt(s, "Pretrained Speech Models (사전학습 음성)", { x: rightX + 0.25, y: 2.75, w: cardW - 0.5, h: 0.38, fontSize: 16, bold: true, color: C.blue });
  txt(s, [
    { text: "목적  ", options: { bold: true, color: C.blue } },
    { text: "대규모 음성으로 미리 학습된 표현이 도움되는지 검토", options: { breakLine: true } },
    { text: "시간 확인  ", options: { bold: true, color: C.blue } },
    { text: "WavLM 약 45.37시간, DPWavLM 약 63.87시간 추정", options: { breakLine: true } },
    { text: "결론  ", options: { bold: true, color: C.blue } },
    { text: "전체 추출·분류기 평가까지 진행하지 않아 성능은 미평가" },
  ], { x: rightX + 0.3, y: 3.38, w: cardW - 0.6, h: 2.35, fontSize: 14, paraSpaceAfter: 8 });
  insight(s, M, 6.52, CW, 0.42, [
    { text: "구분  ", options: { bold: true, color: C.blue } },
    { text: "실행하지 않은 경로의 점수는 없으며, 이를 낮은 성능으로 해석하지 않습니다." },
  ]);
  caption(s, "근거: docs/mission2_experiment_log.md §4.12, Stage 11·12 기록", 7.02);
  s.addNotes("모든 검토가 정확도 실험으로 끝난 것은 아닙니다. 시간×주파수 모델은 팀원이 같은 방향을 맡아 중복을 피하기 위해 개인 실험을 건너뛰었습니다. 사전학습 음성 모델은 전체 Training을 처리하는 데 약 45시간 또는 64시간이 걸릴 것으로 예상돼 실행 가능성 문제로 종료했습니다. 실제 전체 모델 평가를 하지 않았기 때문에 두 경로 모두 성능 미평가이며, 성능 실패로 볼 수 없습니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 아직 보지 않은 분류 방식도 확인", 24);
  coreMsg(s, "새로운 여섯 방식 중 GANDALF가 가장 높았지만, 기존 TabM·ResNet 기준은 넘지 못했습니다");
  table(s, [
    ["방식", "무엇을 다르게 하나", "내부 검증 정확도"],
    ["EBM", "특징별 효과를 더하는 해석형 부스팅", "83.96%"],
    ["Spline-GAM", "부드러운 곡선으로 특징 효과 학습", "85.18%"],
    ["RFF + Logistic SGD", "비선형 관계를 빠르게 근사", "83.15%"],
    ["DCNv2", "특징 간 조합을 직접 학습", "87.78%"],
    ["TabNet", "단계별로 중요한 입력에 집중", "87.75%"],
    [{ text: "GANDALF", options: { bold: true, color: C.blue } }, "Feature gating (특징별 정보 선택)", { text: "88.17% · 신규 계열 중 최고", options: { bold: true, color: C.blue } }],
  ], M + 0.2, 2.37, [2.45, 6.2, 2.95], { rowH: 0.53, fontSize: 11.2 });
  txt(s, "같은 clean 59D 비교에서 GANDALF가 신규 계열 중 가장 높아 F01·F02 결합 후보로 포함했습니다. TabM·ResNet 기준은 넘지 못했습니다.", { x: M + 0.2, y: 6.18, w: CW - 0.4, h: 0.4, fontSize: 11.4, color: C.sub, align: "center" });
  caption(s, "근거: docs/mission2_experiment_log.md §4.13 (고정 설정 6개 모델 비교)", 7.02);
  s.addNotes("이 단계의 목적은 clean 59개 특징에서 이전에 시험하지 않은 분류 방식을 넓게 확인하는 것이었습니다. 여섯 설정을 결과를 보기 전에 고정해 비교했고, 새 후보 중 GANDALF가 88.17%로 가장 높았습니다. 그래도 앞서 보존한 TabM 88.40%와 ResNet 88.38%에는 미치지 못했습니다. 첫 비교 결과이므로 모든 모델 계열의 우열을 확정하는 것은 아닙니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 두 음향 특징을 함께 넣은 모델", 25);
  coreMsg(s, "세 모델에서 각각 개선된 두 특징을 함께 넣어, 결합해도 효과가 이어지는지 확인했습니다");
  const gap = 0.32, cardW = (CW - gap) / 2;
  const leftX = M, rightX = M + cardW + gap;
  card(s, leftX, 2.48, cardW, 3.92);
  card(s, rightX, 2.48, cardW, 3.92);
  txt(s, "새로 더한 두 소리 요약", { x: leftX + 0.25, y: 2.72, w: cardW - 0.5, h: 0.38, fontSize: 18, bold: true, color: C.blue });
  txt(s, [
    { text: "F01 · MFCC Robust Statistics (분포 요약) · 80D", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "Quantiles (분포 위치)와 IQR (가운데 50% 폭)로 20개 계수의 퍼짐을 요약합니다.", options: { breakLine: true } },
    { text: "F02 · Spectral Contrast (대역 대비) · 10D", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "0–4,000Hz 다섯 구간에서 뚜렷한 소리와 조용한 소리의 차이를 요약합니다." },
  ], { x: leftX + 0.28, y: 3.3, w: cardW - 0.56, h: 2.55, fontSize: 14, paraSpaceAfter: 8 });
  txt(s, "결합 입력으로 학습한 세 모델", { x: rightX + 0.25, y: 2.72, w: cardW - 0.5, h: 0.38, fontSize: 18, bold: true, color: C.blue });
  table(s, [
    ["모델", "내부 검증 정확도"],
    ["TabM", "88.64%"],
    ["Tabular ResNet", "88.55%"],
    ["GANDALF", "88.58%"],
  ], rightX + 0.45, 3.3, [2.8, 2.5], { w: 5.3, rowH: 0.55, fontSize: 13 });
  txt(s, "같은 고정 내부 split·seed 42·threshold 0.5 사용; 공식 Validation은 사용하지 않음", { x: rightX + 0.3, y: 5.65, w: cardW - 0.6, h: 0.45, fontSize: 11.5, color: C.sub, align: "center" });
  caption(s, "근거: docs/mission2_experiment_log.md §4.14 (특징 정의·멤버 결과)", 7.02);
  s.addNotes("MFCC 분포 요약은 평균·표준편차만으로 놓치는 값들의 퍼짐을 더 세밀하게 담으려는 특징입니다. 대역별 대비는 낮은 주파수부터 4,000Hz까지 다섯 구간에서 두드러진 소리와 조용한 소리의 차이를 요약합니다. 두 특징을 clean 59개에 추가해 총 149개 입력으로 만들고 세 모델을 학습했습니다. 세 모델의 점수는 모두 같은 고정 내부 분할에서 나온 것이며 공식 검증 점수는 아닙니다.");
}
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 같은 비중으로 평균한 앙상블", 26);
  coreMsg(s, "세 모델의 확률을 똑같이 평균한 후보가 내부 정확도는 가장 높았지만, 모든 위험이 해결된 것은 아닙니다");
  table(s, [
    ["평균한 모델", "내부 정확도", "간단한 해석"],
    ["TabM + ResNet", "88.72%", "두 모델 평균"],
    ["TabM + GANDALF", "88.82%", "두 모델 평균"],
    ["ResNet + GANDALF", "88.82%", "가장 강한 2모델 조합"],
    [{ text: "E_TRG (TabM + ResNet + GANDALF)", options: { bold: true, color: C.blue } }, { text: "88.85%", options: { bold: true, color: C.blue } }, { text: "세 모델 1/3씩 평균; 내부 검증 정확도가 가장 높아 후보로 선택", options: { bold: true, color: C.blue } }],
  ], M + 0.3, 2.27, [4.0, 2.0, 5.8], { rowH: 0.45, fontSize: 10.8 });
  const gap = 0.28, cardW = (CW - 2 * gap) / 3;
  stat(s, M, 5.02, cardW, "84.34%", "119대원 Recall", C.blue, 25);
  stat(s, M + cardW + gap, 5.02, cardW, "93.00%", "신고자 Recall", C.sky, 25);
  stat(s, M + 2 * (cardW + gap), 5.02, cardW, "1,797", "높은 확신 오답(HCE)", C.red, 25);
  txt(s, "같은 내부 검증에서 네 앙상블 후보를 비교해 점수가 낙관적일 수 있습니다. 짧은 119대원 발화 정확도는 76.71%, 같은 길이의 겹침 발화는 67.90%였습니다. Stage 14에서 Official Validation은 사용하지 않았고, Stage 15 튜닝도 하지 않았습니다.", { x: M + 0.12, y: 6.34, w: CW - 0.24, h: 0.55, fontSize: 10.2, color: C.sub, align: "center" });
  caption(s, "근거: docs/mission2_experiment_log.md §4.14 (후보 비교·취약 구간·선택 경계)", 7.02);
  s.addNotes("세 모델은 신고자일 확률을 각각 내고, 이를 같은 비중으로 평균했습니다. 세 모델 평균이 내부 정확도 88.85%로 가장 높았습니다. 가장 높은 2모델 조합과의 차이는 0.04%포인트보다도 작은 약 0.0315%포인트로, 177,891개 중 약 56개 차이입니다. 신고자 Recall은 높지만 119대원 Recall과 높은 확신 오답도 함께 살펴야 합니다. 같은 내부 검증 자료를 여러 단계와 여러 후보 선택에 재사용했으므로 점수가 낙관적일 수 있습니다. 이 결과는 내부에서 채택한 후보이며 공식 검증된 최종 성능은 아닙니다.");
}

// ---------- Mission 2 추가: Official Validation 최종 평가 ----------
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 고정 후보의 Official Validation 결과", 27);
  coreMsg(s, "Stage 14에서 정한 E_TRG를 추가 튜닝 없이 전체 Training으로 다시 학습해 88.49%를 기록했습니다");
  card(s, M, 2.45, CW, 4.35);

  stat(s, M + 0.15, 2.78, 2.55, "88.49%", "Accuracy · Official Validation", C.blue, 36);
  txt(s, "111,947개 발화 · 3,640통화\n오분류 12,883개 (11.51%)\n\n119대원 Recall 84.18%\n신고자 Recall 92.49%\n화자 Recall 차이 8.31%p", {
    x: M + 0.2, y: 4.04, w: 2.45, h: 2.1, fontSize: 12, color: C.text, align: "center", valign: "middle",
  });

  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["TabM", "ResNet", "GANDALF", "E_TRG"], values: [88.0560, 88.0854, 88.1864, 88.4919] }], {
    x: M + 2.85, y: 2.62, w: 4.55, h: 3.35, barDir: "col", chartColors: [C.sky, C.sky, C.sky, C.blue],
    showTitle: true, title: "모델별 Official Validation Accuracy (%)", titleFontSize: 11, titleColor: C.text, titleFontFace: F,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisMinVal: 86, valAxisMaxVal: 90, valAxisLabelColor: C.sub, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 10,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, varyColors: true,
  });

  txt(s, "E_TRG 혼동행렬 (행=실제, 열=예측)", { x: M + 7.55, y: 2.66, w: 4.1, h: 0.3, fontSize: 12, color: C.sub, align: "center" });
  table(s, [
    ["", "예측 0\n119대원", "예측 1\n신고자"],
    ["실제 0\n119대원", { text: "45,326", options: { bold: true, color: C.blue } }, "8,519"],
    ["실제 1\n신고자", "4,364", { text: "53,738", options: { bold: true, color: C.blue } }],
  ], M + 7.48, 3.08, [1.35, 1.65, 1.65], { w: 4.65, h: 1.86, rowH: 0.62, fontSize: 11, align: "center" });
  txt(s, "고정: 149D · 세 모델 확률 1/3씩 평균 · threshold 0.5 · 학습/Scaler/체크포인트 선택에 Validation 미사용", {
    x: M + 7.55, y: 5.45, w: 4.05, h: 0.65, fontSize: 10.5, color: C.sub, align: "center", valign: "middle",
  });
  txt(s, "해석 경계: Stage 15 자체는 고정 후보 평가지만, 과거 E-series 특징 선택에도 Official Validation을 사용해 독립 1회 holdout은 아닙니다.\n복구: GANDALF 18≠17 epoch 기록으로 첫 실행 중단; cache와 TabM/ResNet checkpoint를 재사용하고 GANDALF만 재학습했습니다.", {
    x: M + 0.16, y: 6.23, w: CW - 0.32, h: 0.5, fontSize: 9.2, color: C.sub, align: "center",
  });
  caption(s, "근거: docs/mission2_experiment_log.md Stage 15 · run audit FINAL_VERIFIED · 고정 threshold 0.5", 7.02);
  s.addNotes("Stage 15는 Stage 14에서 고른 세 모델을 추가 튜닝 없이 전체 Training으로 다시 학습한 뒤, 고정 threshold 0.5에서 저장된 Official Validation 예측을 평가했습니다. Accuracy는 88.4919%이며 혼동행렬 순서는 TN/FP/FN/TP입니다. E_TRG는 세 단독 모델 중 가장 높은 GANDALF보다 0.3055%p 높습니다. Official Validation은 이 refit의 scaler, loss, gradient, early stopping, checkpoint 선택에 사용하지 않았습니다. 다만 과거 E-series 특징 비교에도 같은 Official Validation이 사용됐으므로 독립된 1회 holdout이라고 표현하지 않습니다. 1차 실행의 고정 epoch 불일치는 audit에서 중단했고, 복구 때 feature cache와 두 checkpoint를 재사용해 GANDALF만 재학습했습니다.");
}

// ---------- Mission 2 추가: Official Validation 잔여 오류 ----------
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 남아 있는 오분류와 취약 구간", 28);
  coreMsg(s, "119대원·짧은 발화·겹침 구간의 오류가 더 많고, 낮은 확신 구간은 오분류율이 44.94%입니다");

  const gap = 0.3, cardW = (CW - gap) / 2;
  const leftX = M, rightX = M + cardW + gap;
  card(s, leftX, 2.45, cardW, 4.35);
  txt(s, "■ 저장된 예측에서 확인된 취약 구간", { x: leftX + 0.2, y: 2.58, w: cardW - 0.4, h: 0.35, fontSize: 13.5, bold: true, color: C.blue });
  table(s, [
    ["구간", "표본 수", "오류", "오류율"],
    ["전체", "111,947", "12,883", "11.51%"],
    ["발화 0.5–1초", "27,392", "4,299", "15.69%"],
    ["Overlap 포함", "48,863", "7,457", "15.26%"],
    ["확신도 0.50–0.60", "12,253", "5,507", "44.94%"],
    ["확신도 ≥0.90 (HCE)", "75,016", "1,301", "1.73%"],
    ["실제 119대원 × 0.5–1초 × overlap", "5,003", "1,550", "31.0%"],
  ], leftX + 0.15, 3.03, [2.55, 1.25, 1.0, 1.0], { w: 5.8, h: 3.01, rowH: 0.43, fontSize: 9.4, align: "center" });
  txt(s, "마지막 행은 3조건 중 실제 화자 0까지 지정한 교차 결과(정확도 69.02%)입니다. 0.5–1초×overlap×낮은 확신의 동시 교차 성능은 저장 집계가 없어 미확정입니다.", {
    x: leftX + 0.2, y: 6.15, w: cardW - 0.4, h: 0.5, fontSize: 9.8, color: C.sub, align: "center", valign: "middle",
  });

  card(s, rightX, 2.45, cardW, 4.35);
  txt(s, "■ 오류에서 확인한 점과 남은 한계", { x: rightX + 0.2, y: 2.58, w: cardW - 0.4, h: 0.35, fontSize: 13.5, bold: true, color: C.blue });
  txt(s, [
    { text: "119대원 분류가 상대적으로 약함  ", options: { bold: true, color: C.red, breakLine: true } },
    { text: "Recall 84.18%로 신고자 92.49%보다 8.31%p 낮습니다. 119대원을 신고자로 잘못 분류한 FP는 8,519건입니다.", options: { breakLine: true } },
    { text: "짧고 겹치는 발화에서 오류가 큼  ", options: { bold: true, color: C.red, breakLine: true } },
    { text: "0.5–1초와 overlap의 각 집계에서 오류율은 각각 15.69%, 15.26%입니다. 동시 조건의 저확신 성능은 추정하지 않습니다.", options: { breakLine: true } },
    { text: "낮은 확신에서 판정이 불안정함  ", options: { bold: true, color: C.red, breakLine: true } },
    { text: "확신도 0.50–0.60의 정확도는 55.06%입니다. 오류 원인 자체는 집계 결과만으로 확정할 수 없습니다." },
  ], { x: rightX + 0.25, y: 3.02, w: cardW - 0.5, h: 3.5, fontSize: 11, paraSpaceAfter: 6, valign: "middle" });
  caption(s, "근거: docs/mission2_experiment_log.md Stage 15 후속 오류 분석 · 음성 청취/원인 판정 없이 저장 예측만 분석", 7.02);
  s.addNotes("이 장의 수치는 복구 run의 FINAL_VERIFIED Official Validation 저장 예측에서 계산된 기술통계입니다. 0.5–1초, overlap, 낮은 확신 수치는 각각의 marginal slice라서 이 세 조건이 동시에 있는 표본의 성능으로 읽으면 안 됩니다. 실제 speaker 0과 0.5–1초, overlap을 함께 본 별도 slice는 n=5,003, accuracy 69.0186%입니다. 낮은 확신까지 겹친 4중 조건 수치는 저장 집계에서 확인되지 않아 미확정으로 남깁니다. 정확도 저하의 원인을 음성으로 확인하거나 청취한 결과가 아니므로 원인을 단정하지 않습니다.");
}


// Mission 3 — 환자 증상 인식 (29~31장)
// ======================================================================

// 29. 문제와 최종 모델 (main 유지 + 수빈 님 PR #200의 파이프라인·근거 문구)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 문제 정의와 접근", 29);
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
    branches: [["RoBERTa-large", "seed 42·43·44 평균", "× 0.55"], ["TF-IDF + 로지스틱 회귀", "초기 기준선", "× 0.45"]], brW: 2.5, brGap: 0.15,
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

// 30. 최종 결과
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 최종 결과", 30);
  coreMsg(s, "최종 3-seed RoBERTa + TF-IDF 앙상블: 공식 Validation Macro F1 65.34%");
  card(s, M, 2.45, 7.3, 4.35);
  txt(s, "65.34%", { x: M + 0.2, y: 2.65, w: 2.6, h: 0.75, fontSize: 38, color: C.blue, bold: true, align: "center", valign: "middle" });
  txt(s, "Macro F1 · 공식 Val 3,640건", { x: M + 0.2, y: 3.43, w: 2.6, h: 0.5, fontSize: 11.5, color: C.sub, align: "center" });
  insight(s, M + 0.2, 4.28, 2.6, 2.3, [
    { text: "Micro F1  66.90%", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "빈 예측 149건", options: { breakLine: true } },
    { text: "앙상블 가중치\nRoBERTa 평균 0.55\n+ TF-IDF 0.45", options: { fontSize: 10.5 } },
  ]);
  // 증상별 F1 (최종 앙상블, 공식 Val) — 낮은 3개만 빨강. 근거: dcc-lab results/M3-FINAL-002/metrics.json
  const sym = [["열상", 0.8828], ["복통", 0.8271], ["고열", 0.7001], ["호흡곤란", 0.6795], ["어지러움", 0.657], ["구토", 0.6088], ["전신쇠약", 0.5939], ["두통", 0.5315], ["오심", 0.4]];
  barChart(s, sym.map(r => r[0]), sym.map(r => r[1]), {
    name: "F1", barDir: "bar", catAxisOrientation: "maxMin", dataLabelFormatCode: "0.0%", x: M + 2.95, y: 2.6, w: 4.2, h: 4.1,
    chartColors: sym.map((r, i) => (i >= 6 ? C.red : C.sky)), showTitle: true, title: "증상별 F1 (최종 앙상블, 공식 Val)", titleFontSize: 11,
    dataLabelFontSize: 11, dataLabelFontBold: false, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 12,
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
  txt(s, "오심 40.00% · 두통 53.15% · 전신쇠약 59.39%", { x: rx + 0.2, y: 5.92, w: rw - 0.4, h: 0.6, fontSize: 12.2, bold: true, color: C.red, align: "center", valign: "middle" });
  caption(s, "근거: submission/mission3/README.md, m3_final_train.executed.ipynb. 모델별 점수 차이가 특정 원인만의 효과임을 뜻하지 않음.");
  s.addNotes("근거: submission/mission3/README.md 결과 표(seed 42·43·44 0.6479/0.6441/0.6445, 앙상블 0.6534, Micro 0.6690, 빈 예측 149). 개선 흐름: M3-OFFVAL-001(TF-IDF 0.5689, KcBERT 0.6246), M3-OFFVAL-002(RoBERTa-large 0.6433), M3-FINAL-002(앙상블 0.6534). 증상별 약점: M3-FINAL-002(오심·두통·전신쇠약 F1이 가장 낮음).");
}

// 31. 모델 선정과 오류 분석 (기존 17장, 페이지 번호 외 변경 없음)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 모델 선정과 오류 분석", 31);
  coreMsg(s, "낮은 F1을 단순 성능 문제로 보지 않고, 전사와 증상 라벨의 정보 차이를 점검했다.");
  // 상단: 미션 3 데이터 진단의 핵심 발견을 3개 증상 가로 막대그래프로 요약.
  // F1 차트는 30장에 이미 있어 중복 제거. 여기의 미검출률은 최종 모델 예측값이 아니라
  // Training 내부 홀드아웃 5,792건의 라벨-전사 키워드 정규식 탐색 결과(M3-ERR-001).
  const leftX = M, leftW = 6.65, topY = 2.35, topH = 1.73;
  const rightX = M + leftW + 0.30, rightW = CW - leftW - 0.30;
  card(s, leftX, topY, leftW, topH, C.bgGray);
  title(s, "■ 저성능 3개 증상의 직접 표현 미검출률", leftX + 0.19, topY + 0.12, leftW - 0.38, 14);
  txt(s, "Training 내부 홀드아웃 5,792건 · 증상별 정규식 탐색", {
    x: leftX + 0.19, y: topY + 0.43, w: leftW - 0.38, h: 0.18,
    fontSize: 9.1, color: C.sub,
  });
  const noDirectExpression = [
    ["오심", 88.6, C.red],
    ["전신쇠약", 75.7, "E78A47"],
    ["두통", 63.7, C.sky],
  ];
  noDirectExpression.forEach(([symptom, percent, col], i) => {
    const yy = topY + 0.70 + i * 0.33;
    txt(s, symptom, { x: leftX + 0.22, y: yy - 0.045, w: 1.10, h: 0.26, fontSize: 11.8, bold: true, color: C.text, valign: "middle" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: leftX + 1.35, y: yy, w: 3.50, h: 0.20,
      fill: { color: C.line }, rectRadius: 0.055, line: { color: C.line },
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: leftX + 1.35, y: yy, w: 3.50 * percent / 100, h: 0.20,
      fill: { color: col }, rectRadius: 0.055, line: { color: col },
    });
    txt(s, percent.toFixed(1) + "%", {
      x: leftX + 4.99, y: yy - 0.052, w: 0.73, h: 0.30,
      fontSize: 13.2, bold: true, color: col, align: "right", valign: "middle",
    });
  });

  // 실제 발화 카드: 전화 통화 GIF(가벼운 흔들림/파형) + PPT 편집 가능한 말풍선/라벨.
  // 미션 2와 동일한 GIF 삽입 방식. 첫 프레임만 보여도 전체 메시지를 이해할 수 있다.
  // 실제 모델 예측 장면이 아닌, 기존 보고서의 정규식 기반 전사-라벨 탐색 사례이다.
  card(s, rightX, topY, rightW, topH, C.light);
  title(s, "■ 오류 분석에서 읽은 실제 발화", rightX + 0.20, topY + 0.12, rightW - 0.40, 14);
  s.addImage({
    path: "figures/anim/m3_call_wave.gif",
    x: rightX + 0.24, y: topY + 0.60, w: 1.38, h: 0.55,
  });
  txt(s, "119 신고 · 음성 전달", {
    x: rightX + 0.20, y: topY + 1.21, w: 1.46, h: 0.22,
    fontSize: 9.0, color: C.sub, align: "center",
  });

  // 말풍선 안의 전사 원문·라벨 대비 문구는 편집 가능한 개별 PPT 도형/텍스트.
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: rightX + 1.75, y: topY + 0.47, w: rightW - 1.96, h: 0.69,
    fill: { color: C.white }, line: { color: C.sky, width: 1.7 }, rectRadius: 0.13,
  });
  txt(s, "“어지러워서 정신을\\n못 차리겠다”", {
    x: rightX + 1.89, y: topY + 0.52, w: rightW - 2.25, h: 0.58,
    fontSize: 15.0, bold: true, color: C.blue, align: "center", valign: "middle",
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: rightX + 1.76, y: topY + 1.24, w: 1.41, h: 0.29,
    fill: { color: C.blue }, line: { color: C.blue }, rectRadius: 0.10,
  });
  txt(s, "오심 양성 라벨", {
    x: rightX + 1.78, y: topY + 1.27, w: 1.37, h: 0.22,
    fontSize: 10.1, bold: true, color: C.white, align: "center", valign: "middle",
  });
  txt(s, "↔", {
    x: rightX + 3.19, y: topY + 1.29, w: 0.21, h: 0.18,
    fontSize: 11.0, bold: true, color: C.sub, align: "center",
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: rightX + 3.43, y: topY + 1.24, w: rightW - 3.65, h: 0.29,
    fill: { color: C.white }, line: { color: C.red, width: 1.4 }, rectRadius: 0.10,
  });
  txt(s, "오심 직접 표현 미검출", {
    x: rightX + 3.45, y: topY + 1.27, w: rightW - 3.69, h: 0.22,
    fontSize: 9.5, bold: true, color: C.red, align: "center", valign: "middle",
  });
  txt(s, "발화와 증상 라벨 사이의 정보 차이 · 키워드 기준 · 최종 RoBERTa 예측 사례 아님", {
    x: rightX + 0.17, y: topY + 1.57, w: rightW - 0.34, h: 0.18,
    fontSize: 8.8, color: C.sub, align: "center", valign: "middle",
  });
  const lw = 5.0;
  insight(s, M, 4.15, lw, 1.3, [
    { text: "REJECT · 과거 결합 기각", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "KcBERT + RoBERTa: 동일 내부 분할에서 +0.28%p. 사전 채택 기준 +1.00%p에 미달하고 오류도 많이 겹침." },
  ], C.light, 11.5);
  insight(s, M, 5.55, lw, 1.3, [
    { text: "ADOPT · 최종 가중 결합 채택", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "RoBERTa 3-seed 55% + TF-IDF 45%. 결합 비율은 Training 홀드아웃에서 사전 확정, 공식 Val은 채점만 수행." },
  ], C.mintLight, 11.5);
  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, 4.15, rw, 2.7);
  title(s, "■ 오류 분석 — 전사에 직접 표현이 없는 양성 라벨", rx + 0.25, 4.27, rw - 0.5);
  s.addChart(pres.charts.DOUGHNUT, [{ name: "양성 라벨", labels: ["직접 표현 없음", "직접 표현 있음"], values: [52, 48] }], {
    x: rx + 0.1, y: 4.65, w: 2.7, h: 2.15, holeSize: 62, chartColors: [C.red, C.line], showLegend: false, showPercent: false, showValue: false, showLabel: false, dataBorder: { pt: 1, color: "FFFFFF" },
  });
  txt(s, "약 52%", { x: rx + 0.1, y: 5.35, w: 2.7, h: 0.5, fontSize: 19, bold: true, color: C.red, align: "center", valign: "middle" });
  txt(s, "양성 라벨 중\n표현 없음", { x: rx + 0.1, y: 5.82, w: 2.7, h: 0.5, fontSize: 10.5, color: C.sub, align: "center" });
  [["92%", "두 모델이 놓친 양성 중\n직접 표현 없는 라벨", C.red], ["약 90%", "표현 있는 라벨 중\n맞힌 비율", C.sky]].forEach(([n, l, col], i) => {
    const x = rx + 3.0 + i * ((rw - 3.2) / 2);
    stat(s, x, 4.75, (rw - 3.2) / 2 - 0.1, n, "", col, 30);
    txt(s, l, { x, y: 5.5, w: (rw - 3.2) / 2 - 0.1, h: 0.6, fontSize: 11, color: C.sub, align: "center" });
  });
  txt(s, "→ TF-IDF·KcBERT의 오류 양상 차이 확인\n(과거 모델 분석·정규식 기준, 오라벨 단정 아님)", { x: rx + 3.0, y: 6.2, w: rw - 3.2, h: 0.55, fontSize: 10.5, bold: true, color: C.blue, align: "center", valign: "middle" });
  caption(s, "근거: reports/mission3/error_analysis/README.md·M3-ERR-001. 상단은 Training 텍스트·라벨 탐색, 하단은 과거 TF-IDF·KcBERT 분석(최종 RoBERTa 오답 분석 아님).");
  s.addNotes("상단은 공식 Validation 또는 최종 RoBERTa 예측 결과가 아니다. Training 80:20 내부 홀드아웃 5,792건의 증상별 라벨 양성 집합에 대해 정규식으로 직접 표현 탐색: 오심 592/668=88.6%, 전신쇠약 804/1,062=75.7%, 두통 370/581=63.7% 미검출. 인용 발화는 reports/mission3/error_analysis/README.md 1절의 오심 라벨 사례이다. 정규식은 구어체 변형을 모두 찾지 못하므로 직접 표현 미검출을 증상 부재나 오라벨이라고 단정하지 않는다. 하단 52%는 라벨-전사 탐색 비율, 92% 및 약 90%는 이전 TF-IDF·KcBERT 및 해당 앙상블의 오류 분석(공식 Val 3,640건)이며 최종 RoBERTa 모델의 오류 통계가 아님. 모델 결정을 위한 내부 실험 M3-ENS-001(KcBERT+RoBERTa +0.0028 < +0.01 기각)과 M3-FINAL-002(최종 RoBERTa×3+TF-IDF 가중치 0.55/0.45 Training 내부 결정)는 별도 조건으로 해석해야 한다. 출처: reports/mission3/error_analysis/README.md, docs/experiment_log.md M3-ERR-001, docs/decisions.md M3-ENS-001 및 M3-FINAL-002.");
}

// =====================================================================
// 32. 04-(1) 성능 요약과 교훈 (기존 18장)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(1) 성능 요약과 교훈", 32);
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
  s.addNotes("근거: submission/mission1/README.md(98.10%), " + M2.evidence + ", submission/mission3/README.md(0.6534). baseline: M1 95.11%(52D LR, docs/decisions.md 2026-10-05), M2 86.44%(L1 LightGBM, M2-RETRAIN-001), M3 0.5689=56.89%(TF-IDF, M3-OFFVAL-001). M3 최종 Macro F1 0.6534=65.34%, 차이는 0.0845=8.45%p(공식 Val 결과 간 비교, 통제된 인과 효과를 뜻하지 않음).");
}

// =====================================================================
// 33. 04-(2) 사회안전 시사점과 향후 계획 — 사회안전은 이 장 한 곳에만 (기존 19장)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(2) 사회안전 시사점과 향후 계획", 33);
  coreMsg(s, "세 미션을 이으면, 신고 전화가 들어오는 순간부터 출동 판단까지를 돕는 하나의 흐름이 된다.");
  // 신고 전화 → M2 → M1 → M3 → 상황실: 순서대로 켜지는 GIF (ppt/tools/make_gif_flow.py), 첫 프레임은 전부 켜진 상태
  { const gw = 12.0, gh = gw * 232 / 1920; s.addImage({ path: "figures/anim/flow_pipeline.gif", x: (W - gw) / 2, y: 2.4, w: gw, h: gh }); }
  const lw = 5.0, rw = CW - lw - 0.4; // 향후 계획 문장이 길어 오른쪽을 넓게
  card(s, M, 3.95, lw, 2.3);
  title(s, "■ 기대효과", M + 0.25, 4.07, lw - 0.5, 18);
  txt(s, [
    strong("접수 시간 단축"), { text: " — 수기 기록 부담 감소", options: { breakLine: true } },
    strong("누락 방지"), { text: " — 긴박한 통화 속 증상 언급 표시", options: { breakLine: true } },
    strong("판단 보조"), { text: " — 출동 우선순위 결정 지원" },
  ], { x: M + 0.25, y: 4.6, w: lw - 0.5, h: 1.55, fontSize: 14, paraSpaceAfter: 8 });
  const rx = M + lw + 0.4;
  card(s, rx, 3.95, rw, 2.3);
  title(s, "■ 본선까지 향후 계획", rx + 0.25, 4.07, rw - 0.5, 18);
  txt(s, [
    strong("M1  "), { text: "오분류 통화 정성 분석, 피치(F0) 특징 검증", options: { breakLine: true } },
    strong("M2  "), { text: M2.future, options: { breakLine: true } },
    strong("M3  "), { text: "근거 없는 라벨 구간 오류 분석, 오심·두통 개선", options: { breakLine: true } },
    strong("공통  "), { text: "통화 도중 실시간 처리 시나리오 검토" },
  ], { x: rx + 0.25, y: 4.6, w: rw - 0.5, h: 1.6, fontSize: 12.5, paraSpaceAfter: 6 });
  insight(s, M, 6.4, CW, 0.5, [
    strong("유의점  "), { text: "모든 출력은 요원의 판단을 돕는 정보이며, 오분류 가능성을 감안해 사람이 최종 확인한다." },
  ], C.bgGray, 12.5);
  s.addNotes("출제문제 12쪽 필수 항목 '사회안전 시사점'은 이 장 한 곳에만 둔다(미션별 장에는 넣지 않음). 향후 계획 근거: M1 — 10장 후속 과제, M2 — docs/experiment_log.md M2-SSL-003, M3 — M3-ERR-001·M3-FINAL-002.");
}

// =====================================================================
// 34. 별첨 A1 — Mission 3 데이터 특성과 모델 설계 근거 (기존 p17 + p21 통합)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "별첨", "Mission 3", "Mission 3 데이터 특성과 모델 설계 근거", 34);
  coreMsg(s, "데이터의 불균형과 입력 길이를 고려해 학습 손실과 최종 앙상블 방식을 설계했다.");

  // 상단: 데이터 관찰(좌)과 손실 설계(우) — 수식은 모두 편집 가능한 텍스트.
  const topY = 2.36, topH = 2.82, gap = 0.25, leftW = 6.0;
  const rightX = M + leftW + gap, rightW = CW - leftW - gap;
  card(s, M, topY, leftW, topH);
  card(s, rightX, topY, rightW, topH);
  title(s, "■ 데이터 관찰 → 설계 결정", M + 0.2, topY + 0.11, leftW - 0.4, 15);
  const observations = [
    ["라벨 불균형", "9개 증상 양성 비율 9.9~23.2%", "pos_weight BCE + Soft macro-F1"],
    ["입력 길이", "40.7%가 256토큰 초과 · 512토큰 +0.0074", "기준 +0.01 미달·약 4배 시간 → 256 유지"],
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
  s.addNotes("기존 p17 + p21 통합. 데이터: Training 29,200통화, 평가 전용 Validation 3,640통화, 9개 증상, Training 통화당 평균 증상 수 1.43. 라벨별 양성 비율(Training): 열상 23.2, 어지러움 21.7, 전신쇠약 18.2, 고열 17.8, 구토 15.3, 호흡곤란 13.4, 복통 11.9, 오심 11.4, 두통 9.9(%). 입력 길이와 512토큰 실험은 Training 내부 80:20 홀드아웃: 40.7% 256토큰 초과, Macro F1 +0.0074, 사전 기준 +0.01 미달, 학습시간 약 4배. 전사 utterances[].text만 원본 순서로 연결하며 화자 태그와 허용되지 않은 라벨은 사용하지 않음. Soft macro-F1 상세: pᵢⱼ = sigmoid(zᵢⱼ), SoftF1ⱼ = 2Σᵢ(pᵢⱼyᵢⱼ) / [Σᵢpᵢⱼ + Σᵢyᵢⱼ + 10⁻⁸]. 가중치 wⱼ = clip(0.5 × √((N − Pⱼ) / max(Pⱼ,1)),1,6). 실효 batch 16 = micro 8 × 2, soft-F1 micro batch별 계산. 최종 RoBERTa seeds 42·43·44 확률 평균 0.55와 TF-IDF 확률 0.45를 결합, 모든 증상 threshold 0.5 적용(별도 후처리 없음). 근거: reports/eda/eda_report.md; docs/experiment_log.md M3-ROBERTA-007; submission/mission3/README.md; submission/mission3/train/train_final.py; submission/mission3/train/build_ensemble.py; submission/mission3/mission3_model.py.");
}

// =====================================================================
// 35. 별첨 A2 — M3 입력 길이 확대 실험
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "별첨", "Mission 3", "M3 입력 길이 확대 실험", 35);
  coreMsg(s, "512토큰은 성능이 개선됐지만 사전 채택 기준과 계산비용 조건을 충족하지 못함");

  const epochs = ["Epoch 1", "Epoch 2", "Epoch 3"];
  s.addChart(pres.charts.LINE, [
    { name: "512토큰 모델", labels: epochs, values: [0.6439, 0.6504, 0.6515] },
    { name: "256토큰 기준 성능", labels: epochs, values: [0.6441, 0.6441, 0.6441] },
    { name: "사전 채택 기준", labels: epochs, values: [0.6541, 0.6541, 0.6541] },
  ], {
    x: M, y: 2.45, w: 7.0, h: 3.75, showTitle: true,
    title: "512토큰 모델의 epoch별 내부 홀드아웃 Macro F1", titleFontFace: F, titleFontSize: 11.5, titleColor: C.text,
    chartColors: [C.blue, "9E9E9E", C.red], showLegend: true, legendPos: "b", legendFontFace: F, legendFontSize: 9,
    showValue: false, lineSize: 2.2, lineDataSymbol: "circle", lineDataSymbolSize: 5,
    catAxisLabelFontFace: F, catAxisLabelFontSize: 10.5, catAxisLabelColor: C.text,
    valAxisLabelFontFace: F, valAxisLabelFontSize: 9.5, valAxisLabelColor: C.sub,
    valAxisMinVal: 0.63, valAxisMaxVal: 0.66, valAxisMajorUnit: 0.005, valAxisLabelFormatCode: "0.0%",
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" },
    chartArea: { border: { color: C.white, pt: 0 }, fill: { color: C.white, transparency: 100 } },
    plotArea: { border: { color: C.white, pt: 0 }, fill: { color: C.white, transparency: 100 } },
  });
  // 512토큰 계열만 필요한 값 레이블을 별도 편집 가능한 텍스트로 표시한다.
  [["64.39%", 1.38, 4.34], ["65.04%", 3.35, 3.75], ["65.15%", 5.34, 3.63]].forEach(([v, x, y]) => {
    txt(s, v, { x, y, w: 0.78, h: 0.22, fontSize: 9.3, bold: true, color: C.blue, align: "center" });
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
// 36. 별첨 A3 — Mission 3 낮은 성능 3개 라벨의 오류 구조
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "별첨", "Mission 3", "Mission 3  낮은 성능 3개 라벨의 오류 구조", 36);
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
// 37. 마무리 — 기존 20장 공식 템플릿 그대로 사용 + 팀 정보 유지
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

const outputFile = process.argv[2] || "DCC2026_예선_결과정리.pptx";
pres.writeFile({ fileName: outputFile }).then(async f => {
  await styleLengthExperimentLines(outputFile);
  console.log("wrote", f);
});
