// DCC 2026 대학부 예선 결과 정리 PPT — 장별 원고 v2(팀장 검토본)를 옮김. 데이터·규칙 장은 삭제해 20장
// 디자인 규칙은 ppt/design_guide.md, 공식 템플릿 이미지는 ppt/figures/tpl_*.png
const pptxgen = require("pptxgenjs");
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
    ["03", "미션별 수행", ["Mission 1  신고자 성별 분류", "Mission 2  신고자/대원 분류", "Mission 3  환자 증상 인식"]],
    ["04", "종합 및 시사점", ["성능 요약과 교훈", "사회안전 시사점과 향후 계획"]],
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
  s.addNotes("원고 v2 구성 기준 목차 (데이터·규칙 장 삭제 후 20장).");
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
  s.addImage({ path: "figures/m2_synthetic_waveform.png", x: M + 0.08, y: 4.98, w: fw, h: fh });
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
  const iw = 2.6, ih = iw * 744 / 1806;
  s.addImage({ path: "figures/canva/speakers_overlap_crop.png", x: rx + 0.2, y: 3.0, w: iw, h: ih });
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
// Mission 3 — 환자 증상 인식 (15~17장)
// ======================================================================

// 15. 문제와 최종 모델 (main 유지 + 수빈 님 PR #200의 파이프라인·근거 문구)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 문제 정의와 접근", 15);
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

// 16. 최종 결과
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 최종 결과", 16);
  coreMsg(s, "최종 3-seed RoBERTa + TF-IDF 앙상블: 공식 Validation Macro F1 0.6534");
  card(s, M, 2.45, 7.3, 4.35);
  txt(s, "0.6534", { x: M + 0.2, y: 2.65, w: 2.6, h: 0.75, fontSize: 38, color: C.blue, bold: true, align: "center", valign: "middle" });
  txt(s, "Macro F1 · 공식 Val 3,640건", { x: M + 0.2, y: 3.43, w: 2.6, h: 0.5, fontSize: 11.5, color: C.sub, align: "center" });
  insight(s, M + 0.2, 4.28, 2.6, 2.3, [
    { text: "Micro F1  0.6690", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "빈 예측 149건", options: { breakLine: true } },
    { text: "앙상블 가중치\nRoBERTa 평균 0.55\n+ TF-IDF 0.45", options: { fontSize: 10.5 } },
  ]);
  // 증상별 F1 (최종 앙상블, 공식 Val) — 낮은 3개만 빨강. 근거: dcc-lab results/M3-FINAL-002/metrics.json
  const sym = [["열상", 0.8828], ["복통", 0.8271], ["고열", 0.7001], ["호흡곤란", 0.6795], ["어지러움", 0.657], ["구토", 0.6088], ["전신쇠약", 0.5939], ["두통", 0.5315], ["오심", 0.4]];
  barChart(s, sym.map(r => r[0]), sym.map(r => r[1]), {
    name: "F1", barDir: "bar", catAxisOrientation: "maxMin", dataLabelFormatCode: "0.00", x: M + 2.95, y: 2.6, w: 4.2, h: 4.1,
    chartColors: sym.map((r, i) => (i >= 6 ? C.red : C.sky)), showTitle: true, title: "증상별 F1 (최종 앙상블, 공식 Val)", titleFontSize: 11,
    dataLabelFontSize: 11, dataLabelFontBold: false, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 12,
  });
  const rx = M + 7.6, rw = W - M - rx;
  card(s, rx, 2.45, rw, 2.75);
  title(s, "■ 개선 흐름 (모두 공식 Val)", rx + 0.2, 2.56, rw - 0.4);
  [["TF-IDF", "0.5689"], ["KcBERT", "0.6246"], ["RoBERTa-large", "0.6433"], ["앙상블", "0.6534"]].forEach(([m, v], i) => {
    const y = 3.0 + i * 0.52, last = i === 3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.25, y, w: rw - 0.5, h: 0.42, fill: { color: last ? C.blue : C.light }, rectRadius: 0.08, line: { color: last ? C.blue : C.light } });
    txt(s, m, { x: rx + 0.4, y, w: rw - 1.9, h: 0.42, fontSize: 13, bold: last, color: last ? C.white : C.text, valign: "middle" });
    txt(s, v, { x: rx + rw - 1.6, y, w: 1.2, h: 0.42, fontSize: 13, bold: true, color: last ? C.white : C.blue, align: "right", valign: "middle" });
  });
  card(s, rx, 5.35, rw, 1.45, C.bgGray);
  title(s, "■ 가장 약한 증상 (빨강)", rx + 0.2, 5.47, rw - 0.4);
  txt(s, "오심 0.40 · 두통 0.53 · 전신쇠약 0.59", { x: rx + 0.2, y: 5.92, w: rw - 0.4, h: 0.6, fontSize: 15, bold: true, color: C.red, align: "center", valign: "middle" });
  caption(s, "근거: submission/mission3/README.md, m3_final_train.executed.ipynb. 모델별 점수 차이가 특정 원인만의 효과임을 뜻하지 않음.");
  s.addNotes("근거: submission/mission3/README.md 결과 표(seed 42·43·44 0.6479/0.6441/0.6445, 앙상블 0.6534, Micro 0.6690, 빈 예측 149). 개선 흐름: M3-OFFVAL-001(TF-IDF 0.5689, KcBERT 0.6246), M3-OFFVAL-002(RoBERTa-large 0.6433), M3-FINAL-002(앙상블 0.6534). 증상별 약점: M3-FINAL-002(오심·두통·전신쇠약 F1이 가장 낮음).");
}

// 17. 모델 선정과 오류 분석
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 모델 선정과 오류 분석", 17);
  coreMsg(s, "이전 앙상블 기각과 최종 앙상블 채택은 서로 다른 실험 — 검증 조건을 구분");
  // 모델별 공식 Val Macro F1 (가로 막대, 위에서 아래로 진행 순서). 회색 = 기각한 과거 결합
  barChart(s, ["TF-IDF + LR (기준선)", "KcBERT-005", "RoBERTa-003", "과거 3종 앙상블 (기각)", "최종 RoBERTa×3 + TF-IDF"],
    [0.5689, 0.6246, 0.6433, 0.6510, 0.6534], {
      name: "Macro F1", barDir: "bar", catAxisOrientation: "maxMin", x: M, y: 2.3, w: CW, h: 1.75,
      chartColors: [C.sky, C.blue, C.blue, "BDBDBD", C.mint], showTitle: false,
      valAxisMinVal: 0.5, valAxisMaxVal: 0.68, valAxisHidden: true, valGridLine: { style: "none" },
      dataLabelFormatCode: "0.0000", dataLabelFontSize: 11, catAxisLabelFontSize: 11,
    });
  const lw = 5.0;
  insight(s, M, 4.15, lw, 1.3, [
    { text: "선정 근거  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "RoBERTa 3-seed 평균 55% + TF-IDF 45%, 임계값 0.5 고정. 결합 비율은 Training 내부 홀드아웃에서 확정." },
  ], C.light, 11.5);
  insight(s, M, 5.55, lw, 1.3, [
    { text: "해석 주의  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "0.6433 → 0.6534는 학습 데이터량·모델 구성이 모두 달라 통제 실험의 개선 폭으로 해석하지 않음." },
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
  txt(s, "→ 두 모델의 오류가 달라 앙상블로 보완\n(키워드 정규식 기준 탐색, 오라벨 단정 아님)", { x: rx + 3.0, y: 6.2, w: rw - 3.2, h: 0.55, fontSize: 10.5, bold: true, color: C.blue, align: "center", valign: "middle" });
  caption(s, "근거: docs/decisions.md M3-FINAL-002, docs/experiment_log.md M3-ERR-001. 키워드 정규식 기준의 탐색적 결과이며 오라벨로 단정하지 않음.");
  s.addNotes("근거: docs/decisions.md (M3-FINAL-002, 2026-10-11), submission/mission3/README.md, docs/experiment_log.md M3-ENS-001(과거 결합 기각)·M3-ERR-001(양성 라벨 약 52% 근거 없음, 오심 89%·전신쇠약 76%, 두 모델 모두 놓친 양성의 92%가 근거 없는 라벨, 근거 있는 라벨 약 90% 적중), reports/mission3/error_analysis/README.md. '근거 없음'은 키워드 정규식 기준이다. 직접 표기되지 않는 증상도 있을 수 있어 라벨 오류라고 단정하면 안 된다(이수빈 님 PR #200 문구).");
}

// =====================================================================
// 18. 04-(1) 성능 요약과 교훈
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(1) 성능 요약과 교훈", 18);
  coreMsg(s, "공식 Validation은 튜닝에 쓰지 않고, 결과를 보기 전에 고정한 기준만 확인했다.");
  const v = t => strong(t);
  const tiles = [
    ["M1 신고자 성별", "음향 특징 + x-vector + LR", "98.10%", "+2.99%p"],
    [M2.summary.task, M2.summary.model, M2.summary.val, M2.summary.gain],
    ["M3 증상 9종", "RoBERTa-large × 3 + TF-IDF", "0.6534", "+0.0845"],
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
  txt(s, "baseline: M1 95.11%, M2 86.44%, M3 0.5689", { x: M, y: 4.7, w: CW, h: 0.25, fontSize: 11, color: C.sub, align: "right" });
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
  s.addNotes("근거: submission/mission1/README.md(98.10%), " + M2.evidence + ", submission/mission3/README.md(0.6534). baseline: M1 95.11%(52D LR, docs/decisions.md 2026-10-05), M2 86.44%(L1 LightGBM, M2-RETRAIN-001), M3 0.5689(TF-IDF, M3-OFFVAL-001).");
}

// =====================================================================
// 19. 04-(2) 사회안전 시사점과 향후 계획 — 사회안전은 이 장 한 곳에만
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(2) 사회안전 시사점과 향후 계획", 19);
  coreMsg(s, "세 미션을 이으면, 신고 전화가 들어오는 순간부터 출동 판단까지를 돕는 하나의 흐름이 된다.");
  // 신고 전화 → M2 → M1 → M3 → 상황실 (아이콘 흐름)
  const fl = [["flow_call", "신고 전화", "음성 입력"], ["flow_m2", "Mission 2", "신고자 발화 분리"], ["flow_m1", "Mission 1", "신고자 성별"], ["flow_m3", "Mission 3", "증상 태깅"], ["flow_screen", "상황실 화면", "자동 기록"]];
  const fgap = CW / fl.length;
  fl.forEach(([ic, t, sub], i) => {
    const cx = M + fgap * i + fgap / 2;
    s.addImage({ path: IC + ic + ".png", x: cx - 0.4, y: 2.4, w: 0.8, h: 0.8 });
    txt(s, t, { x: cx - fgap / 2 + 0.1, y: 3.25, w: fgap - 0.2, h: 0.3, fontSize: 14, bold: true, color: i === 4 ? "5E8E2A" : C.blue, align: "center" });
    txt(s, sub, { x: cx - fgap / 2 + 0.1, y: 3.55, w: fgap - 0.2, h: 0.28, fontSize: 11.5, color: C.sub, align: "center" });
    if (i < fl.length - 1) arrow(s, cx + fgap / 2 - 0.14, 2.62, 0.28, 0.36);
  });
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
// 20. 마무리 — 공식 템플릿 그대로 사용 + 팀 정보만 빈 여백(2.1~2.9in)에 추가
// =====================================================================
{
  const s = pres.addSlide();
  s.addImage({ path: TPL.closing, x: 0, y: 0, w: W, h: 7.5, sizing: { type: "cover", w: W, h: 7.5 } });
  txt(s, "팀 eigensinn  |  김장수 · 김광희 · 지연우 · 이수빈 · 조지은", { x: M, y: 2.2, w: CW, h: 0.4, fontSize: 16, bold: true, color: C.blue, align: "center" });
  txt(s, "코드 · 실험 기록: GitHub eigensinn-dcc2026/prelim  (docs/experiment_log.md, docs/decisions.md)", { x: M, y: 2.58, w: CW, h: 0.3, fontSize: 11, color: C.sub, align: "center" });
  s.addNotes("공식 템플릿(슬라이드4) 그대로. 흰 여백(약 2.1~2.9in)에만 팀 정보를 얹음 — 배너 그래픽은 건드리지 않음.");
}

pres.writeFile({ fileName: process.argv[2] || "DCC2026_예선_결과정리.pptx" }).then(f => console.log("wrote", f));
