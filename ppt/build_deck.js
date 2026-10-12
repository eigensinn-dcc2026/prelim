// DCC 2026 대학부 예선 결과 정리 PPT 뼈대 — 디자인 규칙은 「ppt 전략 가이드.md」 1~3장을 따름
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
  ph: "9E9E9E", white: "FFFFFF", bigNum: "DCE3EC",
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

const shadow = () => ({ type: "outer", blur: 5, offset: 1, angle: 90, color: "000000", opacity: 0.1 });

// ---------- 공통 요소 ----------
function txt(slide, text, o) {
  slide.addText(text, Object.assign({ isTextBox: true, fontFace: F, fontSize: 14, color: C.text, margin: 0, valign: "top" }, o));
}
// 자리표시자 run (회색 기울임) — 최종 수치·그림이 나오면 교체
const ph = (t, o = {}) => ({ text: `〔${t}〕`, options: Object.assign({ color: C.ph, italic: true }, o) });

// 공식 템플릿 상단 띠(HB=0.704in) + 그 아래 섹션 라벨·구분선·소제목.
// 헤더 블록 전체 높이는 기존과 동일(1.43in)하게 맞춰 본문 카드 y좌표(2.45 등)는 그대로 둠.
function header(slide, sec, secName, sub, page) {
  slide.addImage({ path: TPL.header, x: 0, y: 0, w: W, h: HB, sizing: { type: "cover", w: W, h: HB } });
  txt(slide, `${sec}  ${secName}`, { x: M, y: HB + 0.04, w: 8, h: 0.26, fontSize: 12.5, bold: true, color: C.blue });
  txt(slide, `${TEAM}  |  ${page}`, { x: W - M - 3, y: HB + 0.04, w: 3, h: 0.26, fontSize: 10, color: C.sub, align: "right" });
  slide.addShape(pres.shapes.LINE, { x: M, y: HB + 0.32, w: CW, h: 0, line: { color: C.line, width: 1 } });
  txt(slide, sub, { x: M, y: HB + 0.37, w: 11, h: 0.36, fontSize: 19, bold: true });
}
function coreMsg(slide, runs, y = 1.6) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y, w: CW, h: 0.6, fill: { color: C.light }, rectRadius: 0.1, line: { color: C.light } });
  txt(slide, runs, { x: M + 0.2, y, w: CW - 0.4, h: 0.6, fontSize: 16, bold: true, color: C.blue, align: "center", valign: "middle" });
}
function caption(slide, t, y = 7.0) {
  txt(slide, t, { x: M, y, w: CW, h: 0.3, fontSize: 10, color: C.sub });
}
// 스타일 B 카드: 흰 배경 + 연회색 테두리 + 미세 그림자
function card(slide, x, y, w, h, fill = C.white) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: 0.1, line: { color: C.line, width: 1 }, shadow: shadow() });
}
// 제목바가 달린 프로세스 박스 (가이드 3.5)
function stepBox(slide, x, y, w, h, title, body, barColor = C.blue) {
  card(slide, x, y, w, h);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.5, fill: { color: barColor }, rectRadius: 0.1, line: { color: barColor } });
  slide.addShape(pres.shapes.RECTANGLE, { x, y: 0.3 + y, w, h: 0.2, fill: { color: barColor }, line: { color: barColor } });
  txt(slide, title, { x, y, w, h: 0.5, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle" });
  txt(slide, body, { x: x + 0.15, y: y + 0.62, w: w - 0.3, h: h - 0.75, fontSize: 12, color: C.text, paraSpaceAfter: 4 });
}
function arrow(slide, x, y, w = 0.28, h = 0.4, color = C.sky) {
  slide.addShape(pres.shapes.CHEVRON, { x, y, w, h, fill: { color }, line: { color } });
}
// 그림/차트 자리 (점선 박스) — 실제 그래프가 없을 때만 사용
function figPH(slide, x, y, w, h, label) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.bgGray }, rectRadius: 0.1, line: { color: "BDBDBD", width: 1, dashType: "dash" } });
  txt(slide, [{ text: "그림 자리", options: { bold: true, breakLine: true } }, { text: label }], { x: x + 0.2, y, w: w - 0.4, h, fontSize: 12, color: C.ph, align: "center", valign: "middle" });
}
// 실제 EDA 그림 삽입 (흰 배경 PNG를 카드 안에 맞춤)
function fig(slide, path, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.white }, line: { color: C.line, width: 1 } });
  slide.addImage({ path, x: x + 0.08, y: y + 0.08, w: w - 0.16, h: h - 0.16, sizing: { type: "contain", w: w - 0.16, h: h - 0.16 } });
}
// "잠정" 태그 — 미션 확정 후 수치 교체 대상 표시
function tag(slide, x, y, t = "잠정 · 10/01 기준") {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.7, h: 0.3, fill: { color: C.pink }, rectRadius: 0.08, line: { color: C.pink } });
  txt(slide, t, { x, y, w: 1.7, h: 0.3, fontSize: 10, bold: true, color: C.red, align: "center", valign: "middle" });
}
function insight(slide, x, y, w, h, runs, fill = C.light) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: 0.1, line: { color: fill } });
  txt(slide, "💡", { x: x + 0.15, y, w: 0.4, h, fontSize: 18, valign: "middle" });
  txt(slide, runs, { x: x + 0.6, y, w: w - 0.75, h, fontSize: 13, valign: "middle" });
}
function checkRow(slide, x, y, w, t) {
  slide.addShape(pres.shapes.OVAL, { x, y: y + 0.02, w: 0.32, h: 0.32, fill: { color: C.mint }, line: { color: C.mint } });
  txt(slide, "✓", { x, y: y + 0.02, w: 0.32, h: 0.32, fontSize: 13, bold: true, color: C.white, align: "center", valign: "middle" });
  txt(slide, t, { x: x + 0.45, y, w: w - 0.45, h: 0.36, fontSize: 13, valign: "middle" });
}
function stat(slide, x, y, w, num, label, color = C.blue, size = 40) {
  txt(slide, num, { x, y, w, h: 0.75, fontSize: size, bold: true, color, align: "center", valign: "middle" });
  txt(slide, label, { x, y: y + 0.78, w, h: 0.5, fontSize: 12, color: C.sub, align: "center" });
}
function table(slide, rows, x, y, colW, o = {}) {
  const head = rows[0].map(t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.blue }, align: "center" } }));
  const body = rows.slice(1).map(r => r.map(c => (typeof c === "string" ? { text: c } : c)));
  slide.addTable([head, ...body], Object.assign({
    x, y, colW, fontFace: F, fontSize: 12, color: C.text, valign: "middle", margin: 0.08,
    border: { type: "solid", pt: 1, color: C.line }, rowH: 0.42,
  }, o));
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
  s.background = { color: C.bgGray };
  for (let x = 0; x <= W; x += 0.5) s.addShape(pres.shapes.LINE, { x, y: 0, w: 0, h: 7.5, line: { color: "EEF0F3", width: 0.75 } });
  for (let y = 0; y <= 7.5; y += 0.5) s.addShape(pres.shapes.LINE, { x: 0, y, w: W, h: 0, line: { color: "EEF0F3", width: 0.75 } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: W / 2 - 1.1, y: 0.55, w: 2.2, h: 0.4, fill: { color: C.sky }, rectRadius: 0.08, line: { color: C.sky } });
  txt(s, "대학부 예선", { x: W / 2 - 1.1, y: 0.55, w: 2.2, h: 0.4, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle" });
  txt(s, "2026 데이터+AI 크리에이터 캠프  |  결과 정리", { x: M, y: 1.15, w: CW, h: 0.4, fontSize: 13, color: C.sub, align: "center" });
  txt(s, "전화대화 음성 및 전사 데이터를 이용한", { x: M, y: 2.5, w: CW, h: 0.6, fontSize: 28, bold: true, align: "center" });
  txt(s, "응급상황 인식", { x: M, y: 3.1, w: CW, h: 0.9, fontSize: 44, bold: true, color: C.blue, align: "center" });
  txt(s, ": Mission 1~3 수행 결과 정리", { x: M, y: 4.05, w: CW, h: 0.5, fontSize: 20, color: C.sub, align: "center" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: W / 2 - 3.2, y: 5.0, w: 6.4, h: 0.7, fill: { color: C.white }, rectRadius: 0.1, line: { color: C.line, width: 1 }, shadow: shadow() });
  txt(s, [{ text: "팀 eigensinn", options: { bold: true, color: C.blue } }, { text: "   김장수 · 김광희 · 지연우 · 이수빈 · 조지은", options: { color: C.text } }],
    { x: W / 2 - 3.2, y: 5.0, w: 6.4, h: 0.7, fontSize: 15, align: "center", valign: "middle" });
  txt(s, "제출일 2026. 10. 14", { x: W - M - 4, y: 6.9, w: 4, h: 0.35, fontSize: 12, color: C.sub, align: "right" });
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
    ["02", "데이터 분석", ["데이터 개요와 대회 규칙", "EDA 핵심 발견", "공통 파이프라인"]],
    ["03", "미션별 수행", ["Mission 1  신고자 성별 분류\n   (+ 연구 방법·시행착오)", "Mission 2  신고자/대원 분류\n   (+ 연구 방법·시행착오, 김광희 파트)", "Mission 3  환자 증상 인식\n   (+ 확장 계획)"]],
    ["04", "종합 및 시사점", ["성능 요약과 교훈", "사회안전 시사점과 향후 계획"]],
  ];
  const cw = (CW - 3 * 0.3) / 4;
  secs.forEach(([n, name, items], i) => {
    const x = M + i * (cw + 0.3);
    txt(s, n, { x, y: 2.1, w: cw, h: 1.5, fontSize: 96, bold: true, color: C.bigNum });
    txt(s, name, { x: x + 0.1, y: 3.15, w: cw - 0.1, h: 0.55, fontSize: 22, bold: true });
    s.addShape(pres.shapes.LINE, { x: x + 0.1, y: 3.85, w: cw - 0.3, h: 0, line: { color: C.line, width: 1 } });
    txt(s, items.map((t, k) => ({ text: `${k + 1}. ${t}`, options: { breakLine: k < items.length - 1 } })),
      { x: x + 0.1, y: 4.05, w: cw - 0.1, h: 2.2, fontSize: 14, color: C.sub, paraSpaceAfter: 10 });
  });
}

// =====================================================================
// 5. 01-(1) 사전 학습 로드맵 (step0 저장소)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "01", "준비 과정", "(1) 발대식 전 사전 학습 로드맵", 4);
  coreMsg(s, "처음 보는 데이터로도  QA → Baseline → 검증 → 통제 실험 → 제출  흐름을 끝까지 돌 수 있도록 대회 전에 미리 연습");
  const steps = [
    ["W1  기초", "Python·pandas\n타이타닉 EDA\n(결측치·분포)"],
    ["W2  Baseline", "첫 모델 학습\n실험 로그 작성\n(Titanic 등)"],
    ["W3  데이터 QA", "CIFAR-10\n누락·중복·분포\n점검"],
    ["W4  통제 실험", "baseline 결함\n감사\n변수 하나씩 비교"],
    ["W5  데이터 정제", "이상치 점수\n기준값 사전 고정\n3조건 비교"],
    ["W6  제출 연습", "제출 패키지\n채점 스크립트"],
  ];
  const bw = 1.78, gap = (CW - 6 * bw) / 5;
  steps.forEach(([t, b], i) => {
    const x = M + i * (bw + gap);
    stepBox(s, x, 2.55, bw, 2.1, t, b, i === 0 ? C.sky : C.blue);
    if (i < 5) arrow(s, x + bw + (gap - 0.22) / 2, 3.4, 0.22, 0.35);
  });
  insight(s, M, 5.0, CW, 1.3, [
    { text: "예선에 그대로 가져온 습관  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "데이터 무결성 점검을 모델보다 먼저  ·  실험마다 ID와 로그 남기기  ·  한 번에 변수 하나만 바꿔 비교  ·  기준값은 결과를 보기 전에 고정" },
  ]);
  caption(s, "근거: GitHub eigensinn-dcc2026/step0 (2026.07.14 ~ 08.23), 팀원별 notebooks/ 및 outputs/ 실습 산출물");
  s.addNotes("step0 저장소 README의 W2~W6 로드맵 기준. 필요하면 각 주차 대표 산출물(타이타닉 EDA 그래프 등) 캡처를 작게 추가.");
}

// =====================================================================
// 6. 01-(2) 팀 구성과 협업 체계
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "01", "준비 과정", "(2) 팀 구성과 협업 체계", 5);
  coreMsg(s, "5명이 한 저장소에서 충돌 없이 일하도록  규칙 · 자동 검사 · 기록 체계를 먼저 만들고 미션에 들어감");
  txt(s, "■ 역할 분담", { x: M, y: 2.4, w: 5, h: 0.4, fontSize: 18, bold: true });
  table(s, [
    ["팀원", "준비기 역할", "예선 담당"],
    ["김장수", "이미지 모델링·통합", "Mission 1"],
    ["김광희", "데이터 품질·매핑", "Mission 1 · 2"],
    ["지연우", "검증·오분류 분석", "Mission 2"],
    ["이수빈", "시각화·리포팅", "Mission 3 · 오류분석 취합"],
    ["조지은", "정형 모델링·EDA", "Mission 3"],
  ], M, 2.9, [1.1, 2.0, 2.4]);

  const rx = 6.6, rw = W - M - rx;
  txt(s, "■ GitHub 작업 흐름", { x: rx, y: 2.4, w: rw, h: 0.4, fontSize: 18, bold: true });
  const flow = ["main에서\n브랜치 생성", "실험·코드\n작성", "PR 작성\n(템플릿)", "자동 검사\n(CI·병합 판정)", "팀원 리뷰\n후 병합"];
  const fw = 1.05, fg = (rw - 5 * fw) / 4;
  flow.forEach((t, i) => {
    const x = rx + i * (fw + fg);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.9, w: fw, h: 0.95, fill: { color: i === 3 ? C.blue : C.light }, rectRadius: 0.1, line: { color: i === 3 ? C.blue : C.light } });
    txt(s, t, { x, y: 2.9, w: fw, h: 0.95, fontSize: 11, bold: true, color: i === 3 ? C.white : C.blue, align: "center", valign: "middle" });
    if (i < 4) arrow(s, x + fw + (fg - 0.14) / 2, 3.2, 0.14, 0.3);
  });
  // 기록 체계 2박스
  const bw = (rw - 0.3) / 2;
  [["docs/experiment_log.md", "무엇을 했고 결과는?", "실험 ID · 날짜 · 데이터 규모 · 분할 · seed · 결과"],
   ["docs/decisions.md", "왜 채택·기각했나?", "결정 · 이유 · 후속 조치 (기각한 실험도 기록)"]].forEach(([f, q, d], i) => {
    const x = rx + i * (bw + 0.3);
    card(s, x, 4.1, bw, 1.25);
    txt(s, f, { x: x + 0.15, y: 4.2, w: bw - 0.3, h: 0.3, fontSize: 11, color: C.sub });
    txt(s, q, { x: x + 0.15, y: 4.5, w: bw - 0.3, h: 0.35, fontSize: 14, bold: true, color: C.blue });
    txt(s, d, { x: x + 0.15, y: 4.85, w: bw - 0.3, h: 0.45, fontSize: 11 });
  });
  // 숫자 콜아웃
  const sw = rw / 3;
  [["32+", "Pull Request"], ["40+", "작업 브랜치"], ["M1–M3", "실험 ID 체계"]].forEach(([n, l], i) =>
    stat(s, rx + i * sw, 5.55, sw, n, l, C.blue, 30));
  tag(s, W - M - 1.7, 2.45, "수치 · 10/01 기준");
  caption(s, "근거: GitHub eigensinn-dcc2026/prelim — WORKFLOW.md, .github/workflows (pr-ci, pr-verdict), docs/");
  s.addNotes("PR·브랜치 수는 제출 직전 gh pr list / git branch -r 로 갱신.");
}

// =====================================================================
// 7. 02-(1) 데이터 개요와 대회 규칙
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "02", "데이터 분석", "(1) 데이터 개요와 대회 규칙 준수", 6);
  coreMsg(s, "AI-Hub 119 신고접수 음성·전사 데이터 — 규칙 위반 가능성을 설계 단계에서부터 차단");
  const lw = 5.6;
  card(s, M, 2.45, lw, 4.35);
  txt(s, "■ 데이터셋", { x: M + 0.25, y: 2.6, w: lw - 0.5, h: 0.4, fontSize: 18, bold: true });
  txt(s, "AI-Hub 「위급상황 음성/음향(고도화) — 119 지능형 신고접수 음성 인식 데이터」", { x: M + 0.25, y: 3.05, w: lw - 0.5, h: 0.6, fontSize: 13, color: C.sub });
  [["음성(WAV)", "통화 녹음 원본"], ["전사·라벨(JSON)", "발화 구간·화자·성별·증상 등"]].forEach(([a, b], i) => {
    const x = M + 0.25 + i * ((lw - 0.7) / 2 + 0.2), w = (lw - 0.7) / 2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.75, w, h: 0.85, fill: { color: C.light }, rectRadius: 0.1, line: { color: C.light } });
    txt(s, [{ text: a, options: { bold: true, color: C.blue, breakLine: true } }, { text: b, options: { fontSize: 11 } }], { x: x + 0.1, y: 3.75, w: w - 0.2, h: 0.85, fontSize: 13, align: "center", valign: "middle" });
  });
  const sw = (lw - 0.5) / 3;
  stat(s, M + 0.25, 4.9, sw, "29,200", "학습 통화 수 (서울)", C.blue, 26);
  txt(s, [ph("00,000")], { x: M + 0.25 + sw, y: 4.9, w: sw, h: 0.75, fontSize: 26, bold: true, align: "center", valign: "middle" });
  txt(s, "학습 발화 수", { x: M + 0.25 + sw, y: 5.68, w: sw, h: 0.5, fontSize: 12, color: C.sub, align: "center" });
  txt(s, [ph("0,000")], { x: M + 0.25 + 2 * sw, y: 4.9, w: sw, h: 0.75, fontSize: 26, bold: true, align: "center", valign: "middle" });
  txt(s, "Validation 통화 수", { x: M + 0.25 + 2 * sw, y: 5.68, w: sw, h: 0.5, fontSize: 12, color: C.sub, align: "center" });

  const rx = M + lw + 0.4, rw = W - M - rx;
  card(s, rx, 2.45, rw, 4.35);
  txt(s, "■ 우리가 지킨 대회 규칙", { x: rx + 0.25, y: 2.6, w: rw - 0.5, h: 0.4, fontSize: 18, bold: true });
  [
    "학습에는 서울 Training 데이터만 사용",
    "Validation은 모델 선택·평가에만 사용 (학습 투입 없음)",
    "미션별 지정 입력만 사용 — M2는 발화 조각 하나만",
    "라벨에 따라 다른 전처리, 평가 데이터 전용 전처리 없음",
    "최종 판정 기준값 0.5 고정 (클래스별 조정 없음)",
    "상용 AI API 미사용, 공개 오픈소스 라이브러리만 사용",
    "학습·검증 분할은 통화 단위 — 같은 통화가 양쪽에 섞이지 않게",
  ].forEach((t, i) => checkRow(s, rx + 0.3, 3.15 + i * 0.5, rw - 0.6, t));
  caption(s, "근거: 대학부 예선 출제문제 5·8쪽 조건, 주최측 Q&A 답변(2026.09.15, 기준값 안내 메일)");
  s.addNotes("통계 수치는 build_manifest.py / 데이터 감사 보고서(reports/) 기준으로 채우기.");
}

// =====================================================================
// 8. 02-(2) EDA 핵심 발견 (3단 그리드)
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "02", "데이터 분석", "(2) EDA 핵심 발견", 7);
  coreMsg(s, "세 미션 모두 단일 변수로는 못 가른다 — 성별만 균형 잡혀 있고, 길이·증상 빈도는 모델이 함께 보정해야 할 불균형");
  const cards = [
    ["figures/speaker_duration_buckets.png", "화자별 발화 길이", [
      { text: "발견  ", options: { bold: true, breakLine: true } },
      { text: "신고자가 긴 발화(≥4초)에서 대원보다 비중이 높지만(14.0% vs 7.9%), 짧은 구간은 서로 겹침", options: { breakLine: true } },
      { text: "설계 반영  ", options: { bold: true, color: C.blue, breakLine: true } },
      { text: "길이만으로는 못 가름 → 음향 특징(M2 L1)과 전체 소리 모양(M2 L2)까지 함께 사용" },
    ]],
    ["figures/gender_distribution.png", "성별 구성 (Mission 1)", [
      { text: "발견  ", options: { bold: true, breakLine: true } },
      { text: "Training 53.2:46.8, Validation 53.8:46.2 — 분할 간 차이 1%p 이내로 균형", options: { breakLine: true } },
      { text: "설계 반영  ", options: { bold: true, color: C.blue, breakLine: true } },
      { text: "클래스 불균형 보정이 불필요 → 기본 로지스틱 회귀·LightGBM 그대로 사용" },
    ]],
    ["figures/symptom_prevalence.png", "증상 9종 유병률 (Mission 3)", [
      { text: "발견  ", options: { bold: true, breakLine: true } },
      { text: "복통·어지러움 20%대 vs 두통·오심 10%대 — 최대 2배 이상 차이", options: { breakLine: true } },
      { text: "설계 반영  ", options: { bold: true, color: C.blue, breakLine: true } },
      { text: "Macro F1은 희소 증상에 민감 → 증상별 성능을 따로 점검 (28p)" },
    ]],
  ];
  const cw = (CW - 2 * 0.3) / 3;
  cards.forEach(([path, t, runs], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.45, cw, 4.35);
    txt(s, t, { x: x + 0.2, y: 2.58, w: cw - 0.4, h: 0.4, fontSize: 15, bold: true, color: C.blue });
    fig(s, path, x + 0.2, 3.02, cw - 0.4, 2.15);
    txt(s, runs, { x: x + 0.2, y: 5.3, w: cw - 0.4, h: 1.4, fontSize: 11.5, paraSpaceAfter: 5 });
  });
  caption(s, "근거: reports/eda/eda_report.md, reports/eda/figures/ (PR #23, 2026.09.21)");
  s.addNotes("실제 EDA 산출 그래프(reports/eda/figures). 세 카드 모두 '발견 → 설계 반영' 한 쌍 — 심사위원은 말없이 읽으므로 그래프 제목만 보고도 결론이 보이게.");
}

// =====================================================================
// 9. 02-(3) 공통 파이프라인
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "02", "데이터 분석", "(3) 세 미션 공통 파이프라인", 8);
  coreMsg(s, "데이터 무결성 검증부터 inference.py 1회 실행까지 — 세 미션이 같은 뼈대 위에서 동작");
  const steps = [
    ["원본 데이터", "WAV 음성\nJSON 전사·라벨"],
    ["무결성 검증", "WAV↔JSON 매칭\nmanifest 생성\n누락·중복 점검"],
    ["미션별 데이터", "음성: 발화 구간 추출\n텍스트: 전사 정리"],
    ["학습 · 평가", "통화 단위 분할\n공식 Validation 평가"],
    ["기록 · 제출", "실험 로그 JSON\ninference.py\n1회 실행 → CSV"],
  ];
  const bw = 2.1, gap = (CW - 5 * bw) / 4;
  steps.forEach(([t, b], i) => {
    const x = M + i * (bw + gap);
    stepBox(s, x, 2.5, bw, 2.0, t, b);
    if (i < 4) arrow(s, x + bw + (gap - 0.28) / 2, 3.3);
  });
  // 음성/텍스트 두 경로
  const lx = M + (bw + gap) * 2, lw = bw * 2 + gap;
  [["음성 경로", "M1 · M2  —  음향 특징 · 스펙트로그램", C.sky], ["텍스트 경로", "M3  —  전사 텍스트", C.mint]].forEach(([a, b, col], i) => {
    const y = 4.75 + i * 0.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y, w: lw, h: 0.45, fill: { color: col }, rectRadius: 0.08, line: { color: col } });
    txt(s, [{ text: `${a}   `, options: { bold: true } }, { text: b }], { x: lx + 0.15, y, w: lw - 0.3, h: 0.45, fontSize: 12, color: C.white, valign: "middle" });
  });
  txt(s, "■ 재현성 장치", { x: M, y: 5.95, w: 3, h: 0.4, fontSize: 16, bold: true });
  ["seed = 42 고정", "통화 단위 분할 (call_id · recordId)", "자동 테스트 (pytest · CI)", "결과는 JSON 로그로 저장"].forEach((t, i) => {
    const w = (CW - 3 * 0.2) / 4, x = M + i * (w + 0.2);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 6.4, w, h: 0.45, fill: { color: C.bgGray }, rectRadius: 0.08, line: { color: C.line, width: 1 } });
    txt(s, t, { x, y: 6.4, w, h: 0.45, fontSize: 12, align: "center", valign: "middle" });
  });
  s.addNotes("src/dcc_data 공통 모듈, scripts/ 진입점 기준. 제출 폴더 구조(미션별 3폴더)를 여기 작게 넣어도 좋음.");
}

// =====================================================================
// 미션 공통 틀: A(문제 정의·접근) / B(결과·시사점)
// =====================================================================
function missionA(page, no, title, core, prob, pipe, why) {
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", `Mission ${no}  ${title} — 문제 정의와 접근`, page);
  coreMsg(s, core);
  const lw = 4.1;
  card(s, M, 2.45, lw, 4.35);
  txt(s, "■ 문제 정의", { x: M + 0.25, y: 2.6, w: lw - 0.5, h: 0.4, fontSize: 18, bold: true });
  prob.forEach(([k, v], i) => {
    const y = 3.15 + i * 0.9;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 0.25, y, w: 0.95, h: 0.36, fill: { color: C.blue }, rectRadius: 0.08, line: { color: C.blue } });
    txt(s, k, { x: M + 0.25, y, w: 0.95, h: 0.36, fontSize: 12, bold: true, color: C.white, align: "center", valign: "middle" });
    txt(s, v, { x: M + 1.35, y: y - 0.02, w: lw - 1.55, h: 0.8, fontSize: 12 });
  });
  const rx = M + lw + 0.4, rw = W - M - rx;
  txt(s, "■ 모델 구성", { x: rx, y: 2.45, w: rw, h: 0.4, fontSize: 18, bold: true });
  const n = pipe.length, bw = 1.72, gap = (rw - n * bw) / (n - 1);
  pipe.forEach(([t, b], i) => {
    const x = rx + i * (bw + gap);
    stepBox(s, x, 2.95, bw, 1.75, t, b, i === n - 1 ? C.sky : C.blue);
    if (i < n - 1) arrow(s, x + bw + (gap - 0.22) / 2, 3.6, 0.22, 0.35);
  });
  txt(s, "■ 왜 이 방법인가", { x: rx, y: 4.95, w: rw, h: 0.4, fontSize: 18, bold: true });
  // why 항목: 문자열 또는 run 배열(자리표시자 포함) — 항목마다 글머리표 한 줄
  const runs = [];
  why.forEach((item, i) => {
    const parts = Array.isArray(item) ? item : [{ text: item }];
    parts.forEach((r, k) => {
      const o = Object.assign({}, r.options);
      if (k === 0) o.bullet = true;
      if (k === parts.length - 1 && i < why.length - 1) o.breakLine = true;
      runs.push({ text: r.text, options: o });
    });
  });
  txt(s, runs, { x: rx, y: 5.4, w: rw, h: 1.4, fontSize: 13, paraSpaceAfter: 6 });
  return s;
}
// showTag: 수치가 아직 잠정이면 true(기본), 확정되어 더 안 바뀌면 false로 꺼서 "잠정" 태그 제거
function missionB(page, no, title, core, showTag = true) {
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", `Mission ${no}  ${title} — 결과와 시사점`, page);
  coreMsg(s, core);
  if (showTag) tag(s, W - M - 1.7, 0.95);
  return s;
}
// 미션 1페이지 요약 틀: 미션 정의(한 줄) + 모델 구성(파이프라인) + 핵심 결과(정확도) — 심사위원이
// 이 한 장만 봐도 결론이 보이게. probLine은 "입력 … · 출력 … · 평가 … · 제약 …" 형태의 한 문장.
function missionSummary(page, no, title, core, probLine, pipe, headline, showTag = true) {
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", `Mission ${no}  ${title} — 한눈에 보기`, page);
  coreMsg(s, core);
  if (showTag) tag(s, W - M - 1.7, 0.95);
  txt(s, "■ 미션 정의", { x: M, y: 2.45, w: CW, h: 0.3, fontSize: 16, bold: true, color: C.blue });
  txt(s, probLine, { x: M, y: 2.78, w: CW, h: 0.5, fontSize: 13 });
  s.addShape(pres.shapes.LINE, { x: M, y: 3.32, w: CW, h: 0, line: { color: C.line, width: 1 } });
  txt(s, "■ 모델 구성", { x: M, y: 3.42, w: CW, h: 0.3, fontSize: 16, bold: true, color: C.blue });
  const n = pipe.length, bw = 2.0, gap = (CW - n * bw) / (n - 1);
  pipe.forEach(([t, b], i) => {
    const x = M + i * (bw + gap);
    stepBox(s, x, 3.78, bw, 1.4, t, b, i === n - 1 ? C.sky : C.blue);
    if (i < n - 1) arrow(s, x + bw + (gap - 0.24) / 2, 4.35, 0.24, 0.38);
  });
  s.addShape(pres.shapes.LINE, { x: M, y: 5.3, w: CW, h: 0, line: { color: C.line, width: 1 } });
  txt(s, "■ 핵심 결과", { x: M, y: 5.4, w: CW, h: 0.3, fontSize: 16, bold: true, color: C.blue });
  txt(s, headline.value, { x: M, y: 5.68, w: CW, h: 0.72, fontSize: 42, bold: true, color: C.blue, align: "center" });
  txt(s, headline.label, { x: M, y: 6.4, w: CW, h: 0.3, fontSize: 13, align: "center" });
  txt(s, headline.sub, { x: M, y: 6.68, w: CW, h: 0.3, fontSize: 11, color: C.sub, align: "center" });
  return s;
}
// 미션 결론 틀: 데이터 자체의 어려움(정보가 부족해 남도 못 맞추는 영역) vs 모델 성능의 개선 여지
// (더 잘 맞출 여지가 있는 영역)를 구분 + 사회안전 시사점 + 향후 계획(미착수 포함)
function missionConclusion(page, no, title, core, dataLimit, modelLimit, safety, future, showTag = true) {
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", `Mission ${no}  ${title} — 결론과 다음 계획`, page);
  coreMsg(s, core);
  if (showTag) tag(s, W - M - 1.7, 0.95);
  const hw = (CW - 0.3) / 2;
  card(s, M, 2.45, hw, 2.45);
  txt(s, "■ 데이터 자체의 어려움", { x: M + 0.2, y: 2.58, w: hw - 0.4, h: 0.32, fontSize: 14, bold: true, color: C.blue });
  txt(s, "정보가 부족해 남도 못 맞추는 영역", { x: M + 0.2, y: 2.9, w: hw - 0.4, h: 0.26, fontSize: 10, color: C.sub });
  txt(s, dataLimit, { x: M + 0.2, y: 3.2, w: hw - 0.4, h: 1.6, fontSize: 11, paraSpaceAfter: 6 });
  const rx = M + hw + 0.3;
  card(s, rx, 2.45, hw, 2.45);
  txt(s, "■ 모델 성능의 개선 여지", { x: rx + 0.2, y: 2.58, w: hw - 0.4, h: 0.32, fontSize: 14, bold: true, color: C.blue });
  txt(s, "더 잘 맞출 여지가 있는 영역", { x: rx + 0.2, y: 2.9, w: hw - 0.4, h: 0.26, fontSize: 10, color: C.sub });
  txt(s, modelLimit, { x: rx + 0.2, y: 3.2, w: hw - 0.4, h: 1.6, fontSize: 11, paraSpaceAfter: 6 });
  insight(s, M, 5.1, hw, 1.8, [{ text: "사회안전 시사점  ", options: { bold: true, color: C.blue, breakLine: true } }].concat(safety), C.mintLight);
  card(s, rx, 5.1, hw, 1.8, C.bgGray);
  txt(s, "■ 향후 계획 (제출 기한 내 미착수)", { x: rx + 0.2, y: 5.22, w: hw - 0.4, h: 0.3, fontSize: 13, bold: true, color: C.blue });
  txt(s, future, { x: rx + 0.2, y: 5.55, w: hw - 0.4, h: 1.3, fontSize: 11, paraSpaceAfter: 5 });
  return s;
}
// 오른쪽 열: 결과 분석 / 재현 정보 / 사회안전 시사점
// repro: 재현 정보 줄에 넣을 run 배열. 생략하면 공통 자리표시자 사용.
function rightCol(s, rx, analysis, safety, repro) {
  const rw = W - M - rx;
  card(s, rx, 2.45, rw, 1.75);
  txt(s, "■ 결과 분석", { x: rx + 0.2, y: 2.55, w: rw - 0.4, h: 0.35, fontSize: 15, bold: true });
  txt(s, analysis, { x: rx + 0.2, y: 2.95, w: rw - 0.4, h: 1.2, fontSize: 12, paraSpaceAfter: 4 });
  card(s, rx, 4.35, rw, 1.05, C.bgGray);
  txt(s, "■ 재현 정보", { x: rx + 0.2, y: 4.43, w: rw - 0.4, h: 0.3, fontSize: 13, bold: true });
  txt(s, repro || [{ text: "파라미터 수 " }, ph("0"), { text: "  ·  학습/추론 환경 " }, ph("GPU/CPU", { breakLine: true }),
    { text: "추론 배치 " }, ph("0"), { text: "  ·  Validation 추론 시간 " }, ph("0.0초/샘플")],
    { x: rx + 0.2, y: 4.75, w: rw - 0.4, h: 0.6, fontSize: 11, color: C.sub });
  insight(s, rx, 5.55, rw, 1.25, [{ text: "사회안전 시사점  ", options: { bold: true, color: C.blue, breakLine: true } }].concat(safety), C.mintLight);
}

// ======================================================================
// Mission 1 — 신고자 성별 분류  (①요약 ②③④방법·시행착오 ⑤최종결과 ⑥결론, 6장)
// ======================================================================

// ① 요약: 미션 정의 + 모델 구성 + 핵심 결과를 한 장에
missionSummary(9, 1, "신고자 성별 분류",
  "신고자 발화 음성 → 52차원 음향 특징 → 로지스틱 회귀 → 공식 Validation Accuracy 95.11%",
  "신고 통화에서 신고자의 발화만 듣고, 성별이 남성인지 여성인지 맞힌다.",
  [["구간 추출", "신고자 발화만\n잘라내기"], ["음향 특징", "음높이·음색 등\n52차원 통계"], ["분류 모델", "로지스틱 회귀"], ["판정", "남/여\n(기준값 0.5)"]],
  { value: "95.11%", label: "Accuracy (공식 Validation, 3,640건)", sub: "LightGBM 94.92% 대비 +0.19%p — 더 단순하고 해석 쉬운 로지스틱 회귀 채택 (2026-10-05 고정)" },
  false);

// ② 연구 방법 1 — 음향 특징 구성
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1 — 연구 방법 ① 음향 특징 구성", 10);
  coreMsg(s, "목소리를 52차원 숫자로 압축 — 어떤 정보를 담았는지");
  card(s, M, 2.45, CW, 4.35);
  txt(s, "■ 52차원 음향 특징 구성", { x: M + 0.3, y: 2.65, w: CW - 0.6, h: 0.35, fontSize: 14, bold: true, color: C.blue });
  table(s, [
    ["구성 요소", "차원", "의미"],
    ["MFCC 평균·표준편차", "40", "음색의 핵심 패턴"],
    ["RMS 평균·표준편차", "2", "음량"],
    ["ZCR 평균·표준편차", "2", "음높이 변화"],
    ["Spectral 4종×2", "8", "음색의 \"밝기\""],
  ], M + 0.3, 3.05, [3.0, 1.3, 4.3], { rowH: 0.52, fontSize: 12.5, align: "center" });
  txt(s, [
    { text: "합계 52차원", options: { bold: true, breakLine: true } },
    { text: "duration(발화 길이)는 의도적으로 제외 — 길이로 성별을 \"눈치채는\" 손쉬운 지름길 학습 방지" },
  ], { x: M + 0.3, y: 6.1, w: CW - 0.6, h: 0.7, fontSize: 12, paraSpaceAfter: 6 });
  caption(s, "근거: reports/mission1/m1_exp_log.json \"고정 변수\"");
}

// ③ 연구 방법 2 (시행착오 A) — 모델 비교와 과적합 점검
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1 — 연구 방법 ② 시행착오: 어떤 모델을 왜 골랐나", 11);
  coreMsg(s, "LightGBM과 로지스틱 회귀 차이는 0.19%p — 더 복잡한 모델이 꼭 낫지는 않았다");
  card(s, M, 2.45, 7.3, 4.35);
  txt(s, "로지스틱 회귀 혼동행렬 (오분류 178건)", { x: M + 0.2, y: 2.6, w: 2.6, h: 0.26, fontSize: 10.5, color: C.sub });
  table(s, [
    ["", "예측 F", "예측 M"],
    ["실제 F", { text: "1,878", options: { bold: true, color: C.blue } }, "81"],
    ["실제 M", "97", { text: "1,584", options: { bold: true, color: C.blue } }],
  ], M + 0.2, 2.88, [0.75, 0.925, 0.925], { rowH: 0.36, fontSize: 11, align: "center" });
  txt(s, "95% 신뢰구간\n[94.40, 95.82]", { x: M + 0.2, y: 4.2, w: 2.6, h: 0.65, fontSize: 12, color: C.sub, align: "center" });
  txt(s, "과적합 점검: 내부 94.8~95.0% vs 공식 95.1% — 차이 작아 뚜렷한 leakage 징후 없음(증명은 아님)",
    { x: M + 0.2, y: 4.95, w: 2.6, h: 1.6, fontSize: 10.5 });
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["LightGBM", "로지스틱 회귀\n(채택)"], values: [94.92, 95.11] }], {
    x: M + 2.9, y: 2.6, w: 4.2, h: 4.0, barDir: "col", chartColors: [C.sky, C.blue],
    showTitle: true, title: "모델별 Accuracy (%)", titleFontSize: 12, titleColor: C.text, titleFontFace: F,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisMinVal: 90, valAxisMaxVal: 97, valAxisLabelColor: C.sub, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 12,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, varyColors: true,
  });
  card(s, M + 7.6, 2.45, W - M - (M + 7.6), 4.35);
  txt(s, "■ 왜 로지스틱 회귀를 골랐나", { x: M + 7.8, y: 2.58, w: 3.7, h: 0.35, fontSize: 14, bold: true, color: C.blue });
  txt(s, [
    { text: "정확도 차이 0.19%p (오분류 7건 차이)로 작음", options: { breakLine: true } },
    { text: "로지스틱 회귀가 더 단순하고 해석하기 쉬움 — 계수로 어떤 특징이 영향 줬는지 바로 확인 가능", options: { breakLine: true } },
    { text: "recall_F 95.87% / recall_M 94.23% 모두 93~96%대로 균형 — 다수 클래스(F)만 찍는 편향 없음", options: { breakLine: true } },
    { text: "LightGBM은 폐기가 아니라 향후 feature 확장 실험용 비교 baseline으로 유지" },
  ], { x: M + 7.8, y: 2.95, w: 3.7, h: 3.7, fontSize: 11, paraSpaceAfter: 8 });
  caption(s, "근거: reports/mission1/m1_val_eval.json, m1_val_bootstrap_ci.json, M1_error_analysis_report.md 1·2절");
}

// ④ 연구 방법 3 (시행착오 B) — 오류 원인 탐색
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1 — 연구 방법 ③ 시행착오: 오류 원인을 찾아본 과정", 12);
  coreMsg(s, "길이·발화 수·겹침, 52차원 feature까지 들여다봤지만 뚜렷한 단일 원인은 못 찾음");
  const hw = (CW - 0.3) / 2;
  card(s, M, 2.45, hw, 4.35);
  txt(s, "■ 발화 길이 · 조각 수 · 겹침 비율", { x: M + 0.2, y: 2.58, w: hw - 0.4, h: 0.35, fontSize: 14, bold: true, color: C.blue });
  txt(s, "오분류 178건을 분위(quintile) 5구간으로 나눠 구간별 오류율 비교", { x: M + 0.2, y: 2.92, w: hw - 0.4, h: 0.3, fontSize: 10.5, color: C.sub });
  table(s, [
    ["분석 축", "Q1 오류율", "Q5 오류율", "차이"],
    ["발화 길이(초)", "6.04%\n(최단)", "4.40%\n(최장)", "1.65%p"],
    ["발화 조각 수", "4.12%\n(최소)", "5.88%\n(최대)", { text: "−1.77%p\n(방향 반대)", options: { color: C.red } }],
    ["겹침 비율", "5.08%\n(최소)", "5.49%\n(최대)", "0.41%p"],
  ], M + 0.2, 3.3, [1.5, 1.2, 1.2, 1.4], { rowH: 0.56, fontSize: 10.5, align: "center" });
  txt(s, [
    { text: "짧고 발화 수도 적은 조합(n=1,506) 오류율 4.38% — 전체 평균(4.89%)보다 오히려 낮음", options: { breakLine: true } },
    { text: "겹침 0구간(n=137) 오류율 2.92%는 표본이 작아 참고용 (순위 판단에 쓰지 않음)", options: { color: C.sub } },
  ], { x: M + 0.2, y: 5.65, w: hw - 0.4, h: 0.95, fontSize: 10.5, paraSpaceAfter: 4 });
  const rx2 = M + hw + 0.3;
  card(s, rx2, 2.45, hw, 4.35);
  txt(s, "■ 52차원 feature 탐색 (Cohen's d)", { x: rx2 + 0.2, y: 2.58, w: hw - 0.4, h: 0.35, fontSize: 14, bold: true, color: C.blue });
  txt(s, "로지스틱 회귀 계수 절댓값 상위 15개 안에 피치 직결 지표(zcr_mean·spectral_centroid_mean·mfcc_mean_0~3)는 없음", { x: rx2 + 0.2, y: 2.92, w: hw - 0.4, h: 0.45, fontSize: 10.5, color: C.sub });
  table(s, [
    ["오류 유형", "상위 축 (|d|>1)", "d"],
    ["F→M (n=81)", "mfcc_std_19 / 17 / 16", "−1.24 ~ −1.10"],
    ["M→F (n=97)", "mfcc_std_17 / 16 / 19 외 2개", "+1.34 ~ +1.83"],
  ], rx2 + 0.2, 3.5, [1.5, 2.0, 1.8], { rowH: 0.5, fontSize: 10.5, align: "center" });
  txt(s, "간접 spectral 지표 6개 중 반대 성별 평균에 더 가까운 개수: F 오류 2/6, M 오류 3/6 — 과반에 못 미치거나 절반 수준",
    { x: rx2 + 0.2, y: 4.95, w: hw - 0.4, h: 0.6, fontSize: 10.5 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx2, y: 5.65, w: hw, h: 0.95, fill: { color: C.pink }, rectRadius: 0.1, line: { color: C.pink } });
  txt(s, [
    { text: "⚠ 사후 탐색(post-hoc) 결과 — ", options: { bold: true, color: C.red } },
    { text: "52개 feature를 전부 탐색한 뒤 값이 큰 축만 추려 보여준 것이라 다중비교 편향 가능. 오분류 표본도 81·97건으로 작아 \"이게 원인이다\"라고 확정하지 않음.", options: {} },
  ], { x: rx2 + 0.2, y: 5.65, w: hw - 0.4, h: 0.95, fontSize: 10.5, valign: "middle" });
  caption(s, "근거: M1_error_analysis_report.md 3·4·5절, m1_val_error_analysis.json, m1_val_error_rows.parquet");
  s.addNotes("서면 심사라 '다른 건 안 해봤나' 질문에 답할 기회가 없음 — 분석 과정 자체를 슬라이드에 그대로 옮김.");
}

// ⑤ 최종 연구 결과
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1 — 최종 연구 결과", 13);
  coreMsg(s, "공식 Validation 3,640건 Accuracy 95.11%, 95% CI [94.40, 95.82] — 로지스틱 회귀를 제출 후보로 확정");
  card(s, M, 2.45, CW, 4.35);
  stat(s, M + 0.3, 2.75, 2.6, "95.11%", "Accuracy (공식 Validation)", C.blue, 42);
  txt(s, "95% 신뢰구간 [94.40, 95.82]\nrecall F 95.87% · recall M 94.23%", { x: M + 0.3, y: 4.2, w: 2.6, h: 0.8, fontSize: 12, color: C.sub, align: "center" });
  txt(s, "파라미터 53개(가중치 52+절편) · CPU 추론 · Validation 3,640건 일괄 처리", { x: M + 0.3, y: 5.9, w: 2.6, h: 0.6, fontSize: 9.5, color: C.sub, align: "center" });
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["LightGBM", "로지스틱 회귀\n(채택)"], values: [94.92, 95.11] }], {
    x: M + 3.2, y: 2.6, w: 4.0, h: 3.9, barDir: "col", chartColors: [C.sky, C.blue],
    showTitle: true, title: "모델별 Accuracy (%)", titleFontSize: 12, titleColor: C.text, titleFontFace: F,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisMinVal: 90, valAxisMaxVal: 97, valAxisLabelColor: C.sub, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 12,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, varyColors: true,
  });
  txt(s, "로지스틱 회귀 혼동행렬 (오분류 178건)", { x: M + 7.7, y: 2.6, w: 2.9, h: 0.26, fontSize: 11, color: C.sub });
  table(s, [
    ["", "예측 F", "예측 M"],
    ["실제 F", { text: "1,878", options: { bold: true, color: C.blue } }, "81"],
    ["실제 M", "97", { text: "1,584", options: { bold: true, color: C.blue } }],
  ], M + 7.7, 2.9, [0.85, 1.0, 1.0], { rowH: 0.42, fontSize: 11.5, align: "center" });
  caption(s, "근거: reports/mission1/ m1_val_eval.json, m1_val_bootstrap_ci.json, docs/decisions.md 2026-10-05");
  s.addNotes("제출 상태(2026-10-07 기준): scripts/mission1_inference.py 구현 완료, CSV 스모크 테스트(헤더·행 수·UTF-8)는 아직 미확인 — m1_submission_checklist.md 참고.");
}

// ⑥ 결론 — 데이터 쪽 어려움 vs 모델 쪽 개선 여지 + 사회안전 시사점 + 향후 계획
missionConclusion(14, 1, "신고자 성별 분류",
  "관찰(사실)과 해석(판단)을 구분해서 기록 — 추가 검증 전까지는 확정 결론을 내리지 않음",
  [
    { text: "관찰 가능한 조건(길이·발화 수·겹침)으로는 오류 패턴이 설명되지 않음 — 분위구간 차이 최대 1.8%p, 방향도 엇갈림", options: { breakLine: true } },
    { text: "성별이 애매하게 들리는 통화 자체가 일부 있을 가능성(탐색적 추정, 확정 아님)", options: { breakLine: true } },
    { text: "오분류 표본이 작아(81·97건) 어떤 조건에서 특히 어려운지 더 좁히기 어려움" },
  ],
  [
    { text: "52차원 통계에는 피치(F0, 기본주파수)가 직접 들어있지 않음", options: { breakLine: true } },
    { text: "Cohen's d 탐색에서 일부 고차 MFCC에 패턴이 보였으나 사후 탐색이라 확정 아님 — F0를 직접 추가하면 개선 여지(후속 가설)", options: { breakLine: true } },
    { text: "LightGBM·로지스틱 회귀를 짝지어 검정(McNemar)하지 않아 두 모델 차이의 통계적 유의성은 미확인" },
  ],
  [{ text: "성별 자동 인식으로 신고 접수 기록을 자동화 — 상담 인력 배치 참고 정보로 활용 가능" }],
  [
    { text: "F0(기본 주파수) 추가 — 상대적 우선순위만 상승, 확정 아님", options: { breakLine: true } },
    { text: "LightGBM·로지스틱 회귀 짝지어 검정 (McNemar / paired bootstrap)", options: { breakLine: true } },
    { text: "오분류 화자 단위 정성 분석 — 소수 화자가 오류를 몰아서 만드는지 확인" },
  ],
  false);

// ======================================================================
// Mission 2 — 신고자/119대원 분류 (김광희 담당분: L1 + L2 CNN, 14장)
// 연우 담당분(MLP 트랙)은 마무리되지 않아 이번 판에는 제외 — 비교가 아니라
// "각자의 방법을 소개"하는 구성이므로, 연우 파트 완료 시 별도로 추가한다.
// 내용 출처: ppt 만들기/미션2_PPT_제작의뢰_초안.md (2026-10-08)
// ======================================================================

// ① 한눈에 보기 — 두괄식: 문제 정의 없이 최종 모델·정확도부터 제시
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 (김광희) — 한눈에 보기", 15);
  coreMsg(s, "발화 하나만 듣고 대원인지 신고자인지 — 목소리를 그림(스펙트로그램)으로 바꿔 CNN 3개로 판정");
  txt(s, "■ 최종 모델과 정확도", { x: M, y: 2.45, w: CW, h: 0.3, fontSize: 16, bold: true, color: C.blue });
  txt(s, "91.49%", { x: M, y: 2.75, w: 5.2, h: 1.0, fontSize: 56, bold: true, color: C.blue, align: "center" });
  txt(s, "Accuracy (공식 Validation, 발화 111,947개)", { x: M, y: 3.75, w: 5.2, h: 0.32, fontSize: 12.5, color: C.sub, align: "center" });
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["L1\nLightGBM", "L2 CNN\n3-seed 앙상블"], values: [86.44, 91.49] }], {
    x: M + 5.5, y: 2.5, w: CW - 5.5, h: 1.7, barDir: "bar", chartColors: [C.line, C.blue], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisMinVal: 80, valAxisMaxVal: 95, valAxisHidden: true, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 11.5,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  });
  txt(s, "L1 대비 +5.05%p — CNN 3-seed 앙상블 채택", { x: M + 5.5, y: 4.25, w: CW - 5.5, h: 0.3, fontSize: 11, color: C.sub });
  s.addShape(pres.shapes.LINE, { x: M, y: 4.7, w: CW, h: 0, line: { color: C.line, width: 1 } });
  txt(s, "■ 모델 구성", { x: M, y: 4.82, w: CW, h: 0.3, fontSize: 16, bold: true, color: C.blue });
  const pipe15 = [["발화 추출", "해당 구간만\n잘라내기"], ["스펙트로그램", "log-mel\n그림으로 변환"], ["CNN × 3", "seed 다르게\n학습 후 평균"], ["판정", "평균 확률\n(기준값 0.5)"]];
  const n15 = pipe15.length, bw15 = 2.0, gap15 = (CW - n15 * bw15) / (n15 - 1);
  pipe15.forEach(([t, b], i) => {
    const x = M + i * (bw15 + gap15);
    stepBox(s, x, 5.15, bw15, 1.75, t, b, i === n15 - 1 ? C.sky : C.blue);
    if (i < n15 - 1) arrow(s, x + bw15 + (gap15 - 0.24) / 2, 5.85, 0.24, 0.38);
  });
}

// ③ §2 데이터
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 데이터", 16);
  coreMsg(s, "서울 지역 신고 통화 약 3만 건, 발화 100만 개 — 절반 이상이 1.5초 이하의 짧은 발화");
  card(s, M, 2.45, 6.4, 4.35);
  table(s, [
    ["", "학습용(Training, 서울)", "공식 Validation"],
    ["통화", "29,200", "3,640"],
    ["발화", "889,358", "111,947"],
    ["대원 발화", "427,161", "53,845"],
    ["신고자 발화", "462,197", "58,102"],
  ], M + 0.2, 2.65, [1.7, 2.3, 2.1], { rowH: 0.42, fontSize: 11.5, align: "center" });
  txt(s, [
    { text: "음성 형식  ", options: { bold: true } }, { text: "8kHz, 모노", options: { breakLine: true } },
    { text: "클래스 비율  ", options: { bold: true } }, { text: "거의 반반 (신고자 52%)", options: { breakLine: true } },
    { text: "발화 길이  ", options: { bold: true } }, { text: "평균 1.98초, 중간값 1.44초 — \"네\",\"예\" 같은 짧은 맞장구가 많음. 이게 문제를 어렵게 만드는 핵심" },
  ], { x: M + 0.2, y: 5.0, w: 6.0, h: 1.7, fontSize: 11.5, paraSpaceAfter: 6 });
  const rx = M + 6.6, rw = W - M - rx;
  card(s, rx, 2.45, rw, 4.35);
  txt(s, "발화 길이 구간별 분포 (공식 Validation)", { x: rx + 0.2, y: 2.6, w: rw - 0.4, h: 0.3, fontSize: 11.5, color: C.sub });
  s.addChart(pres.charts.BAR, [{ name: "발화 수", labels: ["0.5초\n미만", "0.5~1초", "1~2초", "2초\n이상"], values: [8954, 27392, 36858, 38743] }], {
    x: rx + 0.1, y: 2.95, w: rw - 0.2, h: 3.6, barDir: "col", chartColors: [C.sky],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10.5, dataLabelFontBold: true, dataLabelFormatCode: "#,##0",
    catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 10.5, valAxisHidden: true,
    catGridLine: { style: "none" }, valGridLine: { style: "none" }, showLegend: false,
  });
  caption(s, "근거: 미션2_PPT_제작의뢰_초안.md 2절, reports/mission2/m2_official_val_spec.md");
}

// ④ §3 첫 번째 접근 — 음향 통계 + LightGBM (L1)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 연구 방법 ① 첫 번째 접근: 음향 통계 + LightGBM", 17);
  coreMsg(s, "발화마다 소리 특징을 숫자 53개로 요약해 분류 → 공식 Validation 86.44%");
  card(s, M, 2.45, CW, 4.35);
  txt(s, "■ 방법", { x: M + 0.3, y: 2.65, w: CW - 0.6, h: 0.32, fontSize: 14, bold: true, color: C.blue });
  txt(s, [
    { text: "특징  ", options: { bold: true } }, { text: "소리 높낮이·세기·음색(MFCC) 등 값의 평균과 분산 등 53개", options: { breakLine: true } },
    { text: "모델  ", options: { bold: true } }, { text: "LightGBM (트리 기반 모델)" },
  ], { x: M + 0.3, y: 3.0, w: CW - 0.6, h: 0.9, fontSize: 13, paraSpaceAfter: 5 });
  s.addShape(pres.shapes.LINE, { x: M + 0.3, y: 4.05, w: CW - 0.6, h: 0, line: { color: C.line, width: 1 } });
  stat(s, M + 0.5, 4.3, 3.3, "86.44%", "공식 Validation Accuracy", C.blue, 42);
  txt(s, "■ 추가로 해 본 것 — 전부 0%p 근처, 큰 개선 없음", { x: M + 4.3, y: 4.3, w: CW - 4.6, h: 0.3, fontSize: 13, bold: true, color: C.blue });
  s.addChart(pres.charts.BAR, [{ name: "정확도 변화(%p)", labels: ["판정 기준 보정", "클래스 가중치 조정", "겹침 구간 제외", "겹침 구간 가중치↓"], values: [0.00, 0.02, -0.25, 0.14] }], {
    x: M + 4.3, y: 4.65, w: CW - 4.6, h: 2.0, barDir: "bar", chartColors: [C.sky],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10.5, dataLabelFontBold: true, dataLabelFormatCode: "+0.00;-0.00",
    valAxisMinVal: -0.5, valAxisMaxVal: 0.5, valAxisHidden: true, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 10.5,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  });
  caption(s, "근거: 미션2_PPT_제작의뢰_초안.md 3절, docs/experiment_log.md M2-EXP-002~005");
}

// ⑤ §4 보완점 발견 → 방향 전환
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 연구 방법 ② 보완점 발견과 방향 전환", 18);
  coreMsg(s, "숫자 몇 개로 요약하는 순간, 목소리의 세부 정보가 사라진다");
  const hw = (CW - 0.3) / 2;
  card(s, M, 2.45, hw, 4.35);
  txt(s, "■ 오류 분석에서 나온 약점", { x: M + 0.2, y: 2.58, w: hw - 0.4, h: 0.32, fontSize: 14, bold: true, color: C.blue });
  txt(s, [
    { text: "짧은 발화  ", options: { bold: true } }, { text: "요약할 정보 자체가 적음", options: { breakLine: true } },
    { text: "목소리 겹침 구간  ", options: { bold: true } }, { text: "두 사람 소리가 섞인 구간" },
  ], { x: M + 0.2, y: 2.95, w: hw - 0.4, h: 0.75, fontSize: 12, paraSpaceAfter: 4 });
  txt(s, "특징을 더 넣어도 이 약점은 그대로였음 (3절 \"추가로 해 본 것\")", { x: M + 0.2, y: 3.75, w: hw - 0.4, h: 0.4, fontSize: 11, color: C.sub });
  s.addShape(pres.shapes.LINE, { x: M + 0.2, y: 4.25, w: hw - 0.4, h: 0, line: { color: C.line, width: 1 } });
  txt(s, [
    { text: "가설  ", options: { bold: true, color: C.blue } }, { text: "평균·분산으로 요약하는 과정에서 시간에 따른 변화와 음색의 결 같은 정보가 버려진다", options: { breakLine: true } },
    { text: "선택  ", options: { bold: true, color: C.blue } }, { text: "요약하지 말고, 소리를 그림(스펙트로그램) 그대로 학습시키자" },
  ], { x: M + 0.2, y: 4.4, w: hw - 0.4, h: 2.3, fontSize: 12, paraSpaceAfter: 8 });
  const rx = M + hw + 0.3;
  card(s, rx, 2.45, hw, 4.35);
  txt(s, "■ 선행 연구 근거", { x: rx + 0.2, y: 2.58, w: hw - 0.4, h: 0.32, fontSize: 14, bold: true, color: C.blue });
  txt(s, [
    { text: "Hershey et al., ", options: { bold: true } }, { text: "CNN Architectures for Large-Scale Audio Classification, ICASSP 2017", options: { italic: true, breakLine: true } },
    { text: "log-mel 스펙트로그램 + CNN이 대규모 오디오 분류에서 효과적임을 보임", options: { color: C.sub, breakLine: true } },
    { text: "Nagrani et al., ", options: { bold: true } }, { text: "VoxCeleb: a large-scale speaker identification dataset, Interspeech 2017", options: { italic: true, breakLine: true } },
    { text: "스펙트로그램 기반 CNN으로 화자 식별", options: { color: C.sub, breakLine: true } },
    { text: "Ozan, ", options: { bold: true } }, { text: "Classification of Audio Segments in Call Center Recordings using CRNN, arXiv:2106.02422 (2021)", options: { italic: true, breakLine: true } },
    { text: "콜센터 통화의 상담원/고객 구간을 CNN+RNN으로 자동 분류 — 본 미션과 가장 가까운 선행 사례", options: { color: C.sub } },
  ], { x: rx + 0.2, y: 2.95, w: hw - 0.4, h: 3.2, fontSize: 10.5, paraSpaceAfter: 6 });
  caption(s, "근거: 미션2_PPT_제작의뢰_초안.md 4절 (인용 2026-10-08 확인)");
}

// ⑥ §5 최종 모델 구조
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 연구 방법 ③ 최종 모델 구조", 19);
  coreMsg(s, "발화를 소리 그림으로 바꿔 CNN으로 분류하고, 3개 모델의 판단을 평균한다");
  const steps = [
    ["① 발화 추출", "해당 구간만\n잘라냄 (다른\n발화는 안 봄)"],
    ["② 스펙트로그램", "log-mel, 주파수\n40칸·20ms 단위\n발화 내 정규화"],
    ["③ 길이 맞춤", "짧은 발화는 무음\n으로 6초까지\n(자기 자신만)"],
    ["④ CNN × 3", "4층 합성곱+전역\n풀링, 각자 확률\n계산"],
    ["⑤ 판정", "3개 확률 평균\n≥0.5 신고자\n(기준값 0.5)"],
  ];
  const bw = 2.1, gap = (CW - 5 * bw) / 4;
  steps.forEach(([t, b], i) => {
    const x = M + i * (bw + gap);
    stepBox(s, x, 2.5, bw, 2.1, t, b, i === 4 ? C.sky : C.blue);
    if (i < 4) arrow(s, x + bw + (gap - 0.24) / 2, 3.35, 0.24, 0.38);
  });
  card(s, M, 4.95, CW, 1.85);
  txt(s, "■ CNN 구조", { x: M + 0.2, y: 5.08, w: CW - 0.4, h: 0.3, fontSize: 13, bold: true, color: C.blue });
  txt(s, [
    { text: "합성곱 층 4개(32→64→128→256 채널) + 전역 평균 풀링 + 분류층", options: { breakLine: true } },
    { text: "모델 1개당 파라미터 421,954개", options: { breakLine: true } },
    { text: "사전학습 모델이나 외부 데이터 없이, 제공된 데이터만으로 처음부터 학습" },
  ], { x: M + 0.2, y: 5.4, w: CW - 0.4, h: 1.3, fontSize: 11.5, paraSpaceAfter: 4 });
  caption(s, "근거: 미션2_PPT_제작의뢰_초안.md 5절, src/dcc_data/mission2_cnn.py (SpeakerCNNLarge)");
}

// ⑦ §6 핵심 아이디어 ① 발화 단독 무음 채우기
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 연구 방법 ④ 핵심 아이디어 ① 발화 단독 무음 채우기", 20);
  coreMsg(s, "짧은 발화를 무음으로 6초까지 채우기만 해도 +1.16%p");
  card(s, M, 2.45, 7.3, 4.35);
  txt(s, [
    { text: "문제  ", options: { bold: true, color: C.blue } },
    { text: "학습은 여러 발화를 묶어 가장 긴 발화에 맞춰 무음을 채움 → 채점은 발화를 하나씩 넣어 짧은 발화가 무음 없이 들어감", options: { breakLine: true } },
    { text: "해결  ", options: { bold: true, color: C.blue } },
    { text: "추론 때 각 발화를 자기 자신만으로 무음을 채워 300프레임(6초)으로 맞춤 — 다른 발화 정보를 안 쓰니 규정에도 맞음" },
  ], { x: M + 0.2, y: 2.6, w: 7.0, h: 1.35, fontSize: 12, paraSpaceAfter: 6 });
  txt(s, "발화 길이 구간별 정확도 (채우기 전 → 후)", { x: M + 0.2, y: 3.95, w: 7.0, h: 0.3, fontSize: 11.5, color: C.sub });
  s.addChart(pres.charts.BAR, [
    { name: "채우기 전", labels: ["0.5초 미만", "0.5~1초", "1~2초"], values: [86.2, 85.0, 88.2] },
    { name: "채우기 후", labels: ["0.5초 미만", "0.5~1초", "1~2초"], values: [90.3, 86.8, 89.1] },
  ], {
    x: M, y: 4.25, w: 7.3, h: 2.15, barDir: "col", barGrouping: "clustered", chartColors: [C.line, C.blue],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFontBold: true, dataLabelFormatCode: "0.0",
    valAxisMinVal: 80, valAxisMaxVal: 94, valAxisHidden: true, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 11,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 10,
  });
  txt(s, "전체 공식 Validation: 89.73% → 90.89% (+1.16%p)  ·  2초 이상은 거의 변화 없음", { x: M + 0.2, y: 6.6, w: 7.0, h: 0.3, fontSize: 10, color: C.sub });
  card(s, M + 7.6, 2.45, W - M - (M + 7.6), 4.35, C.bgGray);
  txt(s, "왜 중요한가", { x: M + 7.8, y: 2.6, w: 3.7, h: 0.3, fontSize: 13, bold: true, color: C.blue });
  txt(s, "재학습 없이 추론 방식만 고쳐서 얻은 이득 — 모델이 아니라 \"학습과 추론의 입력이 서로 달랐다\"는 걸 찾아낸 것", { x: M + 7.8, y: 2.95, w: 3.7, h: 1.8, fontSize: 11 });
  caption(s, "근거: 미션2_PPT_제작의뢰_초안.md 6절, docs/experiment_log.md M2-PAD-001, M2-VAL-004/005");
}

// ⑧ §7 핵심 아이디어 ② seed 앙상블
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 연구 방법 ⑤ 핵심 아이디어 ② seed 앙상블", 21);
  coreMsg(s, "같은 CNN을 시작 조건만 바꿔 3번 학습하고 평균하면, 단일 모델보다 0.5%p 높음");
  card(s, M, 2.45, 6.6, 4.35);
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["모델 1개\n(평균)", "모델 2개\n평균", "모델 3개\n평균"], values: [90.97, 91.41, 91.55] }], {
    x: M + 0.2, y: 2.65, w: 6.2, h: 3.9, barDir: "col", chartColors: [C.line, C.sky, C.blue], varyColors: true,
    showTitle: true, title: "모델 수에 따른 공식 Validation Accuracy (%)", titleFontSize: 12, titleColor: C.text, titleFontFace: F,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisMinVal: 90, valAxisMaxVal: 92, valAxisLabelColor: C.sub, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 11.5,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  });
  const rx = M + 6.9, rw = W - M - rx;
  card(s, rx, 2.45, rw, 4.35);
  txt(s, "■ 방법", { x: rx + 0.2, y: 2.6, w: rw - 0.4, h: 0.3, fontSize: 13, bold: true, color: C.blue });
  txt(s, "모델 구조·데이터·학습 설정은 같고 무작위 시작값(seed 1/2/3)만 다름 — 실력은 비슷해도 틀리는 발화가 서로 달라서, 평균을 내면 서로의 실수를 보완", { x: rx + 0.2, y: 2.95, w: rw - 0.4, h: 1.6, fontSize: 11 });
  table(s, [
    ["구성", "Accuracy"],
    ["모델 1개", "90.90~91.04%"],
    ["모델 2개", "91.39~91.45%"],
    ["모델 3개", { text: "91.55%", options: { bold: true, color: C.blue } }],
  ], rx + 0.2, 4.7, [1.6, 1.7], { rowH: 0.42, fontSize: 11, align: "center" });
  txt(s, "2개→3개는 오르는 폭이 +0.1%p로 줄어서 3개에서 멈춤", { x: rx + 0.2, y: 6.45, w: rw - 0.4, h: 0.4, fontSize: 10.5, color: C.sub });
  caption(s, "근거: 미션2_PPT_제작의뢰_초안.md 7절, docs/experiment_log.md M2-ENS-002/003");
}

// ⑨ §8 검증했지만 채택하지 않은 시도
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 연구 방법 ⑥ 검증했지만 채택하지 않은 시도", 22);
  coreMsg(s, "아이디어마다 seed 3개로 반복 실험해 우연이 아닌지 확인했고, 기준에 못 미치면 버렸다");
  card(s, M, 2.45, CW, 4.1);
  txt(s, "정확도 변화 (%p) — 채택 기준은 +0.5%p 이상 (파랑=양수, 빨강=음수)", { x: M + 0.2, y: 2.6, w: CW - 0.4, h: 0.3, fontSize: 11.5, color: C.sub });
  const tornadoLabels = ["화자인식 전용 구조\n(ECAPA-TDNN)", "더 큰 모델\n(ResNet)", "데이터 보강\n(SpecAugment)", "주파수 눈금 변경\n(mel→linear)", "시간 해상도 2배\n(hop80)*", "주파수 칸 수 2배\n(40→80)"];
  const tornadoVals = [0.27, 0.06, -0.12, -0.39, -0.14, -0.68];
  // 콤보(막대+점선 기준선) 차트는 다른 차트가 많은 덱 안에서 PowerPoint가 파일을 못 여는
  // 문제가 있어(pptxgenjs 콤보차트 축 id 버그, 2026-10-12 확인) 막대 단독으로 대체 —
  // 기준선 설명은 아래 "메시지" 박스 텍스트로 대신함
  s.addChart(pres.charts.BAR, [{ name: "정확도 변화", labels: tornadoLabels, values: tornadoVals }], {
    chartColors: tornadoVals.map(v => v >= 0 ? C.sky : C.red),
    x: M + 0.2, y: 2.95, w: CW - 0.4, h: 3.4, barDir: "bar", barGrouping: "standard",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10.5, dataLabelFontBold: true, dataLabelFormatCode: "+0.00;-0.00",
    valAxisMinVal: -0.8, valAxisMaxVal: 0.8, valAxisHidden: true, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 10.5,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  });
  insight(s, M, 6.65, CW, 0.5, [
    { text: "메시지  ", options: { bold: true, color: C.blue } },
    { text: "판정 기준은 실험 전에 정했고, 결과를 보고 바꾸지 않았음 — 6개 모두 기준선을 못 넘어 기각" },
  ]);
  txt(s, "* hop80은 1차(+0.27%p)·2차(−0.14%p) 확인 결과가 뒤집혀 재현 안 됨 · 근거: 미션2_PPT_제작의뢰_초안.md 8절, experiment_log.md M2-CNN-005~009",
    { x: M, y: 7.18, w: CW, h: 0.25, fontSize: 8.5, color: C.sub });
}

// ⑩ §9 최종 결과
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 최종 연구 결과", 23);
  coreMsg(s, "공식 Validation 정확도 91.49% (L1 LightGBM 대비 +5.05%p)");
  card(s, M, 2.45, 4.2, 4.35);
  stat(s, M + 0.2, 2.7, 3.8, "91.49%", "Accuracy (공식 Validation)", C.blue, 38);
  txt(s, "111,947건 중 102,419건 정답\n대원 recall 87.15% · 신고자 recall 95.51%", { x: M + 0.2, y: 4.0, w: 3.8, h: 0.7, fontSize: 11, color: C.sub, align: "center" });
  txt(s, "혼동행렬 (히트맵, 행 기준 %)", { x: M + 0.2, y: 4.85, w: 3.8, h: 0.26, fontSize: 10.5, color: C.sub });
  table(s, [
    ["실제 ↓ · 예측 →", "대원", "신고자"],
    ["대원 (53,845)",
      { text: "46,926\n(87.15%)", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "6,919\n(12.85%)", options: { color: C.red, fill: { color: C.pink } } }],
    ["신고자 (58,102)",
      { text: "2,609\n(4.49%)", options: { color: C.sub, fill: { color: "FFF4F4" } } },
      { text: "55,493\n(95.51%)", options: { bold: true, color: C.white, fill: { color: C.blue } } }],
  ], M + 0.2, 5.13, [1.3, 1.25, 1.25], { rowH: 0.56, fontSize: 10, align: "center" });
  const mx = M + 4.5, mw = 4.3;
  card(s, mx, 2.45, mw, 4.35);
  txt(s, "발전 과정 (공식 Validation)", { x: mx + 0.2, y: 2.6, w: mw - 0.4, h: 0.3, fontSize: 11.5, color: C.sub });
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["LightGBM\n(L1)", "CNN 1개", "CNN 3개\n앙상블"], values: [86.44, 90.97, 91.49] }], {
    x: mx + 0.1, y: 2.95, w: mw - 0.2, h: 3.7, barDir: "col", chartColors: [C.line, C.sky, C.blue], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisMinVal: 84, valAxisMaxVal: 93, valAxisLabelColor: C.sub, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 11,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  });
  const rx = mx + mw + 0.3, rw = W - M - rx;
  card(s, rx, 2.45, rw, 4.35);
  txt(s, "발화 길이별 정확도", { x: rx + 0.2, y: 2.6, w: rw - 0.4, h: 0.3, fontSize: 11.5, color: C.sub });
  s.addChart(pres.charts.BAR, [{ name: "정확도", labels: ["0.5초\n미만", "0.5~1초", "1~2초", "2초\n이상"], values: [90.95, 87.60, 89.71, 96.05] }], {
    x: rx + 0.1, y: 2.95, w: rw - 0.2, h: 3.0, barDir: "col", chartColors: [C.sky, C.red, C.sky, C.blue], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisMinVal: 84, valAxisMaxVal: 98, valAxisHidden: true, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 10,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  });
  txt(s, "0.5~1초 구간(빨강)이 가장 약함 · [확인 필요] 연우님 모델과 비교 반영 여부", { x: rx + 0.2, y: 6.05, w: rw - 0.4, h: 0.55, fontSize: 9.5, color: C.ph, italic: true });
  caption(s, "근거: 미션2_PPT_제작의뢰_초안.md 9절, docs/experiment_log.md M2-RERUN-001");
}

// ⑪ §10 오류 분석과 보완점
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 오류 분석과 보완점", 24);
  coreMsg(s, "남은 오류는 짧은 맞장구와 \"대원을 신고자로 착각\"하는 경우에 몰려 있음");
  const hw = (CW - 0.3) / 2;
  card(s, M, 2.45, hw, 4.35);
  txt(s, "■ 가장 약한 구간: 0.5~1초 (87.60%)", { x: M + 0.2, y: 2.58, w: hw - 0.4, h: 0.3, fontSize: 13.5, bold: true, color: C.blue });
  txt(s, "구간별 오류율 (%) — \"네\",\"예\" 같은 짧은 대답은 대원·신고자가 똑같이 씀", { x: M + 0.2, y: 2.88, w: hw - 0.4, h: 0.3, fontSize: 10, color: C.sub });
  s.addChart(pres.charts.BAR, [{ name: "오류율", labels: ["0.5초\n미만", "0.5~1초", "1~2초", "2초\n이상"], values: [9.05, 12.40, 10.29, 3.95] }], {
    x: M + 0.2, y: 3.2, w: hw - 0.4, h: 1.75, barDir: "col", chartColors: [C.sky, C.red, C.sky, C.blue], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisHidden: true, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 10,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  });
  txt(s, "오분류 방향 (건수) — 대원→신고자 쏠림 73%", { x: M + 0.2, y: 5.15, w: hw - 0.4, h: 0.3, fontSize: 10, color: C.sub });
  s.addChart(pres.charts.BAR, [{ name: "오분류 건수", labels: ["대원 → 신고자", "신고자 → 대원"], values: [6919, 2609] }], {
    x: M + 0.2, y: 5.45, w: hw - 0.4, h: 1.15, barDir: "bar", chartColors: [C.red, C.sky], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFontBold: true, dataLabelFormatCode: "#,##0",
    valAxisHidden: true, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 10.5,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  });
  txt(s, "라벨 잡음: 라벨이 뒤바뀐 것으로 의심되는 통화는 규정상 고치지 않고 제공된 라벨 그대로 학습, 의심 사례만 별도 기록",
    { x: M + 0.2, y: 6.65, w: hw - 0.4, h: 0.5, fontSize: 9, color: C.sub });
  const rx = M + hw + 0.3;
  card(s, rx, 2.45, hw, 4.35);
  txt(s, "■ 데이터 자체의 어려움 vs 모델 성능의 개선 여지", { x: rx + 0.2, y: 2.58, w: hw - 0.4, h: 0.32, fontSize: 13.5, bold: true, color: C.blue });
  txt(s, [
    { text: "데이터 쪽  ", options: { bold: true } },
    { text: "다른 방법(§8)으로도 비슷한 지점(짧은 발화)에서 성능이 멈춤 — 오류 상당 부분은 \"한 발화만 보고 판단\"이라는 조건 자체에서 비롯된 것으로 봄", options: { breakLine: true } },
    { text: "모델 쪽  ", options: { bold: true } },
    { text: "대원→신고자 쏠림(73%)은 비대칭 — 구조·입력을 여러 방향(§8)으로 바꿔도 못 줄였다는 점에서, 지금 접근으로는 쉽게 안 줄어드는 영역으로 판단" },
  ], { x: rx + 0.2, y: 2.95, w: hw - 0.4, h: 3.2, fontSize: 11.5, paraSpaceAfter: 8 });
  caption(s, "근거: 미션2_PPT_제작의뢰_초안.md 10절, docs/experiment_log.md M2-COMPARE-002, M2-LABEL-001~003");
}

// ⑫ §11 결과의 신뢰성: 규정 준수와 재현성
// (§11 결과의 신뢰성: 규정 준수와 재현성 — 슬라이드 제거. 대회 규칙 준수 내용은
// 본문에서 생략하고, 재현성 관련 핵심 수치는 §12 계산 효율성에서 함께 다룸)

// ⑬ §12 계산 효율성
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 계산 효율성 (대회 권장 항목)", 25);
  coreMsg(s, "학습 속도를 개선해 에폭당 약 25분 → 약 7분(약 3.5배)으로 줄임");
  card(s, M, 2.45, 7.7, 4.35);
  table(s, [
    ["항목", "값"],
    ["파라미터 수", "모델 1개 421,954 × 3 = Total 1,265,862 / Active 1,265,862"],
    ["학습 환경", "Colab NVIDIA Tesla T4 1개, 혼합 정밀도, 에폭당 약 7분, 시드별 20~24에폭(자동 조기 종료)"],
    ["추론 환경", "CPU (Intel Xeon 2.0GHz 2코어), GPU 미사용"],
    ["Validation 추론 batch size", "1 (발화 하나씩)"],
    ["Validation 추론 시간", "전체 85.3분, 발화당 평균 45.7ms"],
  ], M + 0.2, 2.65, [2.6, 4.7], { rowH: 0.63, fontSize: 11 });
  txt(s, "데이터 저장 방식 개선(스펙트로그램을 float16 하나로 묶어 메모리 매핑) + 혼합 정밀도(AMP) 적용 덕분",
    { x: M + 0.2, y: 6.55, w: 7.3, h: 0.3, fontSize: 9.5, color: C.sub });
  const ex = M + 8.0, ew = W - M - ex;
  card(s, ex, 2.45, ew, 4.35);
  txt(s, "에폭당 학습 시간", { x: ex + 0.2, y: 2.6, w: ew - 0.4, h: 0.3, fontSize: 11.5, color: C.sub });
  s.addChart(pres.charts.BAR, [{ name: "분", labels: ["개선 전", "개선 후"], values: [25, 7] }], {
    x: ex + 0.1, y: 2.95, w: ew - 0.2, h: 3.4, barDir: "col", chartColors: [C.line, C.blue], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelFontBold: true, dataLabelFormatCode: "0\"분\"",
    valAxisHidden: true, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 12,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  });
  txt(s, "약 3.5배", { x: ex + 0.1, y: 6.3, w: ew - 0.2, h: 0.3, fontSize: 11, bold: true, color: C.red, align: "center" });
  caption(s, "근거: 미션2_PPT_제작의뢰_초안.md 12절, docs/decisions.md 2026-10-05 (학습 속도 개선 항목)");
}

// ⑭ §13 사회안전 시사점
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2 — 사회안전 시사점", 26);
  coreMsg(s, "목소리만으로 \"누가 말했는지\"를 자동으로 나누면, 신고 접수 기록과 분석을 빠르게 만들 수 있음");
  const hw = (CW - 0.3) / 2;
  insight(s, M, 2.45, hw, 2.0, [
    { text: "신고 내용 자동 정리  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "통화를 대원·신고자 발화로 자동 분리하면, 신고자가 말한 증상·위치만 골라 요약하거나 다음 단계(증상 분류 등, Mission 3)에 넘길 수 있음" },
  ]);
  insight(s, M, 4.6, hw, 2.2, [
    { text: "접수 품질 관리  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "대원 발화만 모아 응대 절차 준수 여부를 점검하는 등, 사후 분석 시간을 줄일 수 있음" },
  ], C.light);
  const rx = M + hw + 0.3;
  insight(s, rx, 2.45, hw, 2.0, [
    { text: "개인정보 측면  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "발화 내용(텍스트) 없이 음향만으로 동작해서, 내용을 보지 않고도 1차 분류가 가능함" },
  ], C.light);
  card(s, rx, 4.6, hw, 2.2, C.bgGray);
  txt(s, "현실 적용 시 개선점", { x: rx + 0.2, y: 4.72, w: hw - 0.4, h: 0.3, fontSize: 13, bold: true, color: C.blue });
  txt(s, [
    { text: "2초 이상 발화는 96%로 믿을 만함", options: { breakLine: true } },
    { text: "짧은 맞장구는 실제 서비스에서 앞뒤 발화 맥락을 함께 쓰면 보완 가능 — 대회에서는 금지된 정보라 쓰지 않음" },
  ], { x: rx + 0.2, y: 5.05, w: hw - 0.4, h: 1.6, fontSize: 11.5, paraSpaceAfter: 6 });
  caption(s, "근거: 미션2_PPT_제작의뢰_초안.md 13절");
}

// ---------- Mission 3 ----------
missionA(27, 3, "환자 증상 인식",
  "증상 단어가 대화에 직접 등장하는 경우가 많음 — 단어 기반 모델로 출발해 문맥 모델로 확장",
  [["입력", "신고자·대원 대화 전사 텍스트\n(그 외 라벨 사용 불가)"], ["출력", "증상 9종 중 0~9개\n(다중 레이블)"], ["평가", "Macro F1\n(증상별 F1의 평균)"], ["대상", "고열·구토·두통·복통·어지러움·열상·오심·전신쇠약·호흡곤란"]],
  [["전처리", "발화 시간순 정렬\n9종 외 증상 제거"], ["텍스트 표현", "TF-IDF\n(단어·2단어 묶음)"], ["분류 모델", "증상별 로지스틱\n회귀 (9개)"], ["판정", "증상별 판정\n(기준값 0.5)"]],
  ["증상 이름·표현(\"머리가 아파요\")이 전사에 직접 드러나 단어 기반이 강한 출발점",
   "희귀 증상이 많아 Macro F1에 불리 → 증상별 성능을 따로 점검",
   [ph("KcBERT 등 문맥 모델 결과 반영 시 갱신")]]);
{
  const s = missionB(28, 3, "환자 증상 인식", [ph("최종 모델과 Macro F1 한 문장", { color: C.sub })]);
  card(s, M, 2.45, 7.3, 4.35);
  txt(s, [ph("0.000")], { x: M + 0.2, y: 2.65, w: 2.6, h: 0.75, fontSize: 40, bold: true, align: "center", valign: "middle" });
  txt(s, "Macro F1 (공식 Validation)", { x: M + 0.2, y: 3.43, w: 2.6, h: 0.5, fontSize: 12, color: C.sub, align: "center" });
  insight(s, M + 0.2, 4.3, 2.6, 2.3, [
    { text: "검증 먼저  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "개선 수치(0.40→0.45)가 데이터 분할 방식 차이일 수 있어, 통화 중복 검사 후에만 채택", options: { fontSize: 11 } },
  ]);
  figPH(s, M + 3.0, 2.65, 4.1, 3.95, "증상 9종별 F1 가로 막대그래프\n(최고/최저 증상 강조)");
  rightCol(s, M + 7.6, [
    { text: "잘 맞춘 증상  ", options: { bold: true } }, ph("예: 호흡곤란", { breakLine: true }),
    { text: "어려운 증상  ", options: { bold: true } }, ph("희귀 증상 · 표현이 다양한 증상"),
  ], [ph("예: 증상 자동 태깅 → 출동 우선순위·병원 선정 보조")]);
  caption(s, "근거: docs/experiment_log.md (M3-EXP-001~), docs/decisions.md 2026-09-20 분할 감사 항목");
}

// ---------- Mission 3 추가: 확장 계획과 검증 일정 ----------
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 확장 계획과 검증 일정", 29);
  coreMsg(s, "분할 감사부터 끝낸 뒤 TF-IDF 모델을 확정하고, 남은 기간 동안 문맥 모델로 확장");
  const steps = [
    ["분할 감사", "recordId 중복·\noverlap 검사", C.sky],
    ["TF-IDF 확정", "공식 Validation\nMacro F1 확정", C.blue],
    ["문맥 모델 비교", "KcBERT 등\nfine-tuning", C.blue],
    ["불균형 보정", "희귀 증상\n보정 (기준값 유지)", C.blue],
    ["최종 선정", "10/14 이전\n모델 확정", C.mint],
  ];
  const bw = 2.1, gap = (CW - 5 * bw) / 4;
  steps.forEach(([t, b, col], i) => {
    const x = M + i * (bw + gap);
    stepBox(s, x, 2.5, bw, 2.0, t, b, col);
    if (i < 4) arrow(s, x + bw + (gap - 0.28) / 2, 3.3);
  });
  insight(s, M, 4.95, CW, 1.0, [
    { text: "지금 위치  ", options: { bold: true, color: C.blue } },
    { text: "① 분할 감사 단계 — 개선 수치(0.40→0.45)가 데이터 분할 차이인지 먼저 확인 중" },
  ]);
  insight(s, M, 6.1, CW, 1.0, [
    { text: "본선 전 열어둔 방향  ", options: { bold: true, color: C.blue } },
    { text: "문맥 모델이 TF-IDF보다 뚜렷이 낫지 않으면, 단순하고 설명 가능한 TF-IDF+로지스틱 회귀를 최종안으로 유지" },
  ], C.mintLight);
  caption(s, "근거: docs/decisions.md 2026-09-20 분할 감사 항목, docs/experiment_log.md");
  s.addNotes("본선 전까지 KcBERT 등 결과가 나오면 M3-B(28p)의 수치를 갱신하고, 이 장은 '실제로 더 해본 것'으로 내용 교체.");
}

// =====================================================================
// 30. 04-(1) 성능 요약과 교훈
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(1) 성능 요약과 교훈", 30);
  coreMsg(s, "세 미션 모두 공식 Validation 기준으로 평가하고, 점수보다 검증의 정직함을 우선함");
  tag(s, W - M - 1.7, 0.95);
  table(s, [
    ["미션", "과제", "최종 모델", "지표", "공식 Validation", "상태"],
    ["Mission 1", "신고자 성별 분류", "음향 특징 + 로지스틱 회귀", "Accuracy", { text: "95.11%", options: { bold: true, color: C.blue } }, "1차 확정"],
    ["Mission 2", "신고자/119대원 분류", "Mel-spectrogram + CNN 앙상블", "Accuracy", { text: "91.49%", options: { bold: true, color: C.blue } }, "1차 확정 (김광희)"],
    ["Mission 3", "환자 증상 인식", { text: "〔최종 모델〕", options: { color: C.ph, italic: true } }, "Macro F1", { text: "〔0.000〕", options: { color: C.ph, italic: true } }, "개선 중"],
  ], M, 2.45, [1.4, 2.6, 3.0, 1.5, 2.2, 1.63], { rowH: 0.5, align: "center" });
  const lessons = [
    ["점수보다 검증 먼저", "같은 통화가 학습·검증에 섞이는 누수를 매번 점검하고, 의심되는 개선은 보류"],
    ["기각도 기록한다", "효과 없던 실험과 그 이유를 decisions.md에 남겨 같은 시도를 반복하지 않음"],
    ["규칙은 설계 단계에서", "규칙 위반 소지가 있는 특징은 성능이 좋아도 제거 (M2 통화 단위 정보)"],
  ];
  const cw = (CW - 2 * 0.3) / 3;
  lessons.forEach(([t, d], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 4.75, cw, 1.95);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 4.95, w: 0.5, h: 0.5, fill: { color: C.blue }, line: { color: C.blue } });
    txt(s, String(i + 1), { x: x + 0.25, y: 4.95, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.white, align: "center", valign: "middle" });
    txt(s, t, { x: x + 0.9, y: 4.95, w: cw - 1.1, h: 0.5, fontSize: 16, bold: true, color: C.blue, valign: "middle" });
    txt(s, d, { x: x + 0.25, y: 5.6, w: cw - 0.5, h: 1.0, fontSize: 12 });
  });
}

// =====================================================================
// 31. 04-(2) 사회안전 시사점과 향후 계획
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(2) 사회안전 시사점과 향후 계획", 31);
  coreMsg(s, "세 미션을 이으면 — 119 신고 접수 순간부터 출동 판단까지를 돕는 하나의 흐름");
  const flow = [
    ["신고 접수", "119 통화 음성", C.sky],
    ["Mission 2", "신고자 발화만\n자동 분리", C.blue],
    ["Mission 1", "신고자 성별\n자동 기록", C.blue],
    ["Mission 3", "환자 증상\n자동 인식", C.blue],
    ["출동 판단 보조", "우선순위 · 병원 선정", C.mint],
  ];
  const bw = 2.1, gap = (CW - 5 * bw) / 4;
  flow.forEach(([t, b, col], i) => {
    const x = M + i * (bw + gap);
    stepBox(s, x, 2.45, bw, 1.55, t, b, col);
    if (i < 4) arrow(s, x + bw + (gap - 0.28) / 2, 3.0);
  });
  const hw = (CW - 0.4) / 2;
  card(s, M, 4.3, hw, 2.5);
  txt(s, "■ 기대효과", { x: M + 0.25, y: 4.42, w: hw - 0.5, h: 0.4, fontSize: 18, bold: true });
  txt(s, [
    { text: "접수 시간 단축", options: { bold: true, color: C.blue } }, { text: " — 상황실 요원의 수기 기록 부담 감소", options: { breakLine: true } },
    { text: "누락 방지", options: { bold: true, color: C.blue } }, { text: " — 긴박한 통화 속 증상 언급을 자동으로 표시", options: { breakLine: true } },
    { text: "판단 보조", options: { bold: true, color: C.blue } }, { text: " — 증상 정보로 출동 우선순위 결정 지원", options: { breakLine: true } },
    ph("추가 시사점"),
  ], { x: M + 0.25, y: 4.9, w: hw - 0.5, h: 1.8, fontSize: 13, paraSpaceAfter: 6 });
  const rx = M + hw + 0.4;
  card(s, rx, 4.3, hw, 2.5);
  txt(s, "■ 본선까지 향후 계획", { x: rx + 0.25, y: 4.42, w: hw - 0.5, h: 0.4, fontSize: 18, bold: true });
  txt(s, [
    { text: "Mission 1  ", options: { bold: true } }, ph("다음 단계", { breakLine: true }),
    { text: "Mission 2  ", options: { bold: true } }, ph("예: 여러 CNN 앙상블 · 짧은 발화 보강", { breakLine: true }),
    { text: "Mission 3  ", options: { bold: true } }, ph("예: KcBERT 문맥 모델 적용", { breakLine: true }),
    { text: "공통  ", options: { bold: true } }, ph("예: 실제 상황실 적용 시나리오 검토"),
  ], { x: rx + 0.25, y: 4.9, w: hw - 0.5, h: 1.8, fontSize: 13, paraSpaceAfter: 6 });
  s.addNotes("출제문제 12쪽 필수 항목 '사회안전 시사점'이 이 장의 핵심. 미션별 결과 장의 시사점 한 줄과 표현을 맞출 것.");
}

// =====================================================================
// 32. 마무리 — 공식 템플릿 그대로 사용 + 팀 정보만 빈 여백(2.1~2.9in)에 추가
// =====================================================================
{
  const s = pres.addSlide();
  s.addImage({ path: TPL.closing, x: 0, y: 0, w: W, h: 7.5, sizing: { type: "cover", w: W, h: 7.5 } });
  txt(s, "팀 eigensinn  |  김장수 · 김광희 · 지연우 · 이수빈 · 조지은", { x: M, y: 2.2, w: CW, h: 0.4, fontSize: 16, bold: true, color: C.blue, align: "center" });
  txt(s, "코드 · 실험 기록: GitHub eigensinn-dcc2026/prelim  (docs/experiment_log.md, docs/decisions.md)", { x: M, y: 2.58, w: CW, h: 0.3, fontSize: 11, color: C.sub, align: "center" });
  s.addNotes("공식 템플릿(슬라이드4) 그대로. 흰 여백(약 2.1~2.9in)에만 팀 정보를 얹음 — 배너 그래픽은 건드리지 않음.");
}

pres.writeFile({ fileName: process.argv[2] || "DCC2026_예선_결과정리_뼈대.pptx" }).then(f => console.log("wrote", f));
