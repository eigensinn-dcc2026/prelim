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
    ["03", "미션별 수행", ["Mission 1  신고자 성별 분류", "Mission 2  신고자/대원 분류\n   (+ 다른 접근 비교)", "Mission 3  환자 증상 인식\n   (+ 확장 계획)"]],
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
      { text: "클래스 불균형 보정이 불필요 → 기본 LR·GBM 그대로 사용" },
    ]],
    ["figures/symptom_prevalence.png", "증상 9종 유병률 (Mission 3)", [
      { text: "발견  ", options: { bold: true, breakLine: true } },
      { text: "복통·어지러움 20%대 vs 두통·오심 10%대 — 최대 2배 이상 차이", options: { breakLine: true } },
      { text: "설계 반영  ", options: { bold: true, color: C.blue, breakLine: true } },
      { text: "Macro F1은 희소 증상에 민감 → 증상별 성능을 따로 점검 (15p)" },
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
function missionB(page, no, title, core) {
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", `Mission ${no}  ${title} — 결과와 시사점`, page);
  coreMsg(s, core);
  tag(s, W - M - 1.7, 0.95);
  return s;
}
// 오른쪽 열: 결과 분석 / 재현 정보 / 사회안전 시사점
function rightCol(s, rx, analysis, safety) {
  const rw = W - M - rx;
  card(s, rx, 2.45, rw, 1.75);
  txt(s, "■ 결과 분석", { x: rx + 0.2, y: 2.55, w: rw - 0.4, h: 0.35, fontSize: 15, bold: true });
  txt(s, analysis, { x: rx + 0.2, y: 2.95, w: rw - 0.4, h: 1.2, fontSize: 12, paraSpaceAfter: 4 });
  card(s, rx, 4.35, rw, 1.05, C.bgGray);
  txt(s, "■ 재현 정보", { x: rx + 0.2, y: 4.43, w: rw - 0.4, h: 0.3, fontSize: 13, bold: true });
  txt(s, [{ text: "파라미터 수 " }, ph("0"), { text: "  ·  학습/추론 환경 " }, ph("GPU/CPU", { breakLine: true }),
    { text: "추론 배치 " }, ph("0"), { text: "  ·  Validation 추론 시간 " }, ph("0.0초/샘플")],
    { x: rx + 0.2, y: 4.75, w: rw - 0.4, h: 0.6, fontSize: 11, color: C.sub });
  insight(s, rx, 5.55, rw, 1.25, [{ text: "사회안전 시사점  ", options: { bold: true, color: C.blue, breakLine: true } }].concat(safety), C.mintLight);
}

// ---------- Mission 1 ----------
missionA(9, 1, "신고자 성별 분류",
  "성별은 목소리 높낮이 차이가 커서 — 단순하고 해석 가능한 모델로 충분히 높은 정확도를 노림",
  [["입력", "신고자 발화 음성\n(startAt · endAt · speaker만 사용)"], ["출력", "남 / 여"], ["평가", "Accuracy"], ["제약", "그 외 라벨 사용 불가"]],
  [["구간 추출", "신고자 발화만\n잘라내기"], ["음향 특징", "음높이·음색 등\n52차원 통계"], ["분류 모델", "로지스틱 회귀\n(LR)"], ["판정", "남/여\n(기준값 0.5)"]],
  ["성별 차이는 음높이 같은 저수준 음향 특징에 이미 크게 드러남",
   "GBM과 정확도 차이가 0.19%p로 작아 더 단순하고 해석 쉬운 LR 채택",
   [ph("최종 모델이 바뀌면 이유 갱신")]]);
{
  const s = missionB(10, 1, "신고자 성별 분류", "공식 Validation 3,640건에서 Accuracy 95.11% — 두 모델 모두 95% 안팎으로 안정적");
  card(s, M, 2.45, 7.3, 4.35);
  stat(s, M + 0.2, 2.65, 2.6, "95.11%", "Accuracy (공식 Validation)", C.blue, 40);
  txt(s, "95% 신뢰구간\n[94.40, 95.82]", { x: M + 0.2, y: 3.95, w: 2.6, h: 0.7, fontSize: 12, color: C.sub, align: "center" });
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["GBM", "LR (채택)"], values: [94.92, 95.11] }], {
    x: M + 2.9, y: 2.6, w: 4.2, h: 4.0, barDir: "col", chartColors: [C.sky, C.blue],
    showTitle: true, title: "모델별 Accuracy (%)", titleFontSize: 12, titleColor: C.text, titleFontFace: F,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisMinVal: 90, valAxisMaxVal: 97, valAxisLabelColor: C.sub, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 12,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, varyColors: true,
  });
  rightCol(s, M + 7.6, [
    { text: "오류가 몰린 곳  ", options: { bold: true } }, ph("오류 분석 결론", { breakLine: true }),
    { text: "과적합 점검  ", options: { bold: true } }, { text: "내부 94.8~95.0% vs 공식 95.1% — 차이 작음" },
  ], [ph("예: 신고자 정보 자동 기록 → 접수 시간 단축")]);
  caption(s, "근거: reports/mission1/ m1_val_eval.json, m1_val_bootstrap_ci.json, M1_error_analysis_report.md");
}

// ---------- Mission 2 ----------
missionA(11, 2, "신고자/119대원 분류",
  "규칙상 '발화 하나만 보고' 판단해야 함 — 통화 전체 정보 없이 목소리 자체에서 화자를 구분",
  [["입력", "한 사람의 한 발화 조각\n(startAt~endAt 구간만)"], ["출력", "신고자(1) / 119대원(0)"], ["평가", "Accuracy"], ["제약", "text · 발화 순서 · 다른 발화 정보 사용 불가"]],
  [["Baseline", "음향 통계\n+ LightGBM"], ["L1", "음향 특징 확장\n+ GBM"], ["L2", "Mel-spectrogram\n+ CNN"], ["판정", "신고자/대원\n(기준값 0.5)"]],
  ["상황실 대원은 전화 헤드셋·일정한 말투, 신고자는 휴대전화·잡음·긴박함 → 음향 차이 존재",
   "통계 특징(L1)으로 한계 확인 후, 소리 모양 전체를 이미지처럼 보는 CNN(L2)으로 확장",
   "통화 단위 정보(같은 통화 다른 발화 비교)는 규칙 위반이라 설계에서 제외"]);
{
  const s = missionB(12, 2, "신고자/119대원 분류", "Baseline 83.78% → L2 CNN 90.89% — 모델 구조 변경으로 +7%p 이상 향상");
  card(s, M, 2.45, 7.3, 4.35);
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["Baseline\n(내부 분할)", "L1 GBM\n(공식 Val)", "L2 CNN\n(공식 Val)"], values: [83.78, 86.40, 90.89] }], {
    x: M + 0.2, y: 2.6, w: 4.6, h: 4.0, barDir: "col", chartColors: [C.line, C.sky, C.blue], varyColors: true,
    showTitle: true, title: "단계별 Accuracy (%)", titleFontSize: 12, titleColor: C.text, titleFontFace: F,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontBold: true, dataLabelFormatCode: "0.00",
    valAxisMinVal: 80, valAxisMaxVal: 93, valAxisLabelColor: C.sub, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 11,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  });
  stat(s, M + 4.9, 3.0, 2.3, "+7.1%p", "Baseline 대비 향상", C.red, 34);
  txt(s, [{ text: "기각한 실험  ", options: { bold: true, breakLine: true } }, ph("N"), { text: "건", options: { breakLine: true } },
    { text: "(기준값 보정, 가중치 조정 등 — decisions.md)", options: { fontSize: 10, color: C.sub } }],
    { x: M + 4.9, y: 4.5, w: 2.3, h: 1.4, fontSize: 13, align: "center" });
  rightCol(s, M + 7.6, [
    { text: "오류가 몰린 곳  ", options: { bold: true } }, { text: "1초 미만 짧은 발화(\"네\", \"여보세요\")", options: { breakLine: true } },
    { text: "데이터 규모  ", options: { bold: true } }, { text: "3천→9천 통화 +1.1%p, 전체 29,200 통화에서는 정체" },
  ], [{ text: "신고자 발화만 자동 분리 → 신고 내용 분석·기록 자동화의 첫 단계" }]);
  caption(s, "근거: docs/experiment_log.md, docs/decisions.md, reports/mission2/  ※ Baseline은 내부 분할, L1·L2는 공식 Validation 기준");
}

// ---------- Mission 2 추가: 시도한 다른 접근 ----------
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/대원 분류 — 다른 접근과 기각/보류 이유", 13);
  coreMsg(s, "채택한 CNN 하나로 끝내지 않고 여러 방향을 검증 — 효과 없거나 규칙에 어긋난 조합은 이유와 함께 기록");
  card(s, M, 2.45, CW, 3.65);
  table(s, [
    ["접근", "무엇을 시도했나", "결과", "상태"],
    ["통화 단위 특징", "같은 통화 내 다른 발화와 비교한 통계(상대 길이 등)", "규칙 위반 소지 확인", { text: "제거", options: { bold: true, color: C.red } }],
    ["L1 + L2 블렌딩", "통계 모델(GBM)과 CNN 예측을 평균", "단독 CNN(L2)보다 낮음", { text: "기각", options: { bold: true, color: C.red } }],
    ["CNN 시드 앙상블", "같은 구조를 다른 seed로 여러 번 학습해 평균", "소규모 검증 통과", { text: "확장 검토", options: { bold: true, color: C.blue } }],
    ["CNN 다중 모델 앙상블", "구조가 다른 CNN 2개 추가 학습", ph("결과 대기"), { text: "보류 (GPU 대기)", options: { bold: true, color: C.sub } }],
    ["사전학습 화자 임베딩", "공개 화자 인식 모델 fine-tuning 파일럿", ph("결과 반영 예정"), { text: "진행 중", options: { bold: true, color: C.sub } }],
  ], M + 0.2, 2.65, [2.3, 5.0, 3.0, 1.9], { rowH: 0.52 });
  insight(s, M, 6.3, CW, 0.65, [
    { text: "왜 이 표를 넣나  ", options: { bold: true, color: C.blue } },
    { text: "서면 심사라 \"더 해본 건 없나\"에 답할 기회가 따로 없음 — 시도와 기각 이유를 미리 남겨 둠" },
  ]);
  caption(s, "근거: docs/decisions.md (M2-ENS-001 2026-09-28 기각), docs/experiment_log.md, colab_scripts/ 파일럿 스크립트");
  s.addNotes("표의 '결과'·'상태' 열은 decisions.md 최신 항목으로 계속 갱신. ECAPA 파일럿 결과가 나오면 바로 반영.");
}

// ---------- Mission 3 ----------
missionA(14, 3, "환자 증상 인식",
  "증상 단어가 대화에 직접 등장하는 경우가 많음 — 단어 기반 모델로 출발해 문맥 모델로 확장",
  [["입력", "신고자·대원 대화 전사 텍스트\n(그 외 라벨 사용 불가)"], ["출력", "증상 9종 중 0~9개\n(다중 레이블)"], ["평가", "Macro F1\n(증상별 F1의 평균)"], ["대상", "고열·구토·두통·복통·어지러움·열상·오심·전신쇠약·호흡곤란"]],
  [["전처리", "발화 시간순 정렬\n9종 외 증상 제거"], ["텍스트 표현", "TF-IDF\n(단어·2단어 묶음)"], ["분류 모델", "증상별 로지스틱\n회귀 (9개)"], ["판정", "증상별 판정\n(기준값 0.5)"]],
  ["증상 이름·표현(\"머리가 아파요\")이 전사에 직접 드러나 단어 기반이 강한 출발점",
   "희귀 증상이 많아 Macro F1에 불리 → 증상별 성능을 따로 점검",
   [ph("KcBERT 등 문맥 모델 결과 반영 시 갱신")]]);
{
  const s = missionB(15, 3, "환자 증상 인식", [ph("최종 모델과 Macro F1 한 문장", { color: C.sub })]);
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
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 확장 계획과 검증 일정", 16);
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
    { text: "문맥 모델이 TF-IDF보다 뚜렷이 낫지 않으면, 단순하고 설명 가능한 TF-IDF+LR을 최종안으로 유지" },
  ], C.mintLight);
  caption(s, "근거: docs/decisions.md 2026-09-20 분할 감사 항목, docs/experiment_log.md");
  s.addNotes("본선 전까지 KcBERT 등 결과가 나오면 M3-B(15p)의 수치를 갱신하고, 이 장은 '실제로 더 해본 것'으로 내용 교체.");
}

// =====================================================================
// 17. 04-(1) 성능 요약과 교훈
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(1) 성능 요약과 교훈", 17);
  coreMsg(s, "세 미션 모두 공식 Validation 기준으로 평가하고, 점수보다 검증의 정직함을 우선함");
  tag(s, W - M - 1.7, 0.95);
  table(s, [
    ["미션", "과제", "최종 모델", "지표", "공식 Validation", "상태"],
    ["Mission 1", "신고자 성별 분류", "음향 특징 + LR", "Accuracy", { text: "95.11%", options: { bold: true, color: C.blue } }, "1차 확정"],
    ["Mission 2", "신고자/119대원 분류", "Mel-spectrogram + CNN", "Accuracy", { text: "90.89%", options: { bold: true, color: C.blue } }, "개선 중"],
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
// 18. 04-(2) 사회안전 시사점과 향후 계획
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "04", "종합 및 시사점", "(2) 사회안전 시사점과 향후 계획", 18);
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
// 19. 마무리 — 공식 템플릿 그대로 사용 + 팀 정보만 빈 여백(2.1~2.9in)에 추가
// =====================================================================
{
  const s = pres.addSlide();
  s.addImage({ path: TPL.closing, x: 0, y: 0, w: W, h: 7.5, sizing: { type: "cover", w: W, h: 7.5 } });
  txt(s, "팀 eigensinn  |  김장수 · 김광희 · 지연우 · 이수빈 · 조지은", { x: M, y: 2.2, w: CW, h: 0.4, fontSize: 16, bold: true, color: C.blue, align: "center" });
  txt(s, "코드 · 실험 기록: GitHub eigensinn-dcc2026/prelim  (docs/experiment_log.md, docs/decisions.md)", { x: M, y: 2.58, w: CW, h: 0.3, fontSize: 11, color: C.sub, align: "center" });
  s.addNotes("공식 템플릿(슬라이드4) 그대로. 흰 여백(약 2.1~2.9in)에만 팀 정보를 얹음 — 배너 그래픽은 건드리지 않음.");
}

pres.writeFile({ fileName: process.argv[2] || "DCC2026_예선_결과정리_뼈대.pptx" }).then(f => console.log("wrote", f));
