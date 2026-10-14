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
  // 그래프 색: 모델 비교 = 제출 빨강 + 나머지 파랑 / 분석 그래프 = 파랑·주황(EDA 그림과 동일)
  hi: "E53935", mid: "009DDA", lo: "B4D2EC",
  // 분석 그래프 색: EDA 그림과 같은 파랑·주황 (an2t = 주황 글자용 진한 색)
  an1: "0072B2", an2: "E69F00", an2t: "C77C00",
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
  // 11장 문제·제약·최종 모델
  core12: "발화 하나의 소리만 쓰는 규칙 안에서, 처음부터 배운 CNN과 사전학습 음성 모델 WavLM을 섞어 92.90%.",
  // 구조도 (flow 함수 입력): 앞 단계 → 앙상블 멤버 → 판정
  flow: {
    pre: [["발화 구간만 입력", "startAt~endAt\ntext·순서·다른 발화 X"]], preW: 2.5,
    branches: [["CNN 3개 평균", "log-mel · 300프레임", "× 0.4"], ["WavLM-large 2개", "16 kHz · 3초 창", "× 0.3 + 0.3"]], brW: 2.6, brGap: 0.1,
    post: [["가중 평균", "확률 × 비중의 합\n≥ 0.5 → 신고자\n< 0.5 → 대원"]], postW: 2.0,
  },
  why: [
    ["음향 차이", "대원 헤드셋·정해진 말투\n신고자 휴대전화·잡음"],
    ["CNN", "log-mel을 처음부터 학습\n강의 교안의 Log-Mel → CNN"],
    ["WavLM", "공개 사전학습 모델 미세조정\n교안의 wav2vec 2.0 계열 후속 모델"],
  ],
  // 12장 성능 개선 과정 (막대 차트: 공식 Validation Accuracy)
  core13: "세 번의 전환점: CNN 도입(+5.05%p) → 제출 방식 차이 보정 → 사전학습 WavLM 결합(+1.41%p).",
  chart: { labels: ["통계 모델\n(LightGBM)", "CNN 3개 앙상블\n(처음부터 학습)", "CNN 3개 + WavLM 2개\n(사전학습 미세조정)"], values: [86.44, 91.49, 92.90] },
  gainTitle: "■ 전환점 ①  CNN의 이득이 몰린 곳",
  gains: [["약 +7%p", "겹친 구간"], ["약 +2%p", "겹치지 않은 구간"]],
  gainNote: "통계 모델 대비 CNN 단일 모델의 개선 폭 (공식 Validation)",
  turn2: "한 발화씩 추론하자 정확도가 떨어졌다(학습 때와 채우기 방식 차이). 발화마다 자기 자신만으로 같은 길이로 채워 회복.",
  turn3: "사전학습 WavLM-large는 단독으로도 CNN보다 높았고(92.40·92.71%), 둘을 섞어 92.90%.",
  // 13장 다른 접근 — 마지막 행이 최종 채택한 WavLM-large 미세조정 + CNN 결합
  core14: "규칙에 어긋날 소지가 있으면 성능이 좋아도 뺐고, 효과가 확인된 것만 남겼다.",
  lastRow: ["WavLM-large 미세조정 2개 + CNN 결합", "✓ 채택", "large·전체 데이터로 단독도 CNN 초과 → 결합 92.90%"],
  // 14장 최종 결과와 보완할 점
  core15: "92.90%. 남은 오류는 \"네\"·\"여보세요\" 같은 짧은 응답에 몰려 있다 — 발화 하나만 보는 규칙의 한계.",
  acc: "92.90%",
  recalls: [["88.36%", "대원 재현율"], ["97.10%", "신고자 재현율"]],
  repro: "파라미터 6.3억 개 · GPU(A100) 1장\n공식 Validation 전체 12.1분 (발화당 6.5ms)\n제출 폴더 그대로 재실행해도 판정 차이 0건",
  improve: "대원 재현율(88.36%)이 신고자(97.10%)보다 낮다.\n남은 오답의 약 78%는 모든 모델이 함께 틀리고,\n대부분 짧은 응답이다.\n규칙상 통화 맥락을 쓸 수 없어,\n소리만으로는 두 화자가 비슷하게 들린다.",
  // 18장 성능 요약 표, 19장 향후 계획
  summary: { task: "M2 신고자/대원", model: "CNN × 3 + WavLM-large × 2", val: "92.90%", gain: "+6.46%p" },
  future: "짧은 응답(\"네\"·\"예\") 보강 — 오답의 약 78%가 모든 모델 공통",
  refs: "대학부 강의 교안 오디오 단원(p.176~182), WavLM(Chen et al., 2022)",
  evidence: "submission/mission2/README.md, docs/experiment_log.md M2-BLEND-001·M2-WLML-001·002·M2-RERUN-001·M2-RETRAIN-001",
  // 슬라이드 노트·캡션의 근거 (12·14장)
  evidence13: "근거: docs/experiment_log.md M2-RETRAIN-001, M2-RERUN-001, M2-BLEND-001, M2-COMPARE-002, M2-PAD-001·002",
  notes13: "근거: L1 86.44%(M2-RETRAIN-001, 제공 라벨 재학습 L1) / CNN 단일 90.98%(M2-ENS-003 ts1) / 3-seed 앙상블 91.49%(M2-RERUN-001) / CNN 3개 + WavLM-large 2개 92.90%(M2-BLEND-001, submission/mission2/README.md). 겹침 구간: M2-COMPARE-002 — 겹침 L1 81.83%→L2 88.88%(+7.05%p), 안 겹침 90.01%→92.44%(+2.43%p), L2 단일 모델 기준. 패딩: M2-PAD-001(발화별 300프레임 채우기), M2-VAL-005, M2-PAD-002.",
  notes15: "근거: submission/mission2/README.md '결과와 계산 효율성' 표(92.90% = 103,994/111,947, 대원 recall 88.36%, 신고자 recall 97.10%, 파라미터 633,220,220, A100 추론 12.1분·발화당 6.5ms). 공통 오답 약 78%: M2-HUBERT-001 분석. 검증: submission/mission2/verify_mission2_final_on_gpu.ipynb(제출 폴더 그대로 Colab A100, requirements만·인터넷 차단, 코드 정리·속도 개선 전후 판정 동일), PR #216·#222. 비율(CNN 0.4 / WavLM 0.3·0.3)은 Training 내부 20,000발화로 고름(공식 Val로 고르지 않음).",
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
// 모델 비교 그래프 색: 제출 모델(winnerIndex)만 빨강, 나머지는 파랑. 분석 그래프는 빨강을 쓰지 않는다.
function winner(n, winnerIndex) {
  return Array.from({ length: n }, (_, i) => (i === winnerIndex ? C.hi : C.mid));
}
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
    // 비중(가중치)은 모델 상자에서 최종 상자로 가는 연결선 한가운데에 알약으로 얹어, 모델의 속성이 아니라 합칠 때 곱하는 값임을 분명히 한다
    if (label) {
      const pw = Math.max(0.62, 0.085 * label.length + 0.22), cx0 = bx + f.brW + gap * 0.4, cy0 = bm + (mid - bm) * 0.4;
      slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx0 - pw / 2, y: cy0 - 0.15, w: pw, h: 0.3, fill: { color: C.white }, rectRadius: 0.15, line: { color: C.blue, width: 1 } });
      txt(slide, label, { x: cx0 - pw / 2, y: cy0 - 0.15, w: pw, h: 0.3, fontSize: 10.5, bold: true, color: C.blue, align: "center", valign: "middle" });
    }
  });
  cx = px;
  post.forEach(([t, b], i) => {
    node(slide, cx, y, f.postW, h, t, b, i === post.length - 1);
    if (i > 0) link(slide, cx - gap, mid, cx, mid);
    cx += f.postW + gap;
  });
}

// ======================================================================
// Mission 2 (김광희 담당) 추가 슬라이드: 모델 선택 근거 + 별첨 3장
// 숫자 출처: submission/mission2/README.md, docs/experiment_log.md (M2-RETRAIN-001, M2-RERUN-001, M2-COMPARE-002,
// M2-PAD-001, M2-WLML-001·002, M2-HUBERT-001, M2-BLEND-001), 발화별 확률 파일(내부 20,000발화·공식 Validation)로 재계산
// ======================================================================

// ---------- 본문: 모델 선택 근거 ----------
function m2ReasonSlide() {
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 모델 선택 근거", 0);
  coreMsg(s, "통계 요약 → 스펙트로그램 CNN → 사전학습 음성 모델. 교안 오디오 단원의 순서를 그대로 밟았다.");

  const colW = 3.95, gap = (CW - 3 * colW) / 2, top = 2.45, h = 3.3;
  const cols = [
    {
      head: "1단계  통계 특징 + 트리 모델(LightGBM)", color: C.line, textColor: C.text,
      rows: [
        ["입력", "발화를 숫자 53개로 요약\n(MFCC·에너지·스펙트럼 통계)"],
        ["교안", "p.173 Mel-spectrogram · MFCC"],
        ["한계", "시간에 따른 소리 변화가 사라짐\n목소리가 겹친 구간 81.83%"],
      ],
      result: "86.44%",
    },
    {
      head: "2단계  log-mel CNN", color: C.sky, textColor: C.white,
      rows: [
        ["입력", "발화의 log-mel 스펙트로그램\n(40 mel × 시간, 그 발화만)"],
        ["교안", "p.176~179 Log-Mel → CNN"],
        ["한계", "89만 발화로 처음부터 학습\n0.5~1초 발화 정확도 87.60%"],
      ],
      result: "91.49%",
    },
    {
      head: "3단계  사전학습 WavLM-large", color: C.blue, textColor: C.white,
      rows: [
        ["입력", "발화 파형 3초 (16 kHz)\n공개 음성 9.4만 시간으로 사전학습"],
        ["교안", "p.182 wav2vec 2.0: 사전학습 → 미세조정\nWavLM: 화자 과제까지 학습한 후속작"],
        ["효과", "단독 92.40·92.71% (CNN보다 높음)\n짧은 발화에서 개선 폭이 가장 큼"],
      ],
      result: "92.90%",
    },
  ];
  cols.forEach((c, i) => {
    const x = M + i * (colW + gap);
    card(s, x, top, colW, h);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: top, w: colW, h: 0.48, fill: { color: c.color }, rectRadius: 0.1, line: { color: c.color } });
    s.addShape(pres.shapes.RECTANGLE, { x, y: top + 0.28, w: colW, h: 0.2, fill: { color: c.color }, line: { color: c.color } });
    txt(s, c.head, { x, y: top, w: colW, h: 0.48, fontSize: 13.5, bold: true, color: c.textColor, align: "center", valign: "middle" });
    c.rows.forEach(([k, v], r) => {
      const y = top + 0.62 + r * 0.72;
      txt(s, k, { x: x + 0.18, y, w: 0.6, h: 0.6, fontSize: 11.5, bold: true, color: C.blue, valign: "middle" });
      txt(s, v, { x: x + 0.8, y, w: colW - 0.95, h: 0.6, fontSize: 11, color: C.text, valign: "middle" });
      s.addShape(pres.shapes.LINE, { x: x + 0.18, y: y + 0.66, w: colW - 0.36, h: 0, line: { color: C.line, width: 0.6 } });
    });
    txt(s, [{ text: c.result, options: { fontSize: 24, bold: true, color: i === 0 ? C.sub : C.blue } },
            { text: i === 2 ? "  CNN과 결합" : "  공식 Validation", options: { fontSize: 11, color: C.sub } }],
        { x, y: top + 2.8, w: colW, h: 0.45, align: "center", valign: "middle" });
    if (i < 2) arrow(s, x + colW + (gap - 0.24) / 2, top + h / 2 - 0.19, 0.24, 0.38);
  });

  insight(s, M, 5.95, CW, 0.88, [
    { text: "출발점이 된 관찰  ", options: { bold: true, color: C.blue } },
    { text: "대원은 헤드셋으로 정해진 말을, 신고자는 휴대전화로 주변 잡음 속에서 말한다. 말의 내용이 아니라 소리의 질감 차이라서, 소리를 직접 보는 모델로 옮겨 갔다." },
  ], C.mintLight, 12);
  caption(s, "참고: 대학부 강의 교안 오디오 단원(p.173, p.176~182), Chen et al., WavLM (IEEE JSTSP, 2022).  근거: docs/experiment_log.md M2-RETRAIN-001·RERUN-001·COMPARE-002·WLML-001·002", 7.02);
  s.addNotes("단계별 공식 Validation: L1 86.44%(M2-RETRAIN-001), CNN 3-seed 91.49%(M2-RERUN-001), CNN+WavLM-large 2개 92.90%(M2-BLEND-001, 제출 코드 92.8957%). 겹친 구간 L1 81.83%(M2-COMPARE-002). CNN 0.5~1초 정확도 87.60%(공식 Validation 발화별 확률 재계산). WavLM-large 단독 92.40%/92.71%(M2-WLML-001/002). 사전학습 데이터 9.4만 시간(Libri-Light·GigaSpeech·VoxPopuli, WavLM 논문). 교안 p.176 학습목표: Log-Mel spectrogram이 CNN에 입력되는 과정, Wav2Vec2·AST 구조. p.182: wav2vec 2.0은 대규모 음성으로 표현을 먼저 학습한 뒤 새 음성 문제에 Fine-tuning.");
}

// ---------- 별첨 1: 모델 구조와 판정식 ----------
function m2Appendix1() {
  const s = pres.addSlide();
  header(s, "별첨", "Mission 2", "별첨 1. Mission 2 모델 구조와 판정식", 0);
  coreMsg(s, "두 모델 모두 발화 하나만 입력으로 받고, 확률을 섞은 뒤 0.5 한 번으로 판정한다.");

  const topY = 2.36, topH = 2.82, gp = 0.25, leftW = 6.0;
  const rightX = M + leftW + gp, rightW = CW - leftW - gp;
  card(s, M, topY, leftW, topH);
  card(s, rightX, topY, rightW, topH);
  title(s, "■ 입력 조건 → 설계 결정", M + 0.2, topY + 0.11, leftW - 0.4, 15);
  const obs = [
    ["발화 하나", "text·순서·같은 통화의 다른 발화 사용 불가", "그 발화 구간(startAt~endAt)의 소리만 입력"],
    ["짧은 발화", "학습은 배치 최장 길이로 채워짐 · 추론은 한 발화씩", "모든 발화를 그 발화만으로 300프레임(6초)까지 채움"],
    ["길이 차이", "0.1초 응답부터 18초 설명까지", "WavLM은 가운데 3초, 짧으면 그 발화를 반복"],
  ];
  obs.forEach(([name, observed, decision], i) => {
    const y = topY + 0.53 + i * 0.73;
    txt(s, name, { x: M + 0.23, y, w: 1.15, h: 0.25, fontSize: 11.3, bold: true, color: C.blue });
    txt(s, observed, { x: M + 1.43, y, w: 4.32, h: 0.25, fontSize: 10.8, color: C.text });
    txt(s, "→ " + decision, { x: M + 1.43, y: y + 0.29, w: 4.32, h: 0.3, fontSize: 10.6, bold: true, color: C.blue });
    if (i < 2) s.addShape(pres.shapes.LINE, { x: M + 0.22, y: y + 0.65, w: leftW - 0.44, h: 0, line: { color: C.line, width: 0.6 } });
  });

  title(s, "■ 두 모델의 구조", rightX + 0.2, topY + 0.11, rightW - 0.4, 15);
  const models = [
    ["CNN × 3 (시작값 1·2·3)", "log-mel(40 mel · 8 kHz) → 합성곱 4단(채널 32→256) → 평균 풀링 → 2클래스", "42만 파라미터 · Adam 1e-3 · 배치 64 · 조기 종료 · T4"],
    ["WavLM-large × 2 (시작값 1·2)", "파형 3초(16 kHz) → WavLM 24층 → 층 가중합 → 평균·표준편차 → MLP → 2클래스", "3.2억 파라미터 · AdamW 2e-5 · 42,500 step · A100 약 2.5시간"],
  ];
  models.forEach(([name, arch, train], i) => {
    const y = topY + 0.52 + i * 1.12;
    txt(s, name, { x: rightX + 0.22, y, w: rightW - 0.44, h: 0.24, fontSize: 11.3, bold: true, color: C.blue });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rightX + 0.18, y: y + 0.27, w: rightW - 0.36, h: 0.5, fill: { color: C.light }, rectRadius: 0.06, line: { color: C.line, width: 0.6 } });
    txt(s, arch, { x: rightX + 0.28, y: y + 0.27, w: rightW - 0.56, h: 0.5, fontSize: 10.4, color: C.text, valign: "middle" });
    txt(s, train, { x: rightX + 0.22, y: y + 0.8, w: rightW - 0.44, h: 0.2, fontSize: 9.6, color: C.sub });
  });

  const bottomY = 5.31, bottomH = 1.55;
  card(s, M, bottomY, CW, bottomH);
  title(s, "■ 최종 확률과 판정", M + 0.21, bottomY + 0.08, CW - 0.42, 15);
  const boxY = bottomY + 0.44, boxH = 0.77, bw = 3.75, arrowW = 0.25;
  [
    ["STEP 1 · CNN 3개 평균", "P_CNN = (p₁ + p₂ + p₃) / 3"],
    ["STEP 2 · 가중 평균", "p = 0.4·P_CNN + 0.3·P_W1 + 0.3·P_W2"],
    ["STEP 3 · 최종 판정", "신고자(1) if p ≥ 0.5"],
  ].forEach(([lab, eq], i) => {
    const x = M + 0.2 + i * (bw + arrowW + 0.1);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: boxY, w: bw, h: boxH, fill: { color: i === 1 ? C.light : C.white }, rectRadius: 0.06, line: { color: i === 1 ? C.sky : C.line, width: 1 } });
    txt(s, lab, { x: x + 0.12, y: boxY + 0.11, w: bw - 0.24, h: 0.2, fontSize: 10.2, color: C.sub, bold: true, align: "center" });
    txt(s, eq, { x: x + 0.07, y: boxY + 0.37, w: bw - 0.14, h: 0.29, fontSize: i === 1 ? 13 : 14, color: C.blue, bold: true, align: "center" });
    if (i < 2) txt(s, "→", { x: x + bw + 0.015, y: boxY + 0.25, w: arrowW, h: 0.3, fontSize: 16, bold: true, color: C.sky, align: "center" });
  });
  txt(s, "판정 기준 0.5 고정 · 비율은 Training 내부 20,000발화로 선택 · 여러 발화를 묶어 계산해도 결과는 한 발화씩과 같음", { x: M + 0.2, y: bottomY + 1.27, w: CW - 0.4, h: 0.2, fontSize: 10.1, bold: true, color: C.blue, align: "center" });
  caption(s, "※ 모든 가중치는 체크포인트 하나(6.3억 파라미터)에 들어 있어 채점 때 아무것도 내려받지 않음. 근거: submission/mission2/README.md", 7.02);
  s.addNotes("CNN: SpeakerCNNLarge 421,954 파라미터 × 3, log-mel 40 mel · FFT 512 · hop 160(8 kHz), 짧은 발화는 바닥값(-80 dB)으로 300프레임까지 채움(M2-PAD-001, 내부 89.58%→90.84%). WavLM-large: 315,977,179 파라미터 × 2, 파형 CNN만 고정, AdamW 본체 2e-5 / head 3e-4, 웜업 500 + 코사인, 배치 32, 42,500 step, best step 선택(seed1 15,000 / seed2 20,000). 합계 633,220,220 파라미터. 묶음 vs 한 발화씩 확률 차이: WavLM 9e-9, CNN 4.8e-7. 근거: submission/mission2/README.md, docs/experiment_log.md M2-WLML-001·002, M2-BLEND-001.");
}

// ---------- 별첨 2: 섞는 비율과 모델 조합 ----------
function m2Appendix2() {
  const s = pres.addSlide();
  header(s, "별첨", "Mission 2", "별첨 2. 섞는 비율과 모델 조합", 0);
  coreMsg(s, "비율은 Training 내부 20,000발화로만 골랐고, 공식 Validation은 고른 뒤 한 번 확인했다.");

  const lw = 6.2;
  card(s, M, 2.45, lw, 4.35);
  barChart(s, ["CNN\n3개", "WavLM\n시작값 1", "WavLM\n시작값 2", "WavLM\n2개", "CNN +\nWavLM 2개"], [91.94, 92.73, 92.99, 93.04, 93.21], {
    x: M + 0.15, y: 2.55, w: lw - 0.3, h: 4.15, chartColors: winner(5, 4),
    title: "Training 내부 20,000발화 정확도 (%)", valAxisMinVal: 91, valAxisMaxVal: 93.5,
  });

  const rx = M + lw + 0.3, rw = W - M - rx;
  table(s, [
    ["조합", "내부 20,000", "공식 Validation"],
    ["CNN 3개", "91.94%", "91.49%"],
    ["WavLM 시작값 1 / 시작값 2", "92.73 / 92.99%", "92.40 / 92.71%"],
    ["WavLM 2개 평균", "93.04%", "92.74%"],
    [strong("CNN 0.4 + WavLM 0.3·0.3", C.text), strong("93.21%"), strong("92.90%")],
    ["+ HuBERT-large (1/4씩)", "93.16%", "92.82%"],
  ], rx, 2.45, [2.85, 1.55, rw - 4.4], { rowH: 0.42, fontSize: 11.5 });

  insight(s, rx, 5.15, rw, 1.65, [
    { text: "비율  ", options: { bold: true, color: C.blue } },
    { text: "0.1 간격 66개 조합 중 7개가 최고와 0.05%p 이내. 비율에 민감하지 않다.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 5, breakLine: true } },
    { text: "더 섞기  ", options: { bold: true, color: C.blue } },
    { text: "HuBERT를 더해도 오르지 않았다. 오답의 78%를 세 모델이 함께 틀린다." },
  ], C.mintLight, 11.5);
  caption(s, "※ 공식 Validation 열은 확인용이며 조합·비율 선택에 쓰지 않음. 최종 행 92.90%는 제출 코드로 잰 값(92.8957%).  근거: docs/experiment_log.md M2-BLEND-001·M2-HUBERT-001", 7.02);
  s.addNotes("내부 20,000발화: CNN·WavLM과 같은 통화 단위 80/20 분할(split seed 42)의 떼어 둔 5,840통화에서 뽑은 고정 표본. 0.1 간격 66개 조합: 최고 0.4/0.3/0.3 93.21%, 0.4/0.2/0.4 93.205%, 1/3씩 93.155%. HuBERT-large(facebook/hubert-large-ll60k) 단독 내부 92.15% / 공식 91.88%, 1/4씩 결합 내부 93.155% / 공식 92.82%. 공통 오답: 공식 Validation 최종 오답 7,955개 중 CNN·WavLM1·WavLM2가 모두 틀린 발화 6,207개(78.0%). 공식 Val 열은 발화별 확률로 재계산한 값(최종 행 연구용 계산 92.894%, 제출 코드 92.8957%).");
}

// ---------- 별첨 3: 발화 길이별 오류 분석 ----------
function m2Appendix3() {
  const s = pres.addSlide();
  header(s, "별첨", "Mission 2", "별첨 3. Mission 2 발화 길이별 오류 분석", 0);
  coreMsg(s, "2초 미만 발화가 전체의 65%인데 오답의 84%를 차지한다. WavLM의 효과도 짧은 발화에서 가장 컸다.");

  const lw = 6.6;
  card(s, M, 2.45, lw, 4.35);
  const labels = ["0.5초 미만", "0.5~1초", "1~2초", "2~3초", "3초 이상"];
  s.addChart(pres.charts.BAR, [
    { name: "CNN 3개", labels, values: [90.95, 87.60, 89.71, 94.21, 97.49] },
    { name: "CNN + WavLM 2개(최종)", labels, values: [93.20, 89.71, 91.16, 95.31, 97.84] },
  ], {
    x: M + 0.15, y: 2.55, w: lw - 0.3, h: 4.15, barDir: "col", barGrouping: "clustered", chartColors: [C.an1, C.an2],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9.5, dataLabelFormatCode: "0.0",
    valAxisMinVal: 84, valAxisMaxVal: 100, valAxisLabelColor: C.sub, catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 11,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 10.5, legendFontFace: F,
    showTitle: true, title: "발화 길이별 공식 Validation 정확도 (%)", titleFontSize: 12, titleColor: C.text, titleFontFace: F,
  });

  const rx = M + lw + 0.3, rw = W - M - rx;
  table(s, [
    ["발화 길이", "발화 비율", "정확도", "오답 비율"],
    ["0.5초 미만", "8.0%", "93.20%", "7.7%"],
    [strong("0.5~1초", C.blue), "24.5%", strong("89.71%", C.blue), strong("35.4%", C.blue)],
    [strong("1~2초", C.blue), "32.9%", strong("91.16%", C.blue), strong("41.0%", C.blue)],
    ["2~3초", "15.2%", "95.31%", "10.0%"],
    ["3초 이상", "19.4%", "97.84%", "5.9%"],
  ], rx, 2.45, [1.5, 1.3, 1.25, rw - 4.05], { rowH: 0.4, fontSize: 11.5 });

  insight(s, rx, 5.0, rw, 1.8, [
    { text: "WavLM의 효과  ", options: { bold: true, color: C.blue } },
    { text: "CNN 대비 0.5초 미만 +2.3%p, 3초 이상 +0.4%p. 짧을수록 사전학습이 도움이 됐다.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 5, breakLine: true } },
    { text: "남은 과제  ", options: { bold: true, color: C.blue } },
    { text: "\"네\", \"여보세요\" 같은 0.5~2초 응답이 오답의 76%를 차지한다." },
  ], C.mintLight, 11.5);
  caption(s, "※ 공식 Validation 111,947발화, 발화 길이 = endAt − startAt. 발화별 확률로 계산한 값(전체 92.89%, 제출 코드 92.90%).  근거: docs/experiment_log.md M2-BLEND-001", 7.02);
  s.addNotes("공식 Validation 발화별 확률(CNN 3-seed, WavLM seed1·2)로 재계산. 길이 구간별 발화 수: 0.5초 미만 8,954 / 0.5~1초 27,392 / 1~2초 36,858 / 2~3초 17,028 / 3초 이상 21,715. 최종(0.4/0.3/0.3) 정확도 93.20/89.71/91.16/95.31/97.84%, CNN 단독 90.95/87.60/89.71/94.21/97.49%. 2초 미만 발화는 전체의 65.4%, 최종 오답 7,955개 중 84.1%. 0.5~2초가 오답의 76.4%(35.4% + 41.0%). Training 내부 20,000발화에서도 같은 모양(0.5~1초 89.63%, 3초 이상 98.72%).");
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
    ["03", "미션별 수행", ["Mission 1  신고자 성별 분류", "Mission 2  신고자/대원 분류\n   (+ 별첨 3장·음향 특징 비교 실험)", "Mission 3  환자 증상 인식"]],
    ["04", "종합 및 시사점", ["성능 요약과 교훈", "활용 흐름과 기대효과", "사회안전 시사점과\n   향후 계획", "별첨 · Mission 3 상세 근거"]],
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
        name: "%", x: x + 0.25, y: 2.9, w: cw - 0.5, h: 3.21, chartColors: [C.an1, C.an2], title: "Training 통화 기준 (%)", titleFontSize: 14,
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
      x: M + 0.15, y: 2.55, w: 7.0, h: 4.15, chartColors: winner(4, 3),
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
  coreMsg(s, "점수가 조금 올라도 미리 정해 둔 목표를 넘지 못하면 채택하지 않았다.");
  table(s, [
    ["시도", "판정", "이유"],
    ["트리 기반 모델 (LightGBM)", strong("✓ 유지"), "점수 차이가 거의 없고, 로지스틱 회귀가 더 단순해서 설명하기 쉬움"],
    ["비선형 분류기 (RBF-SVM)", strong("✗ 기각", C.red), "정확도 향상 +0.33%p로 목표(+0.4%p)에 못 미치고, 예측이 훨씬 느림"],
    ["다른 음성 특징 모델 (ECAPA) 추가·교체", strong("✗ 기각", C.red), "향상 폭이 목표에 못 미침 (교체 비교 +0.28%p, 목표 +0.3%p)"],
    ["두 모델의 예측 확률 평균 (x-vector 모델 + ECAPA 모델)", strong("✗ 기각", C.red), "공식 Validation 98.30%로 조금 높았지만(+6통화) 목표에 못 미치고, 우연과 구분되지 않음"],
  ], M, 2.45, [3.9, 1.5, 6.93], { rowH: 0.62, fontSize: 13 });
  insight(s, M, 5.75, CW, 1.05, [
    { text: "공식 Validation에서 더 높았던 모델을 왜 버렸나?  ", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "몇 통화 차이는 우연과 구분되지 않는다. 결과를 본 뒤 기준을 낮추면 공식 Validation에만 잘 맞는 모델이 되어(과적합) 실제 성능을 믿을 수 없다." },
  ], C.light, 13.5);
  caption(s, "근거: docs/decisions.md 2026-10-05, docs/experiment_log.md M1-HEAD-001, M1-ECAPA-002, M1-FUSE-001-OFFVAL");
  s.addNotes("근거: LightGBM vs LR — docs/decisions.md 2026-10-05 / RBF-SVM — M1-HEAD-001 / ECAPA — M1-ECAPA-001(3차)·M1-ECAPA-002 / 확률 평균 — M1-FUSE-001, M1-FUSE-001-OFFVAL(98.30%, 사전 고정 기준 98.434% 미달, McNemar p=0.345).");
}

// 11. 최종 결과
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 1  신고자 성별 분류 — 최종 결과", 11);
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
  title(s, "■ 오분류는 어느 한쪽으로 쏠리지 않았다", rx + 0.25, 2.62, rw - 0.5, 18);
  // 혼동행렬. 값은 공식 Validation 3,640건(여성 1,959 / 남성 1,681)에 recall을 곱해 복원한 건수.
  const mx = rx + 0.25, lblW = 1.2, cw2 = 2.25, cg = 0.1, ch = 1.1;
  const cellX = [mx + lblW + cg, mx + lblW + cg + cw2 + cg];
  ["예측: 여성", "예측: 남성"].forEach((t, j) => txt(s, t, { x: cellX[j], y: 3.2, w: cw2, h: 0.3, fontSize: 12.5, bold: true, color: C.sub, align: "center", valign: "middle" }));
  const rows2 = [
    ["실제 여성", [["1,925", "정답", true], ["34", "오분류", false]]],
    ["실제 남성", [["35", "오분류", false], ["1,646", "정답", true]]],
  ];
  rows2.forEach(([lab, cells], i) => {
    const y = 3.55 + i * (ch + cg);
    txt(s, lab, { x: mx, y, w: lblW, h: ch, fontSize: 12.5, bold: true, color: C.sub, align: "center", valign: "middle" });
    cells.forEach(([n, note, ok], j) => {
      const fill = ok ? C.light : C.pink, edge = ok ? C.sky : C.pink;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cellX[j], y, w: cw2, h: ch, fill: { color: fill }, rectRadius: 0.08, line: { color: edge, width: 1 } });
      txt(s, [
        { text: n, options: { fontSize: 30, bold: true, color: ok ? C.blue : C.red, breakLine: true } },
        { text: note, options: { fontSize: 11.5, color: C.sub } },
      ], { x: cellX[j], y, w: cw2, h: ch, align: "center", valign: "middle" });
    });
  });
  s.addShape(pres.shapes.LINE, { x: mx, y: 6.0, w: rw - 0.5, h: 0, line: { color: C.line, width: 1 } });
  txt(s, [
    { text: "3,640건 중 69건 오류 ", options: { bold: true, color: C.blue } },
    { text: "· 여성→남성 34건, 남성→여성 35건으로 거의 같다" },
  ], { x: mx, y: 6.12, w: rw - 0.5, h: 0.5, fontSize: 12.5, valign: "middle" });
  caption(s, "근거: submission/mission1/README.md 상태(2026-10-12), docs/experiment_log.md M1-REFIT-OFFVAL-001");
  s.addNotes("근거: submission/mission1/README.md — 공식 Validation 3,640건 98.10%(3,571건), 여성 recall 98.26%, 남성 recall 97.92%. docs/experiment_log.md M1-REFIT-OFFVAL-001(단일 seed). 혼동행렬 건수는 공식 Validation 클래스 수(여성 1,959 / 남성 1,681, reports/mission1/m1_val_eval.json)에 recall을 곱해 복원: 여성 정답 1,925·오분류 34, 남성 정답 1,646·오분류 35 (합 3,571 = 정답 수와 일치).");
}

// ======================================================================
// Mission 2 — 신고자/119대원 분류 (12~15장, 수치·모델명은 상단 M2 상수)
// ======================================================================

// 12. 문제·제약·최종 모델 (광희)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 문제·제약·최종 모델", 11);
  coreMsg(s, M2.core12);
  const lw = 8.9;
  txt(s, [strong("■ 문제   "), { text: "한 발화 조각 → 신고자(1) / 대원(0)   ·   지표 정확도(Accuracy)" }], { x: M, y: 2.5, w: lw, h: 0.4, fontSize: 15 });
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
  caption(s, "※ 왼쪽 도식은 합성 신호.  참고: " + M2.refs + ".  근거: submission/mission2/README.md");
  s.addNotes("근거: " + M2.evidence + ". 제약은 출제문제·주최측 Q&A(2026.09.15), 추론 방식은 submission/mission2/README.md '추론' 절. 참고: 대학부 강의 교안 오디오 단원 p.176(학습목표: Log-Mel spectrogram이 CNN에 입력되는 과정, Wav2Vec2·AST 구조), p.182(wav2vec 2.0: 대규모 음성으로 표현을 먼저 학습한 뒤 새 문제에 Fine-tuning). WavLM: Chen et al., 'WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing', IEEE JSTSP 2022.");
}

m2ReasonSlide();
// 12. 성능 개선 과정
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 성능 개선 과정", 12);
  coreMsg(s, M2.core13);
  const lw = 6.2;
  card(s, M, 2.45, lw, 4.35);
  barChart(s, M2.chart.labels, M2.chart.values, {
    x: M + 0.15, y: 2.55, w: lw - 0.3, h: 4.15, chartColors: winner(M2.chart.values.length, M2.chart.values.length - 1),
    title: "공식 Validation 정확도 (%)", valAxisMinVal: 84, valAxisMaxVal: 94,
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
    { text: "전환점 ②  ", options: { bold: true, color: C.blue } },
    { text: M2.turn2, options: { breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "전환점 ③  ", options: { bold: true, color: C.blue } },
    { text: M2.turn3 },
  ], C.mintLight, 12);
  caption(s, M2.evidence13);
  s.addNotes(M2.notes13);
}

// 13. 다른 접근과 기각·채택 이유
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 다른 접근과 기각·채택 이유", 13);
  coreMsg(s, M2.core14);
  const red = t => strong("✗ " + t, C.red);
  const [lastTry, lastVerdict, lastWhy] = M2.lastRow;
  table(s, [
    ["시도", "판정", "이유"],
    ["같은 통화의 다른 발화와 비교하는 특징", red("제거"), "규칙 위반 소지 (주최 측 답변)"],
    [{ text: "text로 고친 라벨로 학습" }, strong("제공 라벨로 재학습"), "점수는 거의 그대로"],
    ["통계 모델과 CNN 확률 평균", red("기각"), "단독 CNN보다 낮음"],
    ["더 큰 CNN / 빈 구간 제외 평균 / 발화 길이 입력 / 데이터 섞기(Mixup)", red("기각"), "사전 기준 미달 또는 재현 안 됨"],
    ["WavLM-base 고정 특징(임베딩) · 일부 데이터 미세조정(fine-tuning)", red("기각"), "작은 모델·적은 데이터로는 CNN보다 낮음"],
    [strong(lastTry, C.text), strong(lastVerdict, C.blue), strong(lastWhy, C.blue)],
  ], M, 2.45, [5.2, 2.4, 4.73], { rowH: 0.6, fontSize: 13 });
  caption(s, "비교는 Training 내부 통화 단위 홀드아웃 기준이며, 마지막 행의 92.90%만 공식 Validation 결과.", 6.75);
  s.addNotes("근거: 통화 단위 특징 M2-EXP-008 / 제공 라벨 재학습 M2-RETRAIN-001 / 확률 평균 M2-ENS-001 / 더 큰 CNN M2-CNN-008, 패딩 제외 풀링 M2-CNN-009, 길이 입력 M2-CNN-004·006, Mixup M2-MIX-001 / WavLM 고정 임베딩 M2-SSL-001, 기본 미세조정 M2-SSL-002 / WavLM-large 미세조정 2개 + CNN 결합 M2-WLML-001·002, M2-BLEND-001(내부 20,000발화 CNN 단독 91.94% → 93.21%). WavLM 기각→채택 차이: SSL-002는 base 모델·6,000통화(85.65%), WLML은 large 모델·Training 전체(단독 공식 Val 92.40·92.71%). 모두 docs/experiment_log.md·docs/decisions.md.");
}

// 14. 최종 결과와 보완할 점
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 2  신고자/119대원 분류 — 최종 결과와 보완할 점", 14);
  coreMsg(s, M2.core15);
  const lw = 5.6;
  card(s, M, 2.45, lw, 4.35);
  stat(s, M, 2.75, lw, M2.acc, "정확도 (공식 Validation)", C.blue, 48);
  s.addShape(pres.shapes.LINE, { x: M + 0.4, y: 4.25, w: lw - 0.8, h: 0, line: { color: C.line, width: 1 } });
  M2.recalls.forEach(([n, l], i) => {
    const y = 4.45 + i * 1.05, bx = M + 0.4, bw = lw - 0.8, pct = parseFloat(n) / 100, col = i === 0 ? C.an1 : C.an2;
    txt(s, l, { x: bx, y, w: 2.0, h: 0.35, fontSize: 13, bold: true, color: C.sub, valign: "middle" });
    txt(s, n, { x: bx + bw - 1.8, y, w: 1.8, h: 0.35, fontSize: 20, bold: true, color: col, align: "right", valign: "middle" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: y + 0.42, w: bw, h: 0.22, fill: { color: C.line }, rectRadius: 0.1, line: { color: C.line } });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: y + 0.42, w: bw * pct, h: 0.22, fill: { color: col }, rectRadius: 0.1, line: { color: col } });
  });
  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, 2.45, rw, 1.6);
  title(s, "■ 계산 효율성 · 재현성", rx + 0.25, 2.6, rw - 0.5, 16);
  txt(s, M2.repro, { x: rx + 0.25, y: 3.0, w: rw - 0.5, h: 1.0, fontSize: 13 });
  card(s, rx, 4.2, rw, 2.6, C.bgGray);
  title(s, "■ 보완할 점", rx + 0.25, 4.35, rw - 0.5, 16);
  txt(s, M2.improve, { x: rx + 0.25, y: 4.8, w: rw - 0.5, h: 1.9, fontSize: 14 });
  caption(s, "근거: " + M2.evidence);
  s.addNotes(M2.notes15);
}

m2Appendix1(); m2Appendix2(); m2Appendix3();

// ======================================================================
// ---------- Mission 2 음향 특징 모델(지연우) 4장 압축본 ----------
// ======================================================================
// Mission 2 음향 특징 모델(지연우) 4장 압축본
// 숫자 출처: 기존 연우 슬라이드 10장, docs/mission2_experiment_log.md (clean 59D 구성, Stage 15 149D E_TRG 88.49%)
// ======================================================================
const YW = "Mission 2  음향 특징 모델(비교 실험) — ";

function ywChip(s, x, y, w, h, color, head, body) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.white }, rectRadius: 0.06, line: { color: C.line, width: 0.8 } });
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.09, h, fill: { color }, line: { color } });
  txt(s, head, { x: x + 0.2, y, w: 1.9, h, fontSize: 11.5, bold: true, color: C.blue, valign: "middle" });
  txt(s, body, { x: x + 2.1, y, w: w - 2.2, h, fontSize: 10.8, color: C.text, valign: "middle" });
}

// ---------- 1. Baseline에서 59D까지 ----------
function yw1() {
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", YW + "Baseline에서 59개 숫자까지", 0);
  coreMsg(s, "기본 특징과 분류기를 먼저 개선하고, 효과가 확인된 음향 정보만 선별해 59개 숫자를 구성했다.");

  // 상단: 실험 흐름
  const nodes = [
    ["음색 요약(MFCC) 40개", "음색 값 20개의 평균·표준편차"],
    ["+ Rolloff·ZCR = 42개", "고음 분포·파형 떨림 2개 추가"],
    ["MLP(신경망)로 전환", "같은 42개 · 복잡한 관계 학습"],
    ["음향 특징 59개", "가설별로 추가하고 선별"],
  ];
  const nw = 2.55, ny = 2.42, nh = 0.82, ngap = (CW - 4 * nw) / 3;
  nodes.forEach(([t, b], i) => {
    const x = M + i * (nw + ngap);
    node(s, x, ny, nw, nh, t, b, i === 3);
    if (i < 3) link(s, x + nw, ny + nh / 2, x + nw + ngap, ny + nh / 2);
  });

  // 하단 왼쪽: LR vs MLP
  const by = 3.5, bh = 3.35, lw = 5.3;
  card(s, M, by, lw, bh);
  title(s, "■ 같은 42개 입력, 분류기만 교체", M + 0.2, by + 0.1, lw - 0.4, 14);
  barChart(s, ["로지스틱 회귀", "MLP (신경망)"], [81.73, 85.95], {
    x: M + 0.15, y: by + 0.5, w: lw - 1.75, h: bh - 0.6, chartColors: [C.an1, C.an2],
    title: "내부 검증 정확도 (%)", valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20,
  });
  txt(s, [{ text: "+4.22%p", options: { fontSize: 24, bold: true, color: C.blue, breakLine: true } },
          { text: "이 MLP를 고정한 뒤\n음향 특징을 비교", options: { fontSize: 10.5, color: C.sub } }],
      { x: M + lw - 1.6, y: by + 1.2, w: 1.5, h: 1.4, align: "center", valign: "middle" });

  // 하단 오른쪽: 59D 구성
  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, by, rw, bh);
  title(s, "■ 59개 구성 (재현 실험)", rx + 0.2, by + 0.1, rw - 0.4, 14);
  const chips = [
    [C.line, "기본 42개", "음색 요약 40 + Rolloff·ZCR 2"],
    [C.sky, "음높이·성대 울림 5개", "목소리 높낮이(F0) 변화 폭·오르내림, 성대 울린 비율·규칙성"],
    [C.sky, "규칙성·대역 관계 5개", "소리의 주기적 규칙성(CPP) 2 + 대역 간 소리 크기 변화 일관성 3"],
    [C.blue, "활동성·엔트로피 7개", "말한 구간 패턴 4 + 소리 퍼짐 정도(엔트로피) 3"],
  ];
  chips.forEach(([col, h, b], i) => ywChip(s, rx + 0.2, by + 0.55 + i * 0.55, rw - 0.4, 0.46, col, h, b));
  txt(s, [{ text: "87.38%", options: { fontSize: 24, bold: true, color: C.blue } },
          { text: "   59개 · 내부 검증", options: { fontSize: 11, color: C.sub } }],
      { x: rx, y: by + 2.8, w: rw, h: 0.45, align: "center", valign: "middle" });
  caption(s, "※ 수치는 모두 Training 통화 단위 내부 검증 기준. 초기 Baseline(음색 요약 40개 + 로지스틱 회귀)은 공식 Validation 80.9%.", 7.02);
  s.addNotes("화면의 \"N개 숫자\"는 ND(차원), \"재현 실험\"은 clean 계열, 음높이·성대 울림=AF02, 규칙성·대역 관계=AF04+AF09, 활동성·엔트로피=AF10+AF11. 기존 연우 슬라이드 15~20장 압축. Baseline: MFCC 40D(20계수 평균·표준편차) + Logistic Regression, 공식 Validation 80.9%. E05 42D = MFCC40 + Rolloff·ZCR(81.3%). 같은 E05 42D 입력, 같은 통화 단위 내부 분할: LR 81.73% / MLP 85.95%(+4.22%p). clean 59D = clean MFCC40 + E05 2D + AF02 5D(음높이·유성성) + AF04 2D(CPP) + AF09 3D(다중 대역 진폭 포락선 일관성) + AF10 4D(활동성) + AF11 3D(엔트로피), 내부 87.38%(clean 52D 86.96% 대비 +0.42%p). 제외·보류: Duration, Energy·Spectral Flux(제외), Delta-MFCC(보류), eGeMAPS 대체(86.40%, 기존 유지). 근거: docs/mission2_experiment_log.md.");
}

// ---------- 2. 세 모델과 149D 확장 ----------
function yw2() {
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", YW + "세 모델과 149개 입력 확장", 0);
  coreMsg(s, "음색 분포와 대역 대비를 보완한 149개 숫자 입력을 서로 다른 세 모델에 적용했다.");

  const top = 2.36, h = 4.5, lw = 7.2;
  card(s, M, top, lw, h);
  title(s, "■ 두 가지 보완 가설", M + 0.2, top + 0.1, lw - 0.4, 14);
  const hyps = [
    ["① 음색 분포 요약", "평균·표준편차가 놓치는 분포 모양을 보완", ["음색 값 20개", "×", "하위10%·중앙값·상위10%·퍼짐", "=", "80개"]],
    ["② 주파수 대역 대비", "대역 안의 강한·약한 소리 차이를 보완", ["0–4 kHz 5개 대역", "×", "대비의 평균·표준편차", "=", "10개"]],
  ];
  hyps.forEach(([name, why, eq], i) => {
    const y = top + 0.5 + i * 1.18;
    txt(s, name, { x: M + 0.25, y, w: 3.2, h: 0.3, fontSize: 12.5, bold: true, color: C.blue });
    txt(s, why, { x: M + 3.3, y, w: lw - 3.5, h: 0.3, fontSize: 10.8, color: C.sub, align: "right" });
    const parts = [[2.0, false], [0.3, true], [2.35, false], [0.3, true], [1.25, false]];
    let x = M + 0.25;
    eq.forEach((t, k) => {
      const [w, op] = parts[k];
      if (op) txt(s, t, { x, y: y + 0.4, w, h: 0.6, fontSize: 16, bold: true, color: C.sky, align: "center", valign: "middle" });
      else {
        const last = k === eq.length - 1;
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y + 0.4, w, h: 0.6, fill: { color: last ? C.blue : C.light }, rectRadius: 0.06, line: { color: last ? C.blue : C.sky, width: 0.8 } });
        txt(s, t, { x, y: y + 0.4, w, h: 0.6, fontSize: last ? 16 : 11.5, bold: true, color: last ? C.white : C.blue, align: "center", valign: "middle" });
      }
      x += w + 0.08;
    });
  });
  // 149D 구성 막대 (길이 비례)
  const barY = top + 3.45, barX = M + 0.25, barW = lw - 0.5;
  txt(s, "결합 입력", { x: barX, y: barY - 0.36, w: 2, h: 0.3, fontSize: 11.5, bold: true, color: C.text });
  let bx = barX;
  [[59, "기존 59개", C.blue], [80, "① 80개", C.sky], [10, "10개", C.mint]].forEach(([n, lab, col]) => {
    const w = barW * n / 149;
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: barY, w, h: 0.5, fill: { color: col }, line: { color: C.white, width: 1 } });
    txt(s, lab, { x: bx, y: barY, w, h: 0.5, fontSize: 11, bold: true, color: C.white, align: "center", valign: "middle" });
    bx += w;
  });
  txt(s, [{ text: "= 149개 숫자", options: { fontSize: 15, bold: true, color: C.blue } },
          { text: "   (마지막 10개 = ② 대역 대비)", options: { fontSize: 10, color: C.sub } }],
      { x: barX, y: barY + 0.55, w: barW, h: 0.3, align: "right" });

  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, top, rw, h);
  title(s, "■ 149개 입력 · 세 모델 내부 검증", rx + 0.2, top + 0.1, rw - 0.4, 14);
  [["TabM", "88.64%", "여러 MLP를 한 번에 학습하는 정형 모델"],
   ["Tabular ResNet", "88.55%", "잔차 연결을 둔 정형 신경망"],
   ["GANDALF", "88.58%", "게이트로 특징을 고르는 다른 계열"]].forEach(([n, v, d], i) => {
    const y = top + 0.55 + i * 1.02;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.2, y, w: rw - 0.4, h: 0.88, fill: { color: i === 0 ? C.light : C.white }, rectRadius: 0.06, line: { color: C.line, width: 0.8 } });
    txt(s, n, { x: rx + 0.35, y: y + 0.08, w: 2.2, h: 0.35, fontSize: 13, bold: true, color: C.blue });
    txt(s, d, { x: rx + 0.35, y: y + 0.45, w: rw - 2.2, h: 0.35, fontSize: 10.3, color: C.sub });
    txt(s, v, { x: rx + rw - 1.85, y, w: 1.5, h: 0.88, fontSize: 21, bold: true, color: C.blue, align: "right", valign: "middle" });
  });
  txt(s, "선택 이유: 59개 입력에서 가장 높은 두 기준 모델(TabM·ResNet)과, 다른 계열 중 가장 높은 GANDALF", { x: rx + 0.2, y: top + 3.68, w: rw - 0.4, h: 0.6, fontSize: 10.5, color: C.text, valign: "middle" });
  caption(s, "※ Training 통화 단위 내부 검증 기준. 전처리 기준은 Training 데이터만 사용.", 7.02);
  s.addNotes("화면의 ①=F01, ②=F02, \"N개\"=ND(차원). 기존 연우 슬라이드 21·23장 압축. F01 MFCC Robust Stats 80D(20계수의 q10·중앙값·q90·IQR), F02 Spectral Contrast 10D(0–4 kHz 5대역 contrast 평균·표준편차). 149D = clean 59D + F01 80D + F02 10D. 149D 내부: TabM 88.64%, Tabular ResNet 88.55%, GANDALF 88.58%. 59D 동결 비교: TabM 88.40%, ResNet 88.38%, GANDALF 88.17%(신규 계열 중 최고).");
}

// ---------- 3. 최종 E_TRG 구성 ----------
function yw3() {
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", YW + "최종 모델(세 모델 합치기) 구성", 0);
  coreMsg(s, "세 모델의 확률을 같은 비중(1/3씩)으로 평균한 모델을 최종으로 선택했다.");
  flow(s, M, 2.55, CW, 2.5, {
    pre: [["149개 음향 특징", "기존 59개 + ① 80개 + ② 10개\n그 발화만 사용"]], preW: 2.9,
    branches: [["TabM"], ["Tabular ResNet"], ["GANDALF"]].map(b => [b[0], "", "× 1/3"]), brW: 2.9, brGap: 0.12,
    post: [["확률 평균", "확률 × 1/3의 합\n≥ 0.5 → 신고자\n< 0.5 → 대원"]], postW: 2.5,
  });
  const by = 5.35, bh = 1.5, lw = 4.0;
  card(s, M, by, lw, bh, C.light);
  txt(s, [{ text: "88.85%", options: { fontSize: 30, bold: true, color: C.blue, breakLine: true } },
          { text: "최종 모델 내부 검증 정확도", options: { fontSize: 11, color: C.sub } }],
      { x: M, y: by, w: lw, h: bh, align: "center", valign: "middle" });
  const rx = M + lw + 0.3, rw = W - M - rx;
  insight(s, rx, by, rw, bh, [
    { text: "선택 결론  ", options: { bold: true, color: C.blue } },
    { text: "세 단일 모델(88.55~88.64%)보다 높았다. 비중은 조정하지 않고 1/3씩 고정했고, 추가 튜닝 없이 Training 전체로 다시 학습해 확정했다.", options: {} },
  ], C.mintLight, 12);
  s.addNotes("화면의 \"최종 모델\"은 E_TRG. 기존 연우 슬라이드 24장 상단. E_TRG = TabM·Tabular ResNet·GANDALF 확률 1/3씩 평균, threshold 0.5. 내부 검증 88.85%. 149D 입력·동일 가중치·threshold 0.5 고정, 전체 Training으로 재학습(Stage 15).");
}

// ---------- 4. 최종 평가와 결론 ----------
function yw4() {
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", YW + "최종 평가와 결론", 0);
  coreMsg(s, "고정한 모델을 공식 Validation에서 한 번 평가해 88.49%를 확인했다.");
  const top = 2.36, h = 2.9, lw = 4.6;
  card(s, M, top, lw, h);
  stat(s, M, top + 0.75, lw, "88.49%", "정확도 (공식 Validation)", C.blue, 46);
  const rx = M + lw + 0.3, rw = W - M - rx;
  card(s, rx, top, rw, h);
  title(s, "■ 역할별 재현율 (공식 Validation)", rx + 0.2, top + 0.1, rw - 0.4, 14);
  s.addChart(pres.charts.BAR, [{ name: "재현율", labels: ["119대원", "신고자"], values: [84.2, 92.5] }], {
    x: rx + 0.2, y: top + 0.5, w: rw - 0.4, h: h - 0.6, barDir: "bar", varyColors: true, chartColors: [C.an1, C.an2],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelFontBold: true, dataLabelFormatCode: "0.0",
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20, valAxisLabelColor: C.sub,
    catAxisLabelColor: C.text, catAxisLabelFontFace: F, catAxisLabelFontSize: 12,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  });
  const by = 5.45, bh = 1.4;
  card(s, M, by, CW, bh);
  title(s, "■ 결론", M + 0.2, by + 0.08, 3, 14);
  bullets(s, [
    [strong("음향 정보 선별  ", C.text), { text: "가설마다 음향 특징을 하나씩 더하고 효과가 확인된 것만 남겼다 (42개 → 59개 → 149개)." }],
    [strong("남은 오류  ", C.text), { text: "짧은 발화와 목소리가 겹친 구간. 최종 제출 모델(CNN + WavLM, 92.90%)에서도 짧은 발화가 가장 어렵다." }],
  ], M + 0.25, by + 0.45, CW - 0.5, 0.9, 12);
  s.addNotes("기존 연우 슬라이드 24·25장. Stage 15 149D E_TRG 공식 Validation Accuracy 0.884919(88.49%), Recall 0/1 0.841787 / 0.924891, 111,947발화/3,640통화, FINAL_VERIFIED. 내부 88.85%와 평가 집합이 달라 차이를 개선량으로 계산하지 않음. 최종 제출 모델은 CNN 3개 + WavLM-large 2개(공식 Validation 92.90%, 11~15장). 근거: docs/mission2_experiment_log.md.");
}

yw1(); yw2(); yw3(); yw4();

// Mission 3 — 환자 증상 인식 (26~29장)
// ======================================================================

// 26. 문제와 최종 모델 (main 유지 + 수빈 님 PR #200의 파이프라인·근거 문구)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 문제 정의와 접근", 26);
  coreMsg(s, "단어 빈도 모델(TF-IDF)부터 문맥 모델(RoBERTa)까지 검증해, RoBERTa 3개와 TF-IDF를 합친 모델을 채택");
  const prob = [["입력", "119 통화 전사문\n(발화 text만 사용)"], ["출력", "증상 9종 중 0~9개\n(다중 레이블)"], ["평가", "Macro F1\n(증상별 F1 평균)"]];
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
    pre: [["입력 정제", "발화 text만\n원본 순서 유지"]], preW: 1.4,
    branches: [["RoBERTa-large", "시작값 3개로 학습해 평균", "× 0.55"], ["TF-IDF + 로지스틱 회귀", "단어 빈도 기반 기준 모델", "× 0.45"]], brW: 2.3, brGap: 0.15,
    post: [["비중을 둔 합", "확률 × 비중의 합이\n0.5 이상이면 있음"]], postW: 1.75,
  });
  txt(s, "■ 왜 이 방법인가", { x: rx, y: 4.95, w: rw, h: 0.4, fontSize: 18, bold: true });
  bullets(s, [
    "오류 단서: 키워드 탐색에서 전사에 직접 근거가 드러나지 않는 양성 라벨을 발견",
    "모델 비교: 키워드 표현과 문맥 표현이 다른 오류를 낼 가능성을 확인",
    "검증 원칙: 결합 비율은 Training 내부에서 확정하고, 공식 Validation은 채점만 수행",
  ], rx, 5.4, rw, 1.4, 13);
  caption(s, "근거: submission/mission3/README.md, docs/decisions.md M3-FINAL-002");
  s.addNotes("근거: submission/mission3/README.md '모델' 절, docs/decisions.md M3-FINAL-002. 파이프라인·'왜' 문구는 이수빈 님 PR #200(subin/ppt-m3-storytelling-20261013)에서 가져옴.");
}

// =====================================================================
// 27. 최종 결과
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 최종 결과", 27);
  coreMsg(s, "RoBERTa 3개 + TF-IDF 결합 모델: 공식 Validation Macro F1 65.34%");
  card(s, M, 2.45, 7.3, 4.35);
  txt(s, "65.34%", { x: M + 0.2, y: 2.65, w: 2.6, h: 0.75, fontSize: 38, color: C.blue, bold: true, align: "center", valign: "middle" });
  txt(s, "Macro F1(증상별 F1 평균)\n공식 Validation 3,640건", { x: M + 0.2, y: 3.43, w: 2.6, h: 0.5, fontSize: 11.5, color: C.sub, align: "center" });
  insight(s, M + 0.2, 4.28, 2.6, 2.3, [
    { text: "Micro F1  66.90%", options: { bold: true, color: C.blue, breakLine: true } },
    { text: "(모든 판정을 합산한 F1)", options: { fontSize: 10.5, color: C.sub, breakLine: true } },
    { text: "증상을 예측하지 않은\n통화 149건", options: { fontSize: 11.5, breakLine: true } },
    { text: "합치는 비중\nRoBERTa 55% + TF-IDF 45%", options: { fontSize: 10.5 } },
  ]);
  // 증상별 F1 (최종 앙상블, 공식 Validation) — 낮은 3개만 빨강. 근거: dcc-lab results/M3-FINAL-002/metrics.json
  const sym = [["열상",88.28],["복통",82.71],["고열",70.01],["호흡곤란",67.95],["어지러움",65.70],["구토",60.88],["전신쇠약",59.39],["두통",53.15],["오심",40.00]];
  barChart(s, sym.map(r => r[0]), sym.map(r => r[1]), {
    name: "F1 (%)", barDir: "bar", catAxisOrientation: "maxMin", dataLabelFormatCode: '0.00"%"', x: M + 2.95, y: 2.6, w: 4.2, h: 4.1,
    chartColors: sym.map(() => C.an1), showTitle: true, title: "증상별 F1 (%) · 최종 모델 공식 Validation", titleFontSize: 11,
    dataLabelFontSize: 11, dataLabelFontBold: false, valAxisMinVal: 0, valAxisMaxVal: 118, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 12,
  });
  const rx = M + 7.6, rw = W - M - rx;
  // 모델별 점수 흐름은 28장 차트에 있으므로 여기서는 약한 증상과 이유를 한 카드로 보여준다.
  card(s, rx, 2.45, rw, 4.35, C.bgGray);
  title(s, "■ 약한 증상, 왜 낮을까", rx + 0.2, 2.56, rw - 0.4);
  [["오심", "40.00%", "양성의 88.6%에서\n전사에 직접 표현이 없음"],
   ["두통", "53.15%", "직접 표현이 없으면\n재현율이 19.5%로 낮음"],
   ["전신쇠약", "59.39%", "양성의 75.7%에서\n전사에 직접 표현이 없음"]].forEach(([nm, f1, why], i) => {
    const y = 3.1 + i * 1.02;
    if (i > 0) s.addShape(pres.shapes.LINE, { x: rx + 0.25, y: y - 0.06, w: rw - 0.5, h: 0, line: { color: C.line, width: 1 } });
    txt(s, nm, { x: rx + 0.25, y, w: 1.35, h: 0.3, fontSize: 12.5, bold: true, color: C.text });
    txt(s, f1, { x: rx + 0.25, y: y + 0.3, w: 1.35, h: 0.5, fontSize: 21, bold: true, color: C.blue, valign: "middle" });
    txt(s, why, { x: rx + 1.75, y, w: rw - 1.95, h: 0.85, fontSize: 12, valign: "middle" });
  });
  txt(s, "※ 이유는 Training 내부 홀드아웃 5,792건의 TF-IDF 진단 기준 (29장)", { x: rx + 0.2, y: 6.2, w: rw - 0.4, h: 0.45, fontSize: 10, color: C.sub, align: "center", valign: "middle" });
  caption(s, "근거: submission/mission3/README.md, m3_final_train.executed.ipynb. 모델별 점수 차이가 특정 원인만의 효과임을 뜻하지 않음.");
  s.addNotes("근거: submission/mission3/README.md 결과 표(seed 42·43·44 64.79%/64.41%/64.45%, 앙상블 65.34%, Micro 66.90%, 빈 예측 149). 모델별 점수 흐름(M3-OFFVAL-001 TF-IDF 56.89%·KcBERT 62.46%, M3-OFFVAL-002 RoBERTa-large 64.33%, M3-FINAL-002 앙상블 65.34%)은 28장 차트에 있음. 증상별 약점: M3-FINAL-002(오심·두통·전신쇠약 F1이 가장 낮음). 약한 증상의 이유(직접 표현 없는 양성 비율 오심 88.6%·전신쇠약 75.7%, 두통의 표현 없을 때 재현율 19.5%)는 Training 내부 홀드아웃 5,792건의 TF-IDF 진단(29장·35장, M3-ERR-001)이며 최종 앙상블의 공식 Validation 오류 분석이 아니다.");
}

// 28. 환자 증상 인식 — 모델 선정 (네이티브 편집 가능 차트와 카드)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 모델 선정", 28);
  coreMsg(s, "서로 다른 오류를 내는 문맥 모델과 단어 모델을 합쳐 최종 제출 모델을 정했다");
  barChart(s, ["TF-IDF", "KcBERT", "RoBERTa-large", "최종 앙상블"],
    [56.89, 62.46, 64.33, 65.34], {
      name: "Macro F1 (%) · 공식 Validation", barDir: "bar", catAxisOrientation: "maxMin",
      x: M + 0.18, y: 2.25, w: CW - 0.36, h: 1.83,
      chartColors: winner(4, 3), showTitle: true,
      title: "모델별 공식 Validation Macro F1 (%)", titleFontSize: 11.5,
      valAxisMinVal: 0, valAxisMaxVal: 100, valAxisHidden: true,
      valGridLine: { style: "none" }, dataLabelFormatCode: '0.00"%"',
      dataLabelFontSize: 12, catAxisLabelFontSize: 11.5,
    });
  const y = 4.18, h = 2.34, gap = 0.30, w = (CW-gap)/2, rx = M+w+gap;
  card(s, M, y, w, h, C.light);
  title(s, "기각 · 비슷한 문맥 모델 결합", M+0.18, y+0.13, w-0.36, 15);
  txt(s, "KcBERT + RoBERTa", { x:M+0.18, y:y+0.53, w:w-0.36, h:0.27, fontSize:12.6, bold:true, color:C.blue });
  bullets(s, [
    "두 모델 모두 문장을 문맥으로 읽는 같은 계열(Transformer) 모델",
    "전사 근거가 부족한 증상에서 오류가 반복",
    "두 모델이 함께 틀리는 사례가 많아 결합 효과가 제한적",
    "내부 검증 Macro F1 개선폭 +0.28%p"
  ], M+0.20, y+0.89, w-0.40, 1.34, 11.4);
  card(s, rx, y, w, h, C.mintLight);
  title(s, "채택 · 서로 다른 표현 방식 결합", rx+0.18, y+0.13, w-0.36, 15);
  txt(s, "RoBERTa + TF-IDF", { x:rx+0.18, y:y+0.53, w:w-0.36, h:0.27, fontSize:12.6, bold:true, color:C.blue });
  bullets(s, [
    "RoBERTa: 문맥과 부정 표현을 반영",
    "TF-IDF: 직접적인 증상 단어와 어휘 패턴을 포착",
    "서로 다른 오류를 예측 확률 결합으로 보완",
    "RoBERTa 55% + TF-IDF 45%",
    "최종 공식 Validation Macro F1 65.34%"
  ], rx+0.20, y+0.86, w-0.40, 1.43, 10.8);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x:M, y:6.62, w:CW, h:0.39, fill:{ color:C.light }, rectRadius:0.06, line:{ color:C.sky, width:0.8 } });
  txt(s, "비슷한 모델을 추가하기보다 서로 다른 정보를 학습하는 모델을 결합했다", { x:M+0.16, y:6.64, w:CW-0.32, h:0.33, fontSize:14, bold:true, color:C.blue, align:"center", valign:"middle" });
  caption(s, "근거: docs/decisions.md M3-ENS-001·M3-FINAL-002. 내부 홀드아웃과 공식 Validation 차이를 직접 인과 효과로 해석하지 않음.", 7.14);
  s.addNotes("근거: docs/decisions.md M3-ENS-001(Training 내부 80:20: 신경망 결합 64.55%, RoBERTa 64.27%, +0.28%p로 사전 기준 +1.00%p 미달), M3-FINAL-002(최종 RoBERTa 3-seed 55%와 TF-IDF 45% 확률 결합, 공식 Validation Macro F1 65.34%). 상단의 공식 Validation 비교는 서로 다른 학습 조건이므로 인과 효과가 아니며 Training 내부 홀드아웃과 직접 비교하지 않음.");
}

// 29. Mission 3 환자 증상 인식 — 오류 분석 (팀원 수정본에 발화 카드 복원)
{
  const s = pres.addSlide();
  header(s, "03", "미션별 수행", "Mission 3  환자 증상 인식 — 오류 분석", 29);
  const lw = 6.18;
  card(s, M, 1.60, lw, 2.52);
  title(s, "오류 분석 - 전사에 직접 표현이 없는 양성 라벨", M+0.16, 1.72, lw-0.32, 13.7);
  s.addChart(pres.charts.DOUGHNUT, [{ name:"양성 라벨", labels:["직접 표현 없음", "직접 표현 있음"], values:[52,48] }], {
    x:M+0.14, y:2.12, w:2.55, h:1.73, holeSize:63, chartColors:[C.an2,C.line],
    showLegend:false, showPercent:false, showValue:false, showLabel:false,
    dataBorder:{ pt:1, color:"FFFFFF" }
  });
  txt(s, "약 52%", { x:M+0.14, y:2.68, w:2.55, h:0.42, fontSize:19, color:C.an2t, bold:true, align:"center", valign:"middle" });
  txt(s, "직접 표현 없음 52% · 있음 48%", { x:M+0.18, y:3.76, w:2.6, h:0.24, fontSize:9.4, color:C.sub, align:"center" });
  txt(s, "92%", { x:M+2.95, y:2.26, w:1.28, h:0.56, fontSize:27, bold:true, color:C.an2t, align:"center" });
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
    s.addShape(pres.shapes.RECTANGLE,{ x:M+1.42,y:y+0.03,w:3.6*value/100,h:0.20,fill:{color:C.an2},line:{color:C.an2} });
    txt(s,value.toFixed(1)+"%",{ x:M+5.07,y:y-0.01,w:0.75,h:0.31,fontSize:11.2,bold:true,color:C.blue,align:"right" });
  });
  txt(s, "※ 미검출은 증상 부재·오라벨을 뜻하지 않음 (정규식이 구어체를 다 잡지 못함)", { x:M+0.17, y:6.75, w:lw-0.34, h:0.34, fontSize:9.1, color:C.sub });
  caption(s, "근거: docs/experiment_log.md M3-ERR-001; reports/mission3/error_analysis/m3_error_analysis.json · README.md",7.15);
  s.addNotes("과거 TF-IDF·KcBERT 오류 분석: 전사에 직접 표현이 없는 양성 라벨 약 52%, 두 모델이 놓친 양성 중 직접 표현 없는 사례 92%, 직접 표현이 있는 라벨 중 맞힌 비율 약 90%. 이 수치는 최종 RoBERTa 오류 통계가 아님. Training 내부 홀드아웃 5,792건에서 직접 표현 미검출률: 오심 88.6%, 전신쇠약 75.7%, 두통 63.7%. 정규식 키워드 탐색은 구어체 변형을 모두 포착하지 못하므로 직접 표현 없음은 증상 부재 또는 오라벨의 증거가 아님. 근거: docs/experiment_log.md M3-ERR-001; reports/mission3/error_analysis/m3_error_analysis.json; reports/mission3/error_analysis/README.md.");

  // Reviewed Mission 3 case card restored; illustrative GIF, not real model inference.
  card(s,6.82,1.60,6.01,5.12);
  title(s,"■ 오류 분석에서 읽은 실제 발화",7.04,1.73,5.55,14);
  s.addShape(pres.shapes.LINE,{x:7.04,y:2.18,w:5.55,h:0,line:{color:C.line,width:1}});
  s.addImage({path:"figures/anim/speakers_separation.gif",x:7.10,y:2.59,w:2.73,h:1.12});
  txt(s,"119 신고 통화 · 상황 삽화",{x:7.10,y:3.77,w:2.73,h:0.23,fontSize:9,color:C.sub,align:"center"});
  arrow(s,9.91,2.93,0.35,0.28);
  txt(s,"전사",{x:9.80,y:3.32,w:0.60,h:0.21,fontSize:9,bold:true,color:C.blue,align:"center"});
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:10.34,y:2.50,w:2.24,h:1.22,
    fill:{color:C.white},line:{color:C.sky,width:1.6},rectRadius:0.1});
  txt(s,"보고서의 발화 인용",{x:10.43,y:2.56,w:2.06,h:0.21,fontSize:9.5,bold:true,color:C.sky,align:"center"});
  txt(s,"“어지러워서\n정신을 못 차리겠다”",{x:10.42,y:2.78,w:2.10,h:0.87,fontSize:14.6,bold:true,
    color:C.blue,align:"center",valign:"middle"});
  txt(s,"분석 입력: 전사 text",{x:10.28,y:3.78,w:2.39,h:0.20,fontSize:9.4,color:C.sub,align:"center"});
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:7.12,y:4.09,w:2.25,h:0.49,
    fill:{color:C.blue},line:{color:C.blue},rectRadius:0.06});
  txt(s,"오심 양성 라벨",{x:7.20,y:4.15,w:2.08,h:0.37,fontSize:12.0,bold:true,color:C.white,align:"center",valign:"middle"});
  txt(s,"vs",{x:9.48,y:4.15,w:0.42,h:0.37,fontSize:12,bold:true,color:C.sub,align:"center",valign:"middle"});
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:10.01,y:4.09,w:2.54,h:0.49,
    fill:{color:C.white},line:{color:C.an2,width:1.4},rectRadius:0.06});
  txt(s,"오심 직접 표현 미검출",{x:10.05,y:4.15,w:2.47,h:0.37,fontSize:10.5,bold:true,color:C.an2t,align:"center",valign:"middle"});
  s.addShape(pres.shapes.LINE,{x:7.06,y:4.83,w:5.50,h:0,line:{color:C.line,width:1}});
  title(s,"이 사례에서 확인한 문제",7.09,4.97,5.40,12.5);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:7.08,y:5.33,w:5.47,h:0.73,
    fill:{color:C.bgGray},line:{color:C.bgGray,width:0.6},rectRadius:0.1});
  txt(s,"전사에 오심 표현이 없어도 오심 양성 라벨이 부여될 수 있음\n→ 직접 표현 미검출은 증상 부재·오라벨 확정이 아님",{
    x:7.23,y:5.41,w:5.18,h:0.59,fontSize:10.4,color:C.text,valign:"middle"});
  s.addShape(pres.shapes.LINE,{x:7.08,y:6.18,w:5.47,h:0,line:{color:C.line,width:1}});
  txt(s,"출처: M3 오류분석 보고서 1절 · 정규식 기준 · 최종 RoBERTa 오답 아님",{
    x:7.10,y:6.23,w:5.42,h:0.28,fontSize:9,color:C.sub,align:"center"});

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
    ["M1 신고자 성별", "음향 특징 + 화자 임베딩 + 로지스틱 회귀", "98.10%", "+2.99%p"],
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
    txt(s, i === 2 ? "Macro F1 · 공식 Validation" : "Accuracy · 공식 Validation", { x: x + 0.25, y: 4.25, w: tw - 0.5, h: 0.3, fontSize: 10.5, color: C.sub });
  });
  txt(s, "▲ = baseline 대비 개선폭 · baseline: M1 95.11%(음향 특징만) · M2 86.44%(통계 모델) · M3 56.89%(TF-IDF)", { x: M, y: 4.7, w: CW, h: 0.25, fontSize: 11, color: C.sub, align: "right" });
  const lessons = [
    ["기준은 결과를 보기 전에\n고정한다","결과를 본 뒤 기준을 낮추면 Validation에 과적합된다"],
    ["기각도 기록한다", "효과 없던 실험과 그 이유를 결정 기록에 남겨 같은 시도를 반복하지 않음"],
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
    ["오류 원인 분석", ["M2  짧은 응답 발화의 오류 사례 확인", "M3  근거 없는 라벨 구간 오류 분석"]],
    ["미션별 모델 보완", ["M2  짧은 응답 발화 보강", "M3  오심·두통 성능 개선"]],
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
    ["라벨 불균형", "9개 증상 양성 비율 9.9~23.2%", "희귀 증상 가중 BCE(pos_weight) + Soft macro-F1"],
    ["입력 길이", "40.7%가 256토큰 초과 · 512토큰 +0.74%p", "기준(+1.00%p)에 못 미치고 약 4배 느려 256토큰 유지"],
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
  txt(s, "BCE = 증상 유무 예측 오차 · SoftF1 = 학습에 쓸 수 있게 만든 F1 점수 (한 번에 16건씩 학습)", { x: rightX + 0.2, y: topY + 2.54, w: rightW - 0.4, h: 0.18, fontSize: 9.3, color: C.sub });

  // 하단: 3단계 최종 앙상블과 판정 — 가중치·임계값 고정.
  const bottomY = 5.31, bottomH = 1.55;
  card(s, M, bottomY, CW, bottomH);
  title(s, "■ 최종 앙상블과 판정", M + 0.21, bottomY + 0.08, CW - 0.42, 15);
  const boxY = bottomY + 0.44, boxH = 0.77, bw = 3.75, arrowW = 0.25;
  const steps = [
    ["STEP 1 · RoBERTa 3개 평균", "pR = (p42 + p43 + p44) / 3"],
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
  txt(s, "p42·p43·p44 = 시작값 42·43·44로 학습한 모델의 예측 확률 · 모든 증상에 같은 판정 기준(threshold) 0.5 적용, 증상별 조정·후처리 없음", { x: M + 0.2, y: bottomY + 1.27, w: CW - 0.4, h: 0.2, fontSize: 10.1, bold: true, color: C.blue, align: "center" });
  caption(s, "※ 입력 길이와 512토큰 실험은 Training 내부 80:20 홀드아웃 기준이며, 공식 Validation은 모델 선택에 사용하지 않음.", 7.02);
  s.addNotes("기존 p17 + p21 통합. 데이터: Training 29,200통화, 평가 전용 Validation 3,640통화, 9개 증상, Training 통화당 평균 증상 수 1.43. 라벨별 양성 비율(Training): 열상 23.2, 어지러움 21.7, 전신쇠약 18.2, 고열 17.8, 구토 15.3, 호흡곤란 13.4, 복통 11.9, 오심 11.4, 두통 9.9(%). 입력 길이와 512토큰 실험은 Training 내부 80:20 홀드아웃: 40.7% 256토큰 초과, Macro F1 +0.74%p, 사전 기준 +1.00%p 미달, 학습시간 약 4배. 전사 utterances[].text만 원본 순서로 연결하며 화자 태그와 허용되지 않은 라벨은 사용하지 않음. Soft macro-F1 상세: pᵢⱼ = sigmoid(zᵢⱼ), SoftF1ⱼ = 2Σᵢ(pᵢⱼyᵢⱼ) / [Σᵢpᵢⱼ + Σᵢyᵢⱼ + 10⁻⁸]. 가중치 wⱼ = clip(0.5 × √((N − Pⱼ) / max(Pⱼ,1)),1,6). 실효 batch 16 = micro 8 × 2, soft-F1 micro batch별 계산. 최종 RoBERTa seeds 42·43·44 확률 평균 가중치 55%와 TF-IDF 확률 가중치 45%를 결합, 모든 증상 threshold 0.5 적용(별도 후처리 없음). 근거: reports/eda/eda_report.md; docs/experiment_log.md M3-ROBERTA-007; submission/mission3/README.md; submission/mission3/train/train_final.py; submission/mission3/train/build_ensemble.py; submission/mission3/mission3_model.py.");
}

// =====================================================================
// 34. 별첨 2 — M3 입력 길이 확대 실험
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "별첨", "Mission 3", "별첨 2. Mission 3 입력 길이 확대 실험", 34);
  coreMsg(s, "512토큰은 성능이 개선됐지만 사전 채택 기준과 계산비용 조건을 충족하지 못함");

  const epochs = ["Epoch 1", "Epoch 2", "Epoch 3"];
  s.addChart(pres.charts.LINE, [
    { name: "512토큰 모델", labels: epochs, values: [64.39, 65.04, 65.15] },
    { name: "256토큰 기준 성능", labels: epochs, values: [64.41, 64.41, 64.41] },
    { name: "사전 채택 기준", labels: epochs, values: [65.41, 65.41, 65.41] },
  ], {
    x: M, y: 2.45, w: 7.0, h: 3.75, showTitle: true,
    title: "512토큰 모델의 epoch별 내부 홀드아웃 Macro F1 (%)", titleFontFace: F, titleFontSize: 11.5, titleColor: C.text,
    chartColors: [C.an2, C.an1, "9E9E9E"], showLegend: true, legendPos: "b", legendFontFace: F, legendFontSize: 9,
    showValue: false, lineSize: 2.2, lineDataSymbol: "circle", lineDataSymbolSize: 5,
    catAxisLabelFontFace: F, catAxisLabelFontSize: 10.5, catAxisLabelColor: C.text,
    valAxisLabelFontFace: F, valAxisLabelFontSize: 9.5, valAxisLabelColor: C.sub,
    valAxisMinVal: 63, valAxisMaxVal: 66, valAxisMajorUnit: 0.5, valLabelFormatCode: '0.0"%"',
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" },
    chartArea: { border: { color: C.white, pt: 0 }, fill: { color: C.white, transparency: 100 } },
    plotArea: { border: { color: C.white, pt: 0 }, fill: { color: C.white, transparency: 100 } },
  });
  // 512토큰 계열만 필요한 값 레이블을 별도 편집 가능한 텍스트로 표시한다.
  [["64.39%", 1.68, 4.52], ["65.04%", 3.97, 3.95], ["65.15%", 5.93, 3.84]].forEach(([v, x, y]) => {
    txt(s, v, { x, y, w: 0.7, h: 0.22, fontSize: 9.5, bold: true, color: C.an2t, align: "center" });
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
  table(s, expRows, 7.75, 2.48, [2.9, 2.18], {
    w: 5.08, rowH: [0.32, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.33], fontSize: 9.3, margin: 0.04,
    fill: C.white, border: { type: "solid", pt: 0.7, color: C.line },
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.75, y: 6.2, w: 5.08, h: 0.68, fill: { color: C.light }, rectRadius: 0.06, line: { color: C.sky, width: 1 } });
  txt(s, "성능은 개선됐지만 사전 기준에 미달했고 계산비용이 증가해 앞 256토큰만 읽는 방식을 유지", { x: 7.95, y: 6.2, w: 4.68, h: 0.68, fontSize: 11.5, bold: true, color: C.blue, align: "center", valign: "middle" });
  caption(s, "※ Training 내부 80:20 홀드아웃, seed 42 기준. 공식 Validation은 사용하지 않음.", 7.0);
  s.addNotes("근거: docs/experiment_log.md M3-ROBERTA-007, docs/decisions.md의 입력 512토큰 기각 결정");
}

// =====================================================================
// 35. 별첨 3 — Mission 3 낮은 성능 3개 라벨의 오류 구조
// =====================================================================
{
  const s = pres.addSlide();
  header(s, "별첨", "Mission 3", "별첨 3. Mission 3 낮은 성능 3개 증상의 오류 구조", 35);
  coreMsg(s, "직접 표현이 없는 양성이 많았고, 두통과 오심은 표현 유무에 따라 재현율 차이가 컸다");

  const labels = ["오심", "두통", "전신쇠약"];
  s.addChart(pres.charts.BAR, [
    { name: "직접 표현 있음 재현율", labels, values: [0.684, 0.820, 0.907] },
    { name: "직접 표현 없음 재현율", labels, values: [0.356, 0.195, 0.549] },
  ], {
    x: M, y: 2.46, w: 6.35, h: 3.7, barDir: "bar", barGrouping: "clustered", catAxisOrientation: "maxMin",
    chartColors: [C.an1, C.an2], showTitle: true, title: "직접 표현 유무별 재현율 · Training 내부 홀드아웃",
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
      headerCell("최종 모델 F1\n(공식 Validation)", { rowspan: 2 }),
      headerCell("Training 내부 홀드아웃 오류 진단 (5,792건)", { colspan: 3, fill: { color: diagFill } }),
    ],
    [
      headerCell("내부 홀드아웃\n양성 수", { fill: { color: diagFill } }),
      headerCell("직접 표현 없음", { fill: { color: diagFill } }),
      headerCell("내부 TF-IDF\nTP / FP / FN", { fill: { color: diagFill } }),
    ],
    ["오심", "40.00%", "668", "592건 · 88.6%", "263 / 754 / 405"],
    ["두통", "53.15%", "581", "370건 · 63.7%", "245 / 372 / 336"],
    ["전신쇠약", "59.39%", "1,062", "804건 · 75.7%", "675 / 846 / 387"],
  ];
  s.addTable(errRows, {
    x: 7.08, y: 2.52, w: 5.75, colW: [0.72, 1.3, 0.93, 1.3, 1.5], rowH: [0.48, 0.55, 0.56, 0.56, 0.56],
    fontFace: F, fontSize: 9.2, color: C.text, valign: "middle", align: "center", margin: 0.05,
    border: { type: "solid", pt: 0.7, color: C.line },
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.08, y: 5.45, w: 5.75, h: 0.72, fill: { color: C.light }, rectRadius: 0.06, line: { color: C.sky, width: 1 } });
  txt(s, "오심은 양성의 88.6%에서 직접 표현을 찾지 못했고, 두통은 표현이 없을 때 재현율이 19.5%까지 낮아졌다.", { x: 7.28, y: 5.5, w: 5.35, h: 0.35, fontSize: 11.2, bold: true, color: C.blue, align: "center", valign: "middle" });
  txt(s, "통화에서 관찰할 수 없는 정보가 성능을 제한했을 가능성", { x: 7.28, y: 5.86, w: 5.35, h: 0.22, fontSize: 9.5, color: C.sub, align: "center" });
  txt(s, [
    { text: "※ 최종 F1은 최종 모델(RoBERTa 3개 + TF-IDF)의 공식 Validation 결과(3,640건)", options: { breakLine: true } },
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
  coreMsg(s, "가중 BCE와 soft macro-F1, RoBERTa 확장과 함께 성능이 단계적으로 개선됐다.");
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
    rowH: [0.48, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.5], fontFace: F,
    fontSize: 11, color: C.text, margin: 0.065, valign: "middle",
    border: { type: "solid", pt: 0.65, color: C.line } });
  txt(s, "※ 가중 BCE: 희귀 증상을 놓칠 때 더 큰 벌점을 주는 학습 방식 · soft macro-F1: F1 점수 자체를 높이도록 학습 손실에 포함 · One-vs-Rest LR: 증상마다 따로 만든 로지스틱 회귀",
    { x: M, y: 6.3, w: CW, h: 0.3, fontSize: 9.8, color: C.sub, valign: "middle" });
  card(s, M, 6.66, 6.06, 0.38, C.light);
  txt(s, [{ text: "내부 검증  ", options: { bold: true, color: C.blue } },
    { text: "56.76% → 64.27%  (+7.51%p)", options: { bold: true } }],
    { x: M + 0.1, y: 6.68, w: 5.85, h: 0.3, fontSize: 12.5, valign: "middle", align: "center" });
  card(s, 6.77, 6.66, 6.06, 0.38, C.mintLight);
  txt(s, [{ text: "공식 Validation  ", options: { bold: true, color: C.blue } },
    { text: "Macro F1  65.34%", options: { bold: true } }],
    { x: 6.87, y: 6.68, w: 5.85, h: 0.3, fontSize: 12.5, valign: "middle", align: "center" });
  caption(s, "※ 56.76%~64.27%는 Training 내부 80:20 홀드아웃, 65.34%는 전체 Training 학습 후 공식 Validation 결과.", 7.12);
  s.addNotes("화면의 \"RoBERTa 3개\"는 seed 42·43·44. 가중 BCE·soft macro-F1·One-vs-Rest LR 풀이는 화면 하단 각주. 근거: docs/experiment_log.md, docs/decisions.md — M3-REEVAL-P0, M3-KCBERT-002·003·005·012, M3-ROBERTA-003, M3-FINAL-002. 내부 홀드아웃과 공식 Validation 수치를 직접 비교하지 않음.");
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
    ["정규화·도메인 적응", "M3-KCBERT-011·013", "61.32% / 61.18%", "KcBERT-005 60.92%", "개선폭이 시작값에 따른 흔들림 수준"],
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
  card(s, M, 6.68, CW, 0.34, C.light);
  txt(s, "모델 복잡화보다 클래스 불균형과 Macro F1을 반영한 손실 설계 쪽에서 더 큰 개선이 있었다.",
    { x: M + 0.16, y: 6.70, w: CW - 0.32, h: 0.28, fontSize: 12.2, bold: true,
      color: C.blue, align: "center", valign: "middle" });
  caption(s, "근거: docs/experiment_log.md, docs/decisions.md — M3-ROBERTA-007 등 주요 기각 실험", 7.12);
  s.addNotes("근거: docs/experiment_log.md, docs/decisions.md — M3-ROBERTA-007, M3-KCBERT-006·011·013·014·015, M3-E5-001, M3-KCELECTRA-001, M3-QWEN-001, M3-ROBERTA-004, M3-STRUCT-001, M3-ENS-001. 미리 정한 +1.00%p 개선 기준에 못 미치거나 비교 모델보다 낮은 주요 변형만 정리함.");
}

// 38. 마무리 — 기존 마무리 장 공식 템플릿 그대로 사용 + 팀 정보 유지
// =====================================================================
{
  const s = pres.addSlide();
  s.addImage({ path: TPL.closing, x: 0, y: 0, w: W, h: 7.5, sizing: { type: "cover", w: W, h: 7.5 } });
  txt(s, "팀 eigensinn  |  김장수 · 김광희 · 지연우 · 이수빈 · 조지은", { x: M, y: 2.2, w: CW, h: 0.4, fontSize: 16, bold: true, color: C.blue, align: "center" });
  txt(s, "코드 · 실험 기록: 제출 폴더(Mission 1·2·3)의 README와 학습 노트북 참고", { x: M, y: 2.58, w: CW, h: 0.3, fontSize: 11, color: C.sub, align: "center" });
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
const EXPECTED_SLIDE_COUNT = 36;
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
  headerPageNumbers.length !== 32 ||
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
