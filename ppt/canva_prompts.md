# Canva 일러스트 요청서 (Codex용)

PPT(`ppt/build_deck.js`)에 넣을 일러스트 4장을 Canva `generate-image`로 만든다. 결과 파일만 `ppt/figures/canva/`에 저장하고, 그 밖의 파일은 건드리지 않는다.

## 공통 규칙 (모든 프롬프트에 그대로 붙인다)

```
Flat 2D vector illustration, clean minimal corporate infographic style, soft rounded shapes, no gradients, no shadows, no outlines thicker than 2px.
Background: pure white (#FFFFFF), nothing behind the subject.
Color palette strictly: navy blue #2F4E91, sky blue #009DDA, light green #8BC34A, light blue tint #EAF3FA, light gray #E0E0E0, and a small red #E53935 accent only for emergency/alert details.
Absolutely NO text, NO letters, NO numbers, NO logos, NO watermarks. People are faceless simplified figures (no eyes/nose/mouth), no real persons.
```

## 파일·형식 규칙

- 형식: PNG, 가로 1920px 이상. 생성 결과가 200px 썸네일이면 `export-design`으로 1920px PNG를 내보내서 저장한다 (이미지가 실제로 들어갔는지 열어서 확인, 빈 흰 화면이면 실패).
- 저장 경로: `ppt/figures/canva/<파일명>.png`
- 다운로드 전 파일명·출처·크기를 사용자에게 알린다. 새 폴더에 받고, 열어서 확인한 뒤 옮긴다.
- 실제 통화 데이터·개인정보·실제 인물 얼굴은 쓰지 않는다.

## 프롬프트 1 — `hero_call_flow.png` (19장 · 2장용 대표 장면, 16:9)

```
[공통 규칙]
Wide landscape scene, left to right.
Left: a smartphone showing a simple sound-wave icon, with a thin curved line leaving it to the right.
Center: an abstract AI node made of three connected circles inside a soft rounded cloud, with small icons inside (a waveform, a person silhouette, a list), suggesting analysis.
Right: a 119 emergency dispatch operator, faceless, wearing a headset, seated at a desk in front of a monitor that shows generic abstract UI blocks, a small map pin, and a small red medical cross. Generous white space around, balanced composition.
```

## 프롬프트 2 — `speakers_overlap.png` (11장용, 2:1)

```
[공통 규칙]
Two faceless head-and-shoulders figures facing each other on a white background: left one wearing a headset (navy blue, the rescue operator), right one holding a smartphone to the ear (sky blue, the caller).
Between them, two sound-wave bundles, one navy and one sky blue, overlapping in the middle with the overlap area highlighted in a light red tint (#FFCDD2).
Clean, symmetrical, lots of white space.
```

## 프롬프트 3 — `symptoms_icons.png` (6·15장용, 3:2, 투명 배경 아이콘 9개)

```
[공통 규칙]
A tidy 3 x 3 grid of 9 simple round-badge medical symptom icons, each a navy blue circle with a white line icon, equal size and spacing, no labels:
fever (thermometer), vomiting (stomach), headache (head with a lightning mark), abdominal pain (torso with a pain mark), dizziness (spiral), laceration (bandage), nausea (queasy stomach wave), general weakness (drooping battery), shortness of breath (lungs).
Transparent or white background.
```

## 프롬프트 4 — `team_roles.png` (5장용, 5:1)

```
[공통 규칙]
A horizontal row of five faceless person busts in a consistent style, evenly spaced, each in a slightly different shade of navy and sky blue, each paired with one small round badge icon below: a gear, a headphone, a document, a chart, a git-branch symbol. No names, no text.
```

## 완료 후

- 4장을 확인하고 파일 경로를 알려 주면 Claude가 `build_deck.js`에 넣는다 (삽입 위치와 크기는 Claude가 정함).
- 마음에 안 드는 장은 프롬프트 한 줄만 바꿔서 재생성한다. 팔레트·무글자 규칙은 유지한다.
