# 영상 생성 AI용 프롬프트: "한 통화가 처리되는 과정" (새 19장)

`slide_prompt_demo.md`는 코드로 직접 그리는 작업용이고, 이 문서는 **영상 생성 AI에 그대로 붙여 넣는 프롬프트**다. 글자와 정확한 도형은 생성형 AI가 못 지키므로, 영상에는 **글자·숫자 없이 그림과 움직임만** 담고 라벨·캡션은 슬라이드에서 PowerPoint 텍스트로 얹는다.

## 0. 이전 결과에서 얻은 교훈 (프롬프트에 반영됨)

- 둥근 도형 가장자리가 깨지거나 흔들렸다 → "선명한 평면 도형, 질감·노이즈·그라데이션 없음"을 명시한다.
- 움직임이 약했다 → 단계마다 **밝기·색이 켜지는 변화**를 크게 요구한다.
- 첫 장면과 끝 장면이 달라 반복이 튀었다 → "끝 장면이 곧 완성 상태, 마지막에서 처음 상태로 부드럽게 되돌아감"을 요구한다.
- 영상 안 글자는 한글이 깨진다 → 글자 금지.
- **색상 코드(`#2F4E91` 등)를 프롬프트에 적으면 영상 AI가 그 코드를 글자로 그려 넣는다 (실제 발생: 휴대전화와 블록 위에 `#E0E0E0`, `#2F4E91` 같은 글자가 박힘).** → 프롬프트에는 색을 **단어로만** 쓰고 `#`·16진수를 절대 쓰지 않는다. 색 코드는 이 문서의 슬라이드 쪽 팔레트 참고용으로만 §1에 적는다.
- 영상 AI가 배경에 회색 띠를 깔고, 마지막에 어두운 처음 상태로 돌아가 끝났다(완성 화면이 없음) → 배경 전체 순백과 "완성 상태에서 끝"을 명시한다. 반복 연결은 GIF 변환 단계에서 크로스페이드로 처리한다.

## 1. 생성 설정

- 길이 10~12초 (모델 한도가 5초면 아래 "분할 버전"으로 5개 클립 생성 후 이어 붙임)
- 해상도 1920×1080, 16:9, 24~30fps, **카메라 고정**, 오디오 없음
- **구도 제한**: 모든 내용은 화면 **가운데 가로 띠(세로 55%)** 안에만 둔다. 위·아래 22%는 순백 여백 (슬라이드에는 2.8:1로 잘라 쓴다).
- 배경 전체 순백(회색 띠·사각 배경 없음). 색은 네이비, 스카이블루, 연두, 아주 연한 파랑, 연회색, 아주 작은 빨강 강조만.
- (슬라이드 팔레트 참고용 코드 — **영상 프롬프트에는 넣지 말 것**: 네이비 2F4E91, 스카이 009DDA, 민트 8BC34A, 연한 파랑 EAF3FA, 회색 E0E0E0, 빨강 E53935)

## 2. 메인 프롬프트 (영어, 그대로 붙여 넣기)

```
A clean flat 2D vector motion-graphics animation. The entire frame is pure white from edge to edge: no gray bands, no panels, no rectangles behind the content. Static camera, no camera movement.
Everything stays inside the horizontal middle band of the frame, with large empty white space above and below.
Crisp solid-color shapes with sharp smooth edges, no gradients, no textures, no film grain, no noise, no shadows, no glow.
Color palette strictly limited to: deep navy blue, bright sky blue, fresh light green, very pale blue, light gray, and tiny red accents only. Never write any color name or color code as text anywhere in the image.
There is absolutely no text, no letters, no numbers, no logos, no watermark, no human faces. People appear only as faceless simplified silhouettes.

Layout, left to right in one wide row:
LEFT: a rounded smartphone icon. A horizontal sound waveform made of vertical bars flows out of it and runs to the right, splitting into four separate waveform blocks side by side (call it block 1, 2, 3, 4), shown on a thin timeline.
CENTER: three round badges stacked vertically, each a navy circle with a simple white icon (a headset-and-speaker silhouette, a person silhouette, a speech-bubble-with-lines icon). They are dim gray at the start.
RIGHT: a rounded rectangle monitor card representing a dispatch screen, initially empty, with a small person icon slot, two small empty tag slots, and a small gray warning-triangle icon at the bottom.

Animation sequence (about 12 seconds), each step clearly changes color or brightness:
1) 0-2s: the waveform flows from the phone and splits into four blocks, all neutral light gray.
2) 2-5s: the first badge (headset icon) lights up navy. The four blocks are highlighted one at a time, strictly one by one, the other three staying dimmed gray while one is active; each highlighted block turns either navy or sky blue (blocks 1 and 3 navy, blocks 2 and 4 sky blue). No lines or arrows ever connect one block to another.
3) 5-7.5s: the second badge (person icon) lights up. Only the sky blue blocks (2 and 4) slide together toward the center and a small round badge appears on the dispatch card's person slot. The navy blocks fade to light gray and stay unused.
4) 7.5-10s: the third badge (speech bubble icon) lights up. A small pale document panel appears under the waveform with several abstract horizontal text lines (just gray bars, no readable characters); two of the bars get highlighted in light green, and two round symptom icons (a head icon and a spiral icon) light up navy.
5) 10-12s: the dispatch card fills in: the person slot becomes a sky blue badge, two tag slots become the head icon and the spiral icon, and the warning-triangle icon turns small red. All three badges are lit.
The video ends on the completed final state with all three badges lit and the dispatch card filled, and holds still for the last 2 seconds. Do NOT return to the dim starting state; the final frame must look complete and fully readable as a still image.
```

## 3. Negative prompt (생성 AI에 별도 입력란이 있으면)

```
text, letters, numbers, hex color codes, hash symbol, color names written as text, labels, Korean characters, gray background band, gray panel, vignette, captions, subtitles, logo, watermark, human face, realistic photo, 3D render, gradient, texture, noise, film grain, blur, glow, shadow, camera movement, zoom, pan, shaky, flicker, morphing shapes, distorted circles, jagged edges, extra objects, background scene, busy details
```

## 4. 분할 버전 (모델이 5초 이하만 지원할 때 — 클립 5개, 각 2~3초)

공통 앞부분(스타일·레이아웃·금지 사항)은 메인 프롬프트의 첫 두 문단과 Layout을 그대로 붙이고, 아래 한 문장만 바꿔 클립마다 생성한다. 각 클립의 **시작 프레임은 직전 클립의 마지막 프레임**을 이미지 입력으로 준다(지원하는 모델일 때).

| 클립 | 동작 |
|---|---|
| A | 파형이 휴대전화에서 나와 4개 블록으로 갈라진다. 모두 연한 회색. |
| B | 첫 배지가 켜진다. 블록이 하나씩만 강조되며 네이비/스카이로 칠해진다 (1·3 네이비, 2·4 스카이). 블록끼리 연결선 없음. |
| C | 둘째 배지가 켜진다. 스카이 블록 둘만 가운데로 모이고 카드의 사람 슬롯에 배지가 생긴다. 네이비 블록은 회색으로 사라진다. |
| D | 셋째 배지가 켜진다. 문서 패널의 회색 막대 두 개가 연한 초록으로 강조되고 증상 아이콘 둘이 켜진다. |
| E | 상황실 카드가 채워지고 경고 아이콘이 작은 빨강으로 바뀐다. 마지막 2초 완성 화면 유지, 처음 상태로 되돌아가지 않음. |

## 5. 슬라이드에 얹을 글자 (영상이 아니라 PowerPoint 텍스트로)

- 단계 라벨(배지 옆): `Mission 2 · 발화별 화자 분류`, `Mission 1 · 신고자 성별`, `Mission 3 · 증상 태깅`
- 합성 문구 캡션: `※ 합성 예시 — 실제 통화·전사문이 아니며, 모든 출력은 요원의 판단을 돕는 정보로 사람이 최종 확인한다.`
- 핵심 메시지 한 줄: `신고 전화 한 통이 들어오면 세 모델이 순서대로 결과를 정리한다.`

## 6. 결과 검수 체크리스트 (하나라도 어긋나면 재생성)

- [ ] 영상 안에 글자·숫자·**색상 코드(`#`로 시작하는 글자)**·얼굴이 없다 (정지해서 휴대전화, 블록, 카드 위를 확대해 확인).
- [ ] 배경 전체가 순백이다 (회색 띠 없음).
- [ ] 마지막 프레임이 완성 상태다 (어두운 시작 상태로 끝나면 재생성).
- [ ] 블록끼리 연결선·화살표가 없고, 블록이 **하나씩** 판정되는 것으로 보인다 (M2 규칙: 다른 발화 정보 사용 불가).
- [ ] 성별 판정에는 **신고자 블록만** 쓰인다 (대원 블록은 쓰이지 않음).
- [ ] 증상 단계는 **음성이 아니라 텍스트 패널**에서만 일어난다.
- [ ] 새 성능 수치·확률·점수 표기가 없다.
- [ ] 원형 도형 가장자리가 매끈하다 (확대해서 확인).
- [ ] 마지막 화면이 완성 상태이고, 처음으로 돌아갈 때 튀지 않는다.
- [ ] 내용이 화면 가운데 띠 안에 있어 2.8:1로 잘라도 잘리는 요소가 없다.

## 7. 받은 뒤 처리 (Claude/Codex가 수행)

1. `ppt/videos/`에 원본 MP4 저장 후 `ppt/tools/mp4_to_gif.py`로 가운데 띠를 잘라 GIF 변환 (첫 프레임이 완성 화면이 되도록).
2. `ppt/slide_prompt_demo.md`의 §1·§2·§5대로 새 19장 삽입, 페이지 번호·목차 갱신, 기존 19장 상단 흐름 GIF 삭제.
3. 둥근 도형이 깨지면 GIF 팔레트를 늘리거나(`colors=96`) 해상도를 높인다. 그래도 깨지면 정지 완성 화면만 쓴다.
