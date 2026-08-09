# PyMuPDF 대체 검토

작성일: 2026-08-08
갱신: 2026-08-09 — **1·2단계 모두 적용 완료.** PyMuPDF 는 런타임 의존성에서
빠졌고 테스트 전용(`[dev]`)으로만 남는다. 2단계 결과는 §5, §6 에 있다.

PyMuPDF(AGPL v3)를 다른 라이브러리로 교체할 수 있는지, 교체한다면 무엇으로
할지에 대한 조사 기록. 발단은 두 가지다.

- `pyproject.toml` 은 MIT 를 선언하지만 PyMuPDF 는 AGPL v3 다. 배포 시점에
  실제 쟁점이 될 수 있다. **(해결 — pypdfium2(Apache-2.0/BSD) + pikepdf
  (MPL-2.0/QPDF) 로 교체)**
- `fonts.py` 274줄 전체가 "PyMuPDF 내장 폰트가 무엇을 인코딩할 수 있는가"를
  경험적으로 알아내는 데 쓰이고 있다. 이 복잡도가 정말 필요한지 의심스럽다.
  **(해결 — glyphless font 로 요구사항 자체가 사라졌다)**

-----

## 0. 현황: PyMuPDF 가 실제로 하던 일

| 기능 | 무엇으로 대체했나 |
|---|---|
| 페이지 → 이미지 렌더링 | `Page.to_image` → PDFium |
| 텍스트 추출 (`get_text`) | `Page.text` → PDFium |
| 불가시 텍스트 삽입 (`render_mode=3` + `morph` 행렬) | `Tm` 행렬을 직접 조립 (§5-1) |
| 레닥션 (`add_redact_annot` / `apply_redactions`) | 대체하지 않고 삭제 (§5-2) |
| 저장 / 가비지 수집 (`garbage=4`) | `pikepdf.Pdf.save` |
| ~~폰트 메트릭 (`text_length`)~~ | 해소됨 (고정 advance) |
| ~~내장 CJK 폰트 (`korea` / `japan` / `china-s`)~~ | 해소됨 (§1) |
| ~~폰트 서브셋팅 (`subset_fonts`)~~ | 해소됨 (폰트 프로그램 664바이트) |

내장 CJK 폰트가 진짜 잠금장치**였다**. `--font-file` 없이도 한글 OCR 이 동작한
이유는 MuPDF 가 Adobe-Korea1 계열 CMap 을 내장하고 있어서, 폰트 파일을
임베딩하지 않고 참조만으로 CID 폰트를 넣어주기 때문이었다. 이걸 그대로 제공하는
Python 라이브러리는 없다 — 그러나 §1 이후로는 필요하지도 않다.

-----

## 1. 핵심 발견 — 그 요구사항 자체가 불필요하다

기존 `fonts.py:3-5` 에 이미 적혀 있던 대로다.

> The overlay text is drawn with render mode 3 (invisible), so glyph shapes
> never matter -- only whether the PDF font can *encode* a code point.

글리프 모양이 아무 의미가 없다면, OCR 업계 표준 해법인 **glyphless font** 가
정확히 이 문제에 맞는다. Tesseract 와 ocrmypdf 가 쓰는 방식이다.

### 1-1. 정정: Tesseract 방식 그대로는 PyMuPDF 에서 동작하지 않는다

원래 이 문서와 `fonts.py:154-157` 의 주석은 "글리프 하나 + ToUnicode CMap"
구조를 전제했다. PyMuPDF 의 고수준 API 로는 **그 구조를 만들 수 없다.**

Tesseract 의 `pdf.ttf` 는 글리프가 1개뿐이고, CID 를 유니코드 코드포인트와
같게 두는 ToUnicode CMap 을 Tesseract 가 **직접** 써 넣는다. 반면
`Page.insert_text(fontfile=...)` 는 MuPDF 가 폰트의 cmap 을 역방향으로 훑어
ToUnicode 를 생성한다. 여러 코드포인트가 한 글리프를 공유하면 그 역매핑은
하나로 뭉개지고, 추출 결과는 전부 같은 문자가 된다.

**해법: 글리프를 코드포인트마다 따로 두되, 전부 비워 둔다.**

```
Type0 (BaseFont: /GlyphLess)
  └ Encoding: Identity-H
  └ DescendantFonts: CIDFontType2
      └ 글리프 65535개, 전부 윤곽선 없음, advance 500/1000 em 고정
      └ cmap format 4: 세그먼트 1개, idDelta 0  → glyph id = code point
  └ ToUnicode: MuPDF 가 자동 생성 (bfrange 256줄로 접힌다)
```

`loca` 는 전부 0, `hmtx` 는 `numberOfHMetrics = 1` 이라 뒤가 전부 0 이다. 즉
용량의 대부분이 0 의 나열이고, deflate 하면 사라진다.

### 1-2. 정정: 새 의존성이 필요하다는 판단도 틀렸다

`fonts.py` 옛 주석은 "fontTools 나 pikepdf 가 필요하고, `--font-file` 로 이미
답이 있는 문제에 새 의존성을 들이긴 아깝다"고 적었다. 실제로는 테이블이 10개,
그중 2개는 0 의 나열이라 `struct` 만으로 조립된다. 의존성 추가 없이 끝났다.

### 1-3. 실제로 얻은 것

- **BMP 전 영역을 인코딩**한다 → 스크립트 감지 자체가 불필요
- 서브셋팅 불필요, `_probe_batch` 경험적 프로빙 불필요
- `dropped_chars` 폴백 경고, `fallback_lines`, `--font-file`, `.ttc` 거부 로직
  (`check_font_file`) 전부 삭제
- `_save()` 의 xref 캐시 주의사항(서브셋팅이 진행 중 문서를 깨뜨리는 문제) 소멸
- 폭 계산이 `len(text) * 0.5 * fontsize` 산수로 축소
- `span_metrics()` 의 런타임 측정 제거 — 폰트의 ascent/descent 를 우리가
  정하므로 선언값과 뷰어 측정값이 일치한다 (테스트로 고정)

### 1-4. 줄 수는 줄지 않았다

문서의 예상은 "274줄 → 약 30줄"이었으나, 새 `fonts.py` 도 275줄이다. 프로빙
로직이 빠진 자리에 TrueType 테이블 조립이 들어왔기 때문이다.

다만 성격이 다르다. 없어진 쪽은 캐시·폴백·경험적 측정이 얽힌 **런타임 분기**
였고, 들어온 쪽은 입력이 없는 **결정론적 직렬화**다. 후자는 한 번 맞으면 다시
틀리지 않고, `build_font() == build_font()` 로 검증된다. 또 줄어든 곳이
`fonts.py` 만이 아니다 — `pdf_writer.py`, `pipeline.py`, `cli.py`, 테스트
4개에서 폰트 선택·서브셋팅·폴백 경로가 모두 사라졌다.

바이너리 폰트를 gzip+base64 로 소스에 박으면 40줄이 되지만, advance 와
ascent/descent 를 조정할 수 없는 불투명한 블롭이 된다. 조립 코드를 택했다.

### 1-5. 확인된 주의사항

- tesseract#373 의 문자 간격 이슈는 대체로 해당 없음. `width_scale` 이 어차피
  박스 폭에 맞추므로 advance 값 자체는 결과에 남지 않는다. 단 결합 문자와
  큰 스트레치가 겹치면 흔적이 남는다 — §6-4.
- CJK 반각/전각 구분은 사라진다. 불가시 레이어이므로 외관 문제는 없고, 폭 피팅
  결과도 §4 에서 회귀가 없음을 확인했다.
- **BMP 밖(U+FFFF 초과)은 인코딩되지 않는다.** 묵음 처리하지 않고 실행 끝에
  보고한다. PaddleOCR 이 내보내는 문자에는 없다.
- RTL(아랍어 등)은 시각 순서 문제가 있을 수 있다. 이전에는 아예 인코딩되지
  않던 문자라 후퇴는 아니다.
- `--resume` 은 폰트를 한 벌 더 임베딩한다. 다시 연 문서에서 기존 폰트 객체를
  찾아 쓰지 않기 때문이다. 중단 1회당 2 KB 남짓이고 페이지 수와 무관하므로
  고치지 않았다. `tests/test_pipeline.py` 의 크기 상한이 이를 붙잡아 둔다.

### 1-6. 정정(2단계에서): 글리프 65535개는 필요 없었고, 비워 두면 안 됐다

1-1 의 "코드포인트마다 글리프 하나씩" 구조는 **MuPDF 가 ToUnicode 를 자동
생성한다는 제약** 때문에 나온 것이다. 2단계에서 ToUnicode 를 직접 쓰게 되자
그 제약이 사라졌고, 대신 **`CIDToGIDMap` 스트림**으로 65536개 CID 를 전부
글리프 1번에 보내면 된다. 반복되는 두 바이트라 deflate 후 150바이트다.

그리고 그 글리프는 **비어 있으면 안 된다.** PDFium 은 문자 크기를 글리프
바운딩 박스에서 재고, 폭이 0으로 나오는 문자를 버린다. 빈 글리프는 높이가
0이라 가로쓰기에서는 문제가 없지만, 줄이 정확히 90도 돌아가는 순간 — 옆으로
스캔된 페이지나 `/Rotate 90` 이 붙은 페이지 — 그 0이 폭이 되어 **PDFium 이
그 줄을 통째로 추출하지 못한다.** MuPDF 에서는 정상으로 보이므로 그냥 나갈
수 있는 종류의 결함이고, 실제로 PyMuPDF 판 출력도 같은 증상이었다(1단계까지
포함해 계속 있던 문제다). 면적 0짜리 2점 컨투어를 넣으면 바운딩 박스가
생기면서 사라진다. `tests/test_fonts.py` 의
`test_a_line_running_down_the_page_survives_too` 가 이걸 붙잡는다.

결과적으로 폰트는 글리프 2개(`.notdef` + 1개), **664바이트**가 됐다. 1단계의
385 KB / deflate 후 777바이트에서 또 줄었고, `loca`·`hmtx` 의 0 나열과 format 4
identity cmap 도 사라졌다. 문서당 텍스트 레이어 비용의 내역은 이렇다.

| 객체 | 원본 | deflate 후 |
|---|---|---|
| FontFile2 | 664 B | 366 B |
| CIDToGIDMap | 128 KB | 150 B |
| ToUnicode CMap | 5.6 KB | 1,692 B |

(`fonts.py` 는 그래도 275줄 → 402줄로 늘었는데, 줄어든 만큼보다 MuPDF 가 대신
써 주던 Type0/CIDFontType2 딕셔너리와 ToUnicode CMap 이 더 크기 때문이다.
§5-4 참고.)

-----

## 2. 선택지

### A. pypdfium2 + pikepdf — 라이브러리만 교체

| 역할 | 라이브러리 | 라이선스 |
|---|---|---|
| 렌더링 / 텍스트 추출 | `pypdfium2` | Apache-2.0 (PDFium 은 BSD) |
| 텍스트 레이어 작성 | `pikepdf.canvas` | MPL-2.0 (QPDF) |
| 저장 / 병합 | `pikepdf` | MPL-2.0 |
| 폰트 | glyphless font (`fonts.py`, 이미 있음) | — |

`pikepdf.canvas` 의 `ContentStreamBuilder` 에 `set_text_matrix()`,
`set_text_rendering()`, `horiz_scale()` 이 모두 있어서 현재 `morph` 행렬
로직(`pdf_writer.py:190-208`)을 거의 그대로 옮길 수 있다.

**단, pikepdf.canvas 는 TTF 임베딩을 지원하지 않는다.** `SimpleFont.encode()`
는 WinAnsiEncoding / MacRomanEncoding 만 처리한다. 따라서 이 경로는
glyphless font 가 필수 전제인데, 그건 이제 갖춰져 있다. 폰트 딕셔너리는
pikepdf 로 직접 구성해야 하지만 한 번만 짜면 된다. 유의할 점은 MuPDF 가
자동으로 써 주던 **ToUnicode CMap 을 직접 써야 한다**는 것 — identity bfrange
한 줄이면 되므로 부담은 아니다.

### B. fpdf2 추가

ocrmypdf 가 기본 렌더러(`--pdf-renderer auto`)로 쓰는 조합. glyphless font 가
이미 구현되어 있으나, 우리도 이제 갖고 있으므로 A 대비 이점이 사라졌다.

단점: fpdf2 는 LGPL-3.0 이다. Python 패키지 의존은 정적 링크가 아니라 대체로
문제되지 않지만, 배포 형태(단일 바이너리 패키징 등)에 따라 확인이 필요하다.

### C. ocrmypdf 플러그인으로 전환 — 가장 급진적

ocrmypdf 는 `get_ocr_engine()` 훅으로 커스텀 OCR 엔진을 받는다. `OcrEngine`
추상 클래스에서 구현할 것은 `version()`, `languages()`, `generate_hocr()`,
`generate_pdf()`, `get_orientation()` 정도다.
**[OCRmyPDF-EasyOCR](https://github.com/ocrmypdf/OCRmyPDF-EasyOCR) 이 정확히
같은 패턴의 선례다.**

얻는 것: `pdf_writer.py`, `pipeline.py` 의 대부분이 사라지고 PaddleOCR 어댑터만
남는다. 페이지 병렬화, 사이드카, PDF/A, 이미지 최적화가 공짜로 따라온다.

잃는 것 — 이쪽이 더 무겁다:

- **width fitting** (`pdf_writer.py:118-125`). 추출 오류율 16.4% → 대폭 개선을
  만든 바로 그 로직이다. ocrmypdf 의 hOCR 좌표 모델 안에서 재현해야 하는데,
  hOCR 은 단어/줄 바운딩 박스만 전달하므로 현재의 4점 폴리곤 기반 회전·기울기
  처리(`pdf_writer.py:158-169`)를 그대로 표현하기 어렵다.
- `--resume` 체크포인트 설계(`pipeline.py:271-281`)가 ocrmypdf 파이프라인과
  충돌한다.
- `bench/` 기준선을 다시 잡아야 한다.

-----

## 3. 권고 — A, 단 2단계로

순서가 중요하다. 한 번에 다 바꾸면 정확도 회귀의 원인을 특정할 수 없다.

### 1단계: glyphless font 만 먼저, PyMuPDF 위에서 — **완료**

라이브러리 교체 없이 `fonts.py` 를 먼저 갈아끼웠고, `bench/` 로 정확도 회귀를
즉시 측정했다(§4).

적용 범위:

- `fonts.py` — glyphless TrueType 조립 + 고정 메트릭으로 전면 교체
- `pdf_writer.py` — `FontResolver` 제거, `OverlayStats.dropped_chars` 추가
- `pipeline.py` — `font_file` 인자와 `subset_fonts()` 경로 제거
- `cli.py` — `--font-file` 및 검증 콜백 제거
- `tests/` — `test_fonts.py` 전면 재작성, 나머지 3개 파일 정리
- 독립 엔진 교차 검증을 테스트에 상설화 (pypdfium2 = PDFium)

### 2단계: pypdfium2 + pikepdf 로 교체 — **완료**

라이선스 문제(AGPL → MPL + Apache)는 이 단계에서 해소됐다. 적용 범위:

- `pdf_reader.py` 삭제 → `pdf_document.py` 신설. `Document` / `Page` 가
  pikepdf(편집)와 PDFium(렌더링·텍스트 추출) 두 핸들을 함께 들고 있다.
- `fonts.py` — `embed_font()`, ToUnicode CMap, `CIDToGIDMap` 추가.
  `font_path()` 의 임시 파일 배관 삭제. 폰트 자체는 §1-6 대로 축소.
- `pdf_writer.py` — `insert_text(morph=...)` 대신 `Tm` 행렬을 직접 조립.
  `remove_text_layer` 삭제, `rasterize_page` → `Page.replace_with_image`.
- `pipeline.py` — 페이지를 두 번 렌더링하던 것을 한 번으로.
- `pyproject.toml` — PyMuPDF 를 `[dev]` 로 이동, pikepdf/pypdfium2 추가.
- `tests/helpers.py` 신설 — 입력 PDF 생성과 출력 검증을 MuPDF 로 몰아
  "쓰는 엔진과 읽는 엔진이 다르다"를 스위트 전체의 성질로 만들었다.

미리 잡아 둔 숙제(`remove_text_layer`)는 예상대로 레닥션을 흉내낼 필요가
없었다. 그 함수는 `rasterize_page` 안에서만 쓰였고, 의도는 "페이지를 통째로
이미지로 교체"하는 것이었기 때문에 **렌더링한 이미지만 담은 페이지로
덮어쓰는** 쪽이 더 짧고 확실하다. 단독 공개 API 로 유지할 이유도 없어 삭제했다.

-----

## 4. 1단계 실측

`scripts/bench.py run baseline --name glyphless` 기준. 판정 기준은 문서가 정한
대로 **pdf 점수 유지**다. (ocr 점수는 인식기 출력이라 폰트와 무관하다.)

| 코퍼스 | 변형 | ocr CER-ns | pdf CER-ns | out KB |
|---|---|---|---|---|
| sample-1 (한국어 서적) | baseline | 0.0112 | 0.0112 | 5828 |
| sample-1 | **glyphless** | 0.0112 | 0.0112 | 5832 |
| sample-2 (한국어 리플릿) | baseline | 0.0174 | 0.0174 | 4557 |
| sample-2 | **glyphless** | 0.0174 | 0.0174 | 4560 |
| sample-3 (라틴 활판) | baseline | 0.0029 | 0.0029 | 128 |
| sample-3 | **glyphless** | 0.0029 | 0.0029 | 133 |

**세 코퍼스 모두 소수점 이하 전 자릿수가 동일하다.** 회귀 없음.

정확도가 한 자리도 움직이지 않은 것은 우연이 아니라 설계의 결과다. 줄의 크기와
위치는 폰트가 선언한 ascent/descent 로 정해지는데, 우리가 그 값을 정하므로
"박스에 맞춘 span" 이라는 목표가 근사가 아니라 항등식이 된다. 폭도 마찬가지로
`width_scale` 이 박스 폭에 맞추므로 advance 값 선택이 결과에 남지 않는다.

크기 비용은 문서 하나당 **2.5–5 KB** 다. 내역은 FontFile2 777바이트(고정) +
ToUnicode CMap 1.7–4.2 KB 이며, 페이지 수와 무관하게 문서당 한 번만 든다.
sample-3 의 +5 KB 가 가장 큰데, 이전에 라틴 전용으로 임베딩 없는 Base14
`helv` 를 1바이트 인코딩으로 쓰던 문서이기 때문이다. 한국어 문서는 이전에도
CID 폰트였으므로 증가분이 상대적으로 작다.

속도는 사실상 동일하다(측정 노이즈 범위). 폰트 조립은 프로세스당 한 번,
1밀리초 미만이고 런타임 프로빙이 사라진 만큼 상쇄된다.

-----

## 5. 2단계에서 배운 것

### 5-1. `morph` 를 대신한 것은 행렬 하나다

`Page.insert_text(point, ..., morph=(pivot, matrix))` 가 하던 일은 결국
"텍스트를 가로로 쓰고, baseline 을 축으로 회전·수평 스케일을 걸어라"였다.
이걸 PDF 연산자로 직접 쓰면 `Tm` 행렬 하나가 된다.

핵심은 좌표계가 셋이라는 점이다. OCR 이 주는 좌표(페이지 공간: 좌상단 원점,
y 아래로)와 PDF user space(좌하단 원점, y 위로)는 다르고, 그 사이에
`/Rotate` 와 crop box 가 끼어 있다. 이걸 `Page.placement` 한 곳에 몰아넣고,
줄마다 만드는 행렬을 거기에 합성한다.

```
Tm = [s·cosθ  s·sinθ ]  ·  placement
     [  sinθ   -cosθ ]
     [   bx      by  ]
```

`placement` 는 `/Rotate` 4가지 경우의 2×3 행렬 표뿐이다. PyMuPDF 는 이
변환을 내부에서 처리해 주고 있었고, 대신 그게 무엇인지 코드에 남지 않았다.
지금은 `test_page_space_is_the_rendered_image` 가 네 방향 전부에 대해
"페이지 공간 좌표로 사각형을 그리면 렌더링 이미지의 그 픽셀에 나온다"를
직접 확인한다.

### 5-2. 레닥션은 흉내낼 필요가 없었다

§3 에서 예상한 대로다. `rasterize_page` 는 "텍스트를 지우고 렌더링을 덮어
그린다"였는데, 실제로 필요한 것은 "렌더링만 남긴다"였다. 페이지의
`/Contents`·`/Resources` 를 이미지 하나로 갈아끼우면 끝이고, 결과는 오히려
더 작다 — 예전 방식은 레닥션 후에도 원본 이미지가 남아 있어서 원본과
렌더링이 **둘 다** 파일에 들어갔다.

주의점 하나: `/Rotate` 와 `/CropBox` 는 페이지 트리에서 상속될 수 있다.
페이지 딕셔너리의 키를 지우면 부모 것이 드러나 렌더링이 한 번 더 돌아간다.
지우지 말고 덮어써야 한다.

### 5-3. 렌더링 핸들과 편집 핸들은 분리해도 된다

pikepdf 로 편집한 내용은 PDFium 핸들에 보이지 않는다. 얼핏 함정 같지만
파이프라인이 이미 그 순서로 돌아간다 — 페이지는 **쓰기 전에** 렌더링된다.
오히려 이 제약 덕에 `--force-ocr` 경로에서 페이지를 두 번 렌더링하던 것이
한 번으로 줄었다. 예전 코드는 `rasterize_page` 안에서 한 번, 그 다음
`page_to_image` 에서 또 한 번 렌더링하고 있었다.

### 5-4. 줄 수는 늘었다 — MuPDF 가 해 주던 몫만큼

`src/` 합계 1613줄 → 1971줄(+358). 내역은 대략 이렇다.

| 파일 | 변화 | 이유 |
|---|---|---|
| `pdf_reader.py` → `pdf_document.py` | 30 → 287 | 문서/페이지 모델, 좌표계, 이미지 교체 |
| `fonts.py` | 275 → 403 | Type0/CIDFontType2 딕셔너리 + ToUnicode CMap |
| `pdf_writer.py` | 211 → 188 | `morph` → `Tm`, `remove_text_layer` 삭제 |

늘어난 것은 전부 **PDF 파일 형식 자체**를 다루는 코드다. 라이선스를 위해
치른 값이 이 358줄이고, 대신 그 358줄이 하는 일이 코드에 적혀 있다 — §1-6 의
PDFium 결함처럼, PyMuPDF 뒤에 있었을 때는 볼 수도 고칠 수도 없던 것을
고칠 수 있게 된 것도 같은 이유다.

-----

## 6. 2단계 실측

`scripts/bench.py run baseline --name pikepdf` 기준. 비교 대상은 1단계 결과인
`glyphless` 다.

| 코퍼스 | 변형 | ocr CER-ns | pdf CER-ns | 오류 문자 | out KB |
|---|---|---|---|---|---|
| sample-1 (한국어 서적) | glyphless | 0.0112 | 0.0112 | 27 / 2417 | 5832 |
| sample-1 | **pikepdf** | 0.0103 | 0.0103 | 25 / 2417 | **4917** |
| sample-2 (한국어 리플릿) | glyphless | 0.0174 | 0.0174 | 18 / 1037 | 4560 |
| sample-2 | **pikepdf** | 0.0145 | 0.0145 | 15 / 1037 | **3919** |
| sample-3 (라틴 활판) | glyphless | 0.0029 | 0.0029 | 6 / 2072 | 133 |
| sample-3 | **pikepdf** | 0.0039 | 0.0039 | 8 / 2072 | **117** |

### 6-1. 판정 기준은 `ocr == pdf` 다

**세 코퍼스 모두 ocr 과 pdf 가 모든 자릿수까지 같다.** 이게 이 단계에서
실제로 확인해야 할 것이다. 인식기가 읽은 문자열이 텍스트 레이어를 지나
(우리가 쓰지 않는) MuPDF 로 추출될 때까지 한 글자도 변하지 않았다는 뜻이고,
`Tm` 행렬·Identity-H 인코딩·ToUnicode CMap 을 직접 쓰게 된 이 단계에서
깨질 수 있었던 것이 전부 이 한 등식에 걸린다.

### 6-2. 움직인 것은 인식 쪽이고, 원인은 래스터라이저다

절대 오류 수로는 sample-1 27→25, sample-2 18→15, sample-3 6→8 이다. 부호가
엇갈리고 크기가 한 자릿수라 개선도 회귀도 아니다 — 페이지를 MuPDF 대신
PDFium 이 렌더링하면서 픽셀이 미세하게 달라졌고, 인식기가 그만큼 다르게
읽었을 뿐이다. `ocr` 과 `pdf` 가 함께 움직인 것이 그 증거다. 텍스트 레이어
탓이라면 `pdf` 만 움직였을 것이다.

### 6-3. 출력은 12–16% 작아졌다

sample-1 5832→4917 KB, sample-2 4560→3919 KB. 원인은 §5-2 다. 예전
`rasterize_page` 는 레닥션으로 텍스트만 지우고 그 위에 렌더링을 덮어
그렸으므로 **원본 이미지와 렌더링이 둘 다** 파일에 남아 있었다. 지금은
렌더링만 남는다. `--force-ocr` 을 쓰지 않는 sample-3(133→117 KB)의 감소는
폰트 축소와 pikepdf 의 오브젝트 스트림 몫이다.

### 6-4. 덤으로 발견한 것: 폭 스트레치와 결합 문자

교차 검증용 MuPDF 를 1.27.2 → 1.28.2 로 올리자 태국어·데바나가리 라운드트립
테스트가 깨졌다. 원인은 우리 쪽 결함이 아니라 **MuPDF 1.28 의 줄 분할
휴리스틱**이다. 이 폰트는 모든 글리프의 advance 가 0.5 em 이고 `width_scale`
이 그걸 박스 폭에 맞춰 늘리는데, 결합 문자(태국어 모음 부호, 데바나가리
비라마)는 뒤가 아니라 위에 얹히는 것이라 MuPDF 가 기대하는 간격과 어긋난다.
스트레치가 3배쯤 되면 그 간격이 줄바꿈으로 읽힌다.

- 문자가 사라지지는 않는다. `'สวัสดีครับ'` 이 `'สวั\nสดี\nครั\nบ'` 이 될 뿐이다.
  즉 데이터 손실이 아니라 **단어 검색이 안 되는** 문제다.
- PDFium 은 스트레치가 얼마든 온전히 추출한다. 대부분의 뷰어가 PDFium 계열이다.
- 실제 페이지는 이 배율에 닿지 않는다. 검출 박스는 그 안에 든 줄의 폭이다.
  테스트 픽스처가 10글자를 500pt 박스에 넣고 있었을 뿐이다.
- 2단계와 무관하게 1단계부터 있던 성질이다. `width_scale` 은 CER 16.4% →
  1.2% 를 만든 바로 그 로직이라 건드리지 않았다.

`tests/test_fonts.py::TestStretchingALinePastItsBox` 가 이 경계를 그대로
고정해 둔다. MuPDF 가 나중에 이 동작을 바꾸면 그 테스트가 알려 준다.

### 6-5. 속도와 메모리는 사실상 같다

`s/page` 는 sample-1 11.8→13.6, sample-2 24.0→26.5, sample-3 11.8→12.8 이다.
같은 sample-1 을 두고 `baseline` 17.3 / `conf-0.3` 14.6 / `glyphless` 11.8 이
나오는 코퍼스이므로 전부 측정 노이즈 범위다. 피크 RSS 도 마찬가지다 —
대부분이 PaddleOCR 모델이라 PDF 라이브러리 몫은 보이지 않는다.

-----

## 7. 참고

- [pikepdf](https://pikepdf.readthedocs.io/)
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
- [OCRmyPDF plugins](https://ocrmypdf.readthedocs.io/en/latest/plugins.html)
- [OCRmyPDF-EasyOCR PDF generation](https://deepwiki.com/ocrmypdf/OCRmyPDF-EasyOCR/3.1-pdf-generation)
- [OCRmyPDF advanced features](https://ocrmypdf.readthedocs.io/en/latest/advanced.html)
- [Glyphless font 간격 이슈 (tesseract#373)](https://github.com/tesseract-ocr/tesseract/issues/373)
- [PDF 32000-1:2008 §9.4.4 Text space details](https://www.pdfa.org/resource/pdf-specification-index/)
- [OpenType cmap format 4](https://learn.microsoft.com/typography/opentype/spec/cmap#format-4-segment-mapping-to-delta-values)
