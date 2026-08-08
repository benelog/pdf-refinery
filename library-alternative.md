# PyMuPDF 대체 검토

작성일: 2026-08-08

PyMuPDF(AGPL v3)를 다른 라이브러리로 교체할 수 있는지, 교체한다면 무엇으로
할지에 대한 조사 기록. 발단은 두 가지다.

- `pyproject.toml` 은 MIT 를 선언하지만 PyMuPDF 는 AGPL v3 다. 배포 시점에
  실제 쟁점이 될 수 있다.
- `fonts.py` 274줄 전체가 "PyMuPDF 내장 폰트가 무엇을 인코딩할 수 있는가"를
  경험적으로 알아내는 데 쓰이고 있다. 이 복잡도가 정말 필요한지 의심스럽다.

-----

## 0. 현황: PyMuPDF 가 실제로 하는 일

| 기능 | 코드 위치 | 대체 난이도 |
|---|---|---|
| 페이지 → 이미지 렌더링 | `pdf_reader.py:14`, `pdf_writer.py:72` | 쉬움 |
| 텍스트 추출 (`get_text`) | `pdf_writer.py:40,54` | 쉬움 |
| 불가시 텍스트 삽입 (`render_mode=3` + `morph` 행렬) | `pdf_writer.py:180` | 중간 |
| 폰트 메트릭 (`text_length`) | `fonts.py:166` | 중간 |
| 레닥션 (`add_redact_annot` / `apply_redactions`) | `pdf_writer.py:59` | 어려움 |
| **내장 CJK 폰트 (`korea` / `japan` / `china-s`)** | `fonts.py:19-22` | **대체 불가** |
| 폰트 서브셋팅 (`subset_fonts`) | `pipeline.py:121` | 어려움 |

내장 CJK 폰트가 진짜 잠금장치다. `--font-file` 없이도 한글 OCR 이 동작하는
이유는 MuPDF 가 Adobe-Korea1 계열 CMap 을 내장하고 있어서, 폰트 파일을
임베딩하지 않고 **참조만으로** CID 폰트를 넣어주기 때문이다. 이걸 그대로
제공하는 Python 라이브러리는 없다.

-----

## 1. 핵심 발견 — 그 요구사항 자체가 불필요하다

`fonts.py:3-5` 에 이미 적혀 있는 대로다.

> The overlay text is drawn with render mode 3 (invisible), so glyph shapes
> never matter -- only whether the PDF font can *encode* a code point.

글리프 모양이 아무 의미가 없다면, OCR 업계 표준 해법인 **glyphless font** 가
정확히 이 문제에 맞는다. Tesseract 와 ocrmypdf 가 쓰는 방식이다.

구조:

```
Type0 (BaseFont: /GlyphLessFont)
  └ Encoding: Identity-H
  └ DescendantFonts: CIDFontType2   (글리프 아웃라인 없음, 고정 advance)
  └ ToUnicode: CMap stream
```

크기는 약 1KB. 얻는 것:

- **모든 유니코드 코드포인트를 인코딩**한다 → 스크립트 감지 자체가 불필요
- 서브셋팅 불필요 (이미 1KB)
- `_probe_batch` 의 "빈 페이지에 글자를 써 보고 다시 읽어서 확인"하는 경험적
  프로빙 불필요
- `dropped_chars` 경고, `fallback_lines`, `--font-file`, `.ttc` 거부 로직
  (`check_font_file`) 전부 불필요

즉 **`fonts.py` 274줄이 대략 30줄로 줄어든다.** 폭 계산도 고정
advance(통상 0.5em) 덕분에 `len(text) * 0.5 * fontsize` 산수로 끝나고,
`pdf_writer.py` 의 `width_scale` 은 오히려 더 예측 가능해진다.

### 주의할 점

- Tesseract 의 glyphless font 는 `DW`(default width) 설정 때문에 문자 간격
  이슈가 보고된 적이 있다. 우리는 `width_scale` 로 어차피 박스 폭에 맞추므로
  영향이 작을 것으로 보이지만, `bench/` 로 확인해야 한다.
- CJK 는 원래 1em 폭이라 반각/전각 구분이 사라진다. 불가시 레이어이므로
  외관 문제는 없지만, 폭 피팅 결과는 측정으로 확인할 것.

-----

## 2. 선택지

### A. pypdfium2 + pikepdf — 라이브러리만 교체

| 역할 | 라이브러리 | 라이선스 |
|---|---|---|
| 렌더링 / 텍스트 추출 | `pypdfium2` | Apache-2.0 (PDFium 은 BSD) |
| 텍스트 레이어 작성 | `pikepdf.canvas` | MPL-2.0 (QPDF) |
| 저장 / 병합 | `pikepdf` | MPL-2.0 |
| 폰트 | glyphless font 직접 구성 | — |

`pikepdf.canvas` 의 `ContentStreamBuilder` 에 `set_text_matrix()`,
`set_text_rendering()`, `horiz_scale()` 이 모두 있어서 현재 `morph` 행렬
로직(`pdf_writer.py:160-178`)을 거의 그대로 옮길 수 있다.

**단, pikepdf.canvas 는 TTF 임베딩을 지원하지 않는다.** `SimpleFont.encode()`
는 WinAnsiEncoding / MacRomanEncoding 만 처리한다. 따라서 이 경로는
**glyphless font 가 필수 전제**다. 폰트 딕셔너리는 pikepdf 로 직접 구성해야
하지만 한 번만 짜면 된다.

라이선스는 AGPL → MPL + Apache 로 정리되어 MIT 선언과 맞아떨어진다.

### B. fpdf2 추가

ocrmypdf 가 기본 렌더러(`--pdf-renderer auto`)로 쓰는 조합. glyphless font 가
이미 구현되어 있어 A 보다 직접 짤 코드가 적다.

단점: fpdf2 는 LGPL-3.0 이다. Python 패키지 의존은 정적 링크가 아니라 대체로
문제되지 않지만, 배포 형태(단일 바이너리 패키징 등)에 따라 확인이 필요하다.

### C. ocrmypdf 플러그인으로 전환 — 가장 급진적

ocrmypdf 는 `get_ocr_engine()` 훅으로 커스텀 OCR 엔진을 받는다. `OcrEngine`
추상 클래스에서 구현할 것은 `version()`, `languages()`, `generate_hocr()`,
`generate_pdf()`, `get_orientation()` 정도다.
**[OCRmyPDF-EasyOCR](https://github.com/ocrmypdf/OCRmyPDF-EasyOCR) 이 정확히
같은 패턴의 선례다.**

얻는 것: `pdf_writer.py`, `fonts.py`, `pipeline.py` 의 대부분이 사라지고
PaddleOCR 어댑터만 남는다. 페이지 병렬화, 사이드카, PDF/A, 이미지 최적화가
공짜로 따라온다.

잃는 것 — 이쪽이 더 무겁다:

- **width fitting** (`pdf_writer.py:99-107`). 추출 오류율 16.4% → 대폭 개선을
  만든 바로 그 로직이다. ocrmypdf 의 hOCR 좌표 모델 안에서 재현해야 하는데,
  hOCR 은 단어/줄 바운딩 박스만 전달하므로 현재의 4점 폴리곤 기반 회전·기울기
  처리(`pdf_writer.py:129-137`)를 그대로 표현하기 어렵다.
- `--resume` 체크포인트 설계(`pipeline.py:265-275`)가 ocrmypdf 파이프라인과
  충돌한다.
- `bench/` 기준선을 다시 잡아야 한다.

-----

## 3. 권고 — A, 단 2단계로

순서가 중요하다. 한 번에 다 바꾸면 정확도 회귀의 원인을 특정할 수 없다.

### 1단계: glyphless font 만 먼저, PyMuPDF 위에서

`fitz.Font(fontfile=glyphless.ttf)` 로 넣을 수 있으므로 **라이브러리 교체 없이**
`fonts.py` 를 먼저 없앨 수 있다. `bench/` 로 정확도 회귀를 즉시 측정할 수 있는
것이 핵심 이점이다.

이 단계만으로 얻는 것:

- `fonts.py` 274줄 → ~30줄
- plan.md §1-1 (한글 텍스트 레이어 소실) 류의 결함이 원천 차단
- `subset_fonts()` 와 그에 얽힌 `_save()` 의 xref 캐시 주의사항
  (`pipeline.py:115-118`) 소멸
- `--font-file` 옵션 및 관련 CLI 검증 제거 가능

측정 기준: `scripts/bench.py table` 의 **pdf** 점수가 유지되어야 한다.
(ocr 점수는 인식기 출력이라 변하지 않는다.)

### 2단계: pypdfium2 + pikepdf 로 교체

1단계가 끝나면 남는 것은 렌더링과 콘텐츠 스트림 작성뿐이라 위험이 훨씬 작다.

**남는 진짜 숙제는 `remove_text_layer` 다.** 다만 이 함수는 실질적으로
`rasterize_page` 안에서만 쓰이고(`pipeline.py:290`), rasterize 의 의도는
"페이지를 통째로 이미지로 교체"하는 것이다. 따라서 레닥션을 흉내내는 대신
**렌더링한 이미지만 담은 새 페이지로 교체**하는 편이 더 간단하고 확실하다.
어차피 원본 페이지 객체를 버리는 것이 목적이기 때문이다.

(참고: `remove_text_layer` 를 단독 공개 API 로 유지할 필요가 있는지도 이때
같이 판단할 것. 현재 pipeline 은 이 함수를 직접 호출하지 않는다.)

-----

## 4. 참고

- [Canvas — pikepdf documentation](https://pikepdf.readthedocs.io/en/latest/api/canvas.html)
- [OCRmyPDF plugins](https://ocrmypdf.readthedocs.io/en/latest/plugins.html)
- [OCRmyPDF-EasyOCR PDF generation](https://deepwiki.com/ocrmypdf/OCRmyPDF-EasyOCR/3.1-pdf-generation)
- [OCRmyPDF advanced features](https://ocrmypdf.readthedocs.io/en/latest/advanced.html)
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
- [Glyphless font 간격 이슈 (tesseract#373)](https://github.com/tesseract-ocr/tesseract/issues/373)
