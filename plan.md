# OCR 품질 개선 계획

현재 파이프라인을 코드 레벨로 검토하고, 설치된 `paddleocr 3.4.0` / `paddlex` 내부 구현을 확인해
검증한 결과를 정리한다. 각 항목은 **근거 → 개선안 → 검증 방법** 순으로 기록한다.

> **진행 상황은 문서 맨 아래 [작업 로그](#작업-로그) 참조.** 세션이 바뀌면 거기부터 읽을 것.

## 0. 현황 요약

```
PDF ──open_pdf──▶ page
      remove_text_layer(page)          # 기존 텍스트 제거 (렌더링 전!)
      page_to_image(page, dpi=300)     # RGB ndarray
      preprocess_image(img)            # gray → NlMeansDenoise → adaptiveThreshold → RGB
      for engine in engines:           # 언어 개수만큼 전체 OCR 반복
          engine.recognize(...)        # PaddleOCR.predict → conf >= 0.5 필터
      deduplicate_results(...)         # 언어 2개 이상일 때만 IoU 중복 제거
      overlay_text_on_page(...)        # insert_text(render_mode=3) 로 투명 텍스트
```

환경: CPU 전용(GPU 없음, 8코어), `paddlepaddle 3.0.0` (CPU 빌드), `PyMuPDF 1.27.2`, `cv2 4.10.0`.

---

## 1. [P0] 확인된 결함 — 측정 이전에 먼저 고칠 것

### 1-1. 한글/CJK 텍스트 레이어가 실제로 깨진다 ★최우선

`pdf_writer.py:91` 의 `page.insert_text()` 는 폰트를 지정하지 않아 기본 Base14 **Helvetica** 를 쓴다.
Helvetica 에는 한글 글리프가 없어 문자가 통째로 치환된다. 실제 검증:

```python
p.insert_text(..., '한글 검색 테스트 Hello')                  # 기본 helv
#  → get_text() == '·· ·· ··· Hello'      ← 한글 전부 소실

p.insert_text(..., '한글 검색 테스트 Hello', fontname='korea')  # PyMuPDF 내장 CJK
#  → get_text() == '한글 검색 테스트 Hello'  ← 정상
```

즉 README 가 내세우는 `-l korean` 사용 시나리오에서 **검색 가능한 텍스트가 0개**가 된다.
OCR 인식률이 아무리 높아도 결과물이 무의미하다.

- **개선**: 인식된 텍스트의 스크립트를 판별해 폰트를 선택
  (한글 `korea`, 일본어 `japan`, 중국어 간체 `china-s` / 번체 `china-t`, 그 외 `helv`).
  더 견고하게는 Noto Sans CJK 를 `page.insert_font(fontfile=...)` 로 임베드하거나
  `fitz.TextWriter` + `fitz.Font` 조합 사용. 다만 파일 크기 증가 트레이드오프 있음.
- **검증**: 출력 PDF 를 `page.get_text()` 로 재추출해 원본 인식 문자열과 일치하는지 단정하는
  회귀 테스트 추가. 현재 통합 테스트는 영문만 다뤄 이 결함을 잡지 못한다.

### 1-2. `-l en` 지정이 오히려 인식 모델을 다운그레이드한다

`paddleocr/_pipelines/ocr.py:453-482` 의 언어→모델 매핑을 확인한 결과:

| `lang` 값 | 선택되는 rec 모델 |
|---|---|
| (미지정) / `ch` / `japan` / `chinese_cht` | **`PP-OCRv5_server_rec`** (대형, 중·영·일·번체 통합) |
| `en` | `en_PP-OCRv5_mobile_rec` (경량) |
| `korean` | `korean_PP-OCRv5_mobile_rec` (경량) |

det 모델은 모든 경우 `PP-OCRv5_server_det` 로 동일하다.
현재 CLI 기본값이 `-l en` 이므로 **기본 실행이 가장 가벼운 인식 모델을 쓴다.**

- **개선**: 라틴 문자 위주 문서는 `text_recognition_model_name="PP-OCRv5_server_rec"` 를 명시.
  `--model-preset {fast,balanced,accurate}` 같은 옵션으로 mobile/server 를 노출.
- **주의**: 한국어는 PP-OCRv5 계열에 server rec 이 없다(mobile 뿐). 한글 정확도를 더 끌어올리려면
  §4 의 엔진 교체 검토가 필요하다.
- **검증**: 동일 페이지에 대해 mobile vs server rec 의 CER 비교.

### 1-3. 이진화 전처리가 정확도를 떨어뜨릴 가능성이 높다

`preprocess_image()` 는 `fastNlMeansDenoising(h=10)` + `adaptiveThreshold` 로 흑백 이진화한다.
PP-OCRv5 같은 딥러닝 OCR 은 안티에일리어싱된 그레이스케일/컬러 스캔 이미지로 학습되어 있어,
이진화는 획 굵기 정보와 계조를 파괴한다. 특히 작은 글씨·얇은 획·저대비 스캔에서 손실이 크다.
(이진화는 Tesseract 이전 세대 엔진의 관행이다.)
게다가 `fastNlMeansDenoising` 은 300 DPI A4(2480×3508) 기준 페이지당 수 초를 소모하는 무거운 연산이다.

- **개선**: 전처리를 **선택 옵션**(`--preprocess {none,binarize,adaptive}`)으로 내리고 기본값을 `none` 으로.
  필요한 보정은 이진화가 아니라 아래 쪽으로 대체:
  - 기울기 보정(deskew): 최소 외접 사각형 / 투영 프로파일 기반 각도 추정 후 회전
  - 저해상도 페이지에 한해 업스케일
  - 대비 보정은 CLAHE 정도의 비파괴 연산
- **검증**: 벤치마크 코퍼스에서 `none` vs 현행의 CER 비교. **가정이므로 반드시 측정 후 결정.**

### 1-4. 전처리를 제거하면 드러나는 RGB/BGR 채널 뒤바뀜 (잠복 버그)

`paddlex/inference/common/reader/image_reader.py:30` 에서 `ReadImage(format="BGR")` 가 기본이고,
`read()` 는 ndarray 입력을 **이미 BGR 이라고 가정**하고 그대로 통과시킨다.
반면 `page_to_image()` 는 PyMuPDF pixmap 을 **RGB** 로 반환한다.

지금은 `preprocess_image()` 가 그레이스케일을 3채널로 복제(R=G=B)해 넘기므로 채널 순서가
무의미해져 **버그가 가려져 있다.** §1-3 대로 전처리를 끄는 순간 컬러 반전된 이미지가 들어간다.

- **개선**: `page_to_image()` 가 BGR 을 반환하도록 하거나, 엔진 입력 직전에 `cv2.cvtColor(..., COLOR_RGB2BGR)`.
  채널 순서 규약을 함수 docstring 과 타입 주석에 명시.

### 1-5. `remove_text_layer` 를 렌더링 **전에** 호출해 원본 내용이 파괴될 수 있다

`pipeline.py:77-79` 순서는 `remove_text_layer(page)` → `page_to_image(page)` 다.
스캔본의 투명 OCR 텍스트를 지우는 용도로는 문제없지만, **텍스트가 실제로 보이는 디지털 원본**
(혹은 스캔+본문 혼재 PDF)에 적용하면 `apply_redactions()` 가 보이는 글자를 지우고,
그 상태로 래스터화한 뒤 그 위에 OCR 텍스트를 얹는다. 출력물에서 해당 내용이 영구 소실된다.

- **개선**:
  - 페이지에 유의미한 텍스트가 이미 있으면 **기본적으로 건너뛴다**(`ocrmypdf --skip-text` 와 동일 정책).
  - 강제 재OCR 은 `--force-ocr`, 기존 텍스트 유지 후 추가는 `--redo-ocr` 식으로 명시적 옵션화.
  - 최소한 렌더링을 redaction **이전**에 수행해 원본 픽셀을 보존.
- **검증**: 텍스트가 보이는 PDF 를 입력으로 넣어 출력에서 원본 문자열이 살아있는지 테스트.

### 1-6. 다국어 처리가 구조적으로 비효율적이고 부정확하다

현재 언어 N개 → 엔진 N개 → **동일 이미지에 대해 det+rec 을 통째로 N번** 수행 후 IoU 중복 제거.

두 가지 문제:
1. det 결과는 언어와 무관하게 거의 동일한데 N번 반복해 **속도가 N배**.
2. `deduplicate_results()` 가 서로 다른 모델의 confidence 를 직접 비교한다.
   모델 간 confidence 는 캘리브레이션이 다르므로 **비교 근거가 없다.**
   또한 `iou_threshold=0.5` 는 촘촘한 본문 행에서 인접 행을 오탐 제거할 위험이 있다.

- **개선안 A (권장, 단순)**: 한→영 혼재 문서는 `korean_PP-OCRv5_mobile_rec` 단일 모델로 처리.
  해당 사전에 라틴 문자가 포함되어 있는지 확인 후, 포함된다면 `-l korean -l en` 이중 실행 자체가 불필요.
  (→ 사전 파일 확인 필요)
- **개선안 B**: det 은 1회만 수행하고, 잘라낸 crop 에 대해서만 rec 을 언어별로 수행.
  선택 기준은 confidence 비교 대신 **스크립트 판별 + 문자 커버리지**(사전 미포함 문자 비율)를 사용.
- **개선안 C**: 중복 제거 기준을 IoU 대신 중심점 포함 여부 + 높이 비율로 바꿔 인접 행 오탐 방지.

---

## 2. [P1] 파라미터 튜닝 — 측정 기반으로 결정

`PaddleOCR()` 에 넘기는 값과 실제 기본값 비교 (`paddlex/configs/pipelines/OCR.yaml` 확인):

| 항목 | 현재 코드 | 라이브러리 기본 | 검토 |
|---|---|---|---|
| `text_det_thresh` | 0.3 | 0.3 | 동일 |
| `text_det_box_thresh` | 0.5 | 0.6 | 낮춤 → 재현율↑ 오탐↑ |
| `text_det_unclip_ratio` | 1.8 | 1.5 | 넓힘 → 글자 잘림 방지, 인접 행 병합 위험 |
| `text_det_limit_side_len` | 미지정 | **64 / `limit_type: min`** | 다운스케일 없음. 300 DPI 원본이 그대로 det 입력 |
| `use_textline_orientation` | True | - | 행 단위 180° 보정만. 페이지 기울기는 보정 안 됨 |
| `use_doc_orientation_classify` | 미사용 | False | 페이지 90/180/270° 회전 자동 보정 |
| `use_doc_unwarping` | 미사용 | False | 책 제본부 곡면 왜곡 보정 (UVDoc) — **스캔 책에 특히 유효** |

검토 항목:

- **`use_doc_unwarping=True`**: 스캔한 책은 제본 쪽 글자가 휘어 인식률이 크게 떨어진다.
  이 프로젝트의 핵심 유스케이스(scanned book)와 정확히 맞는 기능인데 현재 미사용. 우선 실험 대상.
- **`use_doc_orientation_classify=True`**: 뒤집혀 스캔된 페이지 자동 보정.
- **DPI 스윕**: 현재 300 고정. det 입력이 다운스케일되지 않으므로 DPI 를 올리면 연산량이 제곱으로 증가한다.
  200/300/400 을 측정해 정확도 대비 비용의 실제 곡선을 확인. 작은 각주 글씨만 문제라면
  전체 DPI 인상 대신 **해당 영역만 crop 재인식**하는 2-pass 가 더 저렴하다.
- **`--confidence 0.5` 기본값**: 인식은 됐지만 신뢰도가 낮은 텍스트를 버린다. 검색 용도에서는
  오탐보다 누락이 더 아프다. 0.3 정도로 낮추고 대신 결과에 confidence 를 남기는 방식 검토.
- **`text_det_limit_side_len` 을 명시적으로 상한(`max`) 설정**: 매우 큰 페이지에서 메모리/시간 폭주 방지.

---

## 3. [P0] 평가 체계 — 이 모든 판단의 전제

현재 저장소에는 **샘플 PDF도, 정답 텍스트도, 정확도 지표도 없다.**
통합 테스트는 PyMuPDF 로 렌더링한 합성 이미지의 영문 단어 몇 개를 확인할 뿐이라,
"OCR 성능이 좋아졌는가"를 판정할 수 없다. 위 모든 가설은 측정 없이는 추측이다.

- `bench/` 디렉터리 구성: 실제 스캔 PDF + 페이지별 정답 텍스트(`.gt.txt`).
  - 한글 본문 / 한영 혼재 / 세로쓰기 / 저품질 팩스 스캔 / 2단 조판 / 표·각주 포함
  - 저작권 문제를 피하려면 공개 도메인 자료(국립중앙도서관 공개자료, Project Gutenberg 스캔본 등) 사용.
  - 용량이 크면 저장소에 넣지 말고 다운로드 스크립트 + 체크섬으로 관리.
- 지표: **CER / WER** (`jiwer` 또는 `rapidfuzz`), 검출 재현율/정밀도, 페이지당 처리 시간, 피크 메모리.
  - 한글은 자모 분리 여부에 따라 CER 이 달라지므로 NFC 정규화 후 비교할 것.
- `scripts/bench.py`: 설정(dict)을 받아 파이프라인을 돌리고 지표 표를 출력. A/B 비교를 한 줄로.
- **텍스트 레이어 품질 지표 분리**: "OCR 이 인식한 문자열"과 "출력 PDF 에서 추출되는 문자열"을
  따로 측정한다. §1-1 같은 결함은 후자에서만 드러난다.

---

## 4. [P2] 엔진 교체 검토

한국어 정확도의 상한이 `korean_PP-OCRv5_mobile_rec`(경량 모델)에 걸려 있으므로,
현행 튜닝을 다 해도 한계가 있을 수 있다. 벤치마크 구축 후 아래를 동일 코퍼스로 비교한다.

| 후보 | 장점 | 단점 / 리스크 |
|---|---|---|
| **PaddleOCR 유지 + 튜닝** | 이미 통합됨, 변경 비용 0, CPU 실용적 | 한국어는 mobile rec 상한 |
| **RapidOCR** | PaddleOCR 모델을 ONNXRuntime 으로 구동. `paddlepaddle` 의존성 제거로 설치 난이도·용량 대폭 개선, CPU 추론 더 빠름 | 모델이 같으므로 **정확도 개선은 없음**. 배포 편의 목적 |
| **Surya** | 90+ 언어, 행 검출·읽기 순서·레이아웃 강함. 한국어 평가 좋음 | GPU 없으면 느림. 라이선스 조건 확인 필요(상업 이용 매출 기준 존재) |
| **Tesseract 5 (+ `ocrmypdf`)** | 성숙, CPU 가벼움, PDF 텍스트 레이어 처리가 레퍼런스 수준 | 한글 정확도는 PP-OCRv5 대비 열세일 가능성 높음 |
| **VLM 계열** (PaddleOCR-VL, dots.ocr, olmOCR, MinerU) | 문서 이해 기반 SOTA, 표·수식·읽기 순서까지 | GPU 필수급, 페이지당 수 초~수십 초, **바운딩박스가 부정확해 투명 텍스트 오버레이 정렬에 부적합**할 수 있음 |
| **클라우드 API** (Google Document AI, Azure Document Intelligence, Naver CLOVA OCR) | 한국어 최상위권 정확도 | 유료, 네트워크 필수, 문서 외부 전송 — 이 CLI 의 로컬 처리 성격과 충돌 |

설계상 권고: **엔진을 인터페이스 뒤로 분리**한다.
`OcrEngine` 을 프로토콜(`recognize(image, confidence) -> list[OcrResult]`)로 정의하고
`PaddleEngine` / `RapidEngine` / `TesseractEngine` 구현을 `--engine` 옵션으로 선택.
그러면 벤치마크로 실측한 뒤 교체 여부를 결정할 수 있고, 교체가 파이프라인 전체를 흔들지 않는다.
참고로 `paddlex` 에 `PaddleOCR-VL` 파이프라인 설정이 이미 포함되어 있어 실험 진입 비용은 낮다.

> 주: `ocrmypdf` 는 엔진이 아니라 PDF 배관 처리의 레퍼런스다. 텍스트 레이어 삽입, 기존 텍스트
> 페이지 스킵, 사이드카 출력, PDF/A 변환, 이미지 최적화 등 이 프로젝트가 앞으로 마주칠 문제를
> 이미 해결해 두었으므로 **정책과 옵션 설계를 참고**할 가치가 있다.

---

## 5. [P1] 텍스트 레이어 정렬 품질

인식 정확도와 별개로, 투명 텍스트가 실제 글자 위에 얼마나 잘 겹치는지가 검색 하이라이트와
드래그 선택 경험을 좌우한다. 현재 `overlay_text_on_page()` 의 한계:

- **가로 폭 미보정**: `insert_text` 는 폰트 고유 자간으로 그린다. OCR 박스 폭과 무관하게
  텍스트가 짧거나 길게 삐져나온다. 하이라이트 사각형이 실제 글자와 어긋난다.
  → `fitz.get_text_length()` 로 실제 폭을 구해 `morph` 행렬에 **가로 스케일**을 곱해 박스 폭에 맞춘다.
- **`BASELINE_RATIO = 0.85` 고정**: 폰트/스크립트마다 baseline 위치가 다르다.
  한글은 디센더가 거의 없어 라틴 기준값이 맞지 않는다. 스크립트별 상수 또는 폰트 메트릭 기반 계산.
- **행 단위 삽입**: PaddleOCR 은 행 단위 박스만 준다. 단어 단위 선택 정확도를 높이려면
  문자 폭 비례로 행을 분할해 여러 조각으로 삽입하는 방법이 있다(ocrmypdf 의 hOCR 방식과 유사).
- **`font_size < 1` 조각을 조용히 버린다**(`pdf_writer.py:69-70`): 검색에서 누락된다.
  최소한 verbose 로그에 남기고, 개수를 집계해 보고.

---

## 6. [P2] 처리 속도

정확도와 별개지만, 느리면 고DPI·server 모델 같은 정확도 옵션을 실사용에서 못 켠다.

- 페이지를 순차 처리한다. 렌더링(I/O·CPU)과 추론을 **파이프라인화**하거나
  페이지 배치를 `predict()` 에 한 번에 넘겨 rec 배치 효율을 살린다.
- `fastNlMeansDenoising` 제거만으로도 페이지당 수 초 절감(§1-3).
- 다국어 중복 det 제거(§1-6)로 언어 수만큼 배수 절감.
- CPU 스레드 수 명시(`cpu_threads`), MKL-DNN 활성화 확인.
- GPU 환경 지원: `--device` 옵션 노출 (현재 하드웨어에는 GPU 없음).

---

## 7. 실행 순서

| 단계 | 내용 | 산출물 |
|---|---|---|
| 1 | §3 벤치마크 코퍼스 + `scripts/bench.py` + CER 지표 | 기준선 수치 |
| 2 | §1-1 CJK 폰트 수정 + 회귀 테스트 | 한글 검색 실제 동작 |
| 3 | §1-4 채널 순서, §1-5 텍스트 페이지 스킵 정책 | 정확성 확보 |
| 4 | §1-3 전처리 on/off, §1-2 server rec, §2 `use_doc_unwarping` A/B 측정 | 측정된 개선폭 |
| 5 | §1-6 다국어 구조 개선, §5 정렬 품질 개선 | 품질·속도 동시 개선 |
| 6 | §4 엔진 인터페이스 분리 후 후보 엔진 실측 비교 | 교체 여부 결정 |

**핵심 원칙**: 2·3 단계(확인된 결함)는 측정 없이 바로 고친다.
4단계 이후는 **1단계 벤치마크 없이는 진행하지 않는다.** 지금까지의 "개선" 커밋들
(전처리 추가, 파라미터 조정)도 실제로 정확도를 올렸는지 검증된 바 없다.

---

## 작업 로그

세션 간 인계를 위한 기록. **다음 세션은 "다음 할 일"부터 시작하면 된다.**

### 완료

#### ✅ §1-1 CJK 폰트 선택 (한글 검색 복구)

- 신규 `src/pdf_refinery/fonts.py`
  - `detect_script(text)` — 한글/가나/한자/라틴 판별. 혼재 시 우선순위는
    **한글 > 가나 > 한자**. 한자는 중·일 공용이라 단독으로는 언어를 특정하지 못하고,
    한국어 문헌의 한자 혼용·일본어의 한자 혼용을 각각 한글/가나가 가려내기 때문.
  - `FontSpec` — `(name, file)`. `file=None` 이면 PyMuPDF 내장 폰트.
  - `unsupported_chars()` — **인코딩 가능 여부를 실측 왕복으로 판정**하고 캐싱.
    `Font.has_glyph()` 는 신뢰할 수 없기 때문(아래 실측 근거 참조).
    판정 비용이 문자당 9.7ms(= 한글 2000자에 19.3초)로 과해서, 스크래치 페이지 하나에
    격자로 몰아 넣고 한 번에 추출하는 **배치 프로브**로 바꿔 **1.03초**로 줄였다(19배).
    격자는 페이지 안에 들어가도록 좌표를 계산한다 — 크롭박스 밖 텍스트는 추출되지 않아
    폰트 실패로 오판되기 때문.
  - `FontResolver` — 줄 단위로 폰트를 고르고, 인코딩 실패 문자를 `dropped_chars` 에 수집.
    `--font-file` 이 주어지면 모든 줄에 그 폰트를 쓴다.
- `pdf_writer.overlay_text_on_page(..., font_resolver=None)` 로 폰트 전달.
- CLI `--font-file` 추가. 인코딩 실패 문자가 있으면 종료 시 경고 + 문자 샘플 출력
  (기존에는 조용히 소실됐다).

**실측 근거 (재조사 방지용으로 남김):**

`insert_text` 왕복으로 측정한 실제 커버리지. `Font.has_glyph()` 는 `é`(U+00E9)에 대해
내장 CJK 폰트가 글리프 167번을 가진다고 응답하지만 **실제 `insert_text` 는 이 문자를 버린다.**
내장 CJK 폰트는 CID 인코딩을 거치므로 has_glyph 결과와 어긋난다. 따라서 실측만 신뢰할 것.

| 범위 | `helv` | 내장 CJK (`korea`/`japan`/`china-s`) |
|---|---|---|
| ASCII | 100% | 100% |
| Latin-1 (é, ä…) | 97% | **32%** |
| Greek | 0% | 65% |
| Cyrillic | 0% | 68% |
| 한글 음절 | 0% | **100%** |
| 한글 자모 | 0% | 98% |
| 가나 | 0% | 88% |
| **한자(CJK Ideographs)** | 0% | **31%** |
| Thai / Arabic / Devanagari | 0% | **0%** |

→ 순수 한글 본문은 내장 폰트로 충분하지만, **한자 혼용·태국어·아랍어는 `--font-file` 필수**다.

#### ✅ §1-4 RGB/BGR 채널 순서

`ocr_engine.recognize()` 가 `predict()` 직전에 `cv2.COLOR_RGB2BGR` 변환.
PaddleX `ReadImage` 가 ndarray 입력을 BGR 로 간주(`format="BGR"` 기본)하는데
`page_to_image()` 는 RGB 를 반환한다. 지금은 이진화 전처리가 R=G=B 로 만들어 버그가
가려져 있지만, §1-3 대로 전처리를 끄는 순간 컬러가 반전되므로 선제적으로 수정했다.

#### ✅ §1-5 기존 텍스트 페이지 정책 (데이터 손실 수정)

- `pdf_writer.has_text(page)` 추가.
- `pdf_writer.rasterize_page(page, dpi)` 추가 — **렌더링 → 텍스트 제거 → 렌더링 결과를 다시 삽입**.
  `remove_text_layer()` 만 쓰면 보이는 글자까지 지워져 페이지가 백지가 된다.
  회귀 테스트 `test_redaction_alone_would_lose_the_content` 가 이 대비를 고정한다.
- 파이프라인 기본 동작: **텍스트가 있는 페이지는 건너뛴다**(`ocrmypdf --skip-text` 와 동일).
  `--force-ocr` 로 래스터화 후 재OCR. 종료 시 스킵한 페이지 수를 보고.

#### ✅ 저장 방식 변경

`shutil.copy2` + `incremental=True` 저장을 폐기하고 **입력을 열어 출력으로 저장**
(`garbage=4, deflate=True`)하도록 변경. 이유:
- 증분 저장은 `subset_fonts()` 와 양립하지 않는다(기존 객체를 재작성하지 못함).
  임베드 폰트를 서브셋하지 않으면 출력이 수십 MB 로 불어난다.
  실측: DejaVuSans 임베드 시 410KB → 서브셋 후 **45.7KB**.
- 증분 저장은 원본 바이트를 계속 누적해 파일이 커진다.
- 입출력 경로가 같으면 `ClickException` 으로 거부(원본 파괴 방지).

**주의**: `.ttc`(폰트 컬렉션)는 `subset_fonts()` 가 `Index bounds` 오류로 실패한다.
`--font-file` 에는 단일 `.ttf`/`.otf` 를 쓸 것. (시스템의 `NotoSansCJK-Regular.ttc` 는 부적합)

#### ✅ 테스트

- 신규 `tests/test_fonts.py` — 스크립트 판별, 커버리지 판정, 리졸버, **PDF 왕복 검증**.
- `tests/test_pdf_writer.py` — CJK/라틴 텍스트가 오버레이 후 추출되는지, `has_text`,
  `rasterize_page` 의 외관 보존, 단순 redaction 의 데이터 손실 대비.
- `tests/test_pipeline.py` — 입출력 동일 거부, 원본 불변, 텍스트 페이지 스킵/강제 OCR.
- 결과: **단위 테스트 74개 통과.** 통합 테스트는 별도 확인 필요(모델 다운로드로 오래 걸림).

#### ✅ end-to-end 검증

합성 한글 스캔본(텍스트 렌더 → 래스터화 → 이미지만 재삽입, 텍스트 레이어 없음)에
실제 파이프라인을 돌린 결과: **15개 토큰 중 13개 검색 가능**.
수정 전에는 한글 토큰이 전부 `··` 로 깨져 **0개**였다.
남은 2개(`한글`→`른우`, `한다` 누락)는 텍스트 레이어가 아니라 **인식 오류**다.

### ⚠️ 측정 결과: §1-3 전처리 가설은 현재 반증됨

위 합성 페이지로 이진화 on/off 를 비교한 결과 **이진화 쪽이 더 정확했다.**

| 전처리 | CER | 인식 결과 |
|---|---|---|
| `binarize` (현행) | **0.125** | 른우 검색 테스트 문서 이문장은 그 OCR 로 인식되어야 Mixed 한영 혼용 line 123 |
| `none` (원본) | 0.325 | 른우 검색 테스트 문서 금요곰l0 한다 OCR 로 인식되어야 Mixed 없우 혼용 line 123 |

**이 수치를 일반화하지 말 것.** 합성 렌더 페이지는 이미 고대비·저노이즈라
이진화의 손실이 거의 없고, 오히려 안티에일리어싱을 제거해 유리하게 작용한다.
실제 스캔본(노이즈, 계조, 얼룩, 저대비)에서는 결과가 뒤집힐 수 있다.
→ **§3 실제 스캔 코퍼스를 갖추기 전까지 전처리 기본값을 바꾸지 않는다.**
가설을 세운 근거(딥러닝 OCR 은 그레이스케일 학습)는 여전히 유효하지만 미검증 상태다.

### 다음 할 일 (우선순위 순)

1. **§3 벤치마크 구축** — 이후 모든 판단의 전제. 아직 착수 안 함.
   - `bench/` 코퍼스(공개 도메인 스캔본) + `scripts/bench.py` + CER/WER(NFC 정규화 후).
   - **"OCR 인식 문자열"과 "출력 PDF 추출 문자열"을 분리 측정**할 것.
     §1-1 결함은 후자에서만 드러났다.
2. **§1-3 전처리 A/B** — `--preprocess {none,binarize}` 옵션화 후 **실제 스캔본으로** 재측정.
   합성 페이지에서는 이진화가 이겼다(위 참조). 실제 스캔에서 확인 전까지 기본값 유지.
3. **§1-2 server rec 모델** — `text_recognition_model_name="PP-OCRv5_server_rec"` 를
   라틴 문서에 적용해 mobile 대비 CER 비교. 현재 기본값 `-l en` 은 경량 모델을 쓴다.
4. **§2 `use_doc_unwarping=True`** — 스캔 책 제본부 왜곡 보정. 유스케이스 적합도 높음.
5. **§1-6 다국어 구조** — det 1회 + rec 다중. 현재는 언어 수만큼 전체 OCR 반복.
6. **§5 텍스트 레이어 정렬** — 가로 폭 보정, 스크립트별 baseline.
7. **§4 엔진 인터페이스 분리** 후 후보 엔진 실측 비교.

### 보류 / 검토 필요

- **글리프리스 폰트(Tesseract 방식)** — 투명 텍스트 레이어의 정공법.
  빈 글리프 1개 + ToUnicode CMap 으로 **유니코드 100% 커버 + 폰트 4KB**.
  현재의 내장 폰트 커버리지 문제(한자 31%)를 근본 해결한다.
  다만 PyMuPDF 고수준 API 로는 불가하고 Type0/Identity-H 폰트를 직접 구성해야 한다
  (`fontTools` 로 런타임 생성하거나 `pikepdf` 사용). 의존성이 늘어 보류.
- README 옵션 표에 `--force-ocr`, `--font-file` 미반영.
- `pyproject.toml` 의존성에 `opencv-python` 이 있으나 venv 에는 `opencv-python-headless`
  계열이 설치돼 있다(`cv2 4.10.0` 은 import 가능). CLI 도구에는 headless 가 적합하므로
  의존성 정정 검토.
