# 다음에 할 일

작성일: 2026-10-09

## 1. paddlepaddle 3.2.2 속도 재측정

`.venv` 는 2026-10-09 에 paddlepaddle 3.1.1 → 3.2.2 로 올렸다. paddleocr 3.7.0,
paddlex 3.7.2 는 그대로다. `pyproject.toml` 의 범위 `>=3.1,<3.3` 안이라
의존성 선언은 바꾸지 않았다.

정확도는 3.1.1 과 같다. baseline, server-rec, unwarp, textline-ori 를 세
corpus 에 돌린 10개 조합 모두 페이지별 오류 수까지 일치했다. 결과는
`bench/results/` 의 `pp311-*`, `pp322-*` 에 있다.

속도는 판단하지 못했다. 첫 측정에서 3.2.2 의 baseline 이 10–20% 느리게
나왔지만(sample-2 18.5 → 20.5 s/page), 같은 3.2.2 를 다시 돌린
`pp322r-baseline` 에서는 sample-2 가 34.7 s/page 로 편차가 더 컸다. 측정
당시 load average 가 코어 8개에 10 안팎이었다.

- [ ] 다른 작업이 없을 때 baseline 만 버전별로 두세 번씩 다시 잰다.
      3.1.1 쪽은 venv 를 잠시 되돌리거나 scratch venv 를 따로 만들어 잰다.
- [ ] 3.2.2 가 실제로 느리다면, 새 설치가 3.2.2 를 받지 않도록 상한을
      `<3.2.2` 등으로 좁힐지 정한다. 차이가 없으면 이 항목은 닫는다.

## 2. paddlepaddle 3.3.x 상한 해제 여부 재확인

paddlepaddle 3.3.1(현재 PyPI 최신) + paddleocr 3.7.0 에서도 CPU oneDNN
경로가 여전히 깨진다. 텍스트 검출 중에 다음 오류가 난다.

```
NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
not support [pir::ArrayAttribute<pir::DoubleAttribute>]
(at /paddle/paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.cc:116)
```

업스트림 이슈: https://github.com/PaddlePaddle/Paddle/issues/77340
(관련 보고: PaddleOCR discussion #17350, issue #17539)

- [ ] paddlepaddle 새 릴리스(3.3.2 / 3.4 등)가 나오면 scratch venv 에서
      `PaddleEngine(lang="korean")` 로 `tests/sample-1.pdf` 한 페이지를
      돌려 본다. 위 오류가 사라졌으면 `pp3xx-*` 이름으로 같은 네 variant 를
      재서 정확도와 속도를 비교한다.
- [ ] 통과하면 `pyproject.toml` 의 상한과 그 위 주석(3.3 이 깨진다는 설명)을
      갱신한다.

## 3. paddleocr 하한 점검 (선택)

`paddleocr>=3.0` 은 선언만 되어 있고 3.0 에서 돌려 본 적이 없다. 코드가
쓰는 옵션(`text_recognition_batch_size`, `use_doc_unwarping`,
`text_recognition_model_name`, `DocImgOrientationClassification`)과 비공개
메서드 `PaddleOCR._get_ocr_model_names` 가 3.0 에도 있는지 확인하고, 없으면
하한을 실제로 확인한 버전으로 올린다.
