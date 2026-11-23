# 🚗 테슬라 Model S 매뉴얼 챗봇 (koGPT2 SFT → EXAONE LoRA)

테슬라 Model S 한국어 사용자 매뉴얼(PDF)을 전처리하고, 질문과 관련된 매뉴얼 문단을 검색해서
모델이 **그 문단을 읽고 답하도록** 파인튜닝한 챗봇 프로젝트입니다.

1. **koGPT2**(`skt/kogpt2-base-v2`, 1.25억 파라미터)로 추가 사전학습 → SFT까지 진행
2. 모델 크기의 한계를 확인하고 **EXAONE 3.5 2.4B**(24억 파라미터)로 교체
3. **LoRA** 파인튜닝 + **"모르면 모른다고 답하기"** 학습으로 마무리

---

## 전체 흐름

```
PDF ─▶ 텍스트 추출 ─▶ 전처리 ─▶ Q&A 생성(OpenAI) ─▶ 데이터셋
                                                      │
 koGPT2 ─▶ 추가 사전학습 ─▶ SFT(문단 검색 + Q&A) ─▶ 한계 확인
                                                      │
 EXAONE 3.5 2.4B ─▶ LoRA(같은 데이터) ─▶ "모름" 예시 추가 학습 ─▶ Gradio 챗봇
```

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| 1 | `step1_extract.py` | PDF에서 글자만 추출 (초기 확인용) |
| 2-1 | `step2_1_extract_layout.py` | 위치·글자 크기·굵기까지 포함해 재추출, 제목/대제목/본문 분리 |
| 2-2 | `step2_2_make_paragraphs.py` | 줄 → 문단으로 이어붙이기, 띄어쓰기 복원 |
| 2-3 | `step2_3_clean_noise.py` | 페이지 참조·표 잔해 삭제, `을(를)` 조사 정리 |
| 2-4 | `step2_4_make_sections.py` | 같은 제목 페이지를 섹션으로 묶기, 원본 오류 보정 |
| 3 | `step3_make_qa.py` | OpenAI로 섹션별 Q&A 생성 |
| 4 | `step4_make_dataset.py` | 불량 Q&A 거르기, 학습/검증 분리 |
| 5 | `step5_pretrain.py` | koGPT2 추가 사전학습 (매뉴얼 본문) |
| 6 | `step6_sft.py` | SFT — 질문만 주고 답하게 학습 (비교용) |
| 7-1 | `step7_1_build_passages.py` | 검색용 문단 생성 + BM25 검색기 |
| 7-2 | `step7_2_make_ctx_dataset.py` | 검색된 문단을 붙인 SFT 데이터 생성 |
| 7-3 | `step7_3_sft_ctx.py` | koGPT2 SFT — 문단을 읽고 답하게 학습 |
| 7-4 | `step7_4_app.py` | koGPT2 Gradio 챗봇 (답변 안에서만 반복 금지하도록 수정) |
| 8-0 | `step8_0_convert_exaone35.py` | EXAONE 3.5를 Llama 형식으로 변환 (가중치 값은 그대로, 이름만 변경) |
| 8-1 | `step8_1_exaone_zeroshot.py` | EXAONE 평가 (`--m35`, `--lora`, `--unk` 옵션) |
| 8-3 | `step8_3_lora.py` | EXAONE LoRA 파인튜닝 (`--m35`, `--unk` 옵션) |
| 8-4 | `step8_4_make_unknown_dataset.py` | 정답 없는 문단 + "매뉴얼에서 찾지 못했습니다." 예시 추가 |
| 8-2 | `step8_2_app.py` | **최종 챗봇** (EXAONE 3.5 + LoRA, Gradio) |

실험용 스크립트: `step7_3_sft_ctx_e6.py`(6에폭, 과적합 확인), `step7_2_*_k3.py`·`step7_3_*_k3.py`(문단 3개, 효과 없음),
`step8_2_app_exaone40.py`(EXAONE 4.0 1.2B 학습 없는 버전).

단계별 상세 설명은 [`docs/`](docs/) 폴더에 있습니다.

---

## 결과 (검증 질문 96개)

| 모델 | 평균 F1 | F1 ≥ 0.5 | 답이 없을 때 |
|---|---:|---:|---|
| koGPT2 + 문단 검색 SFT | 0.248 | 10 | 지어냄 |
| EXAONE 4.0 1.2B (학습 없음) | 0.328 | 18 | 답이 있어도 "찾지 못함"이 있음 |
| EXAONE 4.0 1.2B + LoRA | 0.427 | 41 | 지어냄 |
| EXAONE 3.5 2.4B + LoRA | 0.462 | 46 | 지어냄 |
| **EXAONE 3.5 2.4B + LoRA + 모름 학습 (최종)** | 0.441 | 44 | **"매뉴얼에서 찾지 못했습니다."** |

- F1: 정답과 모델 답변의 글자 2개 단위 겹침 정도
- 최종 모델이 "찾지 못했습니다"라고 답한 14개는 **모두 이전 모델이 틀리게 답하던 질문**입니다.
  F1이 조금 낮아진 건 오답 대신 "모름"으로 답해 겹치는 글자가 없어졌기 때문입니다.
- 검색이 정답 문단을 제대로 가져온 63문제 중 크게 틀린 답: koGPT2 23개 → EXAONE 3.5 4개

### 왜 모델을 바꿨나
koGPT2는 학습을 늘리거나(과적합), 문단을 3개로 늘리거나(효과 없음), 생성 설정을 고쳐도(깨진 단어만 해결)
정확도가 오르지 않았습니다. **정답이 문단에 그대로 적혀 있어도** 관계없는 부분을 옮겨 쓰는 경우가 많아,
문제가 데이터·설정이 아니라 **모델 크기(문단을 읽고 이해하는 능력)** 라고 판단했습니다.

### 한계
- 검색(BM25)이 문단 2개 안에 정답 문단을 넣는 비율은 약 66%입니다.
- 검색이 틀린 경우 중 일부는 여전히 답을 지어냅니다.

---

## 폴더 구조

```
SFT/
├─ scripts/          # 단계별 코드
├─ docs/             # 단계별 설명
├─ data/             # (저장소에 없음) 원본 PDF와 가공 데이터
│  ├─ raw/           #   Owners_Manual.pdf
│  └─ processed/     #   각 단계 결과
├─ models/           # (저장소에 없음) 학습된 모델·LoRA 어댑터
├─ requirements.txt
└─ .env              # (저장소에 없음) OpenAI API 키
```

---

## 실행 방법

### 1. 환경 준비 (Python 3.11, Windows, NVIDIA GPU)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 매뉴얼 PDF 받기
저작권 문제로 PDF는 저장소에 포함하지 않았습니다. 테슬라 공식 사이트에서 직접 받아 `data/raw/Owners_Manual.pdf`에 저장하세요.

### 3. OpenAI API 키 (3단계에만 필요)
프로젝트 폴더에 `.env` 파일을 만들고 `OPENAI_API_KEY=sk-...` 를 적어주세요.

### 4. 순서대로 실행
```bash
# 전처리 · 데이터
python scripts\step2_1_extract_layout.py
python scripts\step2_2_make_paragraphs.py
python scripts\step2_3_clean_noise.py
python scripts\step2_4_make_sections.py
python scripts\step3_make_qa.py              # OpenAI 사용
python scripts\step4_make_dataset.py
python scripts\step7_1_build_passages.py
python scripts\step7_2_make_ctx_dataset.py

# (선택) koGPT2
python scripts\step5_pretrain.py
python scripts\step7_3_sft_ctx.py
python scripts\step7_4_app.py

# EXAONE 3.5 (최종)
python scripts\step8_0_convert_exaone35.py
python scripts\step8_4_make_unknown_dataset.py
python scripts\step8_3_lora.py --m35 --unk          # RTX 4080 기준 약 5분, GPU 약 7GB
python scripts\step8_1_exaone_zeroshot.py --m35 --lora --unk   # 평가 약 3분
python scripts\step8_2_app.py                       # 최종 챗봇, 브라우저 자동 실행
```

---

## 라이선스 및 주의사항

- **코드**: 이 저장소의 `scripts/`, `docs/`는 자유롭게 참고하셔도 됩니다.
- **매뉴얼**: 테슬라 Model S 사용자 매뉴얼의 저작권은 Tesla, Inc.에 있습니다. 매뉴얼 원본과 이를 가공한 데이터·모델은 배포하지 않습니다.
- **koGPT2**: `skt/kogpt2-base-v2`는 CC BY-NC-SA 4.0 라이선스(비상업적 이용)를 따릅니다.
- **EXAONE**: LG AI연구원의 EXAONE 모델은 EXAONE AI Model License(비상업적 연구 목적)를 따릅니다.
- 이 챗봇은 개인 학습용 프로젝트이며, 실제 차량 사용 시에는 반드시 공식 매뉴얼을 확인하세요.
