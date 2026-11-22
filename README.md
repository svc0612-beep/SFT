# 🚗 테슬라 Model S 매뉴얼 챗봇 (koGPT2 SFT)

테슬라 Model S 한국어 사용자 매뉴얼(PDF)로 **koGPT2**(`skt/kogpt2-base-v2`, 1.25억 파라미터)를
**추가 사전학습 → SFT(지도 파인튜닝)** 해서, 매뉴얼 내용으로 답하는 챗봇을 만드는 프로젝트입니다.

GPU 메모리 **2GB 노트북(MX450)** 에서 전 과정을 돌릴 수 있도록 메모리 절약 기법을 적용했습니다.

---

## 전체 흐름

```
PDF ─▶ 텍스트 추출 ─▶ 전처리 ─▶ Q&A 생성(OpenAI) ─▶ 데이터셋
                                                      │
 koGPT2 ─▶ 추가 사전학습(매뉴얼 본문) ─▶ SFT(문단 검색 + Q&A) ─▶ Gradio 챗봇
```

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| 1 | `step1_extract.py` | PDF에서 글자만 추출 (초기 확인용) |
| 2-1 | `step2_1_extract_layout.py` | 위치·글자 크기·굵기까지 포함해 재추출, 제목/대제목/본문 분리 |
| 2-2 | `step2_2_make_paragraphs.py` | 줄 → 문단으로 이어붙이기, 띄어쓰기 복원 |
| 2-3 | `step2_3_clean_noise.py` | 페이지 참조·표 잔해 삭제, `을(를)` 조사 정리 |
| 2-4 | `step2_4_make_sections.py` | 같은 제목 페이지를 섹션으로 묶기, 원본 오류 보정 |
| 3 | `step3_make_qa.py` | OpenAI `gpt-6-luna`로 섹션별 Q&A 생성 |
| 4 | `step4_make_dataset.py` | 불량 Q&A 거르기, 학습/검증 분리 |
| 5 | `step5_pretrain.py` | koGPT2 추가 사전학습 (매뉴얼 본문) |
| 6 | `step6_sft.py` | SFT — 질문만 주고 답하게 학습 (비교용) |
| 7-1 | `step7_1_build_passages.py` | 검색용 문단 생성 + BM25 검색기 |
| 7-2 | `step7_2_make_ctx_dataset.py` | 검색된 문단을 붙인 SFT 데이터 생성 |
| 7-3 | `step7_3_sft_ctx.py` | SFT — 문단을 읽고 답하게 학습 (최종 모델) |
| 7-4 | `step7_4_app.py` | Gradio 챗봇 화면 |

단계별 상세 설명은 [`docs/`](docs/) 폴더에 있습니다.

---

## 결과

| 단계 | 지표 | 값 |
|---|---|---|
| 추가 사전학습 | 검증 perplexity | 50.2 → **16.8** |
| SFT (질문만) | 최고 검증 loss | 2.656 |
| SFT (문단 + 질문) | 최고 검증 loss | **1.882** |
| 검색기 (BM25) | 정답 문단 top-2 적중률 | 67% |

- 데이터: Q&A 966개 (학습 869 / 검증 97), 생성 비용 약 $0.07
- 학습 시간 (MX450 2GB): 추가 사전학습 29분, SFT(문단 포함) 69분

**질문만 주는 방식(6단계)** 은 학습 때 못 본 질문에 엉뚱한 답을 지어냈고,
**관련 문단을 검색해서 같이 주는 방식(7단계)** 으로 바꾸자 답변 정확도가 크게 좋아졌습니다.

### 한계
- 검색이 틀린 문단을 가져오면 답도 틀립니다 (검증 질문의 약 1/3).
- 맞는 문단을 줘도 1.25억 파라미터 모델의 한계로 문장이 엉키는 경우가 있습니다.

---

## 폴더 구조

```
STF/
├─ scripts/          # 단계별 코드
├─ docs/             # 단계별 설명
├─ data/             # (저장소에 없음) 원본 PDF와 가공 데이터
│  ├─ raw/           #   Owners_Manual.pdf
│  └─ processed/     #   각 단계 결과
├─ models/           # (저장소에 없음) 학습된 모델
├─ requirements.txt
└─ .env              # (저장소에 없음) OpenAI API 키
```

---

## 실행 방법

### 1. 환경 준비 (Python 3.11, Windows, NVIDIA GPU)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 매뉴얼 PDF 받기
저작권 문제로 PDF는 저장소에 포함하지 않았습니다. 테슬라 공식 사이트에서 직접 받아주세요.

- Model S (2021+) 한국어판: https://www.tesla.com/ownersmanual/models/ko_kr/Owners_Manual.pdf
- 저장 위치: `data/raw/Owners_Manual.pdf`

### 3. OpenAI API 키 (3단계에만 필요)
프로젝트 폴더에 `.env` 파일을 만들고 아래처럼 적어주세요.
```
OPENAI_API_KEY=sk-...
```

### 4. 순서대로 실행
```bash
python scripts\step2_1_extract_layout.py
python scripts\step2_2_make_paragraphs.py
python scripts\step2_3_clean_noise.py
python scripts\step2_4_make_sections.py
python scripts\step3_make_qa.py           # OpenAI 사용 (약 5~10분)
python scripts\step4_make_dataset.py
python scripts\step5_pretrain.py          # 약 30분 (MX450 기준)
python scripts\step7_1_build_passages.py
python scripts\step7_2_make_ctx_dataset.py
python scripts\step7_3_sft_ctx.py         # 약 70분 (MX450 기준)
python scripts\step7_4_app.py             # 브라우저 자동 실행
```
`step1_extract.py`, `step6_sft.py`는 비교·확인용이라 건너뛰어도 됩니다.

### 2GB GPU에서 돌리기 위한 설정
- fp16 혼합 정밀도
- 8비트 AdamW (`bitsandbytes`)
- 그래디언트 체크포인팅
- 작은 배치 + 그래디언트 누적 (실제 배치 효과 16)

메모리가 부족하면 각 학습 스크립트의 `BATCH`를 줄이고 `ACCUM`을 그만큼 늘려주세요.

---

## 라이선스 및 주의사항

- **코드**: 이 저장소의 `scripts/`, `docs/`는 자유롭게 참고하셔도 됩니다.
- **매뉴얼**: 테슬라 Model S 사용자 매뉴얼의 저작권은 Tesla, Inc.에 있습니다. 매뉴얼 원본과 이를 가공한 데이터·모델은 배포하지 않습니다.
- **koGPT2**: `skt/kogpt2-base-v2`는 CC BY-NC-SA 4.0 라이선스(비상업적 이용)를 따릅니다.
- 이 챗봇은 개인 학습용 프로젝트이며, 실제 차량 사용 시에는 반드시 공식 매뉴얼을 확인하세요.
