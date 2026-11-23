# =============================================================
# 7-4단계: Gradio 화면 — 질문하면 매뉴얼 문단을 검색해서 koGPT2가 답변
# 실행: python scripts/step7_4_app.py   (C:\STF, 가상환경 켠 상태에서)
# 준비: pip install gradio
# 사용 모델: models/kogpt2-tesla-sft-ctx/   (7-3 결과)
# 실행하면 브라우저가 자동으로 열림. 끄려면 터미널에서 Ctrl+C
# 화면에 "검색된 문단"도 같이 보여줌 → 답이 틀렸을 때 검색 문제인지 모델 문제인지 바로 확인 가능
# =============================================================

from pathlib import Path                       # 경로 처리
import torch                                   # 파이토치
import gradio as gr                            # 웹 화면 라이브러리
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast   # 모델, 토크나이저
from step7_1_build_passages import BM25, load_passages, passage_text   # 7-1 검색기 재사용

ROOT = Path(__file__).resolve().parent.parent              # 프로젝트 최상위 폴더 (C:\STF)
MODEL_DIR = ROOT / "models" / "kogpt2-tesla-sft-ctx"       # 7-3에서 학습한 모델
CTX, USR, SYS = "<ctx>", "<usr>", "<sys>"                  # 학습 때와 똑같은 표시
TOP_K = 2                                                  # 붙일 문단 수 (학습 때와 똑같이 2개)
MAX_LEN = 512                                              # 학습 때 최대 길이

device = "cuda" if torch.cuda.is_available() else "cpu"    # GPU 있으면 GPU
print("모델 불러오는 중...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
if device == "cuda":
    model = model.half()                                   # 답변 생성만 할 거라 fp16으로 → 메모리 절반
model.eval()                                               # 평가(생성) 모드
passages = load_passages()                                 # 7-1에서 만든 문단 1,820개
bm25 = BM25([passage_text(p) for p in passages])           # 검색기
print("준비 완료!")


def prompt_ids(ctx, q):                                    # 7-3과 똑같은 방식으로 입력 만들기
    q_ids = tokenizer.encode(f"{USR}{q}{SYS}")
    c_ids = tokenizer.encode(f"{CTX}{ctx}")
    room = MAX_LEN - len(q_ids) - 120                      # 답변 자리 남기기
    return c_ids[:room] + q_ids


@torch.no_grad()                                           # 생성할 땐 기울기 계산 안 함
def ask(question, sample):                                 # 화면에서 "질문하기" 누르면 실행되는 함수
    question = question.strip()
    if not question:                                       # 빈 질문이면
        return "질문을 입력해주세요.", ""
    top = bm25.search(question, k=TOP_K)                   # 1) 관련 문단 검색
    ctx = "\n".join(passage_text(passages[i]) for i in top)   # 2) 문단 이어붙이기
    x = torch.tensor([prompt_ids(ctx, question)], device=device)
    out = model.generate(                                  # 3) 답변 생성
        x, max_new_tokens=120,
        do_sample=sample,                                  # 체크하면 매번 조금씩 다른 답 (랜덤), 안 하면 항상 같은 답
        top_p=0.9, temperature=0.7,                        # 랜덤일 때 설정 (do_sample=False면 무시됨)
        no_repeat_ngram_size=4,                            # 같은 말 반복 금지
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    answer = tokenizer.decode(out[0][x.shape[1]:], skip_special_tokens=True).strip()   # 질문 부분 빼고 답변만
    found = "\n\n".join(                                   # 검색된 문단을 화면용으로 정리
        f"**{rank}위 · [{passages[i]['chapter']} > {passages[i]['title']}]**\n\n{passages[i]['text']}"
        for rank, i in enumerate(top, 1))
    return answer, found


with gr.Blocks(title="테슬라 Model S 매뉴얼 챗봇 (koGPT2)") as demo:   # 화면 구성
    gr.Markdown("## 🚗 테슬라 Model S 매뉴얼 챗봇\nkoGPT2 + 매뉴얼 문단 검색 (질문 1개씩 답변)")
    with gr.Row():                                         # 가로로 두 칸
        with gr.Column(scale=1):                           # 왼쪽 칸: 질문/답변
            q_box = gr.Textbox(label="질문", placeholder="예: 와이퍼 블레이드는 어떻게 청소하나요?", lines=2)
            sample_box = gr.Checkbox(label="랜덤 생성 (체크하면 매번 다른 답변)", value=False)
            btn = gr.Button("질문하기", variant="primary")
            a_box = gr.Textbox(label="답변", lines=5)
            gr.Examples(                                   # 클릭하면 질문 칸에 들어가는 예시
                examples=["와이퍼 블레이드는 어떻게 청소하고 손상 여부를 확인하나요?",
                          "에어백이 터진 뒤 가루가 피부나 눈에 묻으면 어떻게 해야 하나요?",
                          "Wi-Fi에 연결하려면 어떻게 하나요?",
                          "계기판에 청색 눈송이 아이콘이 뜨면 무슨 뜻인가요?",
                          "타이어 공기압은 얼마로 맞춰야 하나요?"],
                inputs=q_box)
        with gr.Column(scale=1):                           # 오른쪽 칸: 검색된 문단
            ctx_box = gr.Markdown(label="검색된 매뉴얼 문단")
    btn.click(ask, inputs=[q_box, sample_box], outputs=[a_box, ctx_box])      # 버튼 누르면 ask 실행
    q_box.submit(ask, inputs=[q_box, sample_box], outputs=[a_box, ctx_box])   # 질문 칸에서 Enter 쳐도 실행

demo.launch(inbrowser=True)                                # inbrowser=True → 실행하면 브라우저 자동으로 열림
