# =============================================================
# 7-4단계: Gradio 화면 — 질문하면 매뉴얼 문단을 검색해서 koGPT2가 답변
# 실행: python scripts/step7_4_app.py   (C:\STF, 가상환경 켠 상태에서)
# 준비: pip install gradio
# 사용 모델: models/kogpt2-tesla-sft-ctx/   (7-3 결과)
# 실행하면 브라우저가 자동으로 열림. 끄려면 터미널에서 Ctrl+C
# 화면에 "검색된 문단"도 같이 보여줌 → 답이 틀렸을 때 검색 문제인지 모델 문제인지 바로 확인 가능
# =============================================================

from pathlib import Path                       # 경로 처리
import re                                      # [수정2] 반복 문장 정리용
from collections import Counter                # [수정2] 문장 비슷한지 비교용
import torch                                   # 파이토치
import gradio as gr                            # 웹 화면 라이브러리
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast   # 모델, 토크나이저
from transformers import LogitsProcessor, LogitsProcessorList        # [수정] 생성 규칙을 직접 만들기 위해 사용
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


class GenOnlyNoRepeat(LogitsProcessor):                    # [수정] '답변 안에서만' 같은 n토큰 묶음 반복 금지
    # 기존 no_repeat_ngram_size=4는 입력(문단+질문)까지 포함해서 금지함
    # → 문단에 있는 표현을 그대로 옮겨 쓰는 것도 막혀서 "블uetoth" 같은 깨진 답이 나옴 (검증 정답 84%가 생성 불가)
    # [수정2] 답변 안에서만 막아도, 영어 단어는 토큰 여러 개로 쪼개져서 "Bluetooth"가 두 번째 나오면 "Bluetox"처럼 깨짐
    #        → 문단에 있는 4토큰 묶음이면 반복이어도 허용 (문단 표현을 다시 옮겨 쓰는 건 정상)
    def __init__(self, n, start, prompt_ids):
        self.n = n                                         # 금지할 묶음 길이 (4토큰)
        self.start = start                                 # 답변이 시작되는 위치 (= 입력 길이)
        self.ctx_ngrams = {tuple(prompt_ids[i:i + n])      # [수정2] 입력(문단+질문)에 있는 4토큰 묶음 목록
                           for i in range(len(prompt_ids) - n + 1)}

    def __call__(self, input_ids, scores):
        gen = input_ids[0, self.start:].tolist()           # 지금까지 생성한 답변 토큰만
        if len(gen) >= self.n - 1:
            prefix = tuple(gen[-(self.n - 1):])            # 마지막 3토큰
            banned = {gen[i + self.n - 1]                  # 답변 안에서 같은 3토큰 뒤에 왔던 토큰 = 반복이 되는 토큰
                      for i in range(len(gen) - self.n + 1)
                      if tuple(gen[i:i + self.n - 1]) == prefix}
            banned = {t for t in banned if prefix + (t,) not in self.ctx_ngrams}   # [수정2] 문단에 있는 표현이면 금지 안 함
            if banned:
                scores[0, list(banned)] = -float("inf")    # 그 토큰은 고르지 못하게
        return scores


def _bigrams(s):                                           # [수정2] 띄어쓰기 빼고 두 글자씩 묶음
    s = re.sub(r"\s+", "", s)
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


def _similar(a, b):                                        # [수정2] 두 문장이 60% 넘게 겹치면 "같은 말"로 봄
    A, B = _bigrams(a), _bigrams(b)
    if not A or not B:
        return False
    return 2 * sum((A & B).values()) / (sum(A.values()) + sum(B.values())) > 0.6


def trim_repeats(text):                                    # [수정2] 답변의 반복 정리 (모델은 그대로, 출력만 정리)
    w = text.split()                                       # 단어 목록
    i = 0
    while i < len(w):                                      # 1) 바로 이어서 반복되는 구절(1~6단어)은 두 번째 것을 지움
        for k in range(6, 0, -1):                          #    예: "페달을 밟을 때 페달을 밟을 때" → "페달을 밟을 때"
            while w[i:i + k] and w[i:i + k] == w[i + k:i + 2 * k]:
                del w[i + k:i + 2 * k]
        i += 1
    sents = re.split(r"(?<=[.!?])\s+", " ".join(w))        # 2) 문장 단위로 나눠서
    kept = []
    for s in sents:
        if s and not any(_similar(s, k) for k in kept):    #    앞 문장과 거의 같은 문장은 버림
            kept.append(s)
    return " ".join(kept).strip()


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
        logits_processor=LogitsProcessorList([GenOnlyNoRepeat(4, x.shape[1], x[0].tolist())]),   # [수정2] 답변 안 반복 금지 + 문단 표현은 허용
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    answer = tokenizer.decode(out[0][x.shape[1]:], skip_special_tokens=True).strip()   # 질문 부분 빼고 답변만
    answer = trim_repeats(answer)                          # [수정2] 반복된 구절·문장 정리
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

demo.launch(inbrowser=True,                                # 실행하면 브라우저 자동으로 열림
            server_name="0.0.0.0",                         # 같은 네트워크의 다른 PC에서도 접속 허용 (기본값은 내 PC만)
            server_port=7860)                              # 접속 포트 번호 고정