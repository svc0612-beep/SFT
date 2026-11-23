# =============================================================
# 8-2단계: 최종 챗봇 (Gradio 화면) — EXAONE 3.5 2.4B + LoRA(모름 학습)
# 실행: python scripts\step8_2_app.py   (SFT 폴더, 가상환경 켠 상태)
# 사용 모델: models/exaone35-2.4b-llama      (8-0에서 Llama 형식으로 변환한 EXAONE 3.5 2.4B)
#           models/exaone35-tesla-lora-unk  (8-3 --m35 --unk 결과: 매뉴얼 답변 + "모르면 모른다고" 학습한 어댑터)
# 하는 일: 질문 → 매뉴얼 문단 2개 검색(7-1 BM25) → 모델이 문단을 읽고 답변 (문단에 답이 없으면 "찾지 못했습니다")
# 실행하면 브라우저 자동으로 열림 · 같은 네트워크에서는 http://(내 PC IP):7860 으로 접속
# 끄려면 터미널에서 Ctrl+C
# 이전 버전(EXAONE 4.0 1.2B, 학습 없음)은 step8_2_app_exaone40.py 로 남겨 둠
# =============================================================

from pathlib import Path                       # 경로 처리
import torch                                   # 파이토치
import gradio as gr                            # 웹 화면 라이브러리
from transformers import AutoModelForCausalLM, AutoTokenizer   # 허깅페이스 모델/토크나이저
from peft import PeftModel                     # LoRA 어댑터 얹기
from step7_1_build_passages import BM25, load_passages, passage_text   # 7-1 검색기 재사용

ROOT = Path(__file__).resolve().parent.parent                       # SFT 폴더
MODEL_DIR = ROOT / "models" / "exaone35-2.4b-llama"                 # 원본 모델 (8-0 변환 결과)
LORA_DIR = ROOT / "models" / "exaone35-tesla-lora-unk"              # 최종 LoRA 어댑터 (8-3 --m35 --unk)
TOP_K = 2                                      # 붙일 문단 수 (학습·테스트와 같게)
MAX_NEW = 256                                  # 답변 최대 토큰 수

SYSTEM = ("너는 테슬라 Model S 사용자 매뉴얼 도우미야. "             # 학습 때와 똑같은 지시문 (한 글자도 바꾸면 안 됨)
          "아래 [매뉴얼 문단]에 있는 내용만 근거로 한국어로 1~3문장으로 짧게 답해. "
          "문단에 답이 없으면 '매뉴얼에서 찾지 못했습니다.'라고만 답해.")

device = "cuda" if torch.cuda.is_available() else "cpu"             # GPU 있으면 GPU
print("모델 불러오는 중... (약 30초)")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))                 # 토크나이저
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.bfloat16).to(device)   # bf16으로 GPU에 올림 (약 5GB)
model = PeftModel.from_pretrained(model, str(LORA_DIR))             # LoRA 어댑터 얹기 (원본 파일은 그대로)
model = model.merge_and_unload()                                    # 어댑터를 모델에 합쳐서 답변 속도 향상 (메모리 안에서만, 파일 변경 없음)
model.eval()                                                        # 생성 모드 (학습 안 함)
passages = load_passages()                                          # 7-1 문단 1,820개
bm25 = BM25([passage_text(p) for p in passages])                    # 검색기
print("준비 완료!")


@torch.no_grad()                                                    # 생성할 땐 기울기 계산 안 함
def ask(question, sample):                                          # 화면에서 "질문하기" 누르면 실행
    question = question.strip()
    if not question:                                                # 빈 질문이면
        return "질문을 입력해주세요.", ""
    top = bm25.search(question, k=TOP_K)                            # 1) 관련 문단 검색
    ctx = "\n".join(passage_text(passages[i]) for i in top)         # 2) 문단 이어붙이기
    messages = [{"role": "system", "content": SYSTEM},              # 3) 채팅 형식 입력 (학습 때와 같은 모양)
                {"role": "user", "content": f"[매뉴얼 문단]\n{ctx}\n\n[질문]\n{question}"}]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  enable_thinking=False,            # 학습·테스트 때와 같은 옵션 (3.5는 무시함)
                                  return_dict=True, return_tensors="pt").to(device)   # input_ids, attention_mask
    gen = model.generate(**enc, max_new_tokens=MAX_NEW,             # 4) 답변 생성
                         do_sample=sample,                          # 체크하면 매번 조금씩 다른 답
                         temperature=0.6 if sample else None,       # 랜덤일 때만 사용
                         top_p=0.95 if sample else None,
                         repetition_penalty=1.05)                   # 같은 말 반복 살짝 억제 (테스트와 같은 값)
    answer = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()   # 입력 부분 빼고 답변만
    if "찾지 못했습니다" in answer:                                  # 모델이 모른다고 답했으면
        answer += "\n(질문을 다른 표현으로 바꿔 보거나, 오른쪽 검색 문단을 확인해 보세요.)"   # 사용자 안내 한 줄 추가
    found = "\n\n".join(                                            # 검색된 문단을 화면용으로 정리
        f"**{rank}위 · [{passages[i]['chapter']} > {passages[i]['title']}]**\n\n{passages[i]['text']}"
        for rank, i in enumerate(top, 1))
    return answer, found


with gr.Blocks(title="테슬라 Model S 매뉴얼 챗봇 (EXAONE 3.5)") as demo:   # 화면 구성
    gr.Markdown("## 🚗 테슬라 Model S 매뉴얼 챗봇\nEXAONE 3.5 2.4B + LoRA 파인튜닝 + 매뉴얼 문단 검색 · 실제 차량 사용 시에는 공식 매뉴얼을 확인하세요")
    with gr.Row():                                                  # 가로로 두 칸
        with gr.Column(scale=1):                                    # 왼쪽: 질문/답변
            q_box = gr.Textbox(label="질문", placeholder="예: 와이퍼 블레이드는 어떻게 청소하나요?", lines=2)
            sample_box = gr.Checkbox(label="랜덤 생성 (체크하면 매번 다른 답변)", value=False)
            btn = gr.Button("질문하기", variant="primary")
            a_box = gr.Textbox(label="답변", lines=5)
            gr.Examples(                                            # 클릭하면 질문 칸에 들어가는 예시
                examples=["와이퍼 블레이드는 어떻게 청소하고 손상 여부를 확인하나요?",
                          "요크 스티어링 휠을 따뜻하게 하려면 어떻게 하나요?",
                          "Model S에 연결할 스마트폰을 직접 바꾸려면 어떻게 하나요?",
                          "견인 모드를 취소하려면 어떻게 해야 하나요?",
                          "Model S에서 좌석 커버를 사용해도 되나요?"],
                inputs=q_box)
        with gr.Column(scale=1):                                    # 오른쪽: 검색된 문단
            ctx_box = gr.Markdown(label="검색된 매뉴얼 문단")
    btn.click(ask, inputs=[q_box, sample_box], outputs=[a_box, ctx_box])      # 버튼 누르면 ask 실행
    q_box.submit(ask, inputs=[q_box, sample_box], outputs=[a_box, ctx_box])   # Enter 쳐도 실행

demo.launch(inbrowser=True,                                         # 실행하면 브라우저 자동으로 열림
            server_name="0.0.0.0",                                  # 같은 네트워크의 다른 PC에서도 접속 허용
            server_port=7860)                                       # 접속 포트 번호 고정
