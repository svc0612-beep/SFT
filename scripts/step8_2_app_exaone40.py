# =============================================================
# 8-2단계: EXAONE 4.0 1.2B 챗봇 (Gradio 화면)
# 실행: python scripts\step8_2_app.py   (SFT 폴더, 가상환경 켠 상태)
# 사용 모델: LGAI-EXAONE/EXAONE-4.0-1.2B (8-1에서 이미 내려받음, 학습 없이 그대로 사용)
# 하는 일: 질문 → 매뉴얼 문단 2개 검색(7-1 BM25) → EXAONE이 문단을 읽고 답변
# 실행하면 브라우저 자동으로 열림 · 같은 네트워크에서는 http://(내 PC IP):7860 으로 접속
# 끄려면 터미널에서 Ctrl+C
# 7-4(koGPT2 챗봇)는 그대로 남아 있음 → 둘을 비교해 볼 수 있음
# =============================================================

import torch                                   # 파이토치
import gradio as gr                            # 웹 화면 라이브러리
from transformers import AutoModelForCausalLM, AutoTokenizer   # 허깅페이스 모델/토크나이저
from step7_1_build_passages import BM25, load_passages, passage_text   # 7-1 검색기 재사용

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"       # 사용할 모델
TOP_K = 2                                      # 붙일 문단 수 (8-1 테스트와 같게)
MAX_NEW = 256                                  # 답변 최대 토큰 수

SYSTEM = ("너는 테슬라 Model S 사용자 매뉴얼 도우미야. "             # 모델에게 주는 역할 설명 (8-1과 같음)
          "아래 [매뉴얼 문단]에 있는 내용만 근거로 한국어로 1~3문장으로 짧게 답해. "
          "문단에 답이 없으면 '매뉴얼에서 찾지 못했습니다.'라고만 답해.")

device = "cuda" if torch.cuda.is_available() else "cpu"             # GPU 있으면 GPU
print("모델 불러오는 중...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)                       # 토크나이저
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)   # bf16으로 GPU에 올림 (약 2.5GB)
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
    messages = [{"role": "system", "content": SYSTEM},              # 3) 채팅 형식 입력
                {"role": "user", "content": f"[매뉴얼 문단]\n{ctx}\n\n[질문]\n{question}"}]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  enable_thinking=False,            # 추론(생각) 모드 끄기 → 바로 답변
                                  return_dict=True, return_tensors="pt").to(device)
    gen = model.generate(**enc, max_new_tokens=MAX_NEW,             # 4) 답변 생성
                         do_sample=sample,                          # 체크하면 매번 조금씩 다른 답
                         temperature=0.6 if sample else None,       # 랜덤일 때만 사용 (EXAONE 권장값)
                         top_p=0.95 if sample else None,
                         repetition_penalty=1.05)                   # 같은 말 반복 살짝 억제
    answer = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()   # 답변만
    found = "\n\n".join(                                            # 검색된 문단을 화면용으로 정리
        f"**{rank}위 · [{passages[i]['chapter']} > {passages[i]['title']}]**\n\n{passages[i]['text']}"
        for rank, i in enumerate(top, 1))
    return answer, found


with gr.Blocks(title="테슬라 Model S 매뉴얼 챗봇 (EXAONE)") as demo:   # 화면 구성
    gr.Markdown("## 🚗 테슬라 Model S 매뉴얼 챗봇\nEXAONE 4.0 1.2B + 매뉴얼 문단 검색 · 실제 차량 사용 시에는 공식 매뉴얼을 확인하세요")
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
                          "타이어 공기압은 얼마로 맞춰야 하나요?"],
                inputs=q_box)
        with gr.Column(scale=1):                                    # 오른쪽: 검색된 문단
            ctx_box = gr.Markdown(label="검색된 매뉴얼 문단")
    btn.click(ask, inputs=[q_box, sample_box], outputs=[a_box, ctx_box])      # 버튼 누르면 ask 실행
    q_box.submit(ask, inputs=[q_box, sample_box], outputs=[a_box, ctx_box])   # Enter 쳐도 실행

demo.launch(inbrowser=True,                                         # 실행하면 브라우저 자동으로 열림
            server_name="0.0.0.0",                                  # 같은 네트워크의 다른 PC에서도 접속 허용
            server_port=7860)                                       # 접속 포트 번호 고정
