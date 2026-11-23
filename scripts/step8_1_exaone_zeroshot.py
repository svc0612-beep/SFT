# =============================================================
# 8-1단계: EXAONE 4.0 1.2B를 학습 없이(zero-shot) 써보기
# 실행: python scripts\step8_1_exaone_zeroshot.py          (학습 없는 원본)
#       python scripts\step8_1_exaone_zeroshot.py --lora   (8-3 LoRA 어댑터 적용)
#       python scripts\step8_1_exaone_zeroshot.py --m35    (EXAONE 3.5 2.4B, 학습 없이)
# 입력: data/processed/sft_val.jsonl   (검증 질문 96개 — koGPT2 비교에 쓴 것과 같음)
#       data/processed/passages.jsonl  (7-1 검색용 문단)
# 결과: data/processed/step8_1_exaone_zeroshot.jsonl  (질문, 정답, 모델 답변)
# 하는 일: 챗봇(7-4)과 똑같이 질문으로 문단 2개를 검색해서 붙이고, EXAONE에게 답하게 함
#          → koGPT2 결과와 같은 질문으로 비교 (가중치 학습 없음)
# 처음 실행 때 모델(약 2.5GB)을 허깅페이스에서 자동으로 내려받음
# =============================================================

import json                                    # jsonl 읽기/쓰기
import sys                                     # [LoRA] --lora 옵션 확인용
import time                                    # 시간 측정
from pathlib import Path                       # 경로 처리
import torch                                   # 파이토치
from transformers import AutoModelForCausalLM, AutoTokenizer   # 허깅페이스 모델/토크나이저
from step7_1_build_passages import BM25, load_passages, passage_text   # 7-1 검색기 그대로 재사용

ROOT = Path(__file__).resolve().parent.parent                       # SFT 폴더
VAL_PATH = ROOT / "data" / "processed" / "sft_val.jsonl"            # 검증 질문
USE_LORA = "--lora" in sys.argv                                     # [LoRA] --lora 붙이면 8-3 어댑터를 얹어서 테스트
USE_35 = "--m35" in sys.argv                                        # [3.5] --m35 붙이면 EXAONE 3.5 2.4B로 테스트
USE_UNK = "--unk" in sys.argv                                       # [모름] --unk 붙이면 "모름" 학습한 어댑터(8-4 데이터) 사용
LORA_DIR = ROOT / "models" / (("exaone35-tesla-lora" if USE_35 else "exaone-tesla-lora") + ("-unk" if USE_UNK else ""))   # [LoRA] 8-3 결과 (모델별 어댑터)
MODEL_ID = str(ROOT / "models" / "exaone35-2.4b-llama") if USE_35 else "LGAI-EXAONE/EXAONE-4.0-1.2B"   # 사용할 모델 (3.5는 8-0에서 Llama 형식으로 변환한 것)
OUT_NAME = "step8_1_exaone{}_{}{}.jsonl".format("35" if USE_35 else "", "lora" if USE_LORA else "zeroshot", "_unk" if USE_UNK else "")   # 결과 파일 (옵션별로 다른 파일 → 덮어쓰기 없음)
OUT_PATH = ROOT / "data" / "processed" / OUT_NAME                   # 결과 저장
TOP_K = 2                                                           # 붙일 문단 수 (koGPT2 챗봇과 같게)
MAX_NEW = 200                                                       # 답변 최대 토큰 수

SYSTEM = ("너는 테슬라 Model S 사용자 매뉴얼 도우미야. "             # 모델에게 주는 역할 설명
          "아래 [매뉴얼 문단]에 있는 내용만 근거로 한국어로 1~3문장으로 짧게 답해. "
          "문단에 답이 없으면 '매뉴얼에서 찾지 못했습니다.'라고만 답해.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"         # GPU 사용
    print(f"모델 불러오는 중... ({MODEL_ID}, 처음이면 다운로드에 몇 분 걸림)")
    remote = False                                                   # 변환한 3.5도 transformers 기본 코드로 돌아가서 허깅페이스 코드 필요 없음
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=remote)   # 토크나이저
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,   # bf16으로 GPU에 올림 (4.0: 약 2.5GB / 3.5: 약 5GB)
                                                 trust_remote_code=remote).to(device)
    if USE_LORA:                                                     # [LoRA] 어댑터 얹기 (원본 모델은 그대로)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(LORA_DIR))
        print(f"LoRA 어댑터 사용: {LORA_DIR}")
    model.eval()                                                     # 평가 모드 (학습 안 함)

    passages = load_passages()                                       # 7-1 문단 1,820개
    bm25 = BM25([passage_text(p) for p in passages])                 # 검색기
    rows = [json.loads(l) for l in open(VAL_PATH, encoding="utf-8")] # 검증 질문 96개

    t0 = time.time()
    out = []
    for n, r in enumerate(rows, 1):
        top = bm25.search(r["question"], k=TOP_K)                    # 1) 문단 검색 (챗봇과 같은 방식)
        ctx = "\n".join(passage_text(passages[i]) for i in top)      # 2) 문단 이어붙이기
        messages = [{"role": "system", "content": SYSTEM},           # 3) 채팅 형식으로 입력 만들기
                    {"role": "user", "content": f"[매뉴얼 문단]\n{ctx}\n\n[질문]\n{r['question']}"}]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,   # 모델이 학습된 대화 형식으로 변환
                                      enable_thinking=False,         # 추론(생각) 모드 끄기 → 바로 답변
                                      return_dict=True, return_tensors="pt").to(device)   # input_ids, attention_mask
        with torch.no_grad():                                        # 생성할 땐 기울기 계산 안 함
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,   # 4) 답변 생성 (항상 같은 답이 나오게 Greedy)
                                 repetition_penalty=1.05)            # 같은 말 반복 살짝 억제
        answer = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()   # 입력 부분 빼고 답변만
        out.append({"question": r["question"], "answer_ref": r["answer"], "answer_model": answer,
                    "passages": [passages[i]["title"] for i in top]})
        if n <= 3 or n % 20 == 0:                                    # 진행 상황과 샘플 출력
            print(f"[{n}/{len(rows)}] {(time.time() - t0) / 60:.1f}분\n  Q: {r['question']}\n  모델: {answer}\n  정답: {r['answer']}\n")

    with open(OUT_PATH, "w", encoding="utf-8") as f:                 # 결과 저장
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"완료 ({(time.time() - t0) / 60:.1f}분) → {OUT_PATH.name}  / 모델 가중치 변경 없음")


if __name__ == "__main__":
    main()
