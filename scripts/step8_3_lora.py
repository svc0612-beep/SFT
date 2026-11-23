# =============================================================
# 8-3단계: EXAONE 4.0 1.2B LoRA 파인튜닝 (문단 + 질문 → 답변)
# 실행: python scripts\step8_3_lora.py          (EXAONE 4.0 1.2B)
#       python scripts\step8_3_lora.py --m35    (EXAONE 3.5 2.4B — 8-0에서 변환한 모델)
#       python scripts\step8_3_lora.py --m35 --unk   (8-4 "모름" 예시가 섞인 데이터로 학습 → 폴더 이름 끝에 -unk)
# 준비: pip install peft   (requirements.txt에 추가됨)
# 입력: data/processed/sft_ctx_train.jsonl, sft_ctx_val.jsonl   (7-2 결과: 문단 2개 + 질문 + 답변)
# 결과: models/exaone-tesla-lora/   (LoRA 어댑터만 저장, 약 수십 MB · 원본 EXAONE은 그대로)
# LoRA란? 모델 전체(12억 개)를 바꾸지 않고, 작은 보조 가중치만 붙여서 학습하는 방식
#        → 메모리를 적게 쓰고, 원래 모델이 가진 한국어 실력을 덜 망가뜨림
# 채점: 답변 부분만 (시스템 문구·문단·질문은 채점 안 함) — 7-3과 같은 원칙
# =============================================================

import json                                    # jsonl 읽기
import sys                                     # --m35 옵션 확인용
import math                                    # 올림 계산
import random                                  # 데이터 섞기
import time                                    # 시간 측정
from pathlib import Path                       # 경로 처리
import torch                                   # 파이토치
from transformers import AutoModelForCausalLM, AutoTokenizer   # 모델/토크나이저
from peft import LoraConfig, get_peft_model    # LoRA

ROOT = Path(__file__).resolve().parent.parent                        # SFT 폴더
USE_UNK = "--unk" in sys.argv                                        # --unk: 8-4 "모름" 예시가 섞인 학습 데이터 사용
TRAIN_PATH = ROOT / "data" / "processed" / ("sft_ctx_unk_train.jsonl" if USE_UNK else "sft_ctx_train.jsonl")   # 학습 데이터 (7-2 또는 8-4)
VAL_PATH = ROOT / "data" / "processed" / "sft_ctx_val.jsonl"         # 검증 데이터 (7-2)
USE_35 = "--m35" in sys.argv                                         # --m35: EXAONE 3.5 2.4B로 학습
MODEL_ID = str(ROOT / "models" / "exaone35-2.4b-llama") if USE_35 else "LGAI-EXAONE/EXAONE-4.0-1.2B"   # 원본 모델
OUT_DIR = ROOT / "models" / (("exaone35-tesla-lora" if USE_35 else "exaone-tesla-lora") + ("-unk" if USE_UNK else ""))   # LoRA 어댑터 저장 폴더 (모델·데이터별로 따로 → 기존 어댑터 안 지워짐)

SYSTEM = ("너는 테슬라 Model S 사용자 매뉴얼 도우미야. "              # 8-1·8-2와 똑같은 지시문 (학습과 사용이 같아야 함)
          "아래 [매뉴얼 문단]에 있는 내용만 근거로 한국어로 1~3문장으로 짧게 답해. "
          "문단에 답이 없으면 '매뉴얼에서 찾지 못했습니다.'라고만 답해.")

MAX_LEN = 1024        # 최대 토큰 길이 (EXAONE 토크나이저 기준, 넘으면 문단 뒤쪽을 자름)
BATCH = 2             # 한 번에 GPU에 올리는 샘플 수
ACCUM = 8             # 8번 모아서 업데이트 → 실제 배치 효과 16 (7-3과 같음)
EPOCHS = 2            # 이미 잘 학습된 모델이라 짧게 (에폭마다 검증해서 최고만 저장)
LR = 2e-4             # LoRA는 보조 가중치만 학습해서 학습률을 크게 잡는 게 보통
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda"


def prompt_ids(tok, ctx, q):                                        # 시스템 + 문단 + 질문 → 토큰 (챗봇과 같은 형식)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"[매뉴얼 문단]\n{ctx}\n\n[질문]\n{q}"}]
    return tok.apply_chat_template(messages, add_generation_prompt=True,
                                   enable_thinking=False, tokenize=True, return_dict=True)["input_ids"]


def encode(tok, row):                                               # 샘플 하나 → (입력 토큰, 정답 라벨)
    p = prompt_ids(tok, row["ctx"], row["question"])                # 답변 앞부분
    a = tok(row["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]   # 답변 + 끝 표시
    if len(p) + len(a) > MAX_LEN:                                   # 너무 길면
        cut = len(p) + len(a) - MAX_LEN                             # 넘친 만큼
        p = prompt_ids(tok, row["ctx"][: max(0, len(row["ctx"]) - cut * 2)], row["question"])   # 문단을 줄여서 다시
    ids = (p + a)[:MAX_LEN]
    labels = ([-100] * len(p) + a)[:MAX_LEN]                        # -100 = 채점 안 함 → 답변만 채점
    return ids, labels


def make_batch(items, pad_id):                                      # 길이가 다른 샘플들을 한 묶음으로
    n = max(len(i) for i, _ in items)
    ids = torch.full((len(items), n), pad_id)                       # 빈 자리는 pad
    lab = torch.full((len(items), n), -100)                         # 빈 자리는 채점 안 함
    mask = torch.zeros((len(items), n), dtype=torch.long)           # 실제 토큰 위치 = 1
    for k, (i, l) in enumerate(items):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
        mask[k, :len(i)] = 1
    return ids.to(device), lab.to(device), mask.to(device)


@torch.no_grad()
def val_loss(model, data, pad_id):                                  # 검증 loss 계산
    model.eval()
    losses = []
    for k in range(0, len(data), BATCH):
        ids, lab, mask = make_batch(data[k:k + BATCH], pad_id)
        losses.append(model(input_ids=ids, attention_mask=mask, labels=lab).loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):                 # 이미 학습한 어댑터가 있으면 중단 (덮어쓰기 방지)
        raise SystemExit(f"[중단] 이미 있음: {OUT_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)                   # 토크나이저 (8-1에서 받은 것 재사용)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"끝 표시 토큰: {tok.eos_token!r} / pad id: {pad_id}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)   # bf16
    model.gradient_checkpointing_enable()                           # 메모리 절약
    model.enable_input_require_grads()                              # LoRA + 체크포인팅을 같이 쓸 때 필요
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,       # 보조 가중치 크기(r)와 세기(alpha)
                      target_modules="all-linear", task_type="CAUSAL_LM")   # 모든 선형층에 붙임
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()                              # 학습되는 가중치 비율 출력

    load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
    train = [encode(tok, r) for r in load(TRAIN_PATH)]               # 학습 867개
    val = [encode(tok, r) for r in load(VAL_PATH)]                   # 검증 96개
    print(f"학습 {len(train)}개 / 검증 {len(val)}개 | 최대 길이 {max(len(i) for i, _ in train)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    total = math.ceil(len(train) / BATCH / ACCUM) * EPOCHS          # 총 업데이트 횟수
    sched = torch.optim.lr_scheduler.LambdaLR(                      # 처음 5% 워밍업 후 선형 감소
        opt, lambda s: min((s + 1) / (0.05 * total), max(0.0, (total - s) / (0.95 * total))))

    best = val_loss(model, val, pad_id)                             # 학습 전 검증 loss
    print(f"[학습 전] 검증 loss {best:.3f}  (총 업데이트 {total}번)")
    best = float("inf")
    step, t0 = 0, time.time()
    model.train()
    for ep in range(1, EPOCHS + 1):
        random.shuffle(train)                                       # 에폭마다 순서 섞기
        for k in range(0, len(train), BATCH):
            ids, lab, mask = make_batch(train[k:k + BATCH], pad_id)
            loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss / ACCUM
            loss.backward()                                         # 기울기 누적
            if (k // BATCH + 1) % ACCUM == 0 or k + BATCH >= len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    el = (time.time() - t0) / 60
                    print(f"  epoch {ep} | {step}/{total} | loss {loss.item() * ACCUM:.3f} | 경과 {el:.1f}분, 남은 예상 {el / step * (total - step):.1f}분")
        vl = val_loss(model, val, pad_id)                           # 에폭 끝 검증
        if vl < best:                                               # 최고 기록이면 저장
            best = vl
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)                          # LoRA 어댑터만 저장
            print(f"[epoch {ep}] 검증 loss {vl:.3f}  ← 최고 기록, 저장함")
        else:
            print(f"[epoch {ep}] 검증 loss {vl:.3f}")
    print(f"\n완료 (총 {(time.time() - t0) / 60:.1f}분) → {OUT_DIR}  (최고 검증 loss {best:.3f})")
    print(f"GPU 최대 메모리 사용: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
