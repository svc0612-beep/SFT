# =============================================================
# 6단계: SFT (지도 파인튜닝) — "질문을 받으면 답한다"를 학습
# 실행: python scripts/step6_sft.py   (C:\STF, 가상환경 켠 상태에서)
# 입력: models/kogpt2-tesla-pretrain/        (5단계 결과 — 출발점)
#       data/processed/sft_train.jsonl, sft_val.jsonl   (4단계 결과)
# 결과: models/kogpt2-tesla-sft/             (검증 loss가 가장 낮았던 모델)
# 5단계와 다른 점: 질문 부분은 채점하지 않고 "답변 부분만" loss를 계산함
# =============================================================

import json                                    # jsonl 읽기
import math                                    # 스케줄 계산용 ceil
import random                                  # 데이터 섞기
import time                                    # 소요 시간 측정
from pathlib import Path                       # 경로 처리
import torch                                   # 파이토치
import bitsandbytes as bnb                     # 8비트 Adam (메모리 절약)
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast   # 모델, 토크나이저

ROOT = Path(__file__).resolve().parent.parent                  # 프로젝트 최상위 폴더 (C:\STF)
BASE_DIR = ROOT / "models" / "kogpt2-tesla-pretrain"           # 출발점 = 추가 사전학습된 모델
TRAIN_PATH = ROOT / "data" / "processed" / "sft_train.jsonl"   # 학습 데이터
VAL_PATH = ROOT / "data" / "processed" / "sft_val.jsonl"       # 검증 데이터
OUT_DIR = ROOT / "models" / "kogpt2-tesla-sft"                 # 결과 저장 폴더

USR, SYS = "<usr>", "<sys>"   # 4단계에서 정한 대화 표시 (질문 시작 / 답변 시작)
MAX_LEN = 256         # 한 샘플 최대 토큰 길이 (넘으면 답변 뒤쪽을 자름)
BATCH = 4             # 한 번에 GPU에 올리는 샘플 수
ACCUM = 4             # 4번 모아서 업데이트 → 실제 배치 효과 16
EPOCHS = 5            # 전체 데이터 반복 횟수
LR = 5e-5             # 학습률
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda"


def load(path):                                            # jsonl 읽기
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode(tokenizer, q, a):                               # Q&A 하나 → (입력 토큰, 정답 라벨)
    prompt_ids = tokenizer.encode(f"{USR}{q}{SYS}")        # "<usr>질문<sys>" 부분 토큰
    answer_ids = tokenizer.encode(a) + [tokenizer.eos_token_id]   # "답변" 토큰 + 끝 표시(</s>) → 모델이 여기서 멈추는 법을 배움
    ids = (prompt_ids + answer_ids)[:MAX_LEN]              # 이어붙이고 최대 길이로 자름
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LEN]   # 핵심! 질문 부분은 -100 → loss 계산에서 제외 (답변만 채점)
    return ids, labels


def make_batch(samples, pad_id):                           # 길이가 다른 샘플들을 같은 길이로 맞춰 묶기
    L = max(len(ids) for ids, _ in samples)                # 이 묶음에서 가장 긴 길이
    ids = [s[0] + [pad_id] * (L - len(s[0])) for s in samples]        # 짧은 건 <pad>로 채움
    labels = [s[1] + [-100] * (L - len(s[1])) for s in samples]       # 채운 부분은 채점 제외
    mask = [[1] * len(s[0]) + [0] * (L - len(s[0])) for s in samples] # 진짜 토큰=1, 채운 칸=0 (모델이 <pad>는 무시하게)
    t = lambda v: torch.tensor(v, device=device)           # 리스트 → GPU 텐서
    return t(ids), t(labels), t(mask)


@torch.no_grad()
def evaluate(model, data, pad_id):                         # 검증 데이터 loss (답변 부분만)
    model.eval()
    losses = []
    for i in range(0, len(data), BATCH):
        ids, labels, mask = make_batch(data[i:i + BATCH], pad_id)
        with torch.autocast("cuda", dtype=torch.float16):
            losses.append(model(input_ids=ids, attention_mask=mask, labels=labels).loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def answer(model, tokenizer, q):                           # 질문 하나에 답변 생성 (학습 결과 확인용)
    model.eval()
    x = torch.tensor([tokenizer.encode(f"{USR}{q}{SYS}")], device=device)   # "<usr>질문<sys>"까지만 넣음
    out = model.generate(x, max_new_tokens=100, do_sample=False,            # 가장 확률 높은 단어로 생성 (매번 같은 결과)
                         repetition_penalty=1.2,                            # 같은 말 반복 억제
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model.train()
    return tokenizer.decode(out[0][x.shape[1]:], skip_special_tokens=True).strip()   # 질문 부분 빼고 답변만


def main():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(BASE_DIR)   # 5단계에서 같이 저장한 토크나이저
    model = GPT2LMHeadModel.from_pretrained(BASE_DIR).to(device)    # 추가 사전학습된 모델
    model.gradient_checkpointing_enable()                  # 메모리 절약
    model.config.use_cache = False
    pad_id = tokenizer.pad_token_id                        # <pad> 토큰 번호

    train_rows, val_rows = load(TRAIN_PATH), load(VAL_PATH)
    train = [encode(tokenizer, r["question"], r["answer"]) for r in train_rows]   # 학습 데이터 토큰화
    val = [encode(tokenizer, r["question"], r["answer"]) for r in val_rows]       # 검증 데이터 토큰화
    lens = [len(ids) for ids, _ in train]
    print(f"학습 {len(train)}개 / 검증 {len(val)}개 | 토큰 길이 평균 {sum(lens) // len(lens)}, 최대 {max(lens)}")

    steps_per_epoch = math.ceil(len(train) / BATCH / ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(         # 워밍업 5% → 선형 감소 (5단계와 동일)
        optimizer, lambda s: min((s + 1) / (0.05 * total_steps), max(0.0, (total_steps - s) / (0.95 * total_steps))))
    scaler = torch.amp.GradScaler("cuda")

    best = evaluate(model, val, pad_id)                    # 학습 전 검증 loss
    print(f"[학습 전] 검증 loss {best:.3f}  (총 업데이트 {total_steps}번)")
    best = float("inf")                                    # 최고 기록 초기화 (학습 후 모델만 저장 대상)

    step, t0 = 0, time.time()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train)
        for i in range(0, len(train), BATCH):
            ids, labels, mask = make_batch(train[i:i + BATCH], pad_id)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / ACCUM
            scaler.scale(loss).backward()
            if (i // BATCH + 1) % ACCUM == 0 or i + BATCH >= len(train):   # ACCUM번 모이면 업데이트
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                if step % 20 == 0:
                    el = time.time() - t0
                    print(f"  epoch {epoch} | {step}/{total_steps} | loss {loss.item() * ACCUM:.3f} | "
                          f"경과 {el / 60:.1f}분, 남은 예상 {el / step * (total_steps - step) / 60:.1f}분")

        vl = evaluate(model, val, pad_id)                  # 에폭마다 검증
        mark = ""
        if vl < best:                                      # 지금까지 중 가장 좋으면 저장 (과적합 대비)
            best = vl
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.config.use_cache = True
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
            model.config.use_cache = False
            mark = "  ← 최고 기록, 저장함"
        print(f"[epoch {epoch}] 검증 loss {vl:.3f}{mark}")

    print(f"\n완료 (총 {(time.time() - t0) / 60:.1f}분) → {OUT_DIR}  (최고 검증 loss {best:.3f})")
    print(f"GPU 최대 메모리 사용: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    model = GPT2LMHeadModel.from_pretrained(OUT_DIR).to(device)   # 저장된 최고 모델로 맛보기
    print("\n--- 검증용 질문 3개로 맛보기 (학습 때 안 본 질문) ---")
    for r in val_rows[:3]:
        print(f"Q: {r['question']}\n모델: {answer(model, tokenizer, r['question'])}\n정답: {r['answer']}\n")


if __name__ == "__main__":
    main()
