# =============================================================
# 7-3단계: 매뉴얼 문단을 같이 넣어주는 SFT — "읽고 답하기"를 학습
# 실행: python scripts/step7_3_sft_ctx_e6.py   (C:\STF, 가상환경 켠 상태에서)
# [재학습] step7_3_sft_ctx.py 복사본 — 바꾼 건 EPOCHS(3→6), OUT_DIR(새 폴더) 두 줄뿐. 기존 모델은 그대로 둠
# 입력: models/kogpt2-tesla-pretrain/                       (5단계 결과 — 출발점)
#       data/processed/sft_ctx_train.jsonl, sft_ctx_val.jsonl  (7-2 결과)
# 결과: models/kogpt2-tesla-sft-ctx/                        (검증 loss가 가장 낮았던 모델)
# 6단계와 다른 점: 입력 앞에 <ctx>문단이 붙음 / 샘플이 길어서 배치 2 x 누적 8 / 3에폭 / 생성 설정 변경
# 예상 소요 시간: 노트북(MX450) 기준 60~80분
# =============================================================

import json                                    # jsonl 읽기
import math                                    # ceil
import random                                  # 섞기
import time                                    # 시간 측정
from pathlib import Path                       # 경로 처리
import torch                                   # 파이토치
import bitsandbytes as bnb                     # 8비트 Adam
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast   # 모델, 토크나이저

ROOT = Path(__file__).resolve().parent.parent                       # 프로젝트 최상위 폴더 (C:\STF)
BASE_DIR = ROOT / "models" / "kogpt2-tesla-pretrain"                # 출발점 = 추가 사전학습 모델
TRAIN_PATH = ROOT / "data" / "processed" / "sft_ctx_train.jsonl"    # 7-2 학습 데이터
VAL_PATH = ROOT / "data" / "processed" / "sft_ctx_val.jsonl"        # 7-2 검증 데이터
OUT_DIR = ROOT / "models" / "kogpt2-tesla-sft-ctx-e6"               # [재학습] 결과 저장 폴더 (새 폴더, 기존 모델 덮어쓰기 안 함)

CTX, USR, SYS = "<ctx>", "<usr>", "<sys>"   # 7-2에서 정한 표시
MAX_LEN = 512         # 최대 토큰 길이 (7-2 결과 최대 494 → 다 들어감)
BATCH = 2             # 샘플이 길어서 한 번에 2개만 (6단계는 4)
ACCUM = 8             # 8번 모아서 업데이트 → 실제 배치 효과 16 (6단계와 동일)
EPOCHS = 6            # [재학습] 3에폭까지 검증 loss가 계속 내려가서 6으로 늘림 (검증 loss 최저 에폭만 저장되므로 과적합돼도 최고 모델이 남음)
LR = 5e-5             # 학습률
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prompt_ids(tokenizer, ctx, q):                         # "<ctx>문단<usr>질문<sys>" 토큰 만들기
    q_ids = tokenizer.encode(f"{USR}{q}{SYS}")             # 질문 부분
    c_ids = tokenizer.encode(f"{CTX}{ctx}")                # 문단 부분
    room = MAX_LEN - len(q_ids) - 120                      # 답변 자리(약 120토큰)를 남기고 문단에 쓸 수 있는 길이
    return c_ids[:room] + q_ids                            # 혹시 문단이 너무 길면 뒤를 잘라서 맞춤 (보통은 안 잘림)


def encode(tokenizer, r):                                  # 샘플 하나 → (입력 토큰, 정답 라벨)
    p = prompt_ids(tokenizer, r["ctx"], r["question"])     # 문단 + 질문
    a = tokenizer.encode(r["answer"]) + [tokenizer.eos_token_id]   # 답변 + 끝 표시
    ids = (p + a)[:MAX_LEN]
    labels = ([-100] * len(p) + a)[:MAX_LEN]               # 문단·질문은 채점 제외, 답변만 채점
    return ids, labels


def make_batch(samples, pad_id):                           # 길이 맞춰 묶기 (6단계와 동일)
    L = max(len(ids) for ids, _ in samples)
    ids = [s[0] + [pad_id] * (L - len(s[0])) for s in samples]
    labels = [s[1] + [-100] * (L - len(s[1])) for s in samples]
    mask = [[1] * len(s[0]) + [0] * (L - len(s[0])) for s in samples]
    t = lambda v: torch.tensor(v, device=device)
    return t(ids), t(labels), t(mask)


@torch.no_grad()
def evaluate(model, data, pad_id):                         # 검증 loss (답변 부분만)
    model.eval()
    losses = []
    for i in range(0, len(data), BATCH):
        ids, labels, mask = make_batch(data[i:i + BATCH], pad_id)
        with torch.autocast("cuda", dtype=torch.float16):
            losses.append(model(input_ids=ids, attention_mask=mask, labels=labels).loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def answer(model, tokenizer, ctx, q):                      # 문단 + 질문 → 답변 생성
    model.eval()
    x = torch.tensor([prompt_ids(tokenizer, ctx, q)], device=device)
    out = model.generate(
        x, max_new_tokens=100, do_sample=False,            # 가장 확률 높은 단어로 생성
        no_repeat_ngram_size=4,                            # 같은 4토큰 묶음 반복 금지 (6단계의 repetition_penalty 대신)
        # ↑ repetition_penalty는 "입력에 나온 단어"까지 불리하게 만들어서, 문단에서 그대로 옮겨 답해야 하는 이번 방식과 안 맞음
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model.train()
    return tokenizer.decode(out[0][x.shape[1]:], skip_special_tokens=True).strip()


def main():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(BASE_DIR)
    model = GPT2LMHeadModel.from_pretrained(BASE_DIR).to(device)
    model.gradient_checkpointing_enable()                  # 메모리 절약
    model.config.use_cache = False
    pad_id = tokenizer.pad_token_id

    train_rows, val_rows = load(TRAIN_PATH), load(VAL_PATH)
    train = [encode(tokenizer, r) for r in train_rows]
    val = [encode(tokenizer, r) for r in val_rows]
    print(f"학습 {len(train)}개 / 검증 {len(val)}개 | 최대 길이 {max(len(i) for i, _ in train)}")

    steps_per_epoch = math.ceil(len(train) / BATCH / ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / (0.05 * total_steps), max(0.0, (total_steps - s) / (0.95 * total_steps))))
    scaler = torch.amp.GradScaler("cuda")

    print(f"[학습 전] 검증 loss {evaluate(model, val, pad_id):.3f}  (총 업데이트 {total_steps}번)")
    best = float("inf")

    step, t0 = 0, time.time()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train)
        for i in range(0, len(train), BATCH):
            ids, labels, mask = make_batch(train[i:i + BATCH], pad_id)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / ACCUM
            scaler.scale(loss).backward()
            if (i // BATCH + 1) % ACCUM == 0 or i + BATCH >= len(train):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                if step % 10 == 0:                         # 이번엔 업데이트가 느려서 10번마다 출력
                    el = time.time() - t0
                    print(f"  epoch {epoch} | {step}/{total_steps} | loss {loss.item() * ACCUM:.3f} | "
                          f"경과 {el / 60:.1f}분, 남은 예상 {el / step * (total_steps - step) / 60:.1f}분")

        vl = evaluate(model, val, pad_id)
        mark = ""
        if vl < best:                                      # 최고 기록이면 저장
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

    model = GPT2LMHeadModel.from_pretrained(OUT_DIR).to(device)   # 최고 모델로 맛보기
    print("\n--- 6단계와 같은 검증 질문 3개 (학습 때 안 본 질문, 문단은 검색 결과 그대로) ---")
    for r in val_rows[:3]:
        print(f"Q: {r['question']}\n모델: {answer(model, tokenizer, r['ctx'], r['question'])}\n정답: {r['answer']}\n")


if __name__ == "__main__":
    main()
