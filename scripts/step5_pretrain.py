# =============================================================
# 5단계: koGPT2 추가 사전학습 (매뉴얼 본문을 그대로 읽혀서 테슬라 용어·문체에 익숙해지게)
# 실행: python scripts/step5_pretrain.py   (C:\STF, 가상환경 켠 상태에서)
# 입력: data/processed/clean_corpus.txt
# 결과: models/kogpt2-tesla-pretrain/    (추가 사전학습된 모델 — 6단계 SFT의 출발점)
# 노트북(MX450, 2GB) 메모리 절약: fp16 혼합 정밀도 + 8비트 Adam + 그래디언트 체크포인팅 + 작은 배치·누적
# =============================================================

import math                                    # perplexity(혼란도) 계산용 exp
import random                                  # 데이터 섞기
import time                                    # 소요 시간 측정
from pathlib import Path                       # 경로 처리
import torch                                   # 파이토치
import bitsandbytes as bnb                     # 8비트 Adam 옵티마이저 (메모리 절약)
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast   # koGPT2 모델과 토크나이저

ROOT = Path(__file__).resolve().parent.parent              # 프로젝트 최상위 폴더 (C:\STF)
CORPUS = ROOT / "data" / "processed" / "clean_corpus.txt"   # 매뉴얼 본문
OUT_DIR = ROOT / "models" / "kogpt2-tesla-pretrain"         # 결과 모델 저장 폴더

BASE_MODEL = "skt/kogpt2-base-v2"   # 허깅페이스의 koGPT2 (처음 실행 때 약 500MB 자동 다운로드)
BLOCK = 256           # 한 번에 읽히는 토큰 길이 (길수록 메모리 많이 씀)
BATCH = 4             # 한 번에 GPU에 올리는 덩어리 수 (2GB라 작게)
ACCUM = 4             # 4번 모아서 한 번 업데이트 → 실제 배치 효과 = 4 x 4 = 16
EPOCHS = 3            # 전체 본문을 몇 번 반복해서 읽힐지
LR = 5e-5             # 학습률 (이미 학습된 모델이라 작게)
VAL_RATIO = 0.05      # 검증용으로 떼어둘 비율 (5%)
SEED = 42             # 결과 재현용

random.seed(SEED)                                          # 파이썬 랜덤 고정
torch.manual_seed(SEED)                                    # 파이토치 랜덤 고정
device = "cuda"                                            # GPU 사용


def load_blocks(tokenizer):                                # 본문 → BLOCK 길이 토큰 덩어리 목록
    text = CORPUS.read_text(encoding="utf-8")              # 본문 전체 읽기
    sections = text.split("\n\n")                          # 섹션 단위로 나눔 (2-4에서 섹션 사이를 빈 줄로 구분함)
    ids = []                                               # 전체 토큰을 한 줄로 이어붙일 리스트
    for sec in sections:
        ids += tokenizer.encode(sec) + [tokenizer.eos_token_id]   # 섹션 토큰 + 끝 표시(</s>)
    print(f"전체 토큰 수: {len(ids):,}")
    n = len(ids) // BLOCK                                  # 만들 수 있는 덩어리 개수 (남는 꼬리는 버림)
    return [ids[i * BLOCK:(i + 1) * BLOCK] for i in range(n)]   # BLOCK 길이로 자름


@torch.no_grad()                                           # 평가할 땐 기울기 계산 안 함 (메모리 절약)
def evaluate(model, blocks):                               # 검증 데이터로 손실(loss) 계산
    model.eval()                                           # 평가 모드
    losses = []
    for i in range(0, len(blocks), BATCH):
        x = torch.tensor(blocks[i:i + BATCH], device=device)   # 덩어리 묶음 → 텐서
        with torch.autocast("cuda", dtype=torch.float16):  # fp16으로 계산
            losses.append(model(input_ids=x, labels=x).loss.item())   # labels=입력 그대로 → "다음 토큰 맞히기" 손실
    model.train()                                          # 다시 학습 모드
    return sum(losses) / len(losses)                       # 평균 손실


def main():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(   # koGPT2 토크나이저 (모델 카드에 나온 특수 토큰 설정)
        BASE_MODEL, bos_token="</s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device)   # koGPT2 모델을 GPU로
    model.gradient_checkpointing_enable()                  # 그래디언트 체크포인팅: 중간값을 덜 저장해서 메모리 절약 (대신 조금 느려짐)
    model.config.use_cache = False                         # 체크포인팅과 같이 쓸 땐 캐시 꺼야 함

    blocks = load_blocks(tokenizer)                        # 학습 데이터 준비
    random.shuffle(blocks)                                 # 섞기
    n_val = max(1, int(len(blocks) * VAL_RATIO))           # 검증용 개수
    val, train = blocks[:n_val], blocks[n_val:]            # 나누기
    steps_per_epoch = math.ceil(len(train) / BATCH / ACCUM)   # 에폭당 업데이트 횟수
    total_steps = steps_per_epoch * EPOCHS                 # 전체 업데이트 횟수
    print(f"덩어리: 학습 {len(train)}개 / 검증 {len(val)}개, 총 업데이트 {total_steps}번")

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)   # 8비트 Adam (옵티마이저 메모리 약 1/4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(         # 학습률 스케줄: 처음 5%는 서서히 올리고(워밍업) 이후 선형으로 줄임
        optimizer, lambda s: min((s + 1) / (0.05 * total_steps), max(0.0, (total_steps - s) / (0.95 * total_steps))))
    scaler = torch.amp.GradScaler("cuda")                  # fp16 학습 시 기울기가 너무 작아져 0이 되는 걸 막아줌

    loss0 = evaluate(model, val)                           # 학습 전 검증 손실 (비교 기준)
    print(f"[학습 전] 검증 loss {loss0:.3f} / perplexity {math.exp(loss0):.1f}")

    step, t0 = 0, time.time()                              # 업데이트 횟수, 시작 시각
    model.train()                                          # 학습 모드
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train)                              # 에폭마다 순서 섞기
        for i in range(0, len(train), BATCH):
            x = torch.tensor(train[i:i + BATCH], device=device)   # 덩어리 묶음
            with torch.autocast("cuda", dtype=torch.float16):     # fp16 계산
                loss = model(input_ids=x, labels=x).loss / ACCUM  # 누적할 거라 ACCUM으로 나눔
            scaler.scale(loss).backward()                  # 기울기 계산 (누적됨)
            if (i // BATCH + 1) % ACCUM == 0 or i + BATCH >= len(train):   # ACCUM번 모였거나 에폭 마지막이면
                scaler.unscale_(optimizer)                 # 기울기 원래 크기로
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 기울기 폭주 방지
                scaler.step(optimizer)                     # 가중치 업데이트
                scaler.update()                            # 스케일 조정
                optimizer.zero_grad(set_to_none=True)      # 기울기 초기화
                scheduler.step()                           # 학습률 갱신
                step += 1
                if step % 20 == 0:                         # 20번마다 진행 상황 출력
                    el = time.time() - t0
                    print(f"  epoch {epoch} | {step}/{total_steps} | loss {loss.item() * ACCUM:.3f} | "
                          f"경과 {el / 60:.1f}분, 남은 예상 {el / step * (total_steps - step) / 60:.1f}분")
        vl = evaluate(model, val)                          # 에폭 끝날 때마다 검증
        print(f"[epoch {epoch}] 검증 loss {vl:.3f} / perplexity {math.exp(vl):.1f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)             # 저장 폴더 생성
    model.config.use_cache = True                          # 생성할 땐 캐시 다시 켬
    model.save_pretrained(OUT_DIR)                         # 모델 저장
    tokenizer.save_pretrained(OUT_DIR)                     # 토크나이저도 같이 저장 (6단계에서 그대로 불러씀)
    print(f"\n저장 완료 → {OUT_DIR}  (총 {(time.time() - t0) / 60:.1f}분)")
    print(f"GPU 최대 메모리 사용: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    model.eval()                                           # 맛보기 생성: 매뉴얼 문체로 이어 쓰는지 확인
    prompt = "[주행 > 오토스티어]\n오토스티어는"
    x = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = model.generate(x, max_new_tokens=60, do_sample=True, top_p=0.9, temperature=0.8,
                         repetition_penalty=1.2, pad_token_id=tokenizer.pad_token_id)
    print("\n--- 맛보기 생성 ---\n" + tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
