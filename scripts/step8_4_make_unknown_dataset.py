# =============================================================
# 8-4단계: "모르면 모른다고 답하기" 학습 데이터 만들기
# 실행: python scripts\step8_4_make_unknown_dataset.py
# 입력: data/processed/sft_ctx_train.jsonl   (7-2 결과: 정답 문단이 들어 있는 학습 데이터 867개)
#       data/processed/sft_train.jsonl       (4단계 결과: 질문·답변·섹션 제목)
#       data/processed/passages.jsonl        (7-1 결과: 검색용 문단)
# 결과: data/processed/sft_ctx_unk_train.jsonl  (기존 867개 + "모름" 예시 약 130개)
# 왜?  지금 모델은 학습 때 항상 정답 문단을 받았음 → "문단에 답이 있다"고 믿고 무조건 답을 지어냄
#      → 검색이 엉뚱한 문단을 가져오면 그럴듯한 오답 (예: CP_a101 경고 → "차량 소유자에게 문의")
#      그래서 정답이 없는 문단을 주고 "매뉴얼에서 찾지 못했습니다."라고 답하는 예시를 섞어 줌
# 검증 데이터(sft_ctx_val.jsonl)는 그대로 둠 → 이전 학습과 검증 loss를 같은 기준으로 비교
# =============================================================

import json                                    # jsonl 읽기/쓰기
import random                                  # 샘플 뽑기·섞기
import sys                                     # step7_1을 import하려고 경로 추가
from collections import Counter                # 글자 겹침 계산
from pathlib import Path                       # 경로 처리

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts 폴더를 import 경로에 추가
from step7_1_build_passages import BM25, load_passages, passage_text, gold_passage, tokenize   # 7-1 검색기 재사용

ROOT = Path(__file__).resolve().parent.parent                # SFT 폴더
PROC = ROOT / "data" / "processed"                           # 데이터 폴더
OUT_PATH = PROC / "sft_ctx_unk_train.jsonl"                  # 결과 파일

UNK_ANSWER = "매뉴얼에서 찾지 못했습니다."                   # 8-1·8-3 지시문에 적힌 문구와 정확히 같게
UNK_RATIO = 0.15       # 기존 학습 데이터의 15%만큼 "모름" 예시 추가
TOP_K = 2              # 문단 2개 (실제 사용 때와 같음)
OVERLAP_MAX = 0.2      # 정답 글자의 20% 이상이 들어 있는 문단은 "답이 있을 수도 있는 문단"이라 제외
SEED = 42
random.seed(SEED)


def overlap(answer, text):                                   # 정답 글자 중 문단에 들어 있는 비율 (0~1)
    a = Counter(tokenize(answer))                            # 정답의 단어 조각
    if not a:
        return 0.0
    return sum((a & Counter(tokenize(text))).values()) / sum(a.values())


def main():
    passages = load_passages()                               # 문단 1,820개
    bm25 = BM25([passage_text(p) for p in passages])         # 검색기 (7-1과 같은 설정)

    load = lambda name: [json.loads(l) for l in open(PROC / name, encoding="utf-8")]
    ctx_train = load("sft_ctx_train.jsonl")                  # 기존 학습 데이터 (정답 문단 포함)
    qa_train = load("sft_train.jsonl")                       # 섹션 제목(title)이 필요해서 같이 읽음
    title_of = {r["question"]: r["title"] for r in qa_train} # 질문 → 섹션 제목

    n_unk = int(len(ctx_train) * UNK_RATIO)                  # 만들 "모름" 예시 수 (867 × 0.15 ≈ 130)
    picked = random.sample(ctx_train, n_unk)                 # 무작위로 질문 고르기

    unk_rows, skipped = [], 0
    for r in picked:
        gold = gold_passage(passages, title_of[r["question"]], r["answer"])   # 정답 문단 번호
        cands = bm25.search(r["question"], k=60)             # 질문과 단어가 비슷한 문단 60개 (일부러 "헷갈리는" 문단을 씀)
        chapter = passages[gold]["chapter"]                  # 정답이 있는 장(chapter) 이름
        neg = [i for i in cands                              # 아래 3가지를 모두 만족하는 문단만 사용
               if i != gold                                  # 1) 정답 문단이 아니고
               and passages[i]["chapter"] != chapter         # 2) 아예 다른 장이고 (같은 장엔 비슷한 답이 흩어져 있을 수 있음)
               and overlap(r["answer"], passages[i]["text"]) < OVERLAP_MAX][:TOP_K]   # 3) 정답과 많이 겹치지 않음
        if len(neg) < TOP_K:                                 # 조건에 맞는 문단이 모자라면 건너뜀
            skipped += 1
            continue
        ctx = "\n".join(passage_text(passages[i]) for i in neg)   # 답이 없는 문단 2개
        unk_rows.append({"ctx": ctx, "question": r["question"], "answer": UNK_ANSWER, "unk": True})

    rows = ctx_train + unk_rows                              # 기존 + 모름 예시
    random.shuffle(rows)                                     # 섞기 (모름 예시가 뒤에 몰리지 않게)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"기존 {len(ctx_train)}개 + 모름 {len(unk_rows)}개 (건너뜀 {skipped}개) = {len(rows)}개")
    print(f"→ {OUT_PATH.name} 저장 완료")
    for ex in unk_rows[:3]:                                  # 예시 3개 출력 (눈으로 확인용)
        print("\n--- 모름 예시 ---")
        print("질문:", ex["question"])
        print("문단:", " / ".join(l[:70] for l in ex["ctx"].split("\n") if l.startswith("[")))   # 문단 제목줄만
        print("답변:", ex["answer"])


if __name__ == "__main__":
    main()
