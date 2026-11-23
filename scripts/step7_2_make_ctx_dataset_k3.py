# =============================================================
# 7-2단계: 검색 결과(매뉴얼 문단)를 붙인 SFT 데이터 만들기
# 실행: python scripts/step7_2_make_ctx_dataset.py   (C:\STF, 가상환경 켠 상태에서)
# 입력: data/processed/sft_train.jsonl, sft_val.jsonl   (4단계 결과)
#       data/processed/passages.jsonl                   (7-1 결과)
# 결과: data/processed/sft_ctx3_train.jsonl, sft_ctx3_val.jsonl
# [문단3개] step7_2_make_ctx_dataset.py 복사본 — TOP_K 2→3, 저장 파일 이름만 바꿈 (기존 파일 그대로)
# 형식: <ctx>문단1\n문단2<usr>질문<sys>답변</s>
# =============================================================

import json                                    # jsonl 읽기/쓰기
import random                                  # 문단 순서 섞기
from pathlib import Path                       # 경로 처리
from transformers import PreTrainedTokenizerFast   # 토큰 길이 확인용
from step7_1_build_passages import BM25, load_passages, passage_text, gold_passage   # 7-1에서 만든 검색기·함수 재사용

ROOT = Path(__file__).resolve().parent.parent                        # 프로젝트 최상위 폴더 (C:\STF)
PROC = ROOT / "data" / "processed"                                   # 데이터 폴더
TOKENIZER_DIR = ROOT / "models" / "kogpt2-tesla-pretrain"            # 토큰 길이 재는 용도 (5단계 결과에 같이 저장됨)

CTX, USR, SYS, EOS = "<ctx>", "<usr>", "<sys>", "</s>"   # <ctx> = 참고 문단 시작 표시 (새로 추가), 나머지는 4단계와 동일
TOP_K = 3              # [문단3개] 붙일 문단 수 (2→3: 같은 섹션 옆 문단까지 들어와서 정답 포함 문항 57→71개)
SEED = 42
random.seed(SEED)


def build(rows, bm25, passages, fix_gold, tok):         # [문단3개] tok: 길이 확인용 토크나이저
    # rows: Q&A 목록, fix_gold: True면 검색이 정답 문단을 못 찾았을 때 정답 문단을 끼워 넣음 (학습용에만)
    out, n_hit, n_fixed = [], 0, 0
    for r in rows:
        top = bm25.search(r["question"], k=TOP_K)          # 질문으로 문단 검색 (실제 사용할 때와 같은 방식)
        gold = gold_passage(passages, r["title"], r["answer"])   # 답이 들어 있는 문단 (추정)
        if gold in top:                                    # 검색이 정답 문단을 찾았으면
            n_hit += 1
        elif fix_gold:                                     # 못 찾았는데 학습용이면
            top[-1] = gold                                 # 마지막 자리를 정답 문단으로 교체 → 모델이 "읽고 답하기"를 제대로 배우게
            n_fixed += 1
        if fix_gold:                                       # 학습용은 문단 순서를 섞음
            random.shuffle(top)                            # (정답 문단이 항상 같은 자리에 있으면 모델이 "자리"를 외워버림)
            # [문단3개] 문단이 3개라 길면 7-3에서 뒤쪽이 잘림 → 정답 문단이 잘리는 자리면 맨 앞으로 옮김
            #          (잘린 채로 학습하면 "문단에 없는 답을 지어내는 법"을 배우게 됨. 867개 중 158개가 해당)
            room = 512 - len(tok.encode(f"{USR}{r['question']}{SYS}")) - 120   # 7-3과 같은 계산: 문단에 쓸 수 있는 토큰 수
            upto = top[:top.index(gold) + 1] if gold in top else top      # 정답 문단까지의 문단들
            if gold in top and len(tok.encode(CTX + "\n".join(passage_text(passages[i]) for i in upto))) > room:
                top.remove(gold)                           # 정답 문단을 빼서
                top.insert(0, gold)                        # 맨 앞에 넣음
        ctx = "\n".join(passage_text(passages[i]) for i in top)   # 문단들을 줄바꿈으로 이어붙임
        out.append({
            "ctx": ctx,                                    # 참고 문단
            "question": r["question"],                     # 질문
            "answer": r["answer"],                         # 답변
            "text": f"{CTX}{ctx}{USR}{r['question']}{SYS}{r['answer']}{EOS}",   # 학습에 들어가는 한 줄 (확인용)
        })
    return out, n_hit, n_fixed


def save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    passages = load_passages()                             # 7-1에서 만든 문단
    bm25 = BM25([passage_text(p) for p in passages])       # 검색기 (7-1과 똑같이 제목 포함)

    load = lambda name: [json.loads(l) for l in open(PROC / name, encoding="utf-8")]
    train_rows, val_rows = load("sft_train.jsonl"), load("sft_val.jsonl")

    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)   # [문단3개] 길이 확인용 (원래 아래에 있던 것을 위로)
    train, hit_t, fixed_t = build(train_rows, bm25, passages, fix_gold=True, tok=tok)    # 학습용: 정답 문단 보장
    val, hit_v, _ = build(val_rows, bm25, passages, fix_gold=False, tok=tok)             # 검증용: 검색 결과 그대로
    print(f"학습용 {len(train)}개: 검색 적중 {hit_t}개, 정답 문단 끼워넣음 {fixed_t}개")
    print(f"검증용 {len(val)}개: 검색 적중 {hit_v}개 ({hit_v / len(val):.0%}), 그대로 둠")

    save(PROC / "sft_ctx3_train.jsonl", train)          # [문단3개] 새 파일 이름
    save(PROC / "sft_ctx3_val.jsonl", val)
    print("→ sft_ctx3_train.jsonl, sft_ctx3_val.jsonl 저장 완료")

    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)   # 토큰 길이 확인 (7-3 메모리 설정에 필요)
    lens = sorted(len(tok.encode(r["text"])) for r in train)
    print(f"토큰 길이: 평균 {sum(lens) // len(lens)}, 90% 지점 {lens[int(len(lens) * 0.9)]}, 최대 {lens[-1]}")

    print("\n--- 학습 데이터 샘플 ---")
    print(train[0]["text"])


if __name__ == "__main__":
    main()
