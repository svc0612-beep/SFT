# =============================================================
# 7-1단계: 검색용 문단(passage) 만들기 + 검색기(BM25) + 검색 정확도 확인
# 실행: python scripts/step7_1_build_passages.py   (C:\STF, 가상환경 켠 상태에서)
# 입력: data/processed/clean_sections.jsonl   (2-4 결과)
#       data/processed/sft_val.jsonl          (검색이 잘 되는지 확인용)
# 결과: data/processed/passages.jsonl         (검색 대상 문단 목록)
# 왜 필요? koGPT2가 외워서 답하는 대신, 질문과 관련된 매뉴얼 문단을 "찾아서 같이 넣어주고" 읽고 답하게 하려고.
#         그러려면 (1) 매뉴얼을 짧은 문단으로 잘라두고 (2) 질문으로 문단을 찾는 검색기가 있어야 함.
# 이 파일의 BM25 검색기는 7-2(데이터 만들기), 7-4(채팅 화면)에서도 import 해서 같이 씀
# =============================================================

import json                                    # jsonl 읽기/쓰기
import math                                    # BM25 점수 계산용 log
import re                                      # 단어 뽑기용 정규식
from collections import Counter                # 단어 개수 세기
from pathlib import Path                       # 경로 처리

ROOT = Path(__file__).resolve().parent.parent                       # 프로젝트 최상위 폴더 (C:\STF)
SECTIONS_PATH = ROOT / "data" / "processed" / "clean_sections.jsonl" # 전처리 최종 섹션
VAL_PATH = ROOT / "data" / "processed" / "sft_val.jsonl"            # 검증 Q&A
PASSAGES_PATH = ROOT / "data" / "processed" / "passages.jsonl"      # 결과: 검색용 문단

PASSAGE_CHARS = 350    # 문단 하나의 목표 길이 (koGPT2 토큰으로 약 150개 → 2개 붙여도 메모리 OK)


# ------------------------------------------------------------
# 1) 검색용 문단 만들기
# ------------------------------------------------------------
def build_passages():
    passages = []                                          # 결과 목록
    with open(SECTIONS_PATH, encoding="utf-8") as f:
        sections = [json.loads(line) for line in f]
    for sec in sections:                                   # 섹션마다
        cur, cur_len = [], 0                               # 만드는 중인 문단 묶음, 그 길이
        for para in sec["paras"]:                          # 문단(2-4 결과의 paras) 하나씩
            text = para["text"]
            if cur and cur_len + len(text) > PASSAGE_CHARS and not para["head"]:   # 너무 길어지면 끊기 (소제목 줄은 다음 본문과 붙어 있게 여기서 안 끊음)
                passages.append({"chapter": sec["chapter"], "title": sec["title"], "text": "\n".join(cur)})
                cur, cur_len = [], 0
            if para["head"] and cur:                       # 소제목이 나오면 그 앞에서 끊기 (주제가 바뀌니까)
                passages.append({"chapter": sec["chapter"], "title": sec["title"], "text": "\n".join(cur)})
                cur, cur_len = [], 0
            cur.append(text)
            cur_len += len(text)
        if cur:                                            # 섹션 마지막 묶음
            passages.append({"chapter": sec["chapter"], "title": sec["title"], "text": "\n".join(cur)})
    for i, p in enumerate(passages):                       # 번호 붙이기
        p["pid"] = i
    return passages


def passage_text(p):                                       # 모델에 넣을 때 쓸 형태: "[대제목 > 제목] 본문"
    return f"[{p['chapter']} > {p['title']}] {p['text']}"


def load_passages():                                       # 저장된 문단 불러오기 (다른 단계에서 씀)
    with open(PASSAGES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------------------
# 2) 검색기: BM25 (검색 엔진에서 많이 쓰는 키워드 기반 점수 방식)
# ------------------------------------------------------------
def tokenize(text):
    # 한국어는 형태소 분석기 없이 "글자 2개씩(bigram)" 쪼개면 꽤 잘 맞음
    #   "충전포트" → 충전, 전포, 포트   /   "충전 포트를" → 충전, 포트, 트를
    tokens = []
    for w in re.findall(r"[가-힣]+|[A-Za-z]+|\d+", text.lower()):   # 한글 덩어리 / 영어 단어 / 숫자 덩어리로 나눔
        if re.match(r"[가-힣]", w):                        # 한글이면
            tokens += [w[i:i + 2] for i in range(len(w) - 1)] or [w]   # 2글자씩 (1글자 단어는 그대로)
        else:                                              # 영어·숫자는
            tokens.append(w)                               # 단어 그대로 (예: wi, fi, usb, 5)
    return tokens


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):              # docs = 문서 텍스트 리스트, k1·b = BM25 기본 파라미터
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]    # 문서마다 토큰 등장 횟수
        self.lens = [sum(tf.values()) for tf in self.tfs]  # 문서마다 토큰 수
        self.avg = sum(self.lens) / len(self.lens)         # 평균 문서 길이
        df = Counter(t for tf in self.tfs for t in tf)     # 토큰마다 몇 개 문서에 나오는지
        N = len(docs)
        self.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}   # 드문 토큰일수록 점수 높게

    def search(self, query, k=3):                          # 질문과 가장 관련 있는 문서 k개의 번호 반환
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tfs):                  # 문서마다 점수 계산
            s = 0.0
            for t in q:
                if t in tf:                                # 질문 토큰이 문서에 있으면 점수 추가
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scores.append(s)
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]   # 점수 높은 순 k개


# ------------------------------------------------------------
# 3) 검색 정확도 확인
# ------------------------------------------------------------
def gold_passage(passages, title, answer):
    # "정답 문단" = 같은 섹션(title)의 문단 중 정답 답변과 글자가 가장 많이 겹치는 문단
    ans = Counter(tokenize(answer))
    cands = [p for p in passages if p["title"] == title]
    return max(cands, key=lambda p: sum((ans & Counter(tokenize(p["text"]))).values()))["pid"]


def main():
    passages = build_passages()                            # 문단 만들기
    with open(PASSAGES_PATH, "w", encoding="utf-8") as f:  # 저장
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    lens = [len(p["text"]) for p in passages]
    print(f"문단 {len(passages)}개 (평균 {sum(lens) // len(lens)}자, 최대 {max(lens)}자) → {PASSAGES_PATH.name}")

    bm25 = BM25([passage_text(p) for p in passages])       # 검색기 만들기 (제목까지 포함해서 검색)
    with open(VAL_PATH, encoding="utf-8") as f:
        val = [json.loads(line) for line in f]

    hits = {1: 0, 2: 0, 3: 0}                              # 정답 문단이 상위 1/2/3개 안에 들어온 횟수
    for r in val:                                          # 검증 질문마다
        gold = gold_passage(passages, r["title"], r["answer"])
        top = bm25.search(r["question"], k=3)              # 질문만으로 검색
        for k in hits:
            hits[k] += gold in top[:k]
    for k, v in hits.items():
        print(f"검색 정확도 top-{k}: {v}/{len(val)} = {v / len(val):.0%}")   # 상위 k개 안에 정답 문단이 있는 비율

    print("\n--- 검색 예시 ---")
    for r in val[:2]:
        top = bm25.search(r["question"], k=2)
        print(f"Q: {r['question']}")
        for rank, pid in enumerate(top, 1):
            print(f"  {rank}위: {passage_text(passages[pid])[:90]}...")
        print()


if __name__ == "__main__":
    main()
