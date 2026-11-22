# =============================================================
# 전처리 2-2단계: 줄 단위로 잘린 텍스트 → 문단 단위로 이어붙이기
# 실행: python scripts/step2_2_make_paragraphs.py   (C:\STF 에서 실행)
# 입력: data/processed/step2_1_lines.jsonl   (2-1 결과)
# 결과: data/processed/step2_2_paras.jsonl   (페이지마다 제목, 대제목, 문단 목록)
#       data/processed/step2_2_view.txt      (눈으로 확인용)
# 하는 일: 1) 어디까지가 한 문단인지 판단해서 나누고
#          2) 같은 문단 안의 줄들을 이어붙이면서 띄어쓰기를 복원함
# =============================================================

import json                                   # jsonl 읽기/쓰기용
import re                                     # 정규식 (번호·글머리표로 시작하는지 확인)
from pathlib import Path                      # 경로 처리

ROOT = Path(__file__).resolve().parent.parent           # 프로젝트 최상위 폴더 (C:\STF)
IN_PATH = ROOT / "data" / "processed" / "step2_1_lines.jsonl"   # 2-1 결과
OUT_JSONL = ROOT / "data" / "processed" / "step2_2_paras.jsonl" # 이번 결과
OUT_VIEW = ROOT / "data" / "processed" / "step2_2_view.txt"     # 확인용

# ---- 기준값 ----
BODY_SIZE = 9          # 본문 글자 크기
PARA_GAP = 13.5        # 윗줄과 y좌표 차이가 이보다 크면 새 문단 (같은 문단 안 줄 간격은 약 10.8, 문단 사이는 15 이상)
NEW_PARA_START = r"^(\d+\.|•|◦|참고:|경고:|주의:)"   # 이걸로 시작하는 줄은 무조건 새 문단 (번호, 글머리표, 참고/경고/주의)


def is_heading(line):                         # 소제목 줄인지 판단하는 함수
    return line["bold"] and line["size"] > BODY_SIZE + 0.5   # 굵은 글씨 + 본문(9)보다 큼 → 소제목 (예: "진단" 크기 14)


def join_lines(prev, nxt):                    # 같은 문단 안에서 윗줄(prev)과 아랫줄(nxt)을 이어붙이는 함수
    if prev.endswith(" "):                    # 윗줄 끝에 공백이 있음 → 원래 띄어쓰기 자리에서 줄이 바뀐 것
        return prev + nxt                     # 공백 그대로 두고 이어붙임 → "소프트웨어 " + "및" = "소프트웨어 및"
    if prev.endswith((".", ":", "!", "?")):   # 공백은 없지만 문장이 끝난 경우
        return prev + " " + nxt               # 띄워서 이어붙임 → "…81." + "이 표시등" = "…81. 이 표시등"
    return prev + nxt                         # 그 외 = 단어 중간에서 끊긴 것 → 붙임 → "네" + "트워크" = "네트워크"


def make_paragraphs(lines):                   # 한 페이지의 줄 목록 → 문단 목록으로 바꾸는 함수
    paras = []                                # 완성된 문단들 ({"text": 문단 내용, "head": 소제목 여부})
    cur = None                                # 지금 만들고 있는 문단
    prev = None                               # 바로 윗줄 정보
    for line in lines:                        # 줄 하나씩 (2-1에서 이미 읽는 순서로 정렬돼 있음)
        text = line["text"]                   # 줄 텍스트 (끝 공백 살아 있음)
        head = is_heading(line)               # 이 줄이 소제목인지
        new_para = (                          # 새 문단을 시작해야 하는 조건들 (하나라도 맞으면 새 문단)
            prev is None                                  # 페이지 첫 줄
            or head                                       # 이 줄이 소제목
            or is_heading(prev)                           # 윗줄이 소제목 (소제목 다음 줄은 본문 시작)
            or line["col"] != prev["col"]                 # 왼쪽 단 → 오른쪽 단으로 넘어감
            or line["y"] - prev["y"] > PARA_GAP           # 윗줄과 간격이 벌어짐
            or re.match(NEW_PARA_START, text.strip())     # 번호·글머리표·참고/경고/주의로 시작
        )
        if new_para:                          # 새 문단이면
            if cur:                           # 만들던 문단이 있으면
                paras.append(cur)             # 완성된 문단으로 저장
            cur = {"text": text, "head": head}   # 새 문단 시작
        else:                                 # 같은 문단이면
            cur["text"] = join_lines(cur["text"], text)   # 윗줄과 이어붙임
        prev = line                           # 지금 줄을 "윗줄"로 기억
    if cur:                                   # 마지막 문단도
        paras.append(cur)                     # 저장
    for p in paras:                           # 문단마다
        p["text"] = p["text"].strip()         # 앞뒤 공백 정리 (이제 끝 공백은 필요 없음)
    return paras


def main():
    with open(IN_PATH, encoding="utf-8") as f:            # 2-1 결과 읽기
        pages = [json.loads(line) for line in f]          # 한 줄 = 한 페이지

    results = []                                          # 결과 모음
    for p in pages:                                       # 페이지 하나씩
        paras = make_paragraphs(p["lines"])               # 줄 → 문단
        results.append({"page": p["page"], "title": p["title"], "chapter": p["chapter"], "paras": paras})

    with open(OUT_JSONL, "w", encoding="utf-8") as f:     # jsonl 저장
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUT_VIEW, "w", encoding="utf-8") as f:      # 확인용 저장
        for r in results:
            f.write(f"\n{'=' * 15} p.{r['page']} | 제목: {r['title'] or '(없음)'} | 대제목: {r['chapter'] or '(없음)'} {'=' * 15}\n")
            for para in r["paras"]:
                prefix = "## " if para["head"] else ""   # 소제목은 앞에 "## " 붙여서 구분되게
                f.write(f"{prefix}{para['text']}\n\n")   # 문단 사이 빈 줄

    n_lines = sum(len(p["lines"]) for p in pages)         # 입력 줄 수
    n_paras = sum(len(r["paras"]) for r in results)       # 출력 문단 수
    n_heads = sum(1 for r in results for para in r["paras"] if para["head"])   # 소제목 수
    print(f"줄 {n_lines:,}개 → 문단 {n_paras:,}개 (그중 소제목 {n_heads}개)")
    print(f"→ {OUT_JSONL.name}, {OUT_VIEW.name} 저장 완료")


if __name__ == "__main__":
    main()
