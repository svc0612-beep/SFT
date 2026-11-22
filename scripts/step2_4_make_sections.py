# =============================================================
# 전처리 2-4단계: 같은 제목 페이지끼리 섹션으로 묶고 최종 파일 만들기
# 실행: python scripts/step2_4_make_sections.py   (C:\STF 에서 실행)
# 입력: data/processed/step2_3_paras.jsonl   (2-3 결과)
# 결과: data/processed/clean_sections.jsonl  (섹션 단위 — step3 OpenAI Q&A 생성 입력)
#       data/processed/clean_corpus.txt      (추가 사전학습용 전체 본문)
#       data/processed/step2_4_view.txt      (눈으로 확인용, 소제목은 "## " 표시)
# 하는 일: 1) 같은 제목의 연속된 페이지를 하나의 섹션으로 묶기
#          2) 페이지 사이에서 끊긴 문장 잇기
#          3) 대제목 채우기 + 원본 오류 바로잡기
#          4) 학습에 안 쓸 페이지 빼기 (표지, 저작권, 목차, 색인, 뒤표지)
# =============================================================

import json                                   # jsonl 읽기/쓰기용
from pathlib import Path                      # 경로 처리

ROOT = Path(__file__).resolve().parent.parent                       # 프로젝트 최상위 폴더 (C:\STF)
IN_PATH = ROOT / "data" / "processed" / "step2_3_paras.jsonl"       # 2-3 결과
OUT_SECTIONS = ROOT / "data" / "processed" / "clean_sections.jsonl" # 최종 섹션 파일
OUT_CORPUS = ROOT / "data" / "processed" / "clean_corpus.txt"       # 최종 사전학습용 본문
OUT_VIEW = ROOT / "data" / "processed" / "step2_4_view.txt"         # 확인용

SKIP_TITLES = {"목차", "색인"}                # 이 제목의 페이지는 빼기 (제목이 아예 없는 표지·저작권·뒤표지도 뺌)
CHAPTER_FIX = {                               # 원본 꼬리말 오류 바로잡기 (PDF 꼬리말에 대제목이 잘못 찍힌 곳)
    "치수": "제원",                           # p.187 꼬리말이 "정비"로 찍혀 있음
    "터치스크린": "개요",                     # p.9 꼬리말이 "이 사용자 매뉴얼 사용하기"로 찍혀 있음 (p.11은 "개요")
    "충전 지침": "충전 및 에너지 소비",       # p.159 꼬리말이 "내비게이션 및 엔터테인먼트"로 찍혀 있음 (p.161은 "충전 및 에너지 소비")
}

# 페이지 사이에서 문장이 끊긴 곳 (step2_3 결과를 직접 확인해서 찾은 곳)
#   키 = 이어지는 쪽 페이지 번호, 값 = 앞 페이지 끝과 이 페이지 시작 사이에 넣을 문자
#   자동으로 판단하면 "목록이 끝난 것"과 "문장이 끊긴 것"을 구분 못 해서 엉뚱하게 붙는 경우가 많아 직접 지정함
PAGE_JOIN = {
    134: " ",    # p.133 "…또는 차량이 실내 온도 조절" + p.134 "시스템의 이상을…"  → 띄어서 이음
    142: "",     # p.141 "…내비게이션 경" + p.142 "로를 따라…"                   → 붙여서 이음 ("경로를")
}


def main():
    with open(IN_PATH, encoding="utf-8") as f:            # 2-3 결과 읽기
        pages = [json.loads(line) for line in f]

    sections = []                                         # 섹션 목록
    n_skip = 0                                            # 뺀 페이지 수
    for p in pages:                                       # 페이지 하나씩
        title, paras = p["title"], list(p["paras"])       # 페이지 제목, 문단 목록(복사본)
        if not title or title in SKIP_TITLES or not paras: # 4) 제목 없는 페이지, 목차·색인, 빈 페이지는
            n_skip += 1
            continue                                      # 건너뜀

        same = sections and sections[-1]["title"] == title and sections[-1]["pages"][-1] == p["page"] - 1
        # ↑ 직전 섹션과 제목이 같고 페이지도 바로 이어지면 같은 섹션

        if same:                                          # 1) 같은 섹션이면 이어붙이기
            sec = sections[-1]
            if p["page"] in PAGE_JOIN:                    # 2) 페이지 사이에서 끊긴 문장이면
                first = paras.pop(0)                      # 이 페이지 첫 문단을 꺼내서
                sec["paras"][-1]["text"] += PAGE_JOIN[p["page"]] + first["text"]   # 앞 페이지 마지막 문단에 이어붙임
            sec["paras"] += paras                         # 나머지 문단 추가
            sec["pages"].append(p["page"])                # 페이지 번호 기록
            if not sec["chapter"]:                        # 3) 섹션 대제목이 아직 비어 있으면
                sec["chapter"] = p["chapter"]             # 이 페이지 대제목으로 채움
        else:                                             # 새 섹션 시작
            sections.append({"chapter": p["chapter"], "title": title, "pages": [p["page"]], "paras": paras})

    for i, sec in enumerate(sections):                    # 3) 대제목 마무리
        if not sec["chapter"] and i > 0:                  # 끝까지 대제목이 없는 섹션은 (짝수 페이지 한 장짜리 섹션 등)
            sec["chapter"] = sections[i - 1]["chapter"]   # 바로 앞 섹션 대제목을 이어받음
        sec["chapter"] = CHAPTER_FIX.get(sec["title"], sec["chapter"])   # 원본 오류 바로잡기

    with open(OUT_SECTIONS, "w", encoding="utf-8") as f:  # 섹션 jsonl 저장
        for sec in sections:
            row = {
                "chapter": sec["chapter"],                # 대제목
                "title": sec["title"],                    # 제목
                "pages": sec["pages"],                    # 포함된 페이지 번호들
                "paras": sec["paras"],                    # 문단 목록 (소제목 여부 포함)
                "text": "\n".join(para["text"] for para in sec["paras"]),   # 문단을 줄바꿈으로 합친 전체 텍스트
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(OUT_CORPUS, "w", encoding="utf-8") as f:    # 사전학습용 본문 저장
        blocks = []
        for sec in sections:                              # 섹션마다 "[대제목 > 제목]" 머리줄 + 본문
            body = "\n".join(para["text"] for para in sec["paras"])
            blocks.append(f"[{sec['chapter']} > {sec['title']}]\n{body}")
        f.write("\n\n".join(blocks))                      # 섹션 사이는 빈 줄 하나

    with open(OUT_VIEW, "w", encoding="utf-8") as f:      # 확인용 저장
        for sec in sections:
            f.write(f"\n{'=' * 15} [{sec['chapter']} > {sec['title']}] p.{sec['pages']} {'=' * 15}\n")
            for para in sec["paras"]:
                prefix = "## " if para["head"] else ""   # 소제목 표시
                f.write(f"{prefix}{para['text']}\n\n")

    total = sum(len(para["text"]) for sec in sections for para in sec["paras"])   # 전체 글자 수
    longest = max(sections, key=lambda s: sum(len(x["text"]) for x in s["paras"]))  # 가장 긴 섹션
    print(f"페이지 {len(pages)}개 중 {n_skip}개 제외 → 섹션 {len(sections)}개, 전체 {total:,}자")
    print(f"가장 긴 섹션: [{longest['chapter']} > {longest['title']}] {sum(len(x['text']) for x in longest['paras']):,}자")
    print(f"→ {OUT_SECTIONS.name}, {OUT_CORPUS.name}, {OUT_VIEW.name} 저장 완료")


if __name__ == "__main__":
    main()
