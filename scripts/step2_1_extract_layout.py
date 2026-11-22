# =============================================================
# 전처리 2-1단계: 위치 정보까지 포함해서 PDF 재추출 + 제목/대제목/본문 분리
# 실행: python scripts/step2_1_extract_layout.py   (C:\STF 에서 실행)
# 입력: data/raw/Owners_Manual.pdf
# 결과: data/processed/step2_1_lines.jsonl  (페이지마다 제목, 대제목, 본문 줄 목록 + 각 줄의 위치·크기)
#       data/processed/step2_1_view.txt     (눈으로 확인용)
# 왜 다시 뽑나? step1은 글자만 뽑아서 "제목이 맨 뒤로 밀림", "단어 중간 줄바꿈" 같은 문제를 못 고침.
#              글자의 위치(x,y)·크기·굵기를 같이 뽑아두면 다음 단계에서 정확히 고칠 수 있음.
# =============================================================

import json                                   # jsonl 저장용
from pathlib import Path                      # 경로 처리
import pymupdf                                # PDF 읽기 (step1에서 이미 설치함)

ROOT = Path(__file__).resolve().parent.parent           # 프로젝트 최상위 폴더 (C:\STF)
PDF_PATH = ROOT / "data" / "raw" / "Owners_Manual.pdf"  # 입력 PDF
OUT_DIR = ROOT / "data" / "processed"                   # 출력 폴더
OUT_JSONL = OUT_DIR / "step2_1_lines.jsonl"             # 결과 파일
OUT_VIEW = OUT_DIR / "step2_1_view.txt"                 # 확인용 파일

# ---- 매뉴얼 레이아웃 기준값 (PDF 한 페이지 크기는 가로 612 x 세로 792) ----
HEADER_Y = 55          # 이 y좌표보다 위 = 머리글 영역 → 여기 있는 큰 글씨가 페이지 제목 (예: "안전벨트")
TITLE_SIZE = 17        # 페이지 제목 글자 크기는 18 → 17 이상이면 제목으로 봄
FOOTER_Y = 755         # 이 y좌표보다 아래 = 꼬리말 영역 → 페이지 번호, 대제목(예: "좌석 및 안전 제어 장치"), 고정 문구
FOOTER_FIXED = "MODEL S 사용자 매뉴얼"   # 꼬리말에 매번 찍히는 고정 문구 (버릴 것)
COLUMN_X = 300         # 본문이 두 단일 때 왼쪽/오른쪽 단을 가르는 x좌표


def extract_page(page):                       # 페이지 하나에서 제목, 대제목, 본문 줄 목록을 뽑는 함수
    title, chapter, lines = "", "", []        # 페이지 제목, 대제목, 본문 줄
    data = page.get_text("dict")              # "dict" 모드: 블록 → 줄 → 글자조각(span) 구조로 위치·크기·폰트까지 줌
    for block in data["blocks"]:              # 블록 하나씩
        if block["type"] != 0:                # type 0 = 텍스트, 1 = 이미지 → 이미지는 건너뜀
            continue
        for line in block["lines"]:           # 블록 안의 줄 하나씩
            text = "".join(span["text"] for span in line["spans"])   # 줄을 이루는 글자조각들을 합쳐서 한 줄 텍스트로
            if not text.strip():              # 공백뿐인 줄은 건너뜀
                continue
            x0, y0 = line["bbox"][0], line["bbox"][1]   # 줄의 왼쪽 x, 위쪽 y 좌표 (bbox = 줄을 감싸는 사각형)
            span0 = line["spans"][0]                     # 줄의 첫 글자조각 (글자 크기·폰트 판단용)
            size = round(span0["size"], 1)               # 글자 크기 (본문 9, 소제목 14 등)
            bold = "Bold" in span0["font"]               # 폰트 이름에 Bold가 있으면 굵은 글씨

            if y0 < HEADER_Y and size >= TITLE_SIZE:     # 머리글 영역의 큰 글씨 → 페이지 제목
                title = text.strip()
                continue                                 # 본문 줄 목록엔 넣지 않음
            if y0 > FOOTER_Y:                            # 꼬리말 영역
                s = text.strip()
                if not s.isdigit() and s != FOOTER_FIXED:   # 페이지 번호도 아니고 고정 문구도 아니면 → 대제목
                    chapter = s
                continue                                 # 꼬리말은 본문에 넣지 않음

            lines.append({                               # 본문 줄 저장
                "col": 0 if x0 < COLUMN_X else 1,        # 왼쪽 단(0) / 오른쪽 단(1)
                "x": round(x0, 1),                       # 왼쪽 x좌표
                "y": round(y0, 1),                       # 위쪽 y좌표 (다음 단계에서 줄 간격으로 문단 나눌 때 씀)
                "size": size,                            # 글자 크기
                "bold": bold,                            # 굵은 글씨 여부
                "text": text,                            # 텍스트 — 끝 공백을 일부러 안 지움! (띄어쓰기 판단 단서)
            })
    lines.sort(key=lambda l: (l["col"], l["y"]))         # 읽는 순서로 정렬: 왼쪽 단 위→아래, 그다음 오른쪽 단 위→아래
    return title, chapter, lines


def main():
    doc = pymupdf.open(PDF_PATH)              # PDF 열기
    print(f"총 페이지 수: {len(doc)}")

    pages = []                                # 결과 모음
    for i, page in enumerate(doc):            # 페이지 하나씩
        title, chapter, lines = extract_page(page)
        pages.append({"page": i + 1, "title": title, "chapter": chapter, "lines": lines})

    with open(OUT_JSONL, "w", encoding="utf-8") as f:     # jsonl 저장 (한 줄 = 한 페이지)
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(OUT_VIEW, "w", encoding="utf-8") as f:      # 확인용 파일 저장
        for p in pages:
            f.write(f"\n{'=' * 15} p.{p['page']} | 제목: {p['title'] or '(없음)'} | 대제목: {p['chapter'] or '(없음)'} {'=' * 15}\n")
            for l in p["lines"]:
                mark = f"[단{l['col']} y{l['y']:5.1f} 크기{l['size']}{' 굵게' if l['bold'] else ''}]"   # 줄 정보 표시
                end = "⏎공백" if l["text"].endswith(" ") else ""   # 줄 끝에 공백이 있으면 표시 (2-2에서 중요)
                f.write(f"{mark} {l['text'].rstrip()} {end}\n")

    n_title = sum(1 for p in pages if p["title"])          # 제목을 찾은 페이지 수
    n_chap = sum(1 for p in pages if p["chapter"])         # 대제목을 찾은 페이지 수
    n_lines = sum(len(p["lines"]) for p in pages)          # 본문 줄 총 개수
    print(f"제목 찾은 페이지: {n_title}개 / 대제목 찾은 페이지: {n_chap}개 / 본문 줄: {n_lines:,}개")
    print(f"→ {OUT_JSONL.name}, {OUT_VIEW.name} 저장 완료")


if __name__ == "__main__":
    main()
