# =============================================================
# 1단계: 테슬라 모델 S 매뉴얼 PDF → 텍스트 추출 + 기본 정리
# 실행: python scripts/step1_extract.py   (C:\STF 에서 실행)
# 입력: data/raw/Owners_Manual.pdf
# 결과: data/processed/manual_pages.jsonl (페이지별), data/processed/manual_corpus.txt (전체 본문)
# =============================================================

import json                      # 결과를 jsonl(한 줄에 json 하나) 형식으로 저장하려고 사용
import re                        # 정규식: 불필요한 줄(목차 점선, 페이지 번호 등) 찾을 때 사용
from pathlib import Path         # 경로를 운영체제 상관없이 다루기 위한 모듈
import pymupdf as fitz           # PyMuPDF 라이브러리 (pip install pymupdf) — 새 이름 pymupdf로 불러오고 기존 코드 호환 위해 fitz라는 별명 붙임

ROOT = Path(__file__).resolve().parent.parent      # 프로젝트 최상위 폴더 (C:\STF) — scripts 폴더의 상위
RAW_DIR = ROOT / "data" / "raw"                    # 원본 데이터 폴더
OUT_DIR = ROOT / "data" / "processed"              # 가공된 데이터 저장 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)         # 저장 폴더가 없으면 자동으로 만듦

PDF_PATH = RAW_DIR / "Owners_Manual.pdf"           # 입력 PDF 파일
OUT_JSONL = OUT_DIR / "manual_pages.jsonl"         # 페이지별 결과 저장 파일
OUT_TXT = OUT_DIR / "manual_corpus.txt"            # 전체 본문을 이어붙인 텍스트 파일 (추가 사전학습용)

FOOTER = "MODEL S 사용자 매뉴얼"          # 매 페이지 하단에 반복되는 꼬리말 문구


def clean_page(text):                    # 한 페이지 텍스트를 받아서 정리된 텍스트를 돌려주는 함수
    lines = text.split("\n")             # 줄 단위로 쪼갬
    kept = []                            # 살릴 줄을 모아둘 리스트
    for line in lines:                   # 한 줄씩 검사
        s = line.strip()                 # 앞뒤 공백 제거
        if not s:                        # 빈 줄이면
            kept.append("")              # 문단 구분용으로 빈 줄은 하나 남겨둠
            continue                     # 다음 줄로
        if s == FOOTER:                  # 꼬리말 "MODEL S 사용자 매뉴얼" 이면
            continue                     # 버림
        if re.fullmatch(r"\d{1,3}", s):  # 숫자만 있는 줄(= 페이지 번호)이면
            continue                     # 버림
        if re.search(r"\.{5,}", s):      # 점이 5개 이상 이어지는 줄(= 목차 "개요......6")이면
            continue                     # 버림
        kept.append(s)                   # 위 조건에 안 걸리면 살림
    text = "\n".join(kept)               # 다시 줄바꿈으로 합침
    text = re.sub(r"\n{3,}", "\n\n", text)  # 빈 줄이 여러 개 연속되면 하나로 줄임
    return text.strip()                  # 앞뒤 공백 정리해서 반환


def main():                              # 메인 실행 함수
    doc = fitz.open(PDF_PATH)            # PDF 열기
    print(f"총 페이지 수: {len(doc)}")    # 페이지 수 출력 (274 나와야 정상)

    pages = []                           # 페이지별 결과 저장 리스트
    for i, page in enumerate(doc):       # 페이지를 하나씩 순회 (i는 0부터 시작)
        raw = page.get_text("text")      # 해당 페이지의 텍스트 추출
        cleaned = clean_page(raw)        # 위에서 만든 정리 함수 적용
        if len(cleaned) < 30:            # 정리 후 30자 미만이면 (표지, 빈 페이지, 그림만 있는 페이지)
            continue                     # 건너뜀
        pages.append({"page": i + 1, "text": cleaned})  # 페이지 번호(1부터)와 텍스트를 저장

    with open(OUT_JSONL, "w", encoding="utf-8") as f:   # 페이지별 jsonl 파일 쓰기
        for p in pages:                                  # 페이지 하나씩
            f.write(json.dumps(p, ensure_ascii=False) + "\n")  # 한글 깨지지 않게 ensure_ascii=False

    with open(OUT_TXT, "w", encoding="utf-8") as f:     # 전체 본문 txt 파일 쓰기
        f.write("\n\n".join(p["text"] for p in pages))  # 페이지 사이는 빈 줄로 구분

    total_chars = sum(len(p["text"]) for p in pages)    # 전체 글자 수 계산
    print(f"저장된 페이지: {len(pages)}개")               # 실제로 저장된 페이지 수
    print(f"전체 글자 수: {total_chars:,}자")             # 데이터 양 확인용
    print(f"→ {OUT_JSONL}, {OUT_TXT} 저장 완료")          # 완료 메시지
    print("\n--- 샘플 (20페이지 근처) ---")              # 결과가 잘 나왔는지 눈으로 확인용
    sample = next((p for p in pages if p["page"] >= 20), pages[0])  # 20페이지 이상 첫 페이지 고르기
    print(sample["text"][:500])                          # 앞 500자만 출력


if __name__ == "__main__":               # 이 파일을 직접 실행했을 때만
    main()                               # main() 실행
