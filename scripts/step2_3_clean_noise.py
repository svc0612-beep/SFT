# =============================================================
# 전처리 2-3단계: 문단 안에 남은 잡음 지우기
# 실행: python scripts/step2_3_clean_noise.py   (C:\STF 에서 실행)
# 입력: data/processed/step2_2_paras.jsonl   (2-2 결과)
# 결과: data/processed/step2_3_paras.jsonl   (잡음 지운 문단)
#       data/processed/step2_3_view.txt      (눈으로 확인용, 소제목은 "## " 표시)
# 하는 일: 1) 페이지 참조 문구 삭제   "(라이트 페이지의 70)", "참조 항목: 트랙션 제어 페이지의 81."
#          2) 표 잔해 문단 삭제       "36-4852-64100-140149-165", "✓✓✓✓"
#          3) 조사 묶음 정리          "Model S을(를)" → "Model S를"
# =============================================================

import json                                   # jsonl 읽기/쓰기용
import re                                     # 정규식 (패턴 찾아서 지우기·바꾸기)
from pathlib import Path                      # 경로 처리

ROOT = Path(__file__).resolve().parent.parent                    # 프로젝트 최상위 폴더 (C:\STF)
IN_PATH = ROOT / "data" / "processed" / "step2_2_paras.jsonl"    # 2-2 결과
OUT_JSONL = ROOT / "data" / "processed" / "step2_3_paras.jsonl"  # 이번 결과
OUT_VIEW = ROOT / "data" / "processed" / "step2_3_view.txt"      # 확인용

# 조사 묶음 목록: "원본 표기": (앞 글자에 받침 있을 때, 받침 없을 때)
JOSA = {
    "을(를)": ("을", "를"), "(을)를": ("을", "를"),
    "은(는)": ("은", "는"), "(은)는": ("은", "는"),
    "이(가)": ("이", "가"), "(이)가": ("이", "가"),
    "과(와)": ("과", "와"), "와(과)": ("과", "와"), "(과)와": ("과", "와"),
    "으로(로)": ("으로", "로"), "(으)로": ("으로", "로"),
    "이(나)": ("이나", "나"),
}


# ------------------------------------------------------------
# 1) 페이지 참조 문구 삭제
# ------------------------------------------------------------
def remove_page_refs(text):
    # 괄호 안 참조 삭제: "(라이트 페이지의 70)", "(에어 서스펜션 페이지의 77 참조)", "(#unique_240 페이지의 참조)"
    # 번호가 잘려서 닫는 괄호가 없는 "(참조 후방 카메라 페이지의" 같은 것도 같이 지움
    text = re.sub(r"\s*\([^()]*페이지의\s*\d*[^()]{0,6}\)?", "", text)
    # "참조 항목: 트랙션 제어 페이지의 81." 삭제
    text = re.sub(r"참조 항목:\s*[^.]*?페이지의\s*\d*\s*\.?", "", text)
    # 그래도 남은 "자세한 내용은 회생 제동 페이지의 75을(를) 참조하세요." 같은 문장은 통째로 삭제
    sentences = re.split(r"(?<=[.!?])\s+", text)            # 문장 단위로 쪼갬 (마침표·느낌표·물음표 뒤 공백 기준)
    sentences = [s for s in sentences if "페이지의" not in s]  # "페이지의"가 들어간 문장은 버림
    return " ".join(sentences)                               # 다시 합침


# ------------------------------------------------------------
# 2) 표 잔해 판단
# ------------------------------------------------------------
def is_table_junk(text):
    # 숫자·기호만으로 된 문단이면 표 잔해로 봄 (예: "36-4852-64100-140149-165", "✓✓✓✓")
    return re.fullmatch(r"[\d\s\-–~.,/%✓✗•:()+°]+", text) is not None


# ------------------------------------------------------------
# 3) 조사 묶음 정리
# ------------------------------------------------------------
def has_batchim(ch):                          # 한글 한 글자에 받침이 있는지
    code = ord(ch) - 0xAC00                   # 한글 시작 글자 "가" 기준으로 몇 번째 글자인지
    return 0 <= code <= 11171 and code % 28 != 0   # 한글 범위 안이고, 28로 나눈 나머지(받침 번호)가 0이 아니면 받침 있음


def choose_josa(word, josa):                  # 앞 단어(word)를 보고 조사 묶음(josa) 중 하나를 골라주는 함수
    with_b, without_b = JOSA[josa]            # 받침 있을 때 / 없을 때 조사
    if re.search(r"Model [SXY3]$", word):     # 모델 이름 "Model S"는 "에스"로 읽음 → 받침 없음
        return word + without_b
    base = re.sub(r"\([^()]*\)$", "", word) or word   # "드라이빙(감독형)"처럼 괄호로 끝나면 괄호 앞 "드라이빙" 기준
    base = base.rstrip(")\"'”’")              # 끝에 붙은 닫는 괄호·따옴표 제거
    last = base[-1] if base else word[-1]     # 판단할 마지막 글자
    if "가" <= last <= "힣":                  # 한글이면 받침 여부로 결정
        return word + (with_b if has_batchim(last) else without_b)
    return word + without_b                   # 영어·숫자는 대부분 받침 없는 발음(Wi-Fi→파이, USB→비)이라 뒤쪽 조사


def fix_josa(text):
    text = text.replace("S은(를)", "S를")     # 원본 오타 직접 수정 ("은(를)"은 짝이 안 맞는 표기)
    text = text.replace("S을(를) 을(를)", "S을(를)")   # 원본 오타 직접 수정 (조사 묶음이 두 번 찍힘)
    pattern = r"(\S+?)\s?(" + "|".join(re.escape(k) for k in JOSA) + ")"   # "단어 + (공백 0~1개) + 조사 묶음" 찾기
    return re.sub(pattern, lambda m: choose_josa(m.group(1), m.group(2)), text)   # 찾은 것마다 choose_josa로 바꿈


# ------------------------------------------------------------
# 문단 하나 정리
# ------------------------------------------------------------
def clean_text(text):
    text = remove_page_refs(text)             # 1) 페이지 참조 삭제
    text = fix_josa(text)                     # 3) 조사 정리
    text = re.sub(r"\s{2,}", " ", text)       # 지우고 나서 생긴 연속 공백을 하나로
    return text.strip()                       # 앞뒤 공백 정리


def main():
    with open(IN_PATH, encoding="utf-8") as f:            # 2-2 결과 읽기
        pages = [json.loads(line) for line in f]

    n_before, n_junk, n_empty = 0, 0, 0                   # 통계: 원래 문단 수, 표 잔해로 지운 수, 참조 지우고 비어버린 수
    results = []
    for p in pages:                                       # 페이지 하나씩
        new_paras = []
        for para in p["paras"]:                           # 문단 하나씩
            n_before += 1
            if is_table_junk(para["text"]):               # 2) 표 잔해면
                n_junk += 1
                continue                                  # 버림
            text = clean_text(para["text"])               # 1), 3) 적용
            if not text:                                  # 참조만 있던 문단이라 다 지워져서 비었으면
                n_empty += 1
                continue                                  # 버림
            new_paras.append({"text": text, "head": para["head"]})   # 소제목 여부는 그대로 유지
        results.append({"page": p["page"], "title": p["title"], "chapter": p["chapter"], "paras": new_paras})

    with open(OUT_JSONL, "w", encoding="utf-8") as f:     # jsonl 저장
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUT_VIEW, "w", encoding="utf-8") as f:      # 확인용 저장
        for r in results:
            f.write(f"\n{'=' * 15} p.{r['page']} | 제목: {r['title'] or '(없음)'} | 대제목: {r['chapter'] or '(없음)'} {'=' * 15}\n")
            for para in r["paras"]:
                prefix = "## " if para["head"] else ""   # 소제목 표시
                f.write(f"{prefix}{para['text']}\n\n")

    all_text = "\n".join(para["text"] for r in results for para in r["paras"])   # 남은 잡음 확인용으로 전체 합침
    left_refs = all_text.count("페이지의")                                       # 남은 페이지 참조 수
    left_josa = len(re.findall("|".join(re.escape(k) for k in JOSA), all_text))  # 남은 조사 묶음 수
    print(f"문단 {n_before:,}개 → {n_before - n_junk - n_empty:,}개 (표 잔해 {n_junk}개, 참조만 있던 문단 {n_empty}개 삭제)")
    print(f"남은 '페이지의': {left_refs}개 / 남은 조사 묶음: {left_josa}개")
    print(f"→ {OUT_JSONL.name}, {OUT_VIEW.name} 저장 완료")


if __name__ == "__main__":
    main()
