# =============================================================
# 4단계: Q&A → koGPT2 학습용 데이터셋 만들기
# 실행: python scripts/step4_make_dataset.py   (C:\STF 에서 실행)
# 입력: data/processed/sft_qa.jsonl     (3단계 결과)
# 결과: data/processed/sft_train.jsonl  (학습용 90%)
#       data/processed/sft_val.jsonl    (검증용 10%)
# 하는 일: 1) 불량 데이터 거르기 (빈 줄, 그림 번호 답변, "본문에는~" 답변, 중복 질문)
#          2) koGPT2 학습 형식으로 바꾸기   "<usr>질문<sys>답변</s>"
#          3) 학습용/검증용 나누기
# =============================================================

import json                                   # jsonl 읽기/쓰기용
import random                                 # 데이터 섞기용
import re                                     # 정규식 (불량 패턴 찾기)
from pathlib import Path                      # 경로 처리

ROOT = Path(__file__).resolve().parent.parent                  # 프로젝트 최상위 폴더 (C:\STF)
IN_PATH = ROOT / "data" / "processed" / "sft_qa.jsonl"         # 3단계 결과
TRAIN_PATH = ROOT / "data" / "processed" / "sft_train.jsonl"   # 학습용
VAL_PATH = ROOT / "data" / "processed" / "sft_val.jsonl"       # 검증용

VAL_RATIO = 0.1          # 검증용 비율 (10%)
SEED = 42                # 섞는 순서 고정 (다시 돌려도 같은 결과 나오게)

# koGPT2 대화 형식에 쓸 표시
#   <usr> = 사용자 질문 시작, <sys> = 챗봇 답변 시작, </s> = 문장 끝 (koGPT2에 원래 있는 종료 토큰)
#   학습 후 "<usr>질문<sys>" 까지 넣어주면 모델이 그 뒤 답변을 이어서 생성하고 </s>에서 멈춤
USR, SYS, EOS = "<usr>", "<sys>", "</s>"

# 거를 패턴
BAD_FIGURE = r"\d+번(으로|은|는|이|에)"        # 그림 번호로 답한 것 (예: "충전 포트는 3번으로 표시되어 있습니다")
BAD_SOURCE = r"본문|매뉴얼에서는|문서에"        # "본문에는~", "매뉴얼에서는~" 처럼 출처를 말하는 답변 (챗봇 답변으로 어색)


def main():
    with open(IN_PATH, encoding="utf-8") as f:            # 3단계 결과 읽기
        rows = [json.loads(line) for line in f]
    print(f"원본: {len(rows)}줄")

    removed = {"빈 줄(null)": 0, "그림 번호 답변": 0, "출처 언급 답변": 0, "중복 질문": 0}   # 걸러낸 개수 기록
    seen = set()                                           # 이미 나온 질문 (중복 확인용)
    data = []                                              # 통과한 데이터
    for r in rows:
        q, a = r["question"], r["answer"]                  # 질문, 답변
        if not q or not a:                                 # null 줄
            removed["빈 줄(null)"] += 1
            continue
        if re.search(BAD_FIGURE, a):                       # 그림 번호 답변
            removed["그림 번호 답변"] += 1
            continue
        if re.search(BAD_SOURCE, a):                       # 출처 언급 답변
            removed["출처 언급 답변"] += 1
            continue
        if q in seen:                                      # 이미 나온 질문
            removed["중복 질문"] += 1
            continue
        seen.add(q)                                        # 질문 기록
        data.append({
            "question": q,                                 # 질문 (참고용)
            "answer": a,                                   # 답변 (참고용)
            "text": f"{USR}{q}{SYS}{a}{EOS}",              # 실제 학습에 들어가는 한 줄
            "title": r["title"],                           # 어느 섹션에서 나왔는지 (참고용)
        })

    for k, v in removed.items():                           # 걸러낸 개수 출력
        print(f"  - {k}: {v}개 제거")
    print(f"남은 데이터: {len(data)}개")

    random.seed(SEED)                                      # 섞는 순서 고정
    random.shuffle(data)                                   # 섞기 (같은 섹션 Q&A가 한쪽에 몰리지 않게)
    n_val = round(len(data) * VAL_RATIO)                   # 검증용 개수
    val, train = data[:n_val], data[n_val:]                # 앞쪽 10% = 검증용, 나머지 = 학습용

    for path, part in [(TRAIN_PATH, train), (VAL_PATH, val)]:   # 두 파일 저장
        with open(path, "w", encoding="utf-8") as f:
            for d in part:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"학습용 {len(train)}개 → {TRAIN_PATH.name}")
    print(f"검증용 {len(val)}개 → {VAL_PATH.name}")
    print("\n--- 학습 형식 샘플 ---")
    print(train[0]["text"])                                # 실제 학습 한 줄이 어떻게 생겼는지


if __name__ == "__main__":
    main()
