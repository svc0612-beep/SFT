# =============================================================
# 3단계: 전처리된 섹션 → OpenAI(gpt-6-luna)로 "질문 → 답변" SFT 데이터 생성
# 실행: python scripts/step3_make_qa.py   (C:\STF 에서 실행)
# 입력: data/processed/clean_sections.jsonl   (2-4 결과)
# 결과: data/processed/sft_qa.jsonl            (한 줄에 Q&A 하나)
# 준비: pip install openai python-dotenv
#       C:\STF\.env 파일에 OPENAI_API_KEY=sk-... 적어두기 (환경변수로 이미 있으면 생략)
# 중간에 끊겨도 다시 실행하면 이미 끝난 덩어리는 건너뛰고 이어서 진행함
# =============================================================

import json                                         # jsonl 읽기/쓰기용
import os                                           # 환경변수(API 키) 읽기용
from pathlib import Path                            # 경로 처리
from concurrent.futures import ThreadPoolExecutor, as_completed   # 여러 덩어리를 동시에 요청하려고 사용
from threading import Lock                          # 여러 스레드가 파일에 동시에 쓰지 않게 막는 자물쇠
from dotenv import load_dotenv                      # .env 파일에서 API 키 읽기
from openai import OpenAI                           # OpenAI 공식 라이브러리

ROOT = Path(__file__).resolve().parent.parent       # 프로젝트 최상위 폴더 (C:\STF)
load_dotenv(ROOT / ".env")                          # C:\STF\.env 의 키를 환경변수로 불러옴

IN_PATH = ROOT / "data" / "processed" / "clean_sections.jsonl"   # 전처리 최종 결과
OUT_PATH = ROOT / "data" / "processed" / "sft_qa.jsonl"          # 이번 단계 결과

MODEL = "gpt-6-luna"                                # 사용할 모델 (OpenAI 최신 경량 모델, 저렴)
CHUNK_CHARS = 2500                                  # 섹션을 이 글자 수 정도로 잘라서 보냄 (긴 섹션 대비)
CHARS_PER_QA = 400                                  # 이 글자 수당 Q&A 1개 정도 만들게 함 (2500자 → 약 6개)
WORKERS = 5                                         # 동시에 보낼 요청 수

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))   # API 클라이언트 생성
write_lock = Lock()                                 # 파일 쓰기용 자물쇠

# 모델에게 줄 지시문 (시스템 프롬프트)
SYSTEM_PROMPT = """너는 테슬라 Model S 사용자 매뉴얼로 챗봇 학습 데이터를 만드는 도우미다.
주어진 [매뉴얼 본문]만 근거로, Model S 오너가 실제로 물어볼 법한 질문과 그 답변을 만들어라.
규칙:
- 답변은 반드시 본문 내용에만 근거할 것. 본문에 없는 내용은 절대 지어내지 말 것.
- 답변은 존댓말로 1~3문장, 간결하게.
- 질문은 서로 겹치지 않게, 실제 사용자가 말하듯 자연스럽게.
- 질문만 보고도 무엇을 묻는지 알 수 있게 쓸 것 ("이 기능은?" 처럼 앞뒤 맥락이 필요한 질문 금지).
- 본문이 법적 고지·인증 번호처럼 질문을 만들 내용이 없으면 빈 리스트를 반환할 것.
출력은 JSON 하나: {"qa": [{"question": "...", "answer": "..."}]}"""


def split_into_chunks(paras):                       # 섹션의 문단 목록을 CHUNK_CHARS 정도 크기의 덩어리로 자르는 함수
    chunks, cur, cur_len = [], [], 0                # 완성된 덩어리들, 만드는 중인 덩어리, 그 길이
    for para in paras:                              # 문단 하나씩
        text = para["text"]
        if cur and cur_len + len(text) > CHUNK_CHARS:   # 이 문단을 넣으면 너무 길어지면
            chunks.append("\n".join(cur))           # 지금까지 모은 걸 덩어리로 저장 (문단 중간에서는 안 자름)
            cur, cur_len = [], 0                    # 새 덩어리 시작
        cur.append(text)                            # 문단 추가
        cur_len += len(text)                        # 길이 누적
    if cur:                                         # 남은 것도
        chunks.append("\n".join(cur))               # 덩어리로 저장
    return chunks


def load_done_ids():                                # 이미 처리한 덩어리 id를 읽는 함수 (이어하기용)
    done = set()
    if OUT_PATH.exists():                           # 결과 파일이 있으면
        with open(OUT_PATH, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["chunk_id"])   # 처리한 덩어리 id 기록
    return done


def make_qa(job):                                   # 덩어리 하나로 Q&A를 만드는 함수
    n_qa = max(2, round(len(job["text"]) / CHARS_PER_QA))   # 만들 개수 (짧아도 최소 2개)
    user_msg = (
        f"[위치] {job['chapter']} > {job['title']}\n"        # 이 본문이 매뉴얼 어디인지 알려줌 (질문 만들 때 맥락)
        f"[만들 개수] 최대 {n_qa}개\n\n"
        f"[매뉴얼 본문]\n{job['text']}"
    )
    resp = client.chat.completions.create(          # OpenAI API 호출
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},    # 지시문
            {"role": "user", "content": user_msg},           # 본문
        ],
        response_format={"type": "json_object"},    # JSON으로만 답하게 강제
        reasoning_effort="low",                     # 루나는 추론 모델 → 생각을 짧게 해서 속도·비용 절약
    )
    data = json.loads(resp.choices[0].message.content)   # 응답 문자열 → dict
    return data.get("qa", []), resp.usage           # Q&A 리스트와 토큰 사용량 반환


def process(job):                                   # 덩어리 하나 처리 + 파일 저장
    qa_list, usage = make_qa(job)                   # Q&A 생성
    with write_lock:                                # 다른 스레드와 겹치지 않게
        with open(OUT_PATH, "a", encoding="utf-8") as f:   # 이어쓰기 모드
            base = {"chunk_id": job["chunk_id"], "chapter": job["chapter"], "title": job["title"]}   # 공통 정보
            if not qa_list:                         # 만들 게 없던 덩어리도
                f.write(json.dumps({**base, "question": None, "answer": None}, ensure_ascii=False) + "\n")   # "처리함" 표시
            for qa in qa_list:                      # Q&A 하나씩 저장
                row = {**base, "question": qa["question"].strip(), "answer": qa["answer"].strip()}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return job["chunk_id"], len(qa_list), usage


def main():
    if not os.getenv("OPENAI_API_KEY"):             # API 키 없으면 종료
        raise SystemExit("OPENAI_API_KEY 가 없어. C:\\STF\\.env 에 넣어줘.")

    with open(IN_PATH, encoding="utf-8") as f:      # 섹션 읽기
        sections = [json.loads(line) for line in f]

    jobs = []                                       # 보낼 덩어리 목록
    for si, sec in enumerate(sections):             # 섹션마다
        for ci, text in enumerate(split_into_chunks(sec["paras"])):   # 덩어리로 자르고
            jobs.append({"chunk_id": f"{si}-{ci}", "chapter": sec["chapter"], "title": sec["title"], "text": text})

    done = load_done_ids()                          # 이미 한 것
    todo = [j for j in jobs if j["chunk_id"] not in done]   # 남은 것만
    print(f"섹션 {len(sections)}개 → 덩어리 {len(jobs)}개, 남은 {len(todo)}개 처리 시작 (모델 {MODEL}, 동시 {WORKERS}개)")

    total_qa, fail, tok_in, tok_out = 0, 0, 0, 0    # 통계
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process, j): j["chunk_id"] for j in todo}   # 덩어리마다 작업 등록
        for i, fut in enumerate(as_completed(futures), 1):              # 끝나는 순서대로
            try:
                cid, n, usage = fut.result()
                total_qa += n
                tok_in += usage.prompt_tokens       # 입력 토큰 누적
                tok_out += usage.completion_tokens  # 출력 토큰 누적 (추론 토큰 포함)
                print(f"[{i}/{len(todo)}] {cid} → {n}개")
            except Exception as e:                  # 실패해도 전체는 계속 (재실행하면 다시 시도됨)
                fail += 1
                print(f"[{i}/{len(todo)}] {futures[fut]} 실패: {e}")

    cost = tok_in / 1e6 * 0.10 + tok_out / 1e6 * 0.50   # gpt-6-luna 가격 (입력 $0.10, 출력 $0.50 / 100만 토큰)
    print(f"\n이번 실행: Q&A {total_qa}개 생성, 실패 {fail}개")
    print(f"토큰: 입력 {tok_in:,} / 출력 {tok_out:,} → 약 ${cost:.3f}")
    if fail:
        print("→ 같은 명령어로 다시 실행하면 실패한 덩어리만 다시 처리해.")

    with open(OUT_PATH, encoding="utf-8") as f:     # 최종 결과 다시 읽어서
        rows = [json.loads(l) for l in f if json.loads(l)["question"]]   # "처리함" 표시 줄은 빼고
    print(f"전체 Q&A: {len(rows)}개 → {OUT_PATH.name}")
    print("\n--- 샘플 3개 ---")
    for r in rows[:3]:
        print(f"[{r['chapter']} > {r['title']}]\nQ: {r['question']}\nA: {r['answer']}\n")


if __name__ == "__main__":
    main()
