# =============================================================
# 8-0단계: EXAONE 3.5 2.4B → Llama 형식으로 변환해서 저장
# 실행: python scripts\step8_0_convert_exaone35.py   (SFT 폴더, 가상환경 켠 상태)
# 왜? EXAONE 3.5는 허깅페이스에서 받은 전용 코드(modeling_exaone.py)로 돌아가는데,
#     그 코드가 지금 쓰는 transformers 5.24와 맞지 않아 생성할 때 에러가 남
#     EXAONE 3.5는 구조가 Llama와 같고 가중치 이름만 다름 → 이름만 바꿔 저장하면
#     transformers에 원래 들어 있는 Llama 코드로 그대로 돌릴 수 있음 (가중치 값은 그대로)
# 입력: 허깅페이스 캐시에 받아둔 LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct (8-1에서 이미 다운로드됨)
# 결과: models/exaone35-2.4b-llama/   (약 5GB)
# =============================================================

import json                                    # config.json 읽기
from pathlib import Path                       # 경로 처리
import torch                                   # 파이토치
from huggingface_hub import snapshot_download  # 캐시에 받아둔 모델 폴더 위치 찾기
from safetensors.torch import load_file        # 가중치 파일 읽기
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM   # Llama 형식 모델

ROOT = Path(__file__).resolve().parent.parent                      # SFT 폴더
SRC_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"                    # 원본 모델
OUT_DIR = ROOT / "models" / "exaone35-2.4b-llama"                  # 변환 결과 저장 폴더


def rename(k):                                                     # EXAONE 가중치 이름 → Llama 가중치 이름
    k = k.replace("transformer.wte.", "model.embed_tokens.")       # 단어 임베딩
    k = k.replace("transformer.ln_f.", "model.norm.")              # 마지막 정규화
    k = k.replace("transformer.h.", "model.layers.")               # 레이어 묶음
    k = k.replace(".attn.attention.out_proj.", ".self_attn.o_proj.")   # 어텐션 출력
    k = k.replace(".attn.attention.", ".self_attn.")               # 어텐션 q/k/v
    k = k.replace(".ln_1.", ".input_layernorm.")                   # 어텐션 앞 정규화
    k = k.replace(".ln_2.", ".post_attention_layernorm.")          # MLP 앞 정규화
    k = k.replace(".mlp.c_fc_0.", ".mlp.gate_proj.")               # MLP 게이트
    k = k.replace(".mlp.c_fc_1.", ".mlp.up_proj.")                 # MLP 올리기
    k = k.replace(".mlp.c_proj.", ".mlp.down_proj.")               # MLP 내리기
    return k


def main():
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):                # 이미 변환했으면 중단 (덮어쓰기 방지)
        raise SystemExit(f"[중단] 이미 있음: {OUT_DIR}")
    src = Path(snapshot_download(SRC_ID))                          # 캐시 폴더 경로 (이미 받았으면 다운로드 안 함)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))   # 원본 설정
    print("원본 설정:", {k: cfg.get(k) for k in ("hidden_size", "num_layers", "num_attention_heads",
                                               "num_key_value_heads", "intermediate_size", "rope_theta",
                                               "rope_scaling", "tie_word_embeddings", "max_position_embeddings")})

    config = LlamaConfig(                                          # 같은 크기의 Llama 설정 만들기
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rms_norm_eps=cfg["layer_norm_epsilon"],
        rope_theta=cfg.get("rope_theta", 10000.0),
        rope_scaling=cfg.get("rope_scaling"),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        bos_token_id=cfg.get("bos_token_id"), eos_token_id=cfg.get("eos_token_id"), pad_token_id=cfg.get("pad_token_id"),
        attention_bias=False, mlp_bias=False,
    )

    state = {}                                                     # 이름 바꾼 가중치 모음
    for f in sorted(src.glob("*.safetensors")):                    # 가중치 파일(여러 개로 나뉨) 하나씩
        for k, v in load_file(str(f)).items():
            state[rename(k)] = v
    print(f"가중치 {len(state)}개 읽음")

    torch.set_default_dtype(torch.bfloat16)                        # bf16으로 빈 모델 만들기 (메모리 절약)
    model = LlamaForCausalLM(config)
    if config.tie_word_embeddings:                                 # 입력·출력 임베딩을 공유하는 모델이면
        state.pop("lm_head.weight", None)                          # 출력 가중치는 따로 넣지 않음
    missing, unexpected = model.load_state_dict(state, strict=False)   # 이름이 맞는지 확인하면서 넣기
    missing = [m for m in missing if not (config.tie_word_embeddings and m == "lm_head.weight")]
    if missing or unexpected:                                      # 하나라도 안 맞으면 저장하지 않고 중단
        raise SystemExit(f"[중단] 이름이 안 맞음\n  빠진 것: {missing[:5]}\n  남는 것: {unexpected[:5]}")
    if config.tie_word_embeddings:
        model.tie_weights()                                        # 출력층을 입력 임베딩과 연결

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)                                 # Llama 형식으로 저장
    tok = AutoTokenizer.from_pretrained(SRC_ID, trust_remote_code=True)   # 토크나이저(채팅 형식 포함)는 그대로
    tok.save_pretrained(OUT_DIR)
    print(f"변환 완료 → {OUT_DIR}  (가중치 값은 그대로, 이름만 Llama 형식)")


if __name__ == "__main__":
    main()
