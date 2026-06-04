#!/usr/bin/env python
# coding: utf-8

# # 실습 2 — LoRA 파인튜닝: 뉴스 재밌게 설명하는 말투 학습
# 
# 사전학습된 **Qwen2.5-1.5B** 모델을 뉴스 설명 스타일 데이터셋으로 파인튜닝하여,  
# 딱딱한 뉴스 이슈를 쉽고 재밌게 3문장으로 설명하는 모델을 만듭니다.
# 
# | 항목 | 내용 |
# |---|---|
# | **모델** | `Qwen2.5-1.5B-Instruct` (15억 파라미터) |
# | **데이터셋** | 직접 만든 뉴스 설명 스타일 합성 데이터셋 (64개 문장 쌍) |
# | **태스크** | 뉴스 이슈 → 재밌고 쉬운 3문장 설명 |
# | **방법** | LoRA Fine-tuning (전체의 ~1%만 학습) |
# 
# > **왜 뉴스 설명 말투인가?**  
# > 뉴스는 내용이 딱딱하고 어렵게 느껴질 때가 많습니다.  
# > 베이스 모델도 설명은 할 수 있지만, 매번 같은 구조와 재밌는 톤을 안정적으로 유지하지는 못합니다.  
# > 이것이 말투/스타일 파인튜닝의 핵심 원리입니다.
# 
# ```
# [파인튜닝 전]  "물가가 다시 오른다는 뉴스"
# → "물가 상승은 경제 전반에 영향을 미치는 현상입니다..."  (딱딱한 설명)
# 
# [파인튜닝 후]  "물가가 다시 오른다는 뉴스"
# → "핵심은 장바구니 영수증이 조용히 커지는 상황입니다. 쉽게 말해 같은 돈으로 살 수 있는 물건이 줄어드는 거예요. 그래서 가계 부담과 정책 판단에 영향을 줍니다."  (뉴스 진행자 톤으로 설명!)
# ```
# 

# ## ⚙️ Colab GPU 설정
# 
# **설정 방법:**
# 1. 상단 메뉴 → **런타임** 클릭
# 2. **런타임 유형 변경** 클릭
# 3. 하드웨어 가속기 → **T4 GPU** 선택
# 4. **저장** 후 런타임 재시작
# 
# > ⚠️ GPU 없이 실행하면 학습 속도가 매우 느립니다. 반드시 설정 후 시작하세요.

# ## 📦 라이브러리 설치
# 
# | 라이브러리 | 역할 |
# |---|---|
# | `unsloth` | LLM을 빠르고 가볍게 파인튜닝하는 최적화 라이브러리 |
# | `trl` | SFTTrainer 제공 — LLM 지도 파인튜닝 전용 Trainer |
# | `peft` | LoRA 등 PEFT 기법을 적용하는 도구 |
# | `datasets` | HuggingFace 데이터셋 로딩 |

# In[ ]:


# ✅ STEP 1: 필수 라이브러리 설치
# ➤ Unsloth는 일반 transformers 대비 2배 빠른 속도와 60% 메모리 절약을 제공합니다.
get_ipython().system('pip install bitsandbytes==0.48.0')
get_ipython().system('pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
get_ipython().system('pip install -q --no-deps trl peft accelerate  datasets')
print("✅ 설치 완료")


# ## 1단계. 모델 로딩
# 
# **Qwen2.5-1.5B-Instruct**는 15억 파라미터의 경량 생성 모델입니다.  
# `FastLanguageModel`은 Unsloth가 제공하는 최적화된 로더로, 메모리를 훨씬 적게 사용합니다.
# 
# ---
# ###load_in_4bit 설정
# 
# 4비트 양자화를 활성화하면 모델 크기가 약 1/4로 줄어 Colab 무료 환경에서도 돌아갑니다.  
# Colab T4 환경에서는 반드시 `True`로 설정해야 합니다.
# 
# ---

# In[ ]:


# ✅ STEP 2: 모델 및 토크나이저 로딩
# ➤ Unsloth의 FastLanguageModel은 일반 AutoModelForCausalLM보다 빠르고 메모리 효율적입니다.
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct",  # 포인트: 1.5B 소형 모델 — Colab 무료 환경에 최적
    max_seq_length = 2048,   # 포인트: 한 번에 처리할 최대 토큰 수 (입력+출력 합계)
    load_in_4bit = True,      # ✏️ 4비트 양자화 활성화 여부 (True / False)
)

print("✅ 모델 로딩 완료: Qwen2.5-1.5B-Instruct")
print(f"전체 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")


# ## 2단계. LoRA 어댑터 설정
# 
# LoRA는 전체 모델 가중치를 수정하지 않고, 각 가중치 행렬에 **작은 행렬 A와 B**를 추가로 붙여 그 부분만 학습합니다.
# 
# ```
# 기존: output = W · input           (W는 고정, 수정 안 함)
# LoRA: output = (W + A·B) · input   (A, B만 학습 — 전체의 ~1%)
# ```
# 
# ---
# ###  LoRA 핵심 파라미터
# 
# | 파라미터 | 권장값 | 설명 |
# |---|---|---|
# | `r` | `16` | LoRA 행렬의 내부 차원(rank). 클수록 표현력↑, 메모리↑ |
# | `lora_alpha` | `16` | LoRA 출력 스케일 조정값. 보통 r과 같게 설정 |
# | `lora_dropout` | `0` | 드롭아웃 비율. 요다 720개처럼 작고 일관된 데이터에서는 0이 더 빠름 |
# 
# ---

# In[ ]:


# ✅ STEP 3: LoRA 어댑터 구성
# ➤ LoRA는 모델의 일부 레이어만 미세조정함으로써 빠르고 효율적인 파인튜닝이 가능합니다.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,             # ✏️ LoRA rank — 학습 파라미터 수 조절 (권장: 16)
    lora_alpha = 16,    # ✏️ 스케일 조정값 — 보통 r과 동일하게 설정
    lora_dropout = 0,  # ✏️ 드롭아웃 비율 (작고 일관된 데이터셋이므로 0 권장)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # 포인트: 어텐션 레이어 (Query/Key/Value/Output)
        "gate_proj", "up_proj", "down_proj"         # 포인트: FFN 레이어 (핵심 연산 담당)
        # 이 레이어들은 모델 연산의 핵심 부분입니다.
        # 큰 가중치를 가지므로 LoRA를 적용하면 적은 자원으로도 효과적으로 학습할 수 있습니다.
    ],
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # 포인트: 메모리 절약 기법 — 중간 계산값을 다시 계산해 저장 안 함
    random_state = 42,                       # 포인트: 재현성 확보 — 같은 시드면 같은 결과
)

# ✅ 실제로 학습되는 파라미터 비율 확인
# ➤ 전체의 1~2%만 학습됩니다. 이것이 LoRA의 핵심 장점입니다.
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"학습 파라미터: {trainable:,}")
print(f"전체 파라미터: {total:,}")
print(f"학습 비율    : {100 * trainable / total:.2f}%  ← 이 작은 비율만 학습합니다!")


# ## 3단계. 프롬프트 템플릿 & EOS 토큰
# 
# **Alpaca 스타일 프롬프트**는 모델이 일관된 형식으로 응답하도록 입출력 형식을 고정합니다.
# 
# ```
# ### Instruction:  (태스크 설명 — "무엇을 해라")
# ### Input:        (사용자 입력 — 변환할 원문)
# ### Response:     (모델이 생성할 부분 — 요다 말투 번역)
# ```
# 
# ---
# ### EOS 토큰
# 
# `EOS_TOKEN`은 모델이 응답을 마쳤다는 신호입니다.  
# 이것이 없으면 모델이 응답을 언제 끝내야 할지 몰라 계속 생성합니다.
# 
# ---

# In[ ]:


# ✅ STEP 4: Alpaca 스타일 프롬프트 포맷 정의
# ➤ 학습 데이터의 형식을 통일하기 위한 템플릿입니다.
#    instruction/input/response 세 칸을 고정 틀로 잡아 학습을 안정적으로 만듭니다.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

NEWS_INSTRUCTION = (
    "너는 뉴스를 쉽고 재밌게 설명하는 진행자다. "
    "입력된 뉴스 이슈를 3문장으로 설명하라. "
    "1문장은 핵심, 2문장은 쉬운 비유, 3문장은 왜 중요한지다. "
    "없는 사실이나 숫자는 만들지 않는다."
)

# ✅ EOS(문장 끝) 토큰 설정
# ➤ EOS 토큰을 붙여야 모델이 "여기까지가 정답"임을 명확히 알 수 있습니다.
EOS_TOKEN = tokenizer.eos_token  #  tokenizer에서 EOS 토큰을 가져오세요


print(f"EOS 토큰: {repr(EOS_TOKEN)}")
print("\n템플릿 예시:")
print(alpaca_prompt.format(NEWS_INSTRUCTION, "물가가 다시 오른다는 뉴스", "핵심은 장바구니 부담이 커지는 상황입니다. 쉽게 말해 같은 돈으로 살 수 있는 물건이 줄어드는 거예요. 그래서 가계 소비와 정책 판단에 영향을 줍니다.") + EOS_TOKEN)


# ## 4단계. 데이터 포맷 변환 함수
# 
# 뉴스 설명 스타일 데이터셋 원본은 세 컬럼으로 구성됩니다.
# 
# ```
# instruction: "너는 뉴스를 쉽고 재밌게 설명하는 진행자다..."  ← 역할과 출력 규칙
# input:       "물가가 다시 오른다는 뉴스"                  ← 설명할 뉴스 이슈
# output:      "핵심은 장바구니 부담이 커지는 상황입니다..." ← 원하는 설명 말투
# ```
# 
# 이를 Alpaca 프롬프트 형식으로 변환합니다.
# 

# In[ ]:


# ✅ STEP 5: 데이터 포맷 변환 함수 정의
# ➤ 원시 데이터의 세 컬럼(instruction/input/output)을 하나의 텍스트로 변환합니다.
#    가공 로직을 한 곳에 모아 재사용 가능하게 만드는 것이 핵심입니다.
def format_instruction(example):
    return {
        "text": alpaca_prompt.format(
            example["instruction"],  # ✏️ 빈칸 6: instruction — 모델에게 무엇을 하라고 지시할까요?

            example["input"],        # 포인트: 설명할 뉴스 이슈 (input 자리)
            example["output"]        # 포인트: 뉴스 진행자 톤 설명 (response 자리 — 정답)
        ) + EOS_TOKEN  # 포인트: 문장 끝에 EOS 토큰을 붙여 모델이 응답을 마무리하도록 유도
    }

# ✅ 변환 결과 미리 확인
test_ex = {
    "instruction": NEWS_INSTRUCTION,
    "input": "물가가 다시 오른다는 뉴스",
    "output": "핵심은 장바구니 부담이 커지는 상황입니다. 쉽게 말해 같은 돈으로 살 수 있는 물건이 줄어드는 거예요. 그래서 가계 소비와 정책 판단에 영향을 줍니다.",
}
print(format_instruction(test_ex)["text"])


# ## 5단계. 데이터셋 로딩
# 
# 직접 만든 뉴스 설명 스타일 합성 데이터셋은 64개의 입력-출력 쌍으로 구성됩니다.
# 
# | 컬럼 | 예시 |
# |---|---|
# | `instruction` | "너는 뉴스를 쉽고 재밌게 설명하는 진행자다..." |
# | `input` | "AI 규제가 논의되고 있다는 뉴스" |
# | `output` | "핵심은 AI라는 빠른 기차에 안전벨트를 달지 정하는 문제입니다..." |
# 
# > 뉴스 설명 말투의 핵심 패턴:  
# > - 첫 문장에 핵심을 바로 말함  
# > - 두 번째 문장에 쉬운 비유를 넣음  
# > - 세 번째 문장에 왜 중요한지 연결함
# 

# In[ ]:


# ✅ STEP 6: 데이터셋 로딩 및 포맷 변환
# ➤ 직접 만든 뉴스 설명 스타일 데이터를 Dataset으로 만들고 Alpaca 형식으로 변환합니다.
from datasets import Dataset

base_examples = [
    ("기준금리가 동결됐다는 뉴스", "돈의 속도를 조절하는 금리 스위치가 이번에는 그대로 멈췄다는 뜻", "자동차로 치면 액셀도 브레이크도 더 밟지 않고 현재 속도를 유지하는 상황", "대출 이자와 예금 이자, 소비 분위기까지 이어질 수 있어서 중요합니다"),
    ("물가가 다시 오르고 있다는 뉴스", "같은 장바구니를 채우는 데 필요한 돈이 더 많아지고 있다는 신호", "영수증이 조용히 키가 크는 것처럼 생활비 부담이 슬금슬금 커지는 장면", "가계 소비와 기업 가격 전략, 정부 정책 판단에 모두 영향을 줍니다"),
    ("반도체 수출이 늘었다는 뉴스", "한국 경제의 주력 선수 중 하나가 다시 득점 기회를 잡았다는 이야기", "야구에서 중심 타자가 장타를 치면 팀 분위기가 살아나는 것과 비슷합니다", "수출과 투자, 관련 일자리 기대에 연결될 수 있습니다"),
    ("환율이 올랐다는 뉴스", "원화로 달러를 사는 비용이 더 비싸졌다는 뜻", "해외 쇼핑 카트에 담긴 물건 가격표가 갑자기 커진 느낌", "수입 물가와 해외여행 비용, 기업 실적에 영향을 줄 수 있습니다"),
    ("전기차 보조금 기준이 바뀐다는 뉴스", "전기차를 살 때 받을 수 있는 할인 쿠폰의 조건이 달라지는 상황", "게임 이벤트 보상이 바뀌면 유저들이 공략법을 다시 짜는 것과 같습니다", "소비자 선택과 자동차 회사의 가격 전략이 함께 움직일 수 있습니다"),
    ("AI 규제가 논의되고 있다는 뉴스", "AI라는 빠른 기차에 안전벨트와 신호등을 어디까지 달지 정하는 문제", "놀이공원이 재밌어도 안전 규칙이 있어야 오래 운영되는 것과 비슷합니다", "혁신 속도와 개인정보 보호, 책임 소재가 동시에 걸려 있습니다"),
    ("플랫폼 수수료 논란 뉴스", "온라인 장터를 빌려 쓰는 비용이 적정한지 따져보는 이슈", "시장 입구 자릿세가 너무 비싸면 가게도 손님도 부담을 느끼는 장면", "소상공인 수익과 소비자 가격에 같이 영향을 줄 수 있습니다"),
    ("배달비 부담이 커졌다는 뉴스", "음식값보다 배달비가 더 눈에 띄는 순간이 많아졌다는 이야기", "짜장면을 시켰는데 배달 오토바이 탑승권도 같이 산 기분", "소비 습관과 자영업 매출, 플랫폼 경쟁에 영향을 줍니다"),
    ("청년 고용이 둔화됐다는 뉴스", "사회에 막 들어서는 사람들이 첫 출발선을 통과하기 어려워졌다는 신호", "출발 총성은 울렸는데 경기장 문이 좁아진 상황", "소득, 주거, 소비 계획이 줄줄이 영향을 받을 수 있습니다"),
    ("전세 사기 방지 대책 뉴스", "집을 빌릴 때 보증금을 지키는 안전장치를 더 촘촘히 만들자는 내용", "큰돈을 맡기는 금고에 잠금장치를 하나 더 다는 것과 같습니다", "주거 안정과 부동산 시장 신뢰에 직접 연결됩니다"),
    ("폭염 대비 정책 뉴스", "더운 날씨가 건강 문제로 번지지 않게 미리 방어막을 치는 일", "여름 보스전이 오기 전에 물약과 방패를 챙기는 상황", "취약계층 보호와 전력 수요 관리가 함께 중요해집니다"),
    ("재생에너지 투자가 늘었다는 뉴스", "전기를 만드는 방식을 더 깨끗한 방향으로 바꾸려는 움직임", "오래된 보일러만 쓰던 집에 태양광 창문을 추가하는 느낌", "기후 대응과 에너지 안보, 산업 경쟁력에 모두 관련됩니다"),
    ("사이버 보안 사고 뉴스", "디지털 세상의 문단속이 뚫렸다는 경고음", "집 현관은 잠갔는데 와이파이 창문이 열려 있었던 상황", "개인정보와 기업 신뢰, 서비스 운영 안정성에 영향을 줍니다"),
    ("개인정보 보호 강화 뉴스", "내 데이터가 어디에 쓰이는지 더 엄격하게 관리하자는 흐름", "내 이름표가 붙은 짐을 아무나 열어보지 못하게 잠그는 일", "편리한 서비스와 사생활 보호 사이의 균형을 정하는 문제입니다"),
    ("K-콘텐츠 해외 흥행 뉴스", "한국 콘텐츠가 해외 관객의 리모컨을 붙잡았다는 이야기", "동네 맛집이 갑자기 세계 푸드코트의 인기 매장이 된 느낌", "제작 투자와 관광, 브랜드 이미지에 긍정적인 파급이 생길 수 있습니다"),
    ("우주 발사체 개발 뉴스", "우주로 물건을 보내는 자체 배송 능력을 키우는 일", "남의 택배차만 기다리던 회사가 자기 로켓 택배차를 만드는 상황", "과학기술 경쟁력과 위성 산업의 기반이 됩니다"),
    ("디지털 교과서 도입 뉴스", "교과서가 종이책에서 화면 속 맞춤형 도구로 바뀌는 흐름", "칠판과 문제집이 태블릿 안에서 한 팀이 되는 장면", "학습 격차, 교사 준비, 학생 집중도까지 함께 따져봐야 합니다"),
    ("대중교통 요금 인상 뉴스", "매일 타는 이동 서비스의 기본 가격표가 바뀌는 일", "출근길 커피값이 오른 것처럼 작아 보여도 매일 쌓이면 크게 느껴집니다", "가계 부담과 교통 재정, 서비스 품질 문제가 함께 얽혀 있습니다"),
    ("중고거래 사기 예방 뉴스", "싸게 사려다 비싸게 배우는 일을 줄이자는 이야기", "온라인 벼룩시장에서 계산대와 CCTV를 더 밝게 켜는 느낌", "개인 간 거래가 커질수록 신뢰 장치가 더 중요해집니다"),
    ("게임 확률형 아이템 공개 뉴스", "뽑기 상자의 당첨 확률표를 더 투명하게 보자는 이슈", "자판기에 동전을 넣기 전에 어떤 음료가 얼마나 나오는지 확인하는 셈", "소비자 보호와 게임사의 수익 모델 신뢰가 같이 걸려 있습니다"),
    ("소상공인 대출 지원 뉴스", "작은 가게들이 숨 고를 시간을 벌도록 자금 길을 열어주는 정책", "비 오는 날 장사하는 가게에 잠깐 큰 우산을 씌워주는 장면", "지역 경제와 고용 유지에 영향을 줄 수 있습니다"),
    ("부동산 대출 규제 변화 뉴스", "집을 살 때 빌릴 수 있는 돈의 문턱이 달라지는 문제", "놀이기구 키 제한이 바뀌면 탈 수 있는 사람이 달라지는 것과 비슷합니다", "주택 수요와 가계부채, 시장 안정에 연결됩니다"),
    ("농산물 가격 변동 뉴스", "밥상에 올라오는 재료 가격이 날씨와 유통 상황에 따라 흔들리는 이야기", "김치찌개 재료들이 각자 다른 롤러코스터를 타는 느낌", "가계 장바구니와 농가 소득 모두에 영향을 줍니다"),
    ("택배 자동화 확대 뉴스", "물류센터에서 사람이 하던 반복 작업을 기계가 더 많이 맡는 흐름", "상자들이 컨베이어벨트 위에서 자기 길을 찾아가는 미니 도시 같습니다", "배송 속도와 노동 환경, 일자리 구조가 함께 바뀔 수 있습니다"),
    ("해외여행 수요 증가 뉴스", "사람들이 다시 여권을 꺼내고 여행 계획을 세우는 분위기", "서랍 속에서 잠자던 캐리어가 드디어 출근 준비를 하는 장면", "항공, 숙박, 환율, 소비 흐름에 연쇄 효과가 생길 수 있습니다"),
    ("택시 호출 앱 경쟁 뉴스", "택시를 부르는 화면 안에서 플랫폼들이 손님 잡기 경쟁을 벌이는 일", "길거리 호객 경쟁이 스마트폰 앱 안으로 들어온 셈", "요금, 배차 속도, 기사 수익에 모두 영향을 줄 수 있습니다"),
    ("의료 인력 부족 뉴스", "병원 현장의 일손과 환자 수요가 잘 맞지 않는다는 신호", "식당에 손님은 몰리는데 주방 인력이 부족한 상황과 비슷합니다", "진료 대기와 지역 의료 접근성에 직접 영향을 줍니다"),
    ("기후 변화 대응 회의 뉴스", "지구 온도라는 공용 온도계를 어떻게 낮출지 나라들이 의논하는 자리", "같은 아파트에 사는 사람들이 난방비와 환기 규칙을 함께 정하는 느낌", "산업 정책과 에너지 비용, 미래 재난 위험이 함께 달려 있습니다"),
    ("대형마트 새벽배송 논의 뉴스", "장을 보는 시간을 더 넓힐지, 기존 상권을 어떻게 보호할지 따지는 이슈", "편의성이라는 빠른 엘리베이터와 골목상권이라는 계단을 같이 보는 문제", "소비자 편의와 유통업 경쟁, 소상공인 보호가 맞물려 있습니다"),
    ("학교 급식 물가 상승 뉴스", "학생들의 한 끼 식판을 채우는 비용이 높아지고 있다는 이야기", "급식판 위 반찬들이 조용히 가격표를 들고 있는 장면", "교육 예산과 식사 품질, 학부모 부담에 영향을 줄 수 있습니다"),
    ("로봇 배송 실험 뉴스", "사람 대신 로봇이 짐을 들고 동네를 다니는 서비스 실험", "작은 캐리어가 길을 외워서 혼자 심부름을 가는 느낌", "편리함과 안전 규칙, 보행 공간 관리가 함께 중요해집니다"),
    ("온라인 교육 플랫폼 성장 뉴스", "배움의 교실이 학교 건물 밖 화면으로 더 넓어지는 흐름", "책상 하나가 전국의 강의실 문을 열 수 있는 리모컨이 된 셈", "교육 접근성과 콘텐츠 품질, 학습 관리 방식이 달라질 수 있습니다"),
]

data = []
for input_text, hook, metaphor, why in base_examples:
    data.append({
        "instruction": NEWS_INSTRUCTION,
        "input": input_text,
        "output": f"핵심은 {hook}입니다. 쉽게 말해 {metaphor}. 그래서 {why}.",
    })
    data.append({
        "instruction": NEWS_INSTRUCTION,
        "input": input_text + "를 초등학생도 이해하게 설명해줘",
        "output": f"오늘의 관전 포인트는 {hook}예요. 비유하자면 {metaphor}. 이게 중요한 이유는 {why}.",
    })

dataset = Dataset.from_list(data).shuffle(seed=42)

# ✅ 포맷 변환 적용
# ➤ 전체 데이터에 format_instruction 함수를 일괄 적용합니다.
train_data = dataset.map(format_instruction, batched=False)  # 포인트: batched=False — 샘플 하나씩 변환

print(f"학습 데이터 수: {len(train_data)}개")
print(f"\n{'='*60}")
print("변환된 샘플 예시:")
print('='*60)
print(train_data[0]["text"])


# ## 6단계. 파인튜닝 전 베이스 모델 답변 확인
# 
# 파인튜닝 **전**에 베이스 모델이 어떻게 답하는지 먼저 기록합니다.  
# 파인튜닝 후와 비교하기 위한 기준점(baseline)입니다.
# 

# In[ ]:


# ✅ STEP 7: 추론 함수 정의 (파인튜닝 전후 비교에 사용)
# ➤ 프롬프트를 구성하고 모델에서 응답을 생성하는 과정을 하나의 함수로 묶습니다.
def generate_response(model, tokenizer, news_issue, max_new_tokens=120):
    # ✅ 추론용 프롬프트 구성
    # ➤ Response 칸은 비워두고 모델이 직접 채우도록 합니다.
    prompt = alpaca_prompt.format(
        NEWS_INSTRUCTION,
        news_issue,  # ✏️ 설명할 뉴스 이슈(news_issue)를 넣으세요
        ""    # 포인트: Response 칸을 비워두면 모델이 이 자리를 채워 생성합니다
    )

    # ✅ 텍스트 → 토큰 변환 후 GPU로 이동
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # 포인트: .to(device) 필수 — GPU에서 추론

    with torch.no_grad():  # 포인트: 추론 시 gradient 계산 불필요 — 메모리 절약
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # 포인트: 생성할 최대 토큰 수 제한
            do_sample=False,                # 포인트: 확률 샘플링 없이 가장 확률 높은 토큰 선택 → 재현성 확보
            pad_token_id=tokenizer.eos_token_id,
        )

    # ✅ 입력 프롬프트 부분을 제거하고 생성된 부분만 디코딩
    # ➤ inputs["input_ids"].shape[1] 이후부터가 모델이 새로 생성한 토큰들입니다.
    generated = outputs[0][inputs["input_ids"].shape[1]:]  # 포인트: 슬라이싱으로 프롬프트 제거
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ✅ 파인튜닝 전후 비교에 사용할 테스트 케이스
test_cases = [
    {"issue": "물가가 다시 오른다는 뉴스",
     "target": "장바구니 부담을 쉬운 비유로 설명"},
    {"issue": "AI 규제가 논의되고 있다는 뉴스",
     "target": "AI 안전 규칙을 쉬운 비유로 설명"},
    {"issue": "환율이 올랐다는 뉴스",
     "target": "달러 비용 증가를 쉬운 비유로 설명"},
    {"issue": "전기차 보조금 기준이 바뀐다는 뉴스",
     "target": "보조금 조건 변화를 쉬운 비유로 설명"},
    {"issue": "사이버 보안 사고 뉴스",
     "target": "디지털 문단속 문제를 쉬운 비유로 설명"},
]

print("="*65)
print("[BEFORE] 파인튜닝 전 베이스 모델 답변")
print("="*65)
for tc in test_cases:
    result = generate_response(model, tokenizer, tc["issue"])
    is_news_style = all(keyword in result for keyword in ["핵심", "쉽게", "그래서"])
    mark = "🟡 형식 일부 충족" if is_news_style else "❌ 원하는 뉴스 톤 아님"
    print(f"뉴스  : {tc['issue']}")
    print(f"목표  : {tc['target']}")
    print(f"출력  : {result[:120]}")
    print(f"판정  : {mark}")
    print("-"*65)


# ## 7단계. 학습 설정 및 SFTTrainer 구성
# 
# **SFTTrainer**는 LLM 생성 파인튜닝에 특화된 Trainer입니다.  
# `dataset_text_field`만 지정하면 토크나이징, 배치 구성, 손실 계산을 모두 자동으로 처리합니다.
# 
# ---
# ### dataset_text_field 지정
# 
# `SFTTrainer`에게 학습 데이터에서 **어떤 컬럼을 텍스트로 쓸지** 알려줘야 합니다.  
# `format_instruction` 함수에서 어떤 키로 텍스트를 저장했는지 확인해보세요.
# 
# ---

# In[ ]:


# ✅ STEP 8: 학습 설정 및 SFTTrainer 구성
# ➤ 학습 방식(배치 크기, 학습률, 에폭 등)을 설정하고 Trainer를 초기화합니다.
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    per_device_train_batch_size = 4,     # 포인트: GPU 하나가 한 번에 처리할 샘플 수
    gradient_accumulation_steps = 4,     # 포인트: 4번 모아 큰 배치(4×4=16)처럼 학습 — 메모리 절약
    num_train_epochs = 3,                # 포인트: 전체 데이터를 3바퀴 반복 학습
    max_steps = 100,                     # 포인트: 빠른 테스트용 상한 (실제 학습 시 이 줄 제거)
    learning_rate = 2e-4,                # 포인트: 가중치 업데이트 크기 — 너무 크면 불안정, 너무 작으면 느림
    warmup_steps = 10,                   # 포인트: 초반 10 step 동안 학습률을 서서히 올려 안정적인 시작
    bf16 = is_bfloat16_supported(),      # 포인트: A100 등 최신 GPU에서 bfloat16 사용 (속도↑, 안정성↑)
    fp16 = not is_bfloat16_supported(),  # 포인트: T4 등 구형 GPU에서는 float16 사용
    logging_steps = 10,                  # 포인트: 10 step마다 loss 로그 출력
    optim = "adamw_8bit",               # 포인트: 8bit AdamW — 메모리 절약 + 속도 개선
    weight_decay = 0.01,                 # 포인트: 가중치에 패널티를 주어 과적합 방지
    lr_scheduler_type = "linear",        # 포인트: 학습률을 선형으로 감소 — 후반 과도한 업데이트 방지
    output_dir = "outputs",
    seed = 42,
    report_to = "none"                   # 포인트: wandb 등 외부 로깅 도구 비활성화
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_data,
    dataset_text_field = "text",  #텍스트가 담긴 컬럼 이름 — format_instruction에서 어떤 키를 썼나요?
    max_seq_length = 2048,
    dataset_num_proc = 2,        # 포인트: 전처리 병렬 처리 수
    packing = False,
    args = training_args,
)

print("✅ SFTTrainer 구성 완료")


# ## 8단계. 학습 시작
# 
# 학습 로그에서 **`loss` 값이 점점 줄어드는지** 확인해보세요.  
# loss가 낮아질수록 모델이 요다 말투 패턴을 학습하고 있다는 의미입니다.
# 
# > 예상 소요 시간: Colab T4 기준 약 **3~5분** (max_steps=100)

# In[ ]:


# ✅ STEP 9: 학습 실행
# ➤ trainer.train() 한 줄로 순전파 → 손실 계산 → 역전파 → 파라미터 업데이트가 자동 반복됩니다.
trainer_stats = trainer.train()

print(f"\n✅ 학습 완료!")
print(f"총 학습 시간: {trainer_stats.metrics['train_runtime']:.1f}초")
print(f"최종 Loss   : {trainer_stats.metrics['train_loss']:.4f}")


# ## 9단계. 파인튜닝 후 비교
# 
# ---
# ###FastLanguageModel 추론 모드 전환
# 
# 파인튜닝이 끝난 모델을 추론에 최적화된 모드로 전환해야 합니다.  
# Unsloth에서 추론 모드로 전환하는 메서드를 찾아보세요.
# 
# ---

# In[ ]:


# ✅ STEP 10: 파인튜닝 후 모델 답변 확인
# ➤ 파인튜닝 전과 동일한 테스트 케이스로 비교합니다.

# 모델을 추론(inference) 모드로 전환하세요
# 힌트: FastLanguageModel.for_inference(model)
# 추론 모드에서는 학습 관련 연산이 꺼지고 속도가 빨라집니다.
FastLanguageModel.for_inference(model)

print("="*65)
print("[AFTER] 파인튜닝 후 모델 답변")
print("="*65)
for tc in test_cases:
    result = generate_response(model, tokenizer, tc["issue"])
    # 뉴스 설명 톤 특징: 핵심 제시, 쉬운 비유, 중요성 설명
    is_news_style = all(keyword in result for keyword in ["핵심", "쉽게", "그래서"]) or all(keyword in result for keyword in ["관전 포인트", "비유", "중요"])
    mark = "✅ 뉴스 설명 톤!" if is_news_style else "❌ 형식 보완 필요"
    print(f"뉴스  : {tc['issue']}")
    print(f"목표  : {tc['target']}")
    print(f"출력  : {result[:120]}")
    print(f"판정  : {mark}")
    print("-"*65)


# ## 10단계. 새 문장으로 직접 테스트해보기
# 
# ---
# ### ✏️ 빈칸 10 — 직접 문장을 넣어보세요
# 
# `my_sentence` 변수에 원하는 영어 문장을 입력하고,  
# 모델이 요다 말투로 어떻게 바꾸는지 확인해보세요!
# 
# ---

# In[ ]:


# ✅ STEP 11: 새 문장으로 직접 테스트
# ➤ 학습 데이터에 없던 새로운 뉴스를 넣어 스타일 학습 여부를 확인합니다.
#    모델이 단순히 외운 게 아니라 뉴스 설명 톤 자체를 학습했다면 새 이슈도 잘 설명합니다.

my_sentence = "대중교통 요금 인상 뉴스"  # ✏️ 빈칸 10: 재밌게 설명해보고 싶은 뉴스 이슈를 직접 입력해보세요!
                      # 예: "디지털 교과서 도입 뉴스" / "로봇 배송 실험 뉴스" 등

result = generate_response(model, tokenizer, my_sentence)

print(f"뉴스   : {my_sentence}")
print(f"설명   : {result}")
print()
print("직접 추가로 테스트해보세요:")
custom_tests = [
    "디지털 교과서 도입 뉴스",
    "로봇 배송 실험 뉴스",
    "온라인 교육 플랫폼 성장 뉴스",
]
for sent in custom_tests:
    out = generate_response(model, tokenizer, sent)
    print(f"  뉴스: {sent}")
    print(f"  설명: {out}")
    print()


# ## 11단계. LoRA 어댑터 저장
# 
# Full 모델 전체(수 GB)가 아닌, 학습한 **A, B 행렬만** 저장합니다.  
# 크기가 매우 작습니다(수 MB ~ 수십 MB).
# 
# ---
# ### ✏️ 빈칸 11 — 저장 경로 지정
# 
# 어댑터를 저장할 폴더 이름을 정해보세요. (예: `"./yoda-lora"`)
# 
# ---

# In[ ]:


# ✅ STEP 12: LoRA 어댑터 저장
# ➤ 전체 모델이 아닌 학습된 A, B 행렬(어댑터)만 저장합니다.
#    용량이 작고 나중에 베이스 모델 위에 다시 얹어 사용할 수 있습니다.
import os

SAVE_PATH = "./news-fun-style-lora"  # ✏️ 빈칸 11: 저장할 경로를 입력하세요 (예: "./news-fun-style-lora")

model.save_pretrained(SAVE_PATH)    # 포인트: LoRA A, B 행렬만 저장 (전체 모델 아님)
tokenizer.save_pretrained(SAVE_PATH)

print(f"✅ 저장 완료: {SAVE_PATH}")
print("\n저장된 파일 목록:")
for f in sorted(os.listdir(SAVE_PATH)):
    size = os.path.getsize(f"{SAVE_PATH}/{f}")
    print(f"  {f:45s}: {size/1024:.1f} KB")
print()
print("💡 adapter_model.safetensors가 실제 학습된 LoRA 가중치입니다.")
print("   전체 모델 대비 매우 작은 크기임을 확인하세요!")


# In[ ]:


# ✅ (선택) HuggingFace Hub에 업로드
# ➤ Colab 세션이 종료되면 로컬 파일이 사라집니다.
#    HF Hub에 올려두면 언제든 다시 불러올 수 있고 팀원과 공유도 가능합니다.

# from google.colab import userdata
# from huggingface_hub import login

# 토큰 발급: https://huggingface.co/settings/tokens (write 권한 필요)
# Colab 왼쪽 열쇠 아이콘에서 HF_TOKEN을 등록한 뒤 주석을 해제하세요.
# login(token=userdata.get("HF_TOKEN"))

# HF_REPO_ID = "rud472888-creator/news-fun-style-qwen-lora"
# model.push_to_hub(HF_REPO_ID)
# tokenizer.push_to_hub(HF_REPO_ID)

# print("업로드 완료!")
# print(f"→ https://huggingface.co/{HF_REPO_ID}")


# ## ✅ 실습 2 완료!
# 
# ### 전체 흐름 정리
# 
# ```
# 1. 사전학습 모델 선택
#          ↓
# 2. 데이터셋 준비 (포맷 변환, 전처리)
#          ↓
# 3. 토크나이저 설정 (모델과 동일한 토크나이저, 텍스트 → 토큰)
#          ↓
# 4. 파인튜닝 방식 선택 (Full / LoRA 등, 자원 상황에 따라)
#          ↓
# 5. 하이퍼파라미터 설정 (실험적으로 조정)
#          ↓
# 6. 학습 수행 (Trainer / SFTTrainer)
#          ↓
# 7. 모델 저장 및 배포 (로컬 또는 Hugging Face Hub)
# ```
# 
# ### 핵심 포인트 3가지
# 
# | 포인트 | 설명 |
# |---|---|
# | **LoRA의 효율성** | 전체 파라미터의 ~1%만 학습해도 말투 변환이 가능 |
# | **스타일 파인튜닝의 원리** | 베이스 모델이 일정한 설명 톤을 따르도록 입력-출력 쌍으로 주입 |
# | **데이터 품질이 핵심** | 적은 데이터라도 출력 구조가 일관되면 원하는 말투를 더 잘 배울 수 있다 |
# 
# ### 실제 응용: 뉴스 설명 말투 학습
# 
# 요다 실습에서 배운 원리를 그대로 응용했습니다.
# 
# ```python
# # 뉴스 설명 스타일 데이터 예시
# {
#     "input": "물가가 다시 오른다는 뉴스",
#     "output": "핵심은 장바구니 부담이 커지는 상황입니다. 쉽게 말해 같은 돈으로 살 수 있는 물건이 줄어드는 거예요. 그래서 가계 소비와 정책 판단에 영향을 줍니다."
# }
# ```
# 
# **데이터 만드는 법:** Claude/GPT에 원하는 말투를 설명하고 쌍 데이터 생성 요청 → 손으로 품질 수정 → LoRA 파인튜닝
# 

# In[ ]:


# =====================================================================
# 🚀 퀵 스타트: 허브에 올려둔 내 파인튜닝 모델 바로 써보기
# 앞선 학습 과정을 생략하고, 허브에서 모델을 다운받아 바로 추론합니다.
# =====================================================================

# 아직 HuggingFace Hub에 업로드하지 않았다면 이 셀은 실행하지 않아도 됩니다.
# 업로드 후 아래 HF_MODEL_ID를 본인 repo id로 바꾸고 주석을 해제하세요.

# from unsloth import FastLanguageModel
# import torch

# HF_MODEL_ID = "rud472888-creator/news-fun-style-qwen-lora"

# print("📥 모델 다운로드 및 로딩 중... (약 1~2분 소요)")
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = HF_MODEL_ID,
#     max_seq_length = 2048,
#     dtype = torch.float16,
#     load_in_4bit = True,
# )

# FastLanguageModel.for_inference(model)

# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# 너는 뉴스를 쉽고 재밌게 설명하는 진행자다. 입력된 뉴스 이슈를 3문장으로 설명하라. 1문장은 핵심, 2문장은 쉬운 비유, 3문장은 왜 중요한지다. 없는 사실이나 숫자는 만들지 않는다.

# ### Input:
# {}

# ### Response:
# """

# test_cases = [
#     "물가가 다시 오른다는 뉴스",
#     "AI 규제가 논의되고 있다는 뉴스",
#     "사이버 보안 사고 뉴스",
# ]

# print("\n" + "="*60)
# print("✨ 뉴스 설명 모델 추론 결과 ✨")
# print("="*60)

# for input_text in test_cases:
#     prompt = alpaca_prompt.format(input_text)
#     inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
#     outputs = model.generate(**inputs, max_new_tokens=120, use_cache=True)
#     result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     final_answer = result.split("### Response:\n")[-1].strip()
#     print(f"뉴스: {input_text}")
#     print(f"설명: {final_answer}")
#     print("-" * 60)

