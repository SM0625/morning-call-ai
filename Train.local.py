"""모닝콜 AI 미세 튜닝 (PC용, RTX 3060 12GB 기준)

Smoked-Salmon-s/empathetic_dialogues_ko 로 gemma3:4b 를 LoRA 학습해서
"공감 → 말 걸기 → 질문"으로 대화를 이끌게 만든다. 결과는 Ollama용 GGUF + Modelfile.

WSL(Ubuntu)에서 실행:
    python train_local.py

같은 폴더에 my_wake.jsonl 이 있으면 내가 만든 핑계 대화도 섞는다.
결과물은 윈도우의 C:\\Users\\bi201\\morning-gemma 에 저장된다.
"""
import glob
import json
import os
import random
import re
import shutil

# ─── 설정 ────────────────────────────────────────────────────────────────────────
N_SAMPLES = 3000                                   # 학습에 쓸 대화 수
OUT_DIR = "/mnt/c/Users/bi201/morning-gemma"       # 윈도우 쪽 저장 위치
SYSTEM = "당신은 사용자에게 존댓말로 친근하게 말하며, 공감하고 질문을 던져 대화를 이끌어가는 AI입니다."
WAKE_SYSTEM = "당신은 민수님을 아침에 깨우는 모닝콜 AI입니다. 존댓말로 짧게 1~2문장만 말하고, 일어나도록 이끄세요."
TESTS = ["5분만 더 잘게", "어제 늦게 자서 너무 피곤해", "오늘 회사 가기 싫다"]

# ─── 1. 데이터 ───────────────────────────────────────────────────────────────────
from datasets import Dataset, load_dataset

raw = load_dataset("Smoked-Salmon-s/empathetic_dialogues_ko", split="train")
print(raw)
print("첫 줄:", raw[0])

USER_KEYS = ("question", "instruction", "input", "user", "human", "q")
BOT_KEYS = ("answer", "output", "response", "assistant", "gpt", "bot", "a")


def role_of(key):
    k = re.sub(r"[\d_\-\s]+", "", key.lower())
    if k in USER_KEYS:
        return "user"
    if k in BOT_KEYS:
        return "assistant"
    return None


def turn_no(key):
    m = re.search(r"\d+", key)
    return int(m.group()) if m else 0


def to_messages(row):
    # 경우 1: 대화가 리스트로 들어 있는 경우 (messages / conversations)
    for v in row.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            msgs = []
            for t in v:
                r = t.get("role") or t.get("from")
                c = t.get("content") or t.get("value")
                r = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(r, r)
                if r in ("user", "assistant") and c:
                    msgs.append({"role": r, "content": c.strip()})
            return msgs
    # 경우 2: question1/answer1/... 처럼 열이 나뉘어 있는 경우
    turns = []
    for k, v in row.items():
        r = role_of(k)
        if r and isinstance(v, str) and v.strip():
            turns.append((turn_no(k), 0 if r == "user" else 1, r, v.strip()))
    turns.sort()
    return [{"role": r, "content": c} for _, _, r, c in turns]


POLITE = re.compile(r"(요|니다|니까|죠|세요|까요)[.!?~…\s)]*$")


def ok(msgs):
    if len(msgs) < 2 or msgs[0]["role"] != "user":
        return False
    if any(a["role"] == b["role"] for a, b in zip(msgs, msgs[1:])):
        return False                                       # 순서가 꼬인 대화
    return all(POLITE.search(m["content"]) for m in msgs if m["role"] == "assistant")  # 반말 대답 제외


convs = [m for m in (to_messages(r) for r in raw) if ok(m)]
if not convs:
    raise SystemExit("변환된 대화가 0개예요. 위에 출력된 '첫 줄'의 열 이름을 Claude에게 보여주세요.")
convs = [m if m[-1]["role"] == "assistant" else m[:-1] for m in convs]
print(f"사용 가능한 대화: {len(convs)}개 / 전체 {len(raw)}개")

random.seed(0)
random.shuffle(convs)
data = [{"messages": [{"role": "system", "content": SYSTEM}] + m} for m in convs[:N_SAMPLES]]

if os.path.exists("my_wake.jsonl"):
    mine = [json.loads(l) for l in open("my_wake.jsonl", encoding="utf-8") if l.strip()]
    for d in mine:
        if d["messages"][0]["role"] != "system":
            d["messages"].insert(0, {"role": "system", "content": SYSTEM})
    data += mine * 3                                       # 양이 적으니 3번 반복
    random.shuffle(data)
    print(f"내 대화 {len(mine)}개 추가 → 총 {len(data)}개")

# ─── 2. 모델 (4비트 + LoRA) ──────────────────────────────────────────────────────
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

model, tokenizer = FastModel.from_pretrained("unsloth/gemma-3-4b-it", max_seq_length=1024, load_in_4bit=True)
model = FastModel.get_peft_model(
    model, finetune_vision_layers=False, finetune_language_layers=True,
    r=16, lora_alpha=16, lora_dropout=0, random_state=0,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")


def ask(text):
    FastModel.for_inference(model)
    msgs = [{"role": "system", "content": WAKE_SYSTEM}, {"role": "user", "content": text}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                        return_dict=True).to("cuda")
    out = model.generate(**ids, max_new_tokens=80, temperature=0.7, top_p=0.95, do_sample=True)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


before = {t: ask(t) for t in TESTS}

# ─── 3. 학습 ─────────────────────────────────────────────────────────────────────
from trl import SFTConfig, SFTTrainer

ds = Dataset.from_list(data).map(
    lambda b: {"text": [tokenizer.apply_chat_template(m, tokenize=False).removeprefix("<bos>")
                        for m in b["messages"]]},
    batched=True,
)
FastModel.for_training(model)
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text", per_device_train_batch_size=4, gradient_accumulation_steps=2,
        num_train_epochs=1, learning_rate=2e-4, warmup_steps=10, lr_scheduler_type="linear",
        logging_steps=20, optim="adamw_8bit", seed=0, report_to="none", output_dir="outputs",
    ),
)
# AI 대답 부분만 배우게 한다 (사용자 말까지 따라 하지 않게)
trainer = train_on_responses_only(trainer, instruction_part="<start_of_turn>user\n",
                                  response_part="<start_of_turn>model\n")
trainer.train()

# ─── 4. 전후 비교 ────────────────────────────────────────────────────────────────
print("\n===== 학습 전/후 비교 =====")
for t in TESTS:
    print(f"👤 {t}\n  전: {before[t]}\n  후: {ask(t)}\n")

# ─── 5. Ollama용 GGUF + Modelfile ────────────────────────────────────────────────
model.save_pretrained_gguf("morning-gemma", tokenizer, quantization_method="q4_k_m")
ggufs = [f for f in glob.glob("morning-gemma*/*.gguf") + glob.glob("*.gguf") if "mmproj" not in f.lower()]
q4 = [f for f in ggufs if "q4_k_m" in f.lower()] or ggufs
if not q4:
    raise SystemExit("GGUF 파일을 못 찾았어요. 위 로그의 오류를 Claude에게 보여주세요.")
os.makedirs(OUT_DIR, exist_ok=True)
shutil.copy(q4[0], os.path.join(OUT_DIR, "morning-gemma.gguf"))

MODELFILE = '''FROM ./morning-gemma.gguf
TEMPLATE """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}"""
PARAMETER stop "<end_of_turn>"
PARAMETER temperature 0.7
'''
with open(os.path.join(OUT_DIR, "Modelfile"), "w", encoding="utf-8") as f:
    f.write(MODELFILE)
print(f"\n완료! 윈도우 {OUT_DIR} 에 morning-gemma.gguf 와 Modelfile 저장됨")