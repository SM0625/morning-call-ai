"""모델별 대화 품질 비교. 서버·폰이 같이 쓰는 대화 엔진(engine/dialog.py)에 모델만 바꿔 같은 시나리오를 돌린다.

폰에서 돌릴 작은 모델(gemma3:1b, gemma3n:e2b, gemma4:e2b)이 깨우는 대화를 해 낼 수 있는지 보려고 만들었다.
속도는 이 PC 기준이라 폰 속도와 다르다. 품질(반말·외국어·포기 발언·대체 문장)만 비교에 쓴다.

    python tools/eval_models.py gemma3:4b gemma3:1b            # 결과 표 + 대화 원문(tools/eval-결과.md)
"""
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import main  # noqa: E402  (engine/ 경로 설정, Ollama 스트림)
import dialog  # noqa: E402

USER = {"user_name": "민수", "user_schedule": "오전 10시 팀 미팅", "weather_info": "흐림 12°C, 강수확률 60%"}

# README "반응 속도" 측정과 같은 네 가지 상황
SCENARIOS = {
    "협조적": ["음... 누구세요", "아 일어났어", "씻으러 왔어", "이제 끊어줘"],
    "버팀": ["5분만 더", "졸려 죽겠어", "오늘 재택하면 안되나", "알았어 일어날게", "일어났다니까"],
    "딴 얘기": ["오늘 비 와?", "어제 영화 진짜 재밌었어", "오늘 일정 뭐였지?", "일어났어", "세수했어"],
    "대답 없음": [dialog.SILENT_MESSAGE, dialog.SILENT_MESSAGE, "응...", "일어났어"],
}
# 말을 멈추게 하는 심각한 문제 (dialog.STOP_ISSUES와 같은 기준)
SERIOUS = {"반말", "외국어 또는 깨진 조사", "포기 발언"}
FALLBACKS = [line.format(name=USER["user_name"]) for line in dialog.FALLBACK_LINES["wake"]] + [dialog.FALLBACK_LINES["end"]]


def think_off_stream(model):
    """Gemma 4는 기본으로 길게 '생각'한 뒤 답한다. 폰(PhoneAi.kt)처럼 생각을 끈 채로 잰다.
    서버의 ollama 라이브러리(0.4.4)는 think 설정이 없어 HTTP로 직접 부른다"""
    def stream(messages):
        body = {"model": model, "messages": messages, "stream": True, "think": False,
                "options": {**main.GENERATION_OPTIONS, "num_gpu": main.NUM_GPU}}
        with httpx.stream("POST", "http://127.0.0.1:11434/api/chat", json=body, timeout=300) as r:
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)["message"].get("content", "")
    return stream


def model_stream(model):
    return think_off_stream(model) if model.startswith("gemma4") else main.ollama_stream(model)


def run_turn(model, system, messages, user_msg, stage, first=False):
    start = time.time()
    first_at = None
    phrases = []
    for phrase in dialog.stream_reply(model_stream(model), system, messages, user_msg, stage, user_name=USER["user_name"], first=first):
        if first_at is None:
            first_at = time.time() - start
        phrases.append(phrase)
    reply = " ".join(phrases)
    # 엔진이 고친 뒤의 최종 대답을 다시 검사한다. 대체 문장이 붙었으면 모델이 제대로 말하지 못한 것
    issues = dialog.reply_issues(reply, user_msg, stage, first)
    return {
        "user": user_msg, "reply": reply, "stage": dialog.STAGE_LABELS[stage],
        "first_s": round(first_at or 0, 2), "total_s": round(time.time() - start, 2),
        "serious": sorted(SERIOUS & set(issues)),
        "no_presence": "깨우는 중이라는 표현 없음" in issues,
        "fallback": any(f in reply for f in FALLBACKS),
    }


def run_scenario(model, turns):
    system, messages = dialog.initial_turn(USER["user_name"], USER["user_schedule"], USER["weather_info"])
    greeting = run_turn(model, system, messages, "", 0, first=True)
    history = [{"role": "user", "content": messages[0]["content"]}, {"role": "assistant", "content": greeting["reply"]}]
    results, stage = [greeting], 0
    for text in turns:
        request = dialog.TurnRequest(message=text, history=history, conversation_level=stage, **USER)
        stage, system, messages = dialog.prepare_turn(request)
        result = run_turn(model, system, messages, text, stage)
        results.append(result)
        # 앱처럼, 서버가 실제로 모델에 보낸 내용(지시문 포함)을 대화 기록에 남긴다
        history += [{"role": "user", "content": messages[-1]["content"]}, {"role": "assistant", "content": result["reply"]}]
        if stage == dialog.LAST_STAGE:
            break
    return results


def summarize(model, runs):
    turns = [t for scenario in runs.values() for t in scenario]
    firsts = sorted(t["first_s"] for t in turns)
    return {
        "model": model, "turns": len(turns),
        "serious": sum(bool(t["serious"]) for t in turns),
        "no_presence": sum(t["no_presence"] for t in turns),
        "fallback": sum(t["fallback"] for t in turns),
        "first_median_s": firsts[len(firsts) // 2],
    }


def main_cli(models):
    out = os.path.join(os.path.dirname(__file__), "eval-결과.md")
    summaries, lines = [], []
    for model in models:
        runs = {name: run_scenario(model, turns) for name, turns in SCENARIOS.items()}
        summary = summarize(model, runs)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        lines.append(f"\n## {model}\n")
        for name, turns in runs.items():
            lines.append(f"\n### {name}\n")
            for t in turns:
                marks = " ".join(t["serious"] + (["깨움 표현 없음"] if t["no_presence"] else [])
                                 + (["대체 문장"] if t["fallback"] else []))
                lines.append(f"- 👤 {t['user'] or '(알람)'} → 🤖 {t['reply']}  `{t['stage']} {t['first_s']}s` {marks}")
    table = ["| 모델 | 턴 | 심각한 문제(반말·외국어·포기) | 깨움 표현 없음 | 대체 문장 | 첫 구절 중간값(이 PC) |",
             "|---|---|---|---|---|---|"]
    table += [f"| {s['model']} | {s['turns']} | {s['serious']} | {s['no_presence']} | {s['fallback']} | {s['first_median_s']}초 |"
              for s in summaries]
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 모델별 대화 품질 비교\n\n" + "\n".join(table) + "\n" + "\n".join(lines) + "\n")
    print("\n".join(table))
    print(f"\n대화 원문: {out}")


if __name__ == "__main__":
    main_cli(sys.argv[1:] or [main.DEFAULT_MODEL])
