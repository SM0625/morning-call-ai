"""모델 버전끼리 대화 품질을 비교한다 (v1 vs v2 같은 미세조정 모델). 서버 코드 없이 대화 엔진만 쓰므로 Kaggle·Colab에서도 돈다.

tools/eval_models.py(상황 4가지)와 tools/eval_excuses.py(핑계 10가지)와 같은 문장으로, 같은 대화 엔진을 거친 최종 대답을 잰다.
모델은 stream(messages) -> 글자 조각들 함수로 받는다 (Ollama든 transformers든).

잰 것 (작을수록 좋다)
- 심각: 최종 대답에 반말·외국어·포기 발언이 남은 턴
- 깨움 없음: 깨우는 중이라는 표현이 없는 턴
- 대체 문장: 검사에 걸려 엔진이 대체 문장으로 끝낸 턴 (이미 말한 앞부분 뒤에 붙어 어색하게 들린다)
- 다시 만듦: 모델의 첫 대답이 검사에 걸려 엔진이 다시 생성한 횟수 (첫 소리가 늦어지는 원인 = 끊김)
- 핑계 넘어감: 핑계 10가지에 넘어간 최종 대답
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))
import dialog  # noqa: E402

USER = {"user_name": "민수", "user_schedule": "오전 10시 팀 미팅", "weather_info": "흐림 12°C, 강수확률 60%"}
# tools/eval_models.py SCENARIOS, tools/eval_excuses.py EXCUSES와 같다 (그쪽은 서버 코드를 불러와서 여기 따로 둔다)
SCENARIOS = {
    "협조적": ["음... 누구세요", "아 일어났어", "씻으러 왔어", "이제 끊어줘"],
    "버팀": ["5분만 더", "졸려 죽겠어", "오늘 재택하면 안되나", "알았어 일어날게", "일어났다니까"],
    "딴 얘기": ["오늘 비 와?", "어제 영화 진짜 재밌었어", "오늘 일정 뭐였지?", "일어났어", "세수했어"],
    "대답 없음": [dialog.SILENT_MESSAGE, dialog.SILENT_MESSAGE, "응...", "일어났어"],
}
EXCUSES = [
    "오늘 재택하면 안되나", "5분만 더 잘게", "오늘 휴강일 것 같은데", "조금만 더 누워 있을래", "늦어도 괜찮지 않아?",
    "오늘 결석할까", "10분만 더", "눈 좀 감고 있을게", "주말이었으면 좋겠다", "너무 피곤해서 못 일어나겠어",
]
GREETING = "민수님, 안녕하세요! 지금 전화로 깨우고 있어요. 눈 좀 떠 보세요."
SERIOUS = {"반말", "외국어 또는 깨진 조사", "포기 발언"}
FALLBACKS = [l.format(name=USER["user_name"]) for l in dialog.FALLBACK_LINES["wake"]] + [dialog.FALLBACK_LINES["end"]]


def _turn(stream, system, messages, user_msg, stage, first=False):
    calls = [0]

    def counted(msgs):
        calls[0] += 1
        return stream(msgs)

    reply = " ".join(dialog.stream_reply(counted, system, messages, user_msg, stage,
                                         user_name=USER["user_name"], first=first))
    issues = dialog.reply_issues(reply, user_msg, stage, first)
    return {"user": user_msg, "reply": reply, "stage": dialog.STAGE_LABELS[stage],
            "serious": sorted(SERIOUS & set(issues)), "no_presence": "깨우는 중이라는 표현 없음" in issues,
            "fallback": any(f in reply for f in FALLBACKS), "retries": calls[0] - 1}


def run_scenarios(stream):
    rows = []
    for name, turns in SCENARIOS.items():
        system, messages = dialog.initial_turn(USER["user_name"], USER["user_schedule"], USER["weather_info"])
        t = _turn(stream, system, messages, "", 0, first=True)
        rows.append({**t, "scenario": name})
        history, stage = [messages[0], {"role": "assistant", "content": t["reply"]}], 0
        for text in turns:
            request = dialog.TurnRequest(message=text, history=history, conversation_level=stage, **USER)
            stage, system, msgs = dialog.prepare_turn(request)
            t = _turn(stream, system, msgs, text, stage)
            rows.append({**t, "scenario": name})
            history += [msgs[-1], {"role": "assistant", "content": t["reply"]}]
            if stage == dialog.LAST_STAGE:
                break
    return rows


def run_excuses(stream):
    system, messages = dialog.initial_turn(USER["user_name"], USER["user_schedule"], USER["weather_info"])
    history = [messages[0], {"role": "assistant", "content": GREETING}]
    rows = []
    for text in EXCUSES:
        request = dialog.TurnRequest(message=text, history=history, conversation_level=0, **USER)
        stage, system, msgs = dialog.prepare_turn(request)
        t = _turn(stream, system, msgs, text, stage)
        t["gave_in"] = "포기 발언" in dialog.reply_issues(t["reply"], text, stage, False)
        rows.append({**t, "scenario": "핑계"})
    return rows


def evaluate(stream, repeats=2, log=print):
    """상황 4가지와 핑계 10가지를 repeats번 돌린다 (대답이 매번 달라서). (요약, 원문 행들)"""
    scen, exc = [], []
    for r in range(repeats):
        scen += run_scenarios(stream)
        exc += run_excuses(stream)
        log(f"  {r + 1}/{repeats}회 끝")
    rows = scen + exc
    summary = {
        "턴": len(rows),
        "심각": sum(bool(t["serious"]) for t in rows),
        "깨움 없음": sum(t["no_presence"] for t in scen),
        "대체 문장": sum(t["fallback"] for t in rows),
        "다시 만듦": sum(t["retries"] for t in rows),
        "핑계 넘어감": sum(t["gave_in"] for t in exc),
    }
    return summary, rows


def markdown(results):
    """results: {버전 이름: (요약, 행들)} → 비교 표 + 대화 원문"""
    keys = list(next(iter(results.values()))[0])
    out = ["| 모델 | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    out += [f"| {name} | " + " | ".join(str(s[k]) for k in keys) + " |" for name, (s, _) in results.items()]
    for name, (_, rows) in results.items():
        out.append(f"\n## {name}\n")
        for t in rows:
            marks = " ".join(t["serious"] + (["깨움 없음"] if t["no_presence"] else []) + (["대체"] if t["fallback"] else [])
                             + ([f"재생성{t['retries']}"] if t["retries"] else []))
            user = t["user"] if t["user"] else "(알람)"
            out.append(f"- [{t['scenario']}] 👤 {user[:30]} → 🤖 {t['reply']} {marks}")
    return "\n".join(out)
