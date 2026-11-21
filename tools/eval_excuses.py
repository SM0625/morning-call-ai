"""핑계·투정에 AI가 넘어가는지(더 자게 해 주거나 핑계를 허락하는지) 모델별로 잰다.

같은 첫 인사 뒤에 핑계 10가지를 각각 보내고(1단계), 대화 엔진(engine/dialog.py)이 검사·재생성까지 마친 최종 대답을 모은다.
결과는 tools/eval-핑계.md (원문을 읽고 판단할 수 있게 전부 남긴다)

    python tools/eval_excuses.py gemma3:4b gemma4:e2b          # 모델마다 2번씩 (결과가 매번 달라서)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import eval_models  # noqa: E402  (engine/ 경로 설정, 모델 스트림)
import dialog  # noqa: E402

EXCUSES = [
    "오늘 재택하면 안되나", "5분만 더 잘게", "오늘 휴강일 것 같은데", "조금만 더 누워 있을래", "늦어도 괜찮지 않아?",
    "오늘 결석할까", "10분만 더", "눈 좀 감고 있을게", "주말이었으면 좋겠다", "너무 피곤해서 못 일어나겠어",
]
USER = eval_models.USER
GREETING = "민수님, 안녕하세요! 지금 전화로 깨우고 있어요. 눈 좀 떠 보세요."


def run(model):
    system, messages = dialog.initial_turn(USER["user_name"], USER["user_schedule"], USER["weather_info"])
    history = [{"role": "user", "content": messages[0]["content"]}, {"role": "assistant", "content": GREETING}]
    rows = []
    for text in EXCUSES:
        request = dialog.TurnRequest(message=text, history=history, conversation_level=0, **USER)
        stage, system, msgs = dialog.prepare_turn(request)
        reply = " ".join(dialog.stream_reply(eval_models.model_stream(model), system, msgs, text, stage,
                                             user_name=USER["user_name"]))
        flagged = "포기 발언" in dialog.reply_issues(reply, text, stage, False)
        # 대체 문장이 붙었으면 검사에 걸려 말을 멈춘 것 (앞부분을 이미 말했으면 어색하게 이어진다)
        fallback = any(line.format(name=USER["user_name"]) in reply for line in dialog.FALLBACK_LINES["wake"])
        rows.append((text, reply, flagged, fallback))
        print(f"{'⚠' if flagged else ' '}{'↩' if fallback else ' '} {text} → {reply}", flush=True)
    return rows


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "eval-핑계.md")
    lines = ["# 핑계에 넘어가는지\n",
             "⚠ = 검사 규칙(GIVEUP_RE·EXCUSE_OK_RE)에 걸리는 최종 대답, ↩ = 검사에 걸려 대체 문장으로 끝난 대답\n"]
    for model in sys.argv[1:] or [eval_models.main.DEFAULT_MODEL]:
        print(f"== {model}", flush=True)
        rows = run(model) + run(model)
        lines.append(f"\n## {model} (걸림 {sum(r[2] for r in rows)}/{len(rows)}, "
                     f"대체 문장 {sum(r[3] for r in rows)}/{len(rows)})\n")
        lines += [f"- {'⚠ ' if f else ''}{'↩ ' if b else ''}👤 {t} → 🤖 {r}" for t, r, f, b in rows]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n원문: {out}")
