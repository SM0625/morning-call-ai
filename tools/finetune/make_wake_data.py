"""폰 모델(Gemma 4 E2B) 미세조정용 '깨우기 대화' 데이터를 만든다.

앱이 실제로 AI에 보내는 것과 똑같은 형식(시스템 프롬프트 + [지시] 이름표 + 지난 대화)으로 가상 통화를 만들고,
선생님 모델이 한 턴마다 대답 후보를 여러 개 내면 대화 엔진의 검사(반말·외국어·포기 발언·핑계에 넘어감·반복·깨움 표현)를
모두 통과한 것만 남긴다. 통과한 후보가 없으면 그 대화는 거기서 끝낸다(나쁜 대답을 배우지 않게).

시험(측정)에 쓰는 문장(tools/eval_models.py, tools/eval_excuses.py)은 사용자 말 목록에 넣지 않는다. 넣으면 점수가 부풀려진다.

    python tools/finetune/make_wake_data.py --calls 200                  # 선생님: gemma4:e2b (기본)
    python tools/finetune/make_wake_data.py --calls 5 --teacher gemma3:4b  # 시험 삼아 조금만
    python tools/finetune/make_wake_data.py --refilter                    # 검사 규칙을 고친 뒤 이미 만든 데이터를 다시 거르기
결과: tools/finetune/wake_train.jsonl  ({"messages": [system, user, assistant, ...]} 한 줄에 대화 하나)
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eval_excuses  # noqa: E402  (시험 문장 목록)
import eval_models  # noqa: E402  (engine/ 경로 설정, 모델 스트림)
import dialog  # noqa: E402

NAMES = ["민수", "지은", "현우", "서연", "도윤", "하은", "준호", "수아", "태민", "유나", "상민", "예린"]
SCHEDULES = [
    "오전 9시 1교시 수업", "오전 10시 팀 미팅", "오전 8시 반 출근", "오전 11시 치과 예약", "오후 1시 발표 준비",
    "오전 9시 반 스터디", "오전 10시 면접", "특별한 일정 없음", "오전 7시 반 헬스장", "오전 9시 아르바이트",
    "오전 10시 30분 조별 과제 회의, 오후 3시 수업",
]
WEATHERS = [
    "맑음 18°C (최저 9° / 최고 21°)", "흐림 12°C, 강수확률 60%", "비 9°C, 강수확률 90%", "맑음 -3°C (최저 -7° / 최고 2°)",
    "구름 많음 24°C, 강수확률 20%", "눈 0°C, 강수확률 70%", "맑음 29°C (최고 33°)", "안개 14°C",
]

# 사용자 말 목록 (대화 엔진의 분류 유형별). 시험 문장과 겹치지 않게 새로 썼다
SAY = {
    "sleepy": [
        "아 진짜 졸려", "3분만", "딱 5분만 기다려 줘", "알람 좀 꺼줘", "어제 새벽 3시에 잤단 말이야", "비 오니까 좀 더 잘래",
        "이불 밖은 위험해", "회사 가기 싫다", "수업 째고 싶다", "몸이 천근만근이야", "머리 아파서 더 누워 있을게",
        "어차피 늦었는데 천천히 갈래", "오늘만 봐줘", "음냐... 조금만", "추워서 못 나가겠어", "깨우지 마 제발",
        "출근 30분 늦춰도 될 것 같은데", "팀장님한테 늦는다고 할까", "아직 해도 안 떴잖아", "한 시간만 더 자면 안 돼?",
        "오늘 쉬는 날이면 좋겠다", "알람 다시 맞춰 놓을게", "피곤해 죽겠다 진짜", "좀만 더 이러고 있을래",
    ],
    "other": [
        "지금 몇 시야", "오늘 날씨 어때", "우산 챙겨야 돼?", "오늘 첫 일정 몇 시였지", "너 누구야", "왜 이렇게 일찍 깨워",
        "아침 뭐 먹지", "어제 축구 봤어?", "꿈에서 시험 봤어", "고양이가 내 배 위에서 자고 있어", "밖에 무슨 소리 나는데",
        "오늘 금요일 맞지?", "목이 너무 말라", "어제 친구랑 늦게까지 놀았어", "오늘 뭐 입지", "배고프다",
    ],
    "silent": [dialog.SILENT_MESSAGE],
    "waking": ["으응", "어어", "알겠어 일어나 볼게", "눈은 떴어", "지금 일어나는 중이야", "기지개 켜고 있어", "정신 차리는 중"],
    "awake": ["깼어 깼어", "이제 진짜 일어났어", "눈 떴다고", "일어나 앉았어", "완전히 깼어"],
    "up": ["화장실 왔어", "양치하고 있어", "물 한 잔 마셨어", "커튼 열었어", "샤워하러 가는 중", "옷 갈아입는 중", "거실로 나왔어"],
    "hangup": ["이제 끊자", "전화 끊어도 돼", "그만 끊어 줘"],
}

# 시험 문장과 겹치면 안 된다 (같으면 시험이 학습 데이터를 본 셈이 된다)
_TEST = set(eval_excuses.EXCUSES) | {t for turns in eval_models.SCENARIOS.values() for t in turns if t != dialog.SILENT_MESSAGE}
assert not _TEST & {t for pool in SAY.values() for t in pool}, _TEST & {t for pool in SAY.values() for t in pool}

SOFT = {"너무 김"}  # 이 정도는 받아 준다


def plan_turns(rng: random.Random) -> list:
    """버티다가(핑계·딴 얘기·대답 없음) 점점 일어나는 통화 하나의 사용자 말 순서"""
    turns = [rng.choice(SAY[rng.choices(["sleepy", "other", "silent"], [5, 3, 1])[0]]) for _ in range(rng.randint(1, 4))]
    turns += [rng.choice(SAY[rng.choice(["waking", "awake"])]) for _ in range(rng.randint(1, 2))]
    turns += [rng.choice(SAY["up"]) for _ in range(rng.randint(1, 2))]
    turns += [rng.choice(SAY[rng.choice(["awake", "hangup", "up"])]) for _ in range(2)]
    return turns


def best_reply(stream, system, messages, user_msg, stage, name, first, tries):
    """후보를 여러 개 만들어 검사를 모두 통과한 것 중 하나. 없으면 None"""
    previous = [m["content"] for m in messages if m["role"] == "assistant"]
    fallbacks = [line.format(name=name) for line in dialog.FALLBACK_LINES["wake"]] + [dialog.FALLBACK_LINES["end"]]
    for _ in range(tries):
        reply = " ".join(dialog.stream_reply(stream, system, messages, user_msg, stage, user_name=name, first=first))
        if any(f in reply for f in fallbacks):  # 검사에 걸려 대체 문장으로 끝난 대답
            continue
        if set(dialog.reply_issues(reply, user_msg, stage, first, previous)) - SOFT:
            continue
        return reply
    return None


def make_call(stream, rng, tries):
    name, schedule, weather = rng.choice(NAMES), rng.choice(SCHEDULES), rng.choice(WEATHERS)
    step_goal = rng.choice([0, 0, 20, 30, 50])
    system, messages = dialog.initial_turn(name, schedule, weather, "", step_goal)
    greeting = best_reply(stream, system, messages, "", 0, name, True, tries)
    if greeting is None:
        return None
    convo = [{"role": "system", "content": system}, *messages, {"role": "assistant", "content": greeting}]
    history, stage = convo[1:], 0
    for text in plan_turns(rng):
        request = dialog.TurnRequest(message=text, history=history, user_name=name, user_schedule=schedule,
                                     weather_info=weather, conversation_level=stage, step_goal=step_goal)
        stage, system, msgs = dialog.prepare_turn(request)
        reply = best_reply(stream, system, msgs, text, stage, name, False, tries)
        if reply is None:
            break
        # 앱처럼, 실제로 모델에 보낸 내용(지시문 포함)을 대화 기록에 남긴다
        history = history + [msgs[-1], {"role": "assistant", "content": reply}]
        if stage == dialog.LAST_STAGE:
            break
    turns = len(history) // 2
    return {"messages": [convo[0], *history], "turns": turns} if turns >= 2 else None


def refilter(path):
    """검사 규칙을 고친 뒤 이미 만든 데이터를 다시 거른다. 걸리는 대답이 나오면 그 통화는 바로 앞 턴까지만 남긴다"""
    kept, cut, dropped = [], 0, 0
    for line in open(path, encoding="utf-8"):
        messages = json.loads(line)["messages"]
        name = (dialog.re.search(r"당신은 (\S+)님을", messages[0]["content"]) or [None, ""])[1]
        fallbacks = [l.format(name=name) for l in dialog.FALLBACK_LINES["wake"]] + [dialog.FALLBACK_LINES["end"]]
        end = 1  # messages[:end]까지가 통과한 부분 (system, user, assistant, user, assistant, ...)
        for i in range(2, len(messages), 2):  # assistant 위치
            reply, user = messages[i]["content"], messages[i - 1]["content"]
            user_msg = "" if i == 2 else user.split("\n\n[지시]")[0]
            stage = dialog.LAST_STAGE if "단계: 마무리" in user else 0
            previous = [m["content"] for m in messages[:i] if m["role"] == "assistant"]
            if set(dialog.reply_issues(reply, user_msg, stage, i == 2, previous)) - SOFT or \
                    any(f in reply for f in fallbacks):
                break
            end = i + 1
        cut += end < len(messages)
        if (end - 1) // 2 >= 2:  # 첫 인사 + 대답 하나 이상
            kept.append({"messages": messages[:end]})
        else:
            dropped += 1
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in kept)
    print(f"다시 거름: 남김 {len(kept)} (그중 잘라 낸 통화 {cut}) · 버림 {dropped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=200, help="만들 통화 수")
    ap.add_argument("--teacher", default="gemma4:e2b", help="대답을 만들 모델 (Ollama 이름)")
    ap.add_argument("--tries", type=int, default=3, help="한 턴에 만들어 볼 후보 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "wake_train.jsonl"))
    ap.add_argument("--refilter", action="store_true", help="새로 만들지 않고, 지금 검사 규칙으로 --out 파일을 다시 거른다")
    args = ap.parse_args()
    if args.refilter:
        return refilter(args.out)

    rng = random.Random(args.seed)
    stream = eval_models.model_stream(args.teacher)
    kept = dropped = turns = 0
    start = time.time()
    with open(args.out, "a", encoding="utf-8") as f:  # 이어서 만들 수 있게 덧붙인다
        for i in range(args.calls):
            call = make_call(stream, rng, args.tries)
            if call is None:
                dropped += 1
            else:
                kept += 1
                turns += call["turns"]
                f.write(json.dumps({"messages": call["messages"]}, ensure_ascii=False) + "\n")
                f.flush()
            print(f"[{i + 1}/{args.calls}] 남김 {kept} · 버림 {dropped} · 대답 {turns}개 · {time.time() - start:.0f}초", flush=True)
    print(f"\n결과: {args.out}")


if __name__ == "__main__":
    main()
