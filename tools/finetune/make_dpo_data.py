"""선호 학습(DPO)용 데이터: 같은 상황에서 '검사를 통과한 대답(chosen)'과 '걸린 대답(rejected)'의 짝을 만든다.

make_wake_data.py와 같은 가상 통화를 진행하되, 한 턴마다 모델의 날것 대답(대화 엔진이 고치기 전)을 여러 개 만들어
대화 엔진의 검사(reply_issues)에 걸린 것을 rejected로 남긴다. 대화는 통과한 대답으로 이어 간다.
엔진이 고쳐 주기 전에 모델이 스스로 좋은 대답을 내게 하려는 것이다
(지금은 걸리면 엔진이 이미 말한 앞부분 뒤에 대체 문장을 붙여 어색해진다).

    python tools/finetune/make_dpo_data.py --calls 150
결과:
  tools/finetune/wake_dpo.jsonl         {"prompt": [system, user, assistant, ..., user], "chosen", "rejected", "issues"}
  tools/finetune/wake_train_more.jsonl  통과한 대답으로 이어 간 통화 (SFT 데이터에 더한다, make_wake_data.py와 같은 형식)
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import make_wake_data as W  # noqa: E402  (사용자 말 목록, 통화 순서, 시험 문장과 겹치지 않는지 확인)
from make_wake_data import dialog, eval_models  # noqa: E402

HERE = os.path.dirname(__file__)


def raw_reply(stream, system, messages, name):
    """대화 엔진의 구절 검사·대체 없이 모델이 낸 대답 그대로 (호칭·공백만 정리)"""
    text = "".join(stream([{"role": "system", "content": system}] + messages))
    return dialog.fix_vocative(dialog.clean_reply(text), name)


def turn_candidates(stream, system, messages, user_msg, stage, name, first, tries, max_rejected, extra=None):
    """통과한 대답 하나와 걸린 대답들. 통과한 것이 없으면 chosen은 None.
    extra: 실수를 더 자주 하는 모델(gemma3:4b)의 스트림. 그 대답은 걸렸을 때 rejected로만 쓴다
    (gemma4:e2b는 포기 발언·반말을 거의 안 해서 그런 나쁜 예가 잘 안 모인다)"""
    previous = [m["content"] for m in messages if m["role"] == "assistant"]
    chosen, rejected = None, []
    sources = ([(extra, False)] * 2 if extra else []) + [(stream, True)] * tries
    for src, can_choose in sources:
        reply = raw_reply(src, system, messages, name)
        if not reply:
            continue
        issues = sorted(set(dialog.reply_issues(reply, user_msg, stage, first, previous)) - W.SOFT)
        if issues:
            if len(rejected) < max_rejected and all(reply != r for r, _ in rejected):
                rejected.append((reply, issues))
        elif chosen is None and can_choose:
            chosen = reply
        if chosen is not None and len(rejected) >= max_rejected:
            break
    return chosen, rejected


def make_call(stream, rng, tries, max_rejected, extra=None):
    name, schedule, weather = rng.choice(W.NAMES), rng.choice(W.SCHEDULES), rng.choice(W.WEATHERS)
    step_goal = rng.choice([0, 0, 20, 30, 50])
    system, messages = dialog.initial_turn(name, schedule, weather, "", step_goal)
    pairs = []

    def collect(msgs, user_msg, stage, first):
        chosen, rejected = turn_candidates(stream, system_now[0], msgs, user_msg, stage, name, first, tries, max_rejected, extra)
        if chosen is not None:
            prompt = [{"role": "system", "content": system_now[0]}, *msgs]
            pairs.extend({"prompt": prompt, "chosen": chosen, "rejected": r, "issues": i, "stage": stage}
                         for r, i in rejected)
        return chosen

    system_now = [system]
    greeting = collect(messages, "", 0, True)
    if greeting is None:
        return None, pairs
    history, stage = [*messages, {"role": "assistant", "content": greeting}], 0
    for text in W.plan_turns(rng):
        request = dialog.TurnRequest(message=text, history=history, user_name=name, user_schedule=schedule,
                                     weather_info=weather, conversation_level=stage, step_goal=step_goal)
        stage, system_now[0], msgs = dialog.prepare_turn(request)
        reply = collect(msgs, text, stage, False)
        if reply is None:
            break
        history = history + [msgs[-1], {"role": "assistant", "content": reply}]
        if stage == dialog.LAST_STAGE:
            break
    turns = len(history) // 2
    call = {"messages": [{"role": "system", "content": system_now[0]}, *history]} if turns >= 2 else None
    return call, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=150)
    ap.add_argument("--teacher", default="gemma4:e2b")
    ap.add_argument("--tries", type=int, default=4, help="한 턴에 만들어 볼 후보 수")
    ap.add_argument("--max-rejected", type=int, default=2, help="한 턴에서 남길 걸린 대답 수")
    ap.add_argument("--extra", default="gemma3:4b", help="나쁜 예를 더 모을 모델 (빈 문자열이면 안 씀)")
    ap.add_argument("--public-other", default="", help="'딴 얘기' 사용자 말을 공개 데이터 문장으로 바꿀 파일 (public_other.txt)")
    ap.add_argument("--dpo-out", default="wake_dpo.jsonl")
    ap.add_argument("--sft-out", default="wake_train_more.jsonl")
    ap.add_argument("--seed", type=int, default=1, help="make_wake_data.py(기본 0)와 다른 통화가 나오게")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.public_other:
        # 공개 데이터의 일상 이야기를 통화 중 딴 얘기로 쓴다. 공감하면서도 깨우는 쪽으로 돌아오는 대답을 배우게
        public = [l.strip() for l in open(args.public_other, encoding="utf-8") if l.strip() and not l.startswith("#")]
        assert not W._TEST & set(public)
        W.SAY["other"] = public
    stream = eval_models.model_stream(args.teacher)
    extra = eval_models.model_stream(args.extra) if args.extra else None
    calls = pairs = 0
    issue_count = {}
    start = time.time()
    with open(os.path.join(HERE, args.dpo_out), "a", encoding="utf-8") as fd, \
            open(os.path.join(HERE, args.sft_out), "a", encoding="utf-8") as fs:
        for i in range(args.calls):
            call, new_pairs = make_call(stream, rng, args.tries, args.max_rejected, extra)
            for p in new_pairs:
                fd.write(json.dumps(p, ensure_ascii=False) + "\n")
                for issue in p["issues"]:
                    issue_count[issue] = issue_count.get(issue, 0) + 1
            if call:
                calls += 1
                fs.write(json.dumps(call, ensure_ascii=False) + "\n")
            pairs += len(new_pairs)
            fd.flush()
            fs.flush()
            print(f"[{i + 1}/{args.calls}] 통화 {calls} · 짝 {pairs} · {time.time() - start:.0f}초 · {issue_count}", flush=True)


if __name__ == "__main__":
    main()
