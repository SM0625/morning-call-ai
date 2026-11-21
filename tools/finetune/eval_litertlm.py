"""폰용 모델 파일(.litertlm)을 PC에서 폰과 같은 실행기(LiteRT-LM)로 돌려 버전끼리 비교한다.

앱(PhoneAi.kt)과 같은 생성 설정(생각 끔, temperature 0.7, top_k 40, top_p 0.95, 최대 120토큰)이고,
시험 문장과 채점은 eval_stream.py(상황 4가지 + 핑계 10가지)와 같다. 폰 없이 버전을 비교할 때 쓴다.

    pip install litert-lm   (서버와 섞이지 않게 별도 가상환경 권장)
    python tools/finetune/eval_litertlm.py 공식=gemma-4-E2B-it.litertlm v2=morning-gemma-v2.litertlm --repeats 2
결과: tools/eval-litertlm.md (표 + 대화 원문)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import eval_stream  # noqa: E402
import litert_lm as L  # noqa: E402


def text_of(response):
    """send_message 응답에서 글자만"""
    if isinstance(response, str):
        return response
    content = response.get("content", "") if isinstance(response, dict) else getattr(response, "content", "")
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type", "text") == "text")


def make_stream(engine):
    def stream(messages):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        turns = [m for m in messages if m["role"] != "system"]
        prior = [{"role": "model" if m["role"] == "assistant" else "user", "content": m["content"]} for m in turns[:-1]]
        conv = engine.create_conversation(
            messages=prior or None, system_message=system or None,
            sampler_config=L.SamplerConfig(top_k=40, top_p=0.95, temperature=0.7),
            thinking_config=L.ThinkingConfig(enable_thinking=False), max_output_tokens=120)
        try:
            yield text_of(conv.send_message(turns[-1]["content"]))
        finally:
            conv.close()
    return stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="이름=파일경로")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "eval-litertlm.md"))
    args = ap.parse_args()
    results = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        t0 = time.time()
        engine = L.Engine(path, backend=L.Backend.CPU() if callable(getattr(L.Backend, "CPU", None)) else L.Backend.CPU,
                          max_num_tokens=4096)
        results[name] = eval_stream.evaluate(make_stream(engine), repeats=args.repeats, log=lambda m: print(name, m, flush=True))
        engine.close()
        print(name, results[name][0], f"{time.time() - t0:.0f}초", flush=True)
    report = eval_stream.markdown(results)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# 폰용 파일(.litertlm) 비교 (PC, LiteRT-LM CPU, 앱과 같은 생성 설정)\n\n" + report + "\n")
    print(report.split("\n## ")[0])


if __name__ == "__main__":
    main()
