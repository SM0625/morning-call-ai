"""폰 앱 전용 입구. 안드로이드(PhoneEngine.kt)가 Chaquopy로 부른다.

서버(main.py)와 같은 대화 엔진(dialog.py)을 쓰고, AI만 폰 안의 Gemma(PhoneAi.kt, LiteRT-LM)로 바꾼다.
화면과 주고받는 형식도 서버와 같다: 첫 인사는 {message, user_content}, 대화 한 턴은 meta → sentence… → done 이벤트.
"""
import json

import dialog


def _stream(ai) -> dialog.Stream:
    """PhoneAi.start(메시지 JSON)가 돌려주는 TokenStream을 토큰 반복자로 바꾼다.
    dialog가 생성을 일찍 멈추면(세 문장을 채움·문제 발견) 반복자가 닫히면서 폰의 생성도 취소된다"""
    def stream(messages: list):
        tokens = ai.start(json.dumps(messages, ensure_ascii=False))
        try:
            while True:
                token = tokens.next()
                if token is None:
                    return
                yield token
        finally:
            tokens.close()
    return stream


def _profile(request: dict) -> str:
    # 프로파일은 서버가 지난 아침 기록으로 만든 것을 화면이 받아 두었다가 보낸다 (서버가 꺼져 있으면 마지막으로 받은 것)
    return dialog.build_profile_context(request.get("profile"), request.get("user_name") or "")


def initial(request_json: str, ai) -> str:
    """알람이 울렸을 때 첫 인사"""
    r = json.loads(request_json)
    result = dialog.initial_reply(_stream(ai), r.get("user_name") or "사용자", r.get("user_schedule") or "",
                                  r.get("weather_info") or "", _profile(r), int(r.get("step_goal") or 0))
    return json.dumps({"message": result["message"], "user_content": result["user_content"]}, ensure_ascii=False)


def chat(request_json: str, ai, emitter) -> None:
    """대화 한 턴. 이벤트가 생길 때마다 emitter.emit(JSON)으로 화면에 바로 넘긴다"""
    r = json.loads(request_json)
    for event in dialog.chat_events(dialog.TurnRequest(**r), _stream(ai), _profile(r),
                                    error_detail=lambda e: f"폰 AI 오류: {e}"):
        emitter.emit(json.dumps(event, ensure_ascii=False))
