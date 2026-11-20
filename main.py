import os
import sys
import json
import logging
import threading
import ollama
import telemetry
import gcalendar
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

# 대화 엔진(engine/)은 폰 앱도 같은 파일을 그대로 쓴다. 폰에서는 이 폴더가 Python 경로의 맨 위라 같은 이름으로 불러온다
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine"))
import dialog  # noqa: E402

app = FastAPI(title="AI 모닝콜 서비스")
log = logging.getLogger("uvicorn.error")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
# 모델이 메모리에서 내려가면 다시 올리는 데 약 6초 걸린다. 앱 알람은 앱이 닫혀 있어도 울려서 화면의 예열(알람 2분 전)이
# 돌지 않을 수 있으므로, 서버가 켜져 있는 동안 계속 올려 둔다(-1). VRAM을 비워야 하면 OLLAMA_KEEP_ALIVE=60m 처럼 지정
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", -1)
NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "99"))  # GPU에 올릴 레이어 수 (99 = 전부)


class Message(BaseModel):
    role: str
    content: str


class WeatherIn(BaseModel):
    """날씨 퀴즈용 숫자 날씨 (engine/weather_quiz.py의 Weather와 같은 항목)"""
    place: str = ""
    temp: int
    max: int
    min: int
    rain: int = 0
    code: int = 0
    desc: str = ""


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    user_name: Optional[str] = "사용자"
    user_schedule: Optional[str] = ""
    weather_info: Optional[str] = ""
    conversation_level: int = 0
    model: Optional[str] = None
    user_id: Optional[str] = ""  # 프로파일을 찾는 익명 ID (static/index.html의 userId())
    step_goal: int = Field(0, ge=0, le=1000)  # 통화 중 이만큼 걸으면 통화가 끝난다 (앱에서만, 0이면 없음)
    weather: Optional[WeatherIn] = None  # 날씨 퀴즈용 숫자 날씨. 없으면 퀴즈를 건너뛴다
    quiz_attempts: int = Field(0, ge=0, le=20)  # 퀴즈에서 지금까지 틀린 횟수 (화면이 기억했다가 보낸다)

    def turn(self) -> dialog.TurnRequest:
        return dialog.TurnRequest(**self.model_dump())


class ChatResponse(BaseModel):
    reply: str
    alarm_can_end: bool
    conversation_level: int


# ─── 사용자 프로파일 (telemetry.py가 지난 아침 기록으로 만든 것) ─────────────────────
assert set(dialog.TONE_GUIDES) == set(telemetry.TONES)

# 통화 중에는 프로파일이 갱신돼도 처음 읽은 것을 그대로 쓴다.
# 시스템 프롬프트가 한 글자라도 바뀌면 Ollama 캐시가 깨져 매 턴 첫 소리가 1초 이상 늦어진다
_call_profiles: dict = {}


def profile_context(user_id: str, user_name: str, refresh: bool = False) -> str:
    """refresh=True(예열·첫 인사)일 때 DB에서 새로 읽고, 대화 턴에서는 그 통화에서 읽은 것을 쓴다"""
    if not user_id:
        return ""
    key = (user_id, user_name)
    if refresh or key not in _call_profiles:
        try:
            _call_profiles[key] = dialog.build_profile_context(telemetry.load_profile(user_id), user_name)
        except Exception:
            log.exception("프로파일 읽기 실패 — 프로파일 없이 진행합니다")
            _call_profiles[key] = ""
    return _call_profiles[key]


# ─── Ollama ─────────────────────────────────────────────────────────────────────
def ollama_chat(model: str, messages: list, options: dict, **kwargs):
    # 모든 레이어를 GPU에 올리면 VRAM 4GB에서도 gemma3:4b가 약 40% 빨라진다.
    # 분류/생성/예열이 같은 num_gpu를 써야 모델이 다시 로드되지 않는다. 메모리가 부족하면 기본 설정으로 재시도.
    try:
        return ollama.chat(model=model, messages=messages, options={**options, "num_gpu": NUM_GPU},
                           keep_alive=KEEP_ALIVE, **kwargs)
    except ollama.ResponseError:
        return ollama.chat(model=model, messages=messages, options=options, keep_alive=KEEP_ALIVE, **kwargs)


# 스트리밍은 세 문장을 채우면 생성을 멈추므로 num_predict는 문장이 중간에 잘리지 않을 만큼 넉넉히
GENERATION_OPTIONS = {"temperature": 0.7, "repeat_penalty": 1.15, "num_predict": 120}


def ollama_error_detail(e: Exception) -> str:
    err = str(e)
    if "connect" in err.lower():
        return "Ollama가 실행되지 않고 있습니다. 터미널에서 'ollama serve'를 실행해주세요."
    return f"Ollama 오류: {err}"


def ollama_stream(model: str, stats: dict = None) -> dialog.Stream:
    """대화 엔진에 넘길 토큰 스트림. GPU 메모리가 부족해 실패하면(아직 아무것도 받기 전) 기본 설정으로 다시 시도.
    stats를 넘기면 마지막에 프롬프트 처리·생성 시간을 채운다."""
    def stream(messages: list):
        for options in ({**GENERATION_OPTIONS, "num_gpu": NUM_GPU}, GENERATION_OPTIONS):
            received = False
            try:
                for chunk in ollama.chat(model=model, messages=messages, options=options,
                                         keep_alive=KEEP_ALIVE, stream=True):
                    received = True
                    if chunk.get("done") and stats is not None:
                        stats.update(prompt_tokens=chunk.get("prompt_eval_count"),
                                     prompt_s=round((chunk.get("prompt_eval_duration") or 0) / 1e9, 3),
                                     eval_tokens=chunk.get("eval_count"),
                                     eval_s=round((chunk.get("eval_duration") or 0) / 1e9, 3))
                    yield chunk["message"]["content"]
                return
            except ollama.ResponseError:
                if received:
                    raise
    return stream


# ─── 대화 API ───────────────────────────────────────────────────────────────────
# 동기 Ollama 호출이 서버 이벤트 루프를 막지 않도록 def(스레드풀)로 둔다
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    telemetry.note_call_activity()
    turn = request.turn()
    quiz = dialog.quiz_turn(turn)
    if quiz:
        stage, reply, _ = quiz
        return ChatResponse(reply=reply, alarm_can_end=stage == dialog.LAST_STAGE, conversation_level=stage)
    stage, system_prompt, messages = dialog.prepare_turn(turn, profile_context(request.user_id, request.user_name))
    try:
        reply = dialog.generate_reply(ollama_stream(request.model or DEFAULT_MODEL), system_prompt, messages,
                                      request.message, stage, user_name=request.user_name)
    except Exception as e:
        raise HTTPException(status_code=503 if "connect" in str(e).lower() else 500, detail=ollama_error_detail(e))
    return ChatResponse(reply=reply, alarm_can_end=stage == dialog.LAST_STAGE, conversation_level=stage)


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    """답변을 문장 단위로 보낸다. 화면은 첫 문장이 오자마자 읽기 시작할 수 있다.
    이벤트(한 줄에 JSON 하나): meta(단계) → sentence(문장)… → done, 실패하면 error"""
    telemetry.note_call_activity()
    stats = {}
    events = dialog.chat_events(
        request.turn(), ollama_stream(request.model or DEFAULT_MODEL, stats),
        profile_context(request.user_id, request.user_name), stats=stats, error_detail=ollama_error_detail,
    )
    return StreamingResponse((json.dumps(e, ensure_ascii=False) + "\n" for e in events),
                             media_type="application/x-ndjson")


@app.get("/api/initial-message")
def initial_message(
    user_name: str = "사용자",
    weather_info: str = "",
    user_schedule: str = "",
    model: str = None,
    user_id: str = "",
    step_goal: int = 0,
):
    # 알람이 울릴 때 최신 프로파일을 읽고, 이 통화가 끝날 때까지 그대로 쓴다
    telemetry.note_call_activity()
    model = model or DEFAULT_MODEL
    profile = profile_context(user_id, user_name, refresh=True)
    try:
        result = dialog.initial_reply(ollama_stream(model), user_name, user_schedule, weather_info, profile, step_goal)
    except Exception as e:
        raise HTTPException(status_code=503 if "connect" in str(e).lower() else 500, detail=ollama_error_detail(e))
    # 첫 인사를 다시 생성했거나 다듬었으면 Ollama 캐시에 남은 내용이 실제 대화 기록과 달라, 첫 대답이 약 2초 늦어진다.
    # 확정된 첫 인사까지를 미리 처리해 두면 첫 대답부터 캐시가 이어진다 (벨이 울리는 동안이라 기다리지 않는다)
    try:
        ollama_chat(model, [{"role": "system", "content": result["system"]},
                            {"role": "user", "content": result["user_content"]},
                            {"role": "assistant", "content": result["message"]}], {"num_predict": 1})
    except Exception:
        pass
    # user_content를 대화 기록 맨 앞에 두면 첫 대답부터 Ollama 프롬프트 캐시가 재사용된다
    return {"message": result["message"], "user_content": result["user_content"]}


@app.post("/api/warmup")
def warmup(model: str = None, user_name: str = "", user_schedule: str = "", weather_info: str = "",
           user_id: str = "", step_goal: int = 0):
    # 알람 직전에 모델을 메모리에 올린다. 사용자 정보가 있으면 실제 시스템 프롬프트를 미리 처리해 두어
    # 첫 인사와 첫 대답에서 프롬프트 처리(약 1.8초)를 Ollama 캐시로 건너뛰게 한다.
    telemetry.note_call_activity()
    try:
        system = dialog.build_system_prompt(
            user_name, user_schedule, weather_info, profile_context(user_id, user_name, refresh=True), step_goal,
        ) if user_name else ""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": "안녕"}]
        ollama_chat(model or DEFAULT_MODEL, messages, {"num_predict": 1})
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.on_event("startup")
def preload_model():
    # 서버를 켜자마자 모델을 올려 두어 첫 통화도 바로 대답하게 한다. 서버 시작은 기다리게 하지 않는다
    threading.Thread(target=warmup, daemon=True).start()


@app.get("/api/models")
async def list_models():
    try:
        result = ollama.list()
        models = [m["model"] for m in result.get("models", [])]
        return {"models": models}
    except Exception:
        return {"models": []}


# 텔레메트리 수신과 사용자 프로파일 (telemetry.py). 분류·생성과 같은 GPU 설정으로 Ollama를 부르게 한다
telemetry.init(ollama_chat, DEFAULT_MODEL)
app.include_router(telemetry.router)
# 구글 캘린더 비공개 iCal 주소에서 오늘 일정 읽기 (gcalendar.py)
app.include_router(gcalendar.router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# index.html이 앱(file://)에서도 열리도록 상대 경로로 부르는 스타일 파일
@app.get("/tailwind.css")
async def tailwind_css():
    return FileResponse("static/tailwind.css")


# 쌓인 통화 기록과 프로파일을 보는 화면. 앱(APK)에 들어가지 않도록 static 밖에 둔다
@app.get("/records")
def records_page(request: Request):
    telemetry.require_local(request)
    return FileResponse("records.html")
