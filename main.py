import os
import ollama
from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

app = FastAPI(title="AI 모닝콜 서비스")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    user_name: Optional[str] = "사용자"
    user_schedule: Optional[str] = ""
    weather_info: Optional[str] = ""
    conversation_level: int = 0
    model: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    alarm_can_end: bool
    conversation_level: int


def build_system_prompt(user_name: str, user_schedule: str, weather_info: str) -> str:
    now = datetime.now()
    time_str = now.strftime("%H시 %M분")
    day_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    day_name = day_names[now.weekday()]

    return f"""당신은 친절하고 따뜻한 AI 모닝콜 비서입니다.
현재 시각: {time_str} ({day_name})
사용자 이름: {user_name}
오늘 일정: {user_schedule or "특별한 일정 없음"}
날씨 정보: {weather_info or "날씨 정보 없음"}

## 대화 진행 (4단계)

1단계 - 인사 및 기상 확인
- 따뜻하게 인사하고 일어났는지 묻습니다.

2단계 - 일정 공유
- 기상을 확인하면 오늘 일정 "{user_schedule}"을 자연스럽게 언급합니다.

3단계 - 날씨 정보 및 기상 독려
- "{weather_info}" 날씨를 알려주고 침대에서 일어나길 권합니다.

4단계 - 마무리 (완료 신호)
- 사용자가 완전히 일어났다고 확인되면 따뜻하게 마무리합니다.
- 이 단계에서만 메시지 맨 끝에 정확히 [ALARM_END]를 붙입니다.

## 규칙
- 반드시 한국어로만 대화합니다.
- 2~3문장 이내로 짧고 자연스럽게 말합니다.
- 사용자가 "응", "일어났어", "네", "기상" 등 기상 확인 응답을 하면 마무리 단계로 진행합니다.
- [ALARM_END]는 사용자가 일어났다고 명확히 확인된 마지막 메시지에만 사용합니다.
- 영어를 사용하지 마세요."""


def call_ollama(model: str, system: str, messages: list) -> str:
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return response["message"]["content"].strip()
    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "connect" in err.lower():
            raise HTTPException(
                status_code=503,
                detail="Ollama가 실행되지 않고 있습니다. 터미널에서 'ollama serve'를 실행해주세요.",
            )
        raise HTTPException(status_code=500, detail=f"Ollama 오류: {err}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    model = request.model or DEFAULT_MODEL
    system_prompt = build_system_prompt(
        request.user_name, request.user_schedule, request.weather_info
    )

    messages = [{"role": m.role, "content": m.content} for m in request.history]
    messages.append({"role": "user", "content": request.message})

    reply = call_ollama(model, system_prompt, messages)
    alarm_can_end = "[ALARM_END]" in reply
    reply_clean = reply.replace("[ALARM_END]", "").strip()
    new_level = 4 if alarm_can_end else min(request.conversation_level + 1, 3)

    return ChatResponse(
        reply=reply_clean,
        alarm_can_end=alarm_can_end,
        conversation_level=new_level,
    )


@app.get("/api/initial-message")
async def initial_message(
    user_name: str = "사용자",
    weather_info: str = "",
    user_schedule: str = "",
    model: str = None,
):
    use_model = model or DEFAULT_MODEL
    now = datetime.now()
    time_str = now.strftime("%H시 %M분")

    prompt = f"""당신은 AI 모닝콜 비서입니다.
지금 {user_name}님의 알람이 울려서 전화를 걸었습니다.
현재 시각: {time_str}
날씨: {weather_info}

따뜻하고 친근하게 아침 인사를 하며 일어났는지 확인하는 첫 메시지를 작성하세요.
2~3문장, 반드시 한국어로만 작성하세요."""

    reply = call_ollama(use_model, "", [{"role": "user", "content": prompt}])
    return {"message": reply}


@app.get("/api/models")
async def list_models():
    try:
        result = ollama.list()
        models = [m["model"] for m in result.get("models", [])]
        return {"models": models}
    except Exception:
        return {"models": []}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
