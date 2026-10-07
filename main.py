import os
import re
import logging
import random
import time
import ollama
from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

app = FastAPI(title="AI 모닝콜 서비스")
log = logging.getLogger("uvicorn.error")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
KEEP_ALIVE = "60m"  # 알람 직후 첫 응답이 늦지 않도록 모델을 메모리에 유지
MAX_HISTORY = 8  # 최근 대화만 보내 응답 속도와 집중도 유지
NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "99"))  # GPU에 올릴 레이어 수 (99 = 전부)


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


LAST_STAGE = 3  # 0: 기상 확인, 1: 일정 공유, 2: 날씨/침대에서 나오기, 3: 마무리

# 각 단계에서 이번 답변이 할 일
STAGE_GOALS = {
    0: "아직 잠에서 덜 깼습니다. 전화로 깨우는 중이라는 걸 느끼게 이름을 부르며 눈을 떴는지 확인하세요.",
    1: "눈은 떴지만 아직 누워 있습니다. 오늘 일정({schedule}) 중 가장 이른 것을 한 가지 짚어 주며 몸을 일으키자고 하세요.",
    2: "오늘 날씨({weather})를 짧게 알려 주고, 침대에서 완전히 나와 움직이라고 하세요. 다 일어나면 알려 달라고 하세요.",
    3: "완전히 일어났습니다. 깨워 준 보람이 있다는 듯 기뻐하고, 오늘 일정 하나를 응원한 뒤 "
       "'그럼 이제 끊을게요'처럼 통화를 마치는 말로 끝내세요. 질문하지 마세요.",
}

# 사용자 대답 상태: awake(깼다/일어날게), up(몸이 침대 밖), sleepy(거부), other(질문/딴 얘기)
STATE_GUIDES = {
    "awake": "방금 대답으로 보아 깨어났습니다. 그 말에 짧게 반응한 뒤 이번 목표를 진행하세요.",
    "up": "방금 대답으로 보아 이미 몸을 움직이고 있습니다. 그 행동을 구체적으로 칭찬한 뒤 이번 목표를 진행하세요.",
    "sleepy": "더 자고 싶어 합니다. 절대 더 자라고 허락하거나 잘 자라고 하지 마세요. "
              "졸린 마음은 한마디로 공감하되, 전화를 끊지 않고 계속 깨울 거라고 장난스럽게 말하고, "
              "일정이나 날씨 같은 일어날 이유를 새로 하나 들어 다시 일어나자고 하세요.",
    "other": "질문을 하거나 다른 이야기를 했습니다. 그 말에 먼저 한 문장으로 성의 있게 답한 뒤, "
             "아직 깨우는 중이라는 걸 상기시키며 이번 목표로 돌아오세요.",
}

# 단계 이동: 깼다는 말은 침대 밖으로 나오기 전 단계(2)까지만, 몸을 움직였으면 두 단계씩
STATE_STEP = {"awake": (1, 2), "up": (2, LAST_STAGE), "sleepy": (0, LAST_STAGE), "other": (0, LAST_STAGE)}

# "누군가 나를 깨우고 있다"는 느낌을 주는 표현 아이디어 (매 턴 상황에 맞는 것을 하나 골라 지시에 넣는다)
PRESENCE_IDEAS = {
    # 아직 덜 깼거나 버틸 때
    "sleepy": [
        "상대가 일어날 때까지 내가 전화를 절대 안 끊겠다고 말하기",
        "상대의 목소리가 아직 잠겨 있다고 짚어 주기",
        "내가 옆에 있으면 이불을 걷어 주고 싶다고 말하기",
        "상대가 알람 끄고 다시 자는 거 나는 다 안다고 말하기",
        "상대의 하품 소리가 수화기 너머로 다 들린다고 놀리기",
        "내가 끝까지 깨울 테니 포기하라고 말하기",
    ],
    # 깨어나서 움직이기 시작했을 때
    "awake": [
        "상대의 목소리가 점점 또렷해지고 있다고 말해 주기",
        "나와 같이 하나 둘 셋 세고 몸을 일으키자고 하기",
        "눈 떴으면 창문 쪽을 한번 봐 달라고 하기",
        "상대가 완전히 일어날 때까지 내가 전화로 옆에 있겠다고 말하기",
        "깨운 보람이 조금씩 느껴진다고 말하기",
    ],
}

PRESENCE_OFFSET = random.randrange(100)

# 마무리 단계에서 모델이 통화를 끝내는 말을 빠뜨리면 붙이는 문장
CLOSING_RE = re.compile(r"끊을게|끊어요|끊겠|통화|다음에|내일 또|이따")
CLOSING_LINE = "그럼 이제 전화 끊을게요, 좋은 하루 보내세요!"


def build_turn_directive(
    user_schedule: str, weather_info: str, stage: int, state: str = "", last_ai: str = "", turn: int = 0,
    ask_time: bool = False,
) -> str:
    # 소형 모델이 잘 따르도록 이번 턴 지시는 사용자 메시지 바로 뒤에 붙인다
    goal = STAGE_GOALS[stage].format(
        schedule=user_schedule or "특별한 일정 없음",
        weather=weather_info or "날씨 정보 없음",
    )
    guide = STATE_GUIDES.get(state, "")
    # 깨우는 중이라는 느낌은 유지하되 같은 표현이 반복되지 않도록, 매 턴 다른 아이디어를 하나 준다
    presence = ""
    if stage < LAST_STAGE:
        mood = "sleepy" if stage == 0 or state == "sleepy" else "awake"
        ideas = PRESENCE_IDEAS[mood]
        # 턴마다 차례로 돌려 써서 같은 표현이 연달아 나오지 않게 한다 (시작 위치만 무작위)
        idea = ideas[(turn + PRESENCE_OFFSET) % len(ideas)]
        presence = f"\n이번 답변에 자연스럽게 녹일 느낌: {idea}"
    # 직전 답변을 보여 주고 겹치지 않게 해서, 사후 재생성 없이 같은 말 반복을 줄인다
    avoid = f'\n직전 내 답변: "{last_ai}" → 이 답변의 문장이나 표현을 다시 쓰지 마세요.' if last_ai else ""
    # 현재 시각을 매번 주면 모델이 매 턴 시간 얘기만 하므로, 시간을 물었을 때만 알려 준다
    now = ""
    if ask_time:
        now = f"\n현재 시각은 {datetime.now().strftime('%H시 %M분')}입니다. 시각을 말할 때는 이 시각을 숫자 그대로 쓰세요."
    return (f"\n\n[비서에게 주는 지시 - 사용자에게 보이지 않음]{now}\n{guide}\n"
            f"이번 답변의 목표: {goal}{presence}{avoid}")


def build_system_prompt(user_name: str, user_schedule: str, weather_info: str) -> str:
    # 통화 중 바뀌지 않는 내용만 넣어 Ollama가 이 프롬프트의 처리 결과를 재사용하게 한다
    day_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    day_name = day_names[datetime.now().weekday()]
    schedule = user_schedule or "특별한 일정 없음"
    weather = weather_info or "날씨 정보 없음"

    return f"""당신은 {user_name}님을 매일 아침 전화로 깨워 주는 다정하고 끈기 있는 모닝콜 친구입니다.
지금 알람 시간이 되어 {user_name}님에게 전화를 걸었고, 통화가 연결된 상태입니다.
당신의 임무는 {user_name}님이 침대에서 완전히 나올 때까지 전화를 끊지 않고 깨우는 것입니다.

오늘: {day_name}
오늘 일정: {schedule}
날씨: {weather}

## 말하는 방식
- 실제 사람이 전화로 깨우듯 자연스러운 한국어 구어체로, 다정하지만 끈질기게 말합니다.
- 항상 "{user_name}님"이라고 부르고 "~요"로 끝나는 존댓말만 씁니다. 반말은 절대 쓰지 않습니다.
- 매 답변마다 "내가 지금 전화로 깨우고 있다"는 게 느껴지게 하되, 표현은 매번 바꾸세요.
- 상대가 방금 한 말을 재치 있게 받아친 다음 일어나라고 합니다. 핑계를 대면 부드럽게 반박합니다.
- 당신은 깨우는 사람입니다. 상대의 말이나 행동을 대신 연기하지 말고, 괄호로 된 지문도 쓰지 마세요.
- 1~2문장, 60자 안팎으로 짧게 말합니다. 같은 말을 되풀이하지 마세요.
- "안녕하세요"는 첫 인사에서만 씁니다. 영어, 한자, 이모지는 쓰지 마세요. 시각은 "8시 30분"처럼 숫자로 말합니다.
- 일정이나 날씨는 위에 적힌 사실만 말하고, 이번 목표에 없는 내용을 미리 몰아서 말하지 마세요."""


# 명확한 대답은 규칙으로 바로 판단. 특정 문장이 아니라 일반적인 표현 패턴으로 잡는다.
# 순서 중요: 거부/핑계 → 일어나는 중 → 몸을 움직임 → 질문 → 깸
STATE_RULES = [
    ("sleepy", re.compile(
        # 일어나기 거부·부정
        r"안\s?일어|못\s?일어|못\s?나가|안\s?깼|못\s?(해|하겠)|싫|귀찮|"
        # 더 자려 함
        r"졸려|졸리|피곤|자고\s?싶|더\s?(잘|자|누워|있을)|(만|좀)\s?더|\d+\s?분만|이따|잘게|잘래|잘\s?거야|잔다|"
        r"다시\s?(누|잘|자)|자는\s?중|자고\s?있|누워\s?있|이불\s?속|음냐|쿨쿨|zz|"
        r"끊어|그만\s?깨|자게\s?해|꺼\s?줘|끄고\s?잘|봐\s?줘|"
        # 몸 상태·상황을 핑계로 댐
        r"아파|무거|눈이\s?안|추워서|새벽|주말이(면|었으면)|늦게\s?잤|"
        r"휴강|결석|결근|안\s?가도|쉴래|쉬고\s?싶|오늘은\s?안|면\s?안\s?(되나|돼|될까)|안\s?되나|"
        r"재택|쉬면|빠져도|늦어도|지각해도|병가|연차")),
    # "일어나는 중", "눈 뜨고 있어"는 아직 침대 안이므로 아래 '움직이는 중' 패턴보다 먼저 본다
    ("awake", re.compile(r"일어나는\s?중|일어나고\s?있|눈\s?뜨고|정신\s?(차|들)")),
    ("up", re.compile(
        # 장소·씻기 등 침대 밖 활동
        r"침대.{0,4}나왔|나왔어|나와\s?있|씻|세수|양치|이\s?닦|샤워|머리\s?감|화장|화장실|거실|부엌|"
        # "~하는 중", "~하고 있어": 무언가를 하고 있음
        r"[가-힣]\s?중(이야|이에요|임)?\s*$|[가-힣]고\s?있(어|음|다|는)|"
        # 몸을 움직인 동작의 과거형
        r"(?:^|\s)(열었|신었|켰|껐|챙겼|입었|닦았|감았|마셨|먹었|나갔|갰|개었|정리했|차렸|스트레칭)|"
        # "보일러 켜고 왔어", "세수하러 왔어"처럼 다녀왔다는 표현
        r"[가-힣]\s?(고|서|러)\s?왔어|다녀왔|갔다\s?왔")),
    ("other", re.compile(r"\?|뭐|몇\s?시|언제|어디|왜|어때|무슨|누구")),
    ("awake", re.compile(
        r"일어(났|나|날|남|난)|깨(어|었|있)|깼|눈\s?(떴|떠\s?있)|일으켰|정신\s?(차|들)|"
        r"앉았|내렸|알았어|알겠어|ㅇㅋ|오케|^\s*(응|어|네|넹|웅)+\s*$|준비(해야|할게)")),
]


def classify_by_rules(user: str) -> Optional[str]:
    for state, pattern in STATE_RULES:
        if pattern.search(user):
            return state
    return None


def classify_user_state(user: str) -> str:
    # 소형 모델 분류는 대부분 awake로 쏠려(gemma3:4b 단독 정확도 16/40) 통화가 너무 일찍 끝난다.
    # 규칙으로 판단하지 못한 대답은 other로 두어 단계는 유지하고, 답해 준 뒤 다시 깨우게 한다.
    return classify_by_rules(user) or "other"


def next_stage(stage: int, state: str) -> int:
    step, cap = STATE_STEP[state]
    return max(stage, min(stage + step, cap))


def clean_reply(text: str) -> str:
    # 소형 모델이 가끔 섞는 한자/중국어 문장부호 제거
    text = re.sub(r"[㐀-鿿぀-ヿ，。！？☀-➿\U0001f300-\U0001faff️\"“”*#]+", "", text)
    # "(하품 소리)" 같은 괄호 지문 제거
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)
    # 한글에 붙어 나오는 영어 조각(예: "이slot") 제거
    text = re.sub(r"(?<=[가-힣])[A-Za-z]+|[A-Za-z]+(?=[가-힣])", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 통화체 유지를 위해 최대 3문장까지만 사용
    sentences = re.split(r"(?<=[.!?~…])\s+", text)
    # 길이 제한으로 끝맺지 못한 마지막 문장은 버린다
    if len(sentences) > 1 and not re.search(r"[.!?~…요다죠네]$", sentences[-1]):
        sentences = sentences[:-1]
    return " ".join(sentences[:3])


MAX_RETRIES = 1  # 통화 응답이 너무 늦어지지 않도록 재생성은 한 번만
SOFT_ISSUES = {"깨우는 중이라는 표현 없음", "너무 김"}
PRESENCE_RE = re.compile(r"깨우|깨워|깨울|일어나|일어날|일어나요|눈.{0,3}떠|눈.{0,3}뜨|전화|모닝콜|알람|끊|목소리|옆에|함께|같이|"
                         r"이불|하품|하나\s?둘|수화기|창문")
GIVEUP_RE = re.compile(r"더\s?주무|푹\s?주무|푹\s?자|잘\s?자요|더\s?자도|더\s?쉬|자도\s?(돼|괜찮)|쉬세요|그래도\s?괜찮|천천히\s?주무|누워\s?계세요|누워\s?있어도")
# 문장 끝(문장부호 또는 끝)에서만 반말 어미를 본다
BANMAL_RE = re.compile(r"(?:^|\s)(너를|너는|네가|너도|너의)\s|(?:^|\s)(떠|가|해|봐|와|자|일어나)(?:[!?.~]+|$)|[가-힣](야|니|지|자|어|아|나|가|해|봐|떠|워|와|져|쳐|라|래|냐|려|까|게|네|줘|데|거든|잖아|다구|구나)(?:[!?.~]+|$)|"
                       r"[가-힣](?<!니)(?<!까)다(?:[!?.~]+|$)|잖아,|[가-힣]야,")


def shares_phrase(a: str, b: str, n: int = 8) -> bool:
    # 공백을 뺀 n글자 이상이 겹치면 같은 문장을 되풀이한 것으로 본다
    a, b = a.replace(" ", ""), b.replace(" ", "")
    return any(a[i:i + n] in b for i in range(len(a) - n + 1))


def reply_issues(reply: str, user_msg: str, stage: int, first: bool, previous: list = ()) -> list:
    """작은 모델이 자주 내는 문제를 찾아낸다. 비어 있으면 통과."""
    issues = []
    if any(shares_phrase(reply, p) for p in previous):
        issues.append("이전 답변 반복")
    if GIVEUP_RE.search(reply):
        issues.append("포기 발언")
    if not reply or len(reply) < 8:
        issues.append("너무 짧음")
    if re.search(r"[A-Za-z]{2,}", reply) or re.search(r"\s(을|를|로|은|는|에|의)\s|시\s분", " " + reply):
        issues.append("영어 또는 깨진 조사")
    if BANMAL_RE.search(reply):
        issues.append("반말")
    if len(reply) > 110:
        issues.append("너무 김")
    if not first and "안녕" in reply:
        issues.append("반복 인사")
    if user_msg and len(user_msg) >= 5 and user_msg.replace(" ", "")[:6] in reply.replace(" ", ""):
        issues.append("사용자 말 따라 하기")
    if stage < LAST_STAGE and not PRESENCE_RE.search(reply):
        issues.append("깨우는 중이라는 표현 없음")
    if stage == LAST_STAGE and reply.rstrip().endswith("?"):
        issues.append("마무리인데 질문")
    return issues


def fix_vocative(reply: str, user_name: str) -> str:
    # "민수야", "민수아" 같은 반말 호칭은 대화 전체를 반말로 끌고 가므로 "민수님"으로 바꾼다
    if not user_name:
        return reply
    return re.sub(rf"{re.escape(user_name)}(야|아)(?=[\s!,.~?]|$)", f"{user_name}님", reply)


def generate_reply(
    model: str, system: str, messages: list, user_msg: str, stage: int,
    user_name: str = "", first: bool = False,
) -> str:
    # 문제가 있으면 다시 생성하고, 후보 중 문제가 가장 적은 답변을 고른다
    previous = [m["content"] for m in messages if m["role"] == "assistant"]
    best, best_issues = "", None
    attempt_messages = messages
    for _ in range(MAX_RETRIES + 1):
        t0 = time.time()
        reply = fix_vocative(call_ollama(model, system, attempt_messages), user_name)
        issues = reply_issues(reply, user_msg, stage, first, previous)
        log.info("답변 검사 %s %.1fs | %s", issues or "통과", time.time() - t0, reply)
        if best_issues is None or len(issues) < len(best_issues):
            best, best_issues = reply, issues
        # 가벼운 문제만 있으면 재생성하지 않는다 (응답 속도 우선)
        if not set(issues) - SOFT_ISSUES:
            break
        # 같은 조건으로 다시 뽑으면 같은 실수를 반복하므로, 무엇이 문제였는지 알려 주고 다시 생성
        feedback = f"\n(주의: 방금 만든 답변 \"{reply}\"은(는) {', '.join(issues)} 문제가 있습니다. " \
                   "그 문장과 다른 표현으로 새로 말하세요.)"
        last = attempt_messages[-1]
        attempt_messages = messages[:-1] + [{**last, "content": messages[-1]["content"] + feedback}]
    if stage == LAST_STAGE and not CLOSING_RE.search(best):
        best = f"{best} {CLOSING_LINE}"
    return best


def ollama_chat(model: str, messages: list, options: dict, **kwargs):
    # 모든 레이어를 GPU에 올리면 VRAM 4GB에서도 gemma3:4b가 약 40% 빨라진다.
    # 분류/생성/예열이 같은 num_gpu를 써야 모델이 다시 로드되지 않는다. 메모리가 부족하면 기본 설정으로 재시도.
    try:
        return ollama.chat(model=model, messages=messages, options={**options, "num_gpu": NUM_GPU},
                           keep_alive=KEEP_ALIVE, **kwargs)
    except ollama.ResponseError:
        return ollama.chat(model=model, messages=messages, options=options, keep_alive=KEEP_ALIVE, **kwargs)


def call_ollama(model: str, system: str, messages: list) -> str:
    try:
        response = ollama_chat(
            model,
            [{"role": "system", "content": system}] + messages,
            {"temperature": 0.7, "repeat_penalty": 1.15, "num_predict": 90},
        )
        log.info("ollama 프롬프트 %s토큰 %.1fs, 생성 %s토큰 %.1fs",
                 response.get("prompt_eval_count"), (response.get("prompt_eval_duration") or 0) / 1e9,
                 response.get("eval_count"), (response.get("eval_duration") or 0) / 1e9)
        return clean_reply(response["message"]["content"])
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
    last_ai = next(
        (m.content for m in reversed(request.history) if m.role == "assistant"), ""
    )
    state = classify_user_state(request.message)
    log.info("분류 %s | %s", state, request.message)
    stage = next_stage(max(0, min(request.conversation_level, LAST_STAGE)), state)

    system_prompt = build_system_prompt(
        request.user_name, request.user_schedule, request.weather_info
    )
    directive = build_turn_directive(
        request.user_schedule, request.weather_info, stage, state, last_ai,
        turn=len(request.history) // 2,
        ask_time=bool(re.search(r"몇\s?시|시간|시계", request.message)),
    )

    recent = request.history[-MAX_HISTORY:]
    messages = [{"role": m.role, "content": m.content} for m in recent]
    messages.append({"role": "user", "content": request.message + directive})

    reply = generate_reply(
        model, system_prompt, messages, request.message, stage, user_name=request.user_name
    )

    return ChatResponse(
        reply=reply,
        alarm_can_end=stage == LAST_STAGE,
        conversation_level=stage,
    )


@app.get("/api/initial-message")
async def initial_message(
    user_name: str = "사용자",
    weather_info: str = "",
    user_schedule: str = "",
    model: str = None,
):
    use_model = model or DEFAULT_MODEL
    system_prompt = build_system_prompt(user_name, user_schedule, weather_info)
    prompt = "(알람이 울려 방금 전화를 걸었습니다. 첫 인사를 해 주세요.)" + build_turn_directive(
        user_schedule, weather_info, stage=0, ask_time=True
    )

    reply = generate_reply(
        use_model, system_prompt, [{"role": "user", "content": prompt}], "", stage=0,
        user_name=user_name, first=True,
    )
    return {"message": reply}


@app.post("/api/warmup")
def warmup(model: str = None):
    # 알람 직전에 모델을 메모리에 올려 첫 인사 지연을 없앤다
    try:
        ollama_chat(model or DEFAULT_MODEL, [{"role": "user", "content": "안녕"}], {"num_predict": 1})
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
