"""대화 엔진: 사용자 대답 분류 → 단계 → 프롬프트 → 구절 단위 생성·검사 → 날씨 퀴즈.

서버(main.py)와 폰 앱(Chaquopy로 이 폴더를 그대로 넣는다)이 같은 코드를 쓴다. 그래서 웹에서 테스트한 대화가 폰에서도 똑같이 흘러간다.
AI 모델은 직접 부르지 않고, 메시지를 받아 토큰을 하나씩 돌려주는 함수(stream)를 받아 쓴다.
서버는 Ollama를, 폰은 LiteRT-LM(Gemma)을 넣는다. 이 파일은 표준 라이브러리만 쓴다(폰에는 pydantic이 없다).
"""
import logging
import random
import re
import time
from datetime import datetime
from typing import Callable, Iterable, Optional

import weather_quiz

log = logging.getLogger("uvicorn.error")

# 메시지 목록([{"role", "content"}], 맨 앞은 system)을 받아 토큰을 하나씩 내주는 함수
Stream = Callable[[list], Iterable[str]]

# 보낼 대화 기록 수. 기록 앞부분이 잘려 나가면 프롬프트 캐시가 깨져 매 턴 1초 이상 늦어지므로,
# 한 통화가 끝날 때까지 잘리지 않을 만큼 넉넉히 둔다
MAX_HISTORY = 30

# 0: 기상 확인, 1: 일정 공유, 2: 날씨/침대에서 나오기, 3: 날씨 퀴즈, 4: 마무리.
# 다 일어났다고 판단되면 바로 끝내지 않고 날씨 퀴즈를 맞혀야 마무리한다(weather_quiz.py). 날씨 정보가 없으면 퀴즈를 건너뛴다
QUIZ_STAGE = 3
LAST_STAGE = 4

# 각 단계에서 이번 답변이 할 일.
# 지시문은 매 턴 새로 붙어 프롬프트 캐시를 쓰지 못한다(설명 문장만으로 첫 토큰이 약 0.26초 늦어짐).
# 그래서 단계별·상황별 설명은 시스템 프롬프트에 표로 두고(캐시됨), 지시문에는 이름표만 넣는다.
STAGE_LABELS = {0: "1단계", 1: "2단계", 2: "3단계", 3: "날씨 퀴즈", 4: "마무리"}
STATE_LABELS = {"awake": "깸", "waking": "일어나는 중", "up": "움직임", "sleepy": "버팀", "hangup": "끊자는 말",
                "silent": "대답 없음", "other": "딴 얘기"}
STAGE_GOALS = {
    0: "아직 덜 깼으니 눈을 뜨게 하기.",
    1: "눈은 떴지만 누워 있으니, 일정({schedule}) 중 가장 이른 것 하나를 들어 몸을 일으키게 하기.",
    2: "날씨({weather})를 짧게 말하고 침대에서 나와 움직이게 하기. 다 일어나면 알려 달라고.",
    4: "완전히 일어났으니 기뻐하며 일정 하나 응원하고 '그럼 이제 끊을게요'로 마무리. 질문 금지.",
}

# 사용자 대답 상태: awake(깼다고 분명히 말함), waking(일어날게·응 같은 짧은 대답), up(몸이 침대 밖), sleepy(거부),
# hangup(끊자고 함), silent(대답 없음), other(질문/딴 얘기).
# 상태별로 "그 말에 어떻게 답할지"를 정한다. 정해진 재촉으로 넘어가지 않고 상대 말의 내용에 진짜로 답하게 하는 게 핵심
STATE_GUIDES = {
    "awake": "깼다는 말을 반가워하기.",
    "waking": "일어나려는 걸 반가워하되, 정말 일어났는지 행동 하나를 시켜 확인하기.",
    "up": "말한 행동을 구체적으로 짚어서 칭찬하기.",
    # 쓴 뒤에 검사(EXCUSE_OK_RE)로 막으면 이미 말한 앞부분 뒤에 대체 문장이 붙어 어색해지므로("재택은 그러니까
    # 민수님, 이불부터…") 애초에 덜 쓰게 한다. 금지 표현을 예시 문장('재택은 괜찮아요')으로 적으면 gemma3:4b가 오히려
    # 따라 해서("괜찮아요, 5분만 더 실컷 누워도 돼요") 원칙만 적는다
    "sleepy": "졸린 이유나 핑계의 내용을 구체적으로 받아치기. 더 자라는 허락, 핑계를 받아 주는 말, "
              "눈을 감으라는 말은 금지.",
    # 다 깼다고 한 뒤(3단계)에 끊자고 하면 마무리로 넘어가므로, 이 지시는 아직 덜 깼을 때만 쓰인다
    "hangup": "아직 다 일어난 걸 확인하지 못해 못 끊는다고 웃으며 버티고, 침대에서 나오면 끊겠다고 약속하기.",
    "silent": "대답이 없어 다시 잠든 것 같으니 이름을 부르며 아직 전화 안 끊었다고 하기.",
    "other": "질문엔 답, 감정엔 공감, 이야기엔 그 내용으로 대화.",
}

# 단계 이동: 깼다는 말은 한 단계씩, 몸을 움직였으면 두 단계씩 (마무리 조건은 next_stage 참고).
# 예전에는 깼다는 말로는 3단계에서 멈춰 "일어났다니까?"를 몇 번 해도 통화가 끝나지 않았다.
# "응", "일어날게" 같은 짧은 대답은 대충 해도 나오므로 예전처럼 3단계까지만
STATE_STEP = {"awake": (1, QUIZ_STAGE), "waking": (1, 2), "up": (2, QUIZ_STAGE), "sleepy": (0, QUIZ_STAGE),
              "hangup": (0, QUIZ_STAGE), "silent": (0, QUIZ_STAGE), "other": (0, QUIZ_STAGE)}

# 화면에서 음성 인식이 아무 말도 듣지 못했을 때 보내는 약속된 메시지 (static/index.html의 SILENT_MESSAGE)
SILENT_MESSAGE = "(대답 없음)"

# "누군가 나를 깨우고 있다"는 느낌을 주는 표현 아이디어 (매 턴 상황에 맞는 것을 하나 골라 지시에 넣는다)
PRESENCE_IDEAS = {
    # 아직 덜 깼거나 버틸 때
    "sleepy": [
        "일어날 때까지 절대 안 끊는다",
        "목소리가 아직 잠겨 있다",
        "옆에 있으면 이불을 걷어 주고 싶다",
        "알람 끄고 다시 자는 거 다 안다",
        "하품 소리가 수화기 너머로 다 들린다고 놀리기",
        "끝까지 깨울 테니 포기하라",
    ],
    # 깨어나서 움직이기 시작했을 때
    "awake": [
        "목소리가 점점 또렷해진다",
        "같이 하나 둘 셋 세고 일어나자",
        "창문 쪽을 한번 봐 달라",
        "다 일어날 때까지 전화로 옆에 있겠다",
        "깨운 보람이 조금씩 느껴진다",
    ],
}

PRESENCE_OFFSET = random.randrange(100)

# 마무리 단계에서 모델이 통화를 끝내는 말을 빠뜨리면 붙이는 문장
CLOSING_RE = re.compile(r"끊을게|끊어요|끊겠|통화|다음에|내일 또|이따")
CLOSING_LINE = "그럼 이제 전화 끊을게요, 좋은 하루 보내세요!"


def build_turn_directive(
    user_schedule: str, weather_info: str, stage: int, state: str = "", last_ai: str = "", turn: int = 0,
    ask_time: bool = False, user_message: str = "", step_goal: int = 0,
) -> str:
    # 소형 모델이 잘 따르도록 이번 턴 지시는 사용자 메시지 바로 뒤에 붙인다
    goal = STAGE_GOALS[stage].format(
        schedule=user_schedule or "특별한 일정 없음",
        weather=weather_info or "날씨 정보 없음",
    )
    # 깨우는 중이라는 느낌은 유지하되 같은 표현이 반복되지 않도록, 매 턴 다른 아이디어를 하나 준다
    presence = ""
    if stage < LAST_STAGE:
        mood = "sleepy" if stage == 0 or state in ("sleepy", "silent", "hangup") else "awake"
        ideas = PRESENCE_IDEAS[mood]
        # 턴마다 차례로 돌려 써서 같은 표현이 연달아 나오지 않게 한다 (시작 위치만 무작위)
        idea = ideas[(turn + PRESENCE_OFFSET) % len(ideas)]
        # 걸음 목표는 시스템 프롬프트에만 두면 소형 모델이 쓰지 않는다(4번 중 0번). 버티는 턴에 번갈아 아이디어로 준다
        if step_goal and mood == "sleepy" and user_message and turn % 2 == 1:
            idea = f"{step_goal}걸음만 걸으면 바로 끊어 주겠다"
        presence = f" / 느낌: {idea}"
    # 현재 시각을 매번 주면 모델이 매 턴 시간 얘기만 하므로, 시간을 물었을 때만 알려 준다
    now = ""
    if ask_time:
        now = f"\n지금 {datetime.now().strftime('%H시 %M분')}. 시각은 이 숫자 그대로 말하기."

    if not user_message and state != "silent":  # 첫 인사
        return f"\n\n[지시]{now} 첫 인사. {goal}{presence}"
    # 상황별 답하는 법과 단계별 깨우는 법은 시스템 프롬프트 표에 있다. 여기에는 이번 턴의 이름표만 넣어
    # 매 턴 새로 처리할 프롬프트를 줄인다 (설명 문장을 넣으면 첫 소리가 약 0.26초 늦어진다)
    return f"\n\n[지시]{now} 유형: {STATE_LABELS.get(state, '딴 얘기')} / 단계: {STAGE_LABELS[stage]}{presence}"


# ─── 사용자 프로파일 주입 (telemetry.py가 지난 아침 기록으로 만든 것) ─────────────────
# 말투 이름(telemetry.TONES)마다 어떻게 깨우는지. 프로파일에는 이름만 있으므로 여기서 구체적인 지시로 바꾼다
TONE_GUIDES = {
    "다정하게 달래기": "부드럽게 공감하며 달래듯 깨우기",
    "장난스럽게 놀리기": "가볍게 놀리며 웃게 만들어 깨우기",
    "단호하게 재촉하기": "짧고 단호하게 지금 당장 일어나라고 하기",
    "일정으로 현실 일깨우기": "오늘 일정과 남은 시간을 구체적으로 들어 현실을 일깨우기",
    "칭찬으로 북돋기": "작은 행동도 크게 칭찬하며 북돋기",
}
MIN_SESSIONS_FOR_TONE = 3  # 기록이 이보다 적으면 말투 분석을 믿기 어려워 넣지 않는다


def days_ago(date: str) -> str:
    try:
        n = (datetime.now().date() - datetime.fromisoformat(date).date()).days
    except ValueError:
        return date
    return {0: "오늘", 1: "어제", 2: "그저께"}.get(n, f"{n}일 전")


def build_profile_context(profile: Optional[dict], user_name: str) -> str:
    """프로파일을 시스템 프롬프트 맨 앞에 넣을 문단으로 만든다. 프로파일이 없으면 빈 문자열 (지금과 같은 프롬프트)"""
    if not profile:
        return ""
    name = user_name or profile.get("user_name") or "사용자"
    stats = profile.get("stats") or {}
    lines = []
    if stats.get("sessions"):
        streak = stats.get("streak") or {}
        line = f"- 최근 {stats['sessions']}번 아침 중 {stats.get('successes', 0)}번 제대로 일어났어요."
        if streak.get("count", 0) >= 2:
            line += f" 지금 {streak['count']}번 연속 {'성공' if streak['type'] == 'success' else '실패'} 중이에요."
        lines.append(line)
    failures = profile.get("recent_failures") or []
    if failures:
        last = failures[0]
        lines.append(f"- 최근 실패: {days_ago(last.get('date', ''))} — {last.get('reason', '')}.")
        if "다시 잠든" in last.get("reason", ""):
            lines.append("- 통화를 끊고 다시 잠든 적이 있으니, 침대에서 나왔다고 할 때까지 확실히 확인하세요.")
    if stats.get("sessions", 0) >= MIN_SESSIONS_FOR_TONE:
        good, bad = profile.get("effective_tone"), profile.get("ineffective_tone")
        if good in TONE_GUIDES:
            lines.append(f"- 잘 통한 방식: {TONE_GUIDES[good]}. 이 방식을 중심으로 깨우세요.")
        if bad in TONE_GUIDES and bad != good:
            lines.append(f"- 잘 안 통한 방식: {TONE_GUIDES[bad]}. 이 방식은 줄이세요.")
    excuses = profile.get("excuse_keywords") or []
    if excuses:
        lines.append(f"- 자주 대는 핑계: {', '.join(excuses)}. 이 핑계가 나오면 미리 알고 있었다는 듯 받아치세요.")
    memories = profile.get("long_term_memory") or []
    if memories:
        lines.append(f"- 기억해 둘 것: {', '.join(memories)}. 어울릴 때 한 번만 안부처럼 꺼내세요.")
    if not lines:
        return ""
    return f"## {name}님에 대해 알고 있는 것 (지난 아침 기록)\n" + "\n".join(lines) + "\n\n"

def build_system_prompt(user_name: str, user_schedule: str, weather_info: str, profile: str = "",
                        step_goal: int = 0) -> str:
    # 통화 중 바뀌지 않는 내용만 넣어 Ollama가 이 프롬프트의 처리 결과를 재사용하게 한다
    day_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    day_name = day_names[datetime.now().weekday()]
    schedule = user_schedule or "특별한 일정 없음"
    weather = weather_info or "날씨 정보 없음"
    # 매 턴 지시문에는 이름표만 들어가므로, 이름표별로 할 일을 여기에 표로 둔다 (캐시되어 매 턴 비용이 없다)
    state_table = "\n".join(f"- {STATE_LABELS[s]}: {g}" for s, g in STATE_GUIDES.items())
    stage_table = "\n".join(
        f"- {STAGE_LABELS[n]}: {g.format(schedule=schedule, weather=weather)}" for n, g in STAGE_GOALS.items()
    )
    # 걸음 목표가 없으면 이 줄이 빠져 예전과 같은 프롬프트가 된다
    steps = (f"\n걸음 목표: 폰을 들고 {step_goal}걸음 걸으면 통화가 바로 끝납니다. "
             f"버티면 \"{step_goal}걸음만 걸으면 끊어 드릴게요\"처럼 가끔 활용하세요.") if step_goal else ""

    return f"""{profile}당신은 {user_name}님을 매일 아침 전화로 깨워 주는 다정하고 끈기 있는 모닝콜 친구입니다.
지금 알람 시간이 되어 {user_name}님에게 전화를 걸었고, 통화가 연결된 상태입니다.
당신의 임무는 {user_name}님이 침대에서 완전히 나올 때까지 전화를 끊지 않고 깨우는 것입니다.

오늘: {day_name}
오늘 일정: {schedule}
날씨: {weather}{steps}

## 말하는 방식
- 실제 사람이 전화로 깨우듯 자연스러운 한국어 구어체로, 다정하지만 끈질기게 말합니다.
- 항상 "{user_name}님"이라고 부르고 "~요"로 끝나는 존댓말만 씁니다. 반말은 절대 쓰지 않습니다.
- 매 답변마다 "내가 지금 전화로 깨우고 있다"는 게 느껴지게 하되, 표현은 매번 바꾸세요.
- 상대가 방금 한 말을 재치 있게 받아친 다음 일어나라고 합니다. 핑계를 대면 부드럽게 반박합니다.
- 당신은 깨우는 사람입니다. 상대의 말이나 행동을 대신 연기하지 말고, 괄호로 된 지문도 쓰지 마세요.
- 1~2문장, 60자 안팎으로 짧게 말합니다. 같은 말을 되풀이하지 마세요.
- "안녕하세요"는 첫 인사에서만 씁니다. 영어, 한자, 이모지는 쓰지 마세요.
- 지금 시각은 [지시]에 시각이 적혀 있을 때만 그 숫자 그대로 말하고, 그 밖에는 지금 몇 시인지 말하지 마세요.
- 단계가 '마무리'가 아니면 "끊을게요", "좋은 하루 보내세요"처럼 통화를 끝내는 말은 하지 마세요.
- 일정이나 날씨는 위에 적힌 사실만 말하고, 이번 목표에 없는 내용을 미리 몰아서 말하지 마세요.

## 대답하는 순서 (매번 지키기)
상대 말 끝에 [지시] 유형 / 단계 / 느낌이 붙어 옵니다. 상대에게는 보이지 않으니 언급하지 마세요.
1) 먼저 상대가 방금 한 말의 내용에 진짜로 답합니다. 유형별로 이렇게 답합니다.
{state_table}
2) 그다음 한 문장으로 깨웁니다. 단계별로 이렇게 깨웁니다.
{stage_table}
- "일어나셔야죠" 같은 짧은 재촉을 반복하지 말고 일정·날씨·행동을 구체적으로 들어 말합니다.
- '느낌'은 어울릴 때만 자기 말로 녹입니다.
- 정해진 틀처럼 말하지 말고, 앞에서 한 표현은 다시 쓰지 마세요."""


# 명확한 대답은 규칙으로 바로 판단. 특정 문장이 아니라 일반적인 표현 패턴으로 잡는다.
# 순서 중요: 거부/핑계 → 일어나는 중 → 몸을 움직임 → 깼다고 함 → 끊자고 함 → 질문 → 짧은 대답
STATE_RULES = [
    ("sleepy", re.compile(
        # 일어나기 거부·부정
        r"안\s?일어|못\s?일어|못\s?나가|안\s?깼|못\s?(해|하겠)|싫|귀찮|힘들|깨우지\s?마|"
        # 더 자려 함
        r"졸려|졸리|피곤|자고\s?싶|더\s?(잘|자|누워|있을)|(만|좀)\s?더|\d+\s?분만|이따|잘게|잘래|잘\s?거야|잔다|"
        r"다시\s?(누|잘|자)|자는\s?중|자고\s?있|누워\s?있|이불\s?속|음냐|쿨쿨|zz|"
        r"그만\s?깨|자게\s?해|꺼\s?줘|끄고\s?잘|봐\s?줘|"
        # 몸 상태·상황을 핑계로 댐
        r"아파|무거|눈이\s?안|추워서|새벽|주말이(면|었으면)|늦게\s?잤|"
        r"휴강|결석|결근|안\s?가도|쉴래|쉬고\s?싶|오늘은\s?안|면\s?안\s?(되나|돼|될까)|안\s?되나|"
        r"재택|쉬면|빠져도|늦어도|지각해도|병가|연차")),
    # "일어나는 중", "눈 뜨고 있어"는 아직 침대 안이므로 아래 '움직이는 중' 패턴보다 먼저 본다
    ("waking", re.compile(r"일어나는\s?중|일어나고\s?있|눈\s?뜨고|정신\s?(차|들)|기지개|침대에\s?앉")),
    # 꿈 이야기는 "나왔어" 같은 말이 섞여도 몸을 움직였다는 뜻이 아니다
    ("other", re.compile(r"꿈")),
    ("up", re.compile(
        # 장소·씻기 등 침대 밖 활동
        r"침대.{0,4}나왔|나왔어|나와\s?있|씻|세수|양치|이\s?닦|샤워|머리\s?감|화장|화장실|거실|부엌|"
        # "~하는 중", "~하고 있어": 무언가를 하고 있음 ("다이어트 중"처럼 명사 + 중은 제외)
        r"[가-힣]는\s?중|[가-힣]고\s?있(어|음|다|는)|(출근|외출|등교|나갈)\s?준비|"
        # 몸을 움직인 동작의 과거형
        r"(?:^|\s)(열었|신었|켰|껐|챙겼|입었|닦았|감았|마셨|먹었|나갔|갰|개었|정리했|차렸|스트레칭)|"
        # "보일러 켜고 왔어", "세수하러 왔어"처럼 다녀왔다는 표현
        r"[가-힣]\s?(고|서|러)\s?왔어|다녀왔|갔다\s?왔")),
    # 깼다고 분명히 말함. "아니 일어났다니까?"처럼 물음표가 붙어도 질문이 아니므로 질문 규칙보다 먼저 본다
    ("awake", re.compile(r"일어(났|남)|깨(어|었)|깼|눈\s?(떴|떠\s?있)|일으켰|앉았|내렸")),
    # "끊어 줘", "끊을 거라면서" (거부·핑계와 함께 나오면 위에서 버팀으로 잡힌다)
    ("hangup", re.compile(r"끊")),
    ("other", re.compile(r"\?|뭐|몇\s?시|언제|어디|왜|어때|무슨|누구")),
    ("waking", re.compile(
        r"일어(나|날|난)|깨\s?있|정신\s?(차|들)|알았어|알겠어|ㅇㅋ|오케|^\s*(응|어|네|넹|웅)+\s*$|준비(해야|할게)")),
]


def classify_by_rules(user: str) -> Optional[str]:
    for state, pattern in STATE_RULES:
        if pattern.search(user):
            return state
    return None


def classify_user_state(user: str) -> str:
    # 소형 모델 분류는 대부분 awake로 쏠려(gemma3:4b 단독 정확도 16/40) 통화가 너무 일찍 끝난다.
    # 규칙으로 판단하지 못한 대답은 other로 두어 단계는 유지하고, 답해 준 뒤 다시 깨우게 한다.
    # 빈 메시지는 음성 인식이 소리만 듣고 글자를 못 알아들은 것 (예전 화면은 그대로 보냈다). 첫 인사로 착각하지 않게 대답 없음으로
    if user.strip() in ("", SILENT_MESSAGE):
        return "silent"
    return classify_by_rules(user) or "other"


def next_stage(stage: int, state: str) -> int:
    # 규칙 분류는 처음 보는 표현에서 약 60%만 맞고, 가끔 질문이나 투정을 "up"으로 오판한다.
    # 오판 한 번으로 통화가 끝나지 않도록, 마무리는 3단계("다 일어나면 알려 달라")에서만 한다.
    # 거기서 몸을 움직였다(up)·깼다(awake)·이제 끊자(hangup)고 하면 마무리로 넘어간다
    if state == "hangup":
        return QUIZ_STAGE if stage >= 2 else stage
    if state == "up" and stage < 2:
        return 2
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
GIVEUP_RE = re.compile(r"더\s?주무|푹\s?주무|푹\s?자|잘\s?자요|더\s?자도|더\s?쉬|자도\s?(돼|괜찮)|쉬세요|그래도\s?괜찮|천천히\s?주무|누워\s?계세요|누워\s?있어도|더\s?누우|더\s?누워|더\s?계시|더\s?있어도|더\s?계셔도|더\s?잘\s?계|좀\s?더\s?주무|더\s?자요")
# 위 규칙을 빠져나간 넘어가는 말 (tools/eval_excuses.py로 핑계 10가지를 보내 모음):
# 다시 자라는 말("조금만 더 눈을 감고 계세요"), 잠깐은 괜찮다는 말("잠깐이라도 괜찮아요"),
# 핑계 허락("재택은 괜찮지만", "휴강이라니, 정말 좋겠네요"). "재택은 안 돼요"처럼 사이에 안·못·아니가 있으면 깨우는 말이라 제외
EXCUSE_OK_RE = re.compile(
    r"눈을?\s?(좀\s?)?(더\s?)?감(고\s?(계세요|있어도|계셔도|있으세요|쉬)|으세요|아도|으셔도|으시면)|눈\s?붙(이|였)|"
    r"(쉬셔도|쉬어도)\s?(돼|괜찮)|"
    r"잠깐(이라도|만)?\s?(더\s?)?(괜찮아요|자도|쉬어도|누워\s?있어도|눈\s?감아도)|누워도\s?(돼|괜찮)|"
    r"누우셔도\s?(돼|괜찮)|더\s?자야(죠|해요|겠)|(졸리|피곤하)시면\s?괜찮|"
    r"(재택|휴강|결석|지각|늦어도|늦게|수업|출근|회사|학교|알바)(?:(?!안|못|아니)[^.!?]){0,10}"
    r"(괜찮아요|괜찮지만|괜찮죠|괜찮습니다|좋겠네요|좋죠|좋아요|돼요|되죠|됩니다)")
# "늦어도 괜찮지 않아?"처럼 허락을 구했는데 "괜찮아요, 민수님."으로 시작하는 대답 (대답만 봐서는 알 수 없어 질문과 같이 본다)
PERMISSION_ASK_RE = re.compile(r"괜찮지|괜찮을까|괜찮겠|해도\s?(돼|되|괜찮)|안\s?되나|안\s?될까|되지\s?않")
PERMISSION_YES_RE = re.compile(r"\s*(네|응|그럼요|물론|괜찮아요|괜찮습니다|좋아요|돼요|됩니다)[\s,.!~]")
# 소형 모델이 가끔 섞는 키릴·태국·아랍·그리스 문자 ("열 пятнадцать 분"). 지우면 말이 깨지므로 문제로 잡아 다시 생성한다
FOREIGN_RE = re.compile(r"[\u0370-\u03FF\u0400-\u04FF\u0600-\u06FF\u0E00-\u0E7F]")
# "지금 열한 시 정분입니다"처럼 묻지도 않은 지금 시각을 지어내는 말. 일정의 시각("10시 발표")은 잡지 않도록
# '지금·현재·벌써' 바로 뒤의 시각과, 실제로는 없는 말인 '정분'만 본다
HOUR_WORD = r"(\d{1,2}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열[가-힣]?|스물[가-힣]?|이십[가-힣]{0,2}|십[가-힣]?)"
NOW_TIME_RE = re.compile(rf"(지금|현재|벌써)\s?(은|는|이)?\s?(오전|오후|아침)?\s?{HOUR_WORD}\s?시(?![간작])|정분")
# 마무리 단계가 아닌데 통화를 끝내는 말 ("전화 안 끊을게요"는 깨우는 말이라 제외)
EARLY_CLOSE_RE = re.compile(r"(?<!안 )(?<!안)(?<!못 )(?<!못)(끊을게|끊겠|끊어요|끊습니다)|"
                            r"좋은\s?하루\s?(보내|시작|되)|하루\s?(잘|즐겁게|힘차게)\s?보내|다음에\s?또|내일\s?또\s?(전화|깨워)")
# 문장 끝(문장부호 또는 끝)에서만 반말 어미를 본다
BANMAL_RE = re.compile(r"(?:^|\s)(너를|너는|네가|너도|너의)\s|(?:^|\s)(떠|가|해|봐|와|자|일어나)(?:[!?.~]+|$)|[가-힣](야|니|지|자|어|아|나|가|해|봐|떠|워|와|져|쳐|라|래|냐|려|까|게|네|줘|데|거든|잖아|다구|구나)(?:[!?.~]+|$)|"
                       r"[가-힣](?<!니)(?<!까)다(?:[!?.~]+|$)|잖아,|[가-힣]야,")


def shares_phrase(a: str, b: str, n: int = 8) -> bool:
    # 공백·문장부호를 뺀 n글자 이상이 겹치면 같은 문장을 되풀이한 것으로 본다
    # ("아이고, 지은님!" 같은 감탄과 호칭은 문장부호를 빼면 짧아서 반복으로 보지 않는다)
    a, b = normalize(a), normalize(b)
    return any(a[i:i + n] in b for i in range(len(a) - n + 1))


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]", "", text)


def sentence_key(sentence: str) -> str:
    # 호칭("지은님")을 뺀 문장 내용. "아이고, 지은님!"처럼 감탄·호칭뿐인 짧은 문장은 비교하지 않는다
    key = normalize(re.sub(r"[가-힣]+님", "", sentence))
    return key if hangul_count(key) >= 4 else ""


def repeats_previous(reply: str, previous: list) -> bool:
    # 긴 구절이 겹치거나, "일어나셔야죠!"처럼 짧아도 앞에서 똑같이 한 문장이면 반복이다
    if any(shares_phrase(reply, p) for p in previous):
        return True
    split = lambda text: re.split(r"(?<=[.!?~…])\s+", text)
    said = {sentence_key(s) for p in previous for s in split(p)} - {""}
    return any(sentence_key(s) in said for s in split(reply))


def reply_issues(reply: str, user_msg: str, stage: int, first: bool, previous: list = ()) -> list:
    """작은 모델이 자주 내는 문제를 찾아낸다. 비어 있으면 통과."""
    issues = []
    if repeats_previous(reply, previous):
        issues.append("이전 답변 반복")
    if GIVEUP_RE.search(reply) or EXCUSE_OK_RE.search(reply) or \
            (PERMISSION_ASK_RE.search(user_msg) and PERMISSION_YES_RE.match(reply)):
        issues.append("포기 발언")
    if not reply or len(reply) < 8:
        issues.append("너무 짧음")
    if re.search(r"[A-Za-z]{2,}", reply) or FOREIGN_RE.search(reply) or \
            re.search(r"\s(을|를|로|은|는|에|의)\s|시\s분", " " + reply):
        issues.append("외국어 또는 깨진 조사")
    if BANMAL_RE.search(reply):
        issues.append("반말")
    if len(reply) > 110:
        issues.append("너무 김")
    if not first and "안녕" in reply:
        issues.append("반복 인사")
    # 상대 말을 통째로 앵무새처럼 따라 할 때만 문제로 본다.
    # "새벽 3시에 주무셨다니"처럼 일부를 받아서 대답하는 건 자연스러운 대화다
    said = normalize(user_msg)
    if len(said) >= 8 and said in normalize(reply):
        issues.append("사용자 말 따라 하기")
    if stage < LAST_STAGE and not PRESENCE_RE.search(reply):
        issues.append("깨우는 중이라는 표현 없음")
    if stage == LAST_STAGE and reply.rstrip().endswith("?"):
        issues.append("마무리인데 질문")
    # 첫 인사와 "지금 몇 시야"에는 [지시]에 실제 시각을 주므로 말해도 된다
    if not first and not asks_current_time(user_msg) and NOW_TIME_RE.search(reply):
        issues.append("묻지 않은 시각")
    if stage < LAST_STAGE and EARLY_CLOSE_RE.search(reply):
        issues.append("이른 마무리")
    return issues


def fix_vocative(reply: str, user_name: str) -> str:
    # "민수야", "민수아" 같은 반말 호칭은 대화 전체를 반말로 끌고 가므로 "민수님"으로 바꾼다
    if not user_name:
        return reply
    return re.sub(rf"{re.escape(user_name)}(야|아)(?=[\s!,.~?]|$)", f"{user_name}님", reply)


def generate_reply(
    stream: Stream, system: str, messages: list, user_msg: str, stage: int,
    user_name: str = "", first: bool = False,
) -> str:
    # 문제가 있으면 다시 생성하고, 후보 중 문제가 가장 적은 답변을 고른다
    previous = [m["content"] for m in messages if m["role"] == "assistant"]
    best, best_issues = "", None
    attempt_messages = messages
    for _ in range(MAX_RETRIES + 1):
        t0 = time.time()
        raw = "".join(stream([{"role": "system", "content": system}] + attempt_messages))
        reply = fix_vocative(clean_reply(raw), user_name)
        issues = reply_issues(reply, user_msg, stage, first, previous)
        log.debug("답변 검사 %s %.1fs | %s", issues or "통과", time.time() - t0, reply)
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


# ─── 스트리밍: 문장이 완성되는 대로 보낸다 ─────────────────────────────────────────
# 답변 전체를 기다리면 첫 소리까지 3~5초(재생성 시 두 배)가 걸린다. 생성 속도(한글 약 20음절/초)가
# 말하는 속도(약 6음절/초)보다 훨씬 빠르므로, 첫 문장만 기다렸다가 읽기 시작해도 중간에 끊기지 않는다.
MAX_SENTENCES = 3
# 문장 단위에서도 걸러야 하는 문제. 길이·깨우는 표현 같은 답변 전체 기준 검사는 제외
SENTENCE_ISSUES = {"반말", "외국어 또는 깨진 조사", "포기 발언", "이전 답변 반복", "사용자 말 따라 하기", "반복 인사",
                   "묻지 않은 시각", "이른 마무리"}
SENTENCE_END_RE = re.compile(r"[.!?~…]+[\"')\]]*\s")
# 사람처럼 "아하," "에이," 정도에서 바로 말을 시작하도록 쉼표에서도 끊는다 (너무 짧은 조각은 붙여서)
PHRASE_END_RE = re.compile(r"(?:[.!?~…]+[\"')\]]*|,)\s")
MIN_PHRASE_SYLLABLES = 2
# 구절이 쉼표 없이 길면("오늘 날씨는 맑고 12도예요!") 문장 끝까지 기다리느라 첫 소리가 늦거나,
# 앞 구절을 다 읽고 나서 말이 끊긴다. 폰이 지금까지 보낸 말을 다 읽을 때가 다가오면
# 이만큼 모인 말을 띄어쓰기 위치에서 미리 보낸다.
EARLY_CHUNK_SYLLABLES = 3
SPEAK_SYLLABLES_PER_SEC = 8.0  # 폰 음성 합성 속도 (빠른 쪽으로 잡아 일찍 보내게)
EARLY_FLUSH_MARGIN_S = 0.3     # 읽을 말이 이 시간보다 적게 남으면 미리 보낸다
COMPLETE_END_RE = re.compile(r"[.!?~…요다죠네]$")
# 모델이 아무 말도 못 했거나 심각한 문제(포기 발언·반말·외국어)로 문장 중간에서 멈췄을 때만 쓰는 문장.
# 같은 문장이 반복되면 정해진 대본처럼 들리므로 돌려 쓴다
FALLBACK_LINES = {
    "wake": [
        "{name}님, 제 목소리 들리세요? 아직 전화 안 끊었어요.",
        "아무튼 {name}님, 이제 눈 좀 떠 봐요!",
        "{name}님, 저 아직 여기 있어요. 일어날 때까지 기다릴게요.",
        "그러니까 {name}님, 이불부터 살짝 걷어 볼까요?",
    ],
    "end": "깨운 보람이 있네요! " + CLOSING_LINE,
}
# 말을 멈춰야 하는 심각한 문제. 반복·지어낸 시각·이른 마무리는 그 문장만 건너뛰고 계속 말한다
STOP_ISSUES = SENTENCE_ISSUES - {"이전 답변 반복", "묻지 않은 시각", "이른 마무리"}



def hangul_count(text: str) -> int:
    return sum(1 for ch in text if "가" <= ch <= "힣")


def split_phrases(tokens):
    """토큰을 이어 붙이다가 문장 끝이나 쉼표가 나오면 한 구절씩 (구절, 문장 중간인지) 로 내보낸다.
    음성 합성 대기열에 바로 넣을 수 있도록 작게 끊되, 2음절 미만 조각은 다음 구절에 붙인다."""
    buffer = ""
    start = 0  # buffer 안에서 아직 끊을 곳을 찾지 않은 위치
    speak_end = 0.0  # 지금까지 보낸 말을 폰이 다 읽을 것으로 예상되는 시각 (0이면 아직 보낸 말 없음)

    def emit(text: str):
        nonlocal speak_end
        now = time.time()
        speak_end = max(now, speak_end) + hangul_count(text) / SPEAK_SYLLABLES_PER_SEC

    for token in tokens:
        buffer += token
        while (match := PHRASE_END_RE.search(buffer, start)):
            if hangul_count(buffer[:match.end()]) < MIN_PHRASE_SYLLABLES:
                start = match.end()
                continue
            phrase = buffer[:match.end()].strip()
            emit(phrase)
            yield phrase, False
            buffer, start = buffer[match.end():], 0
        # 읽을 말이 곧 떨어지면(또는 아직 아무 말도 안 했으면) 문장부호를 기다리지 않고 띄어쓰기에서 보낸다
        if time.time() + EARLY_FLUSH_MARGIN_S >= speak_end:
            cut = buffer.rfind(" ")
            if cut > 0 and hangul_count(buffer[:cut]) >= EARLY_CHUNK_SYLLABLES:
                phrase = buffer[:cut].strip()
                emit(phrase)
                yield phrase, True
                buffer, start = buffer[cut + 1:], 0
    rest = buffer.strip()
    # 길이 제한으로 중간에 끊긴 마지막 구절은 버린다
    if rest and COMPLETE_END_RE.search(rest):
        yield rest, False


def stream_reply(
    stream: Stream, system: str, messages: list, user_msg: str, stage: int,
    user_name: str = "", first: bool = False,
):
    """구절을 하나씩 검사하며 내보낸다.
    첫 구절에 문제가 있으면 아직 아무 말도 안 했으므로 문제를 알려 주고 다시 생성하고,
    이미 말하기 시작한 뒤의 구절에 문제가 있으면 그 구절을 버리고 거기서 말을 끝낸다."""
    previous = [m["content"] for m in messages if m["role"] == "assistant"]
    spoken = []
    sentence_so_far = ""  # 아직 끝나지 않은 문장에서 이미 보낸 부분
    sentence_start = 0    # spoken에서 그 문장이 시작된 위치
    sentence_count = 0
    attempt_messages = messages
    for attempt in range(MAX_RETRIES + 1):
        retry_feedback = None
        stopped = False  # 문제가 있어 말을 중간에 멈췄는지
        tokens = iter(stream([{"role": "system", "content": system}] + attempt_messages))
        phrases = split_phrases(tokens)
        skipping = False  # 반복이라 건너뛰기로 한 문장의 나머지 구절을 버리는 중
        try:
            for raw, mid_sentence in phrases:
                phrase = fix_vocative(clean_reply(raw), user_name)
                if not phrase:
                    continue
                ends_sentence = not mid_sentence and not phrase.endswith(",")
                if skipping:
                    skipping = not ends_sentence
                    continue
                # 구절이 문장 중간이나 쉼표에서 나뉘면 "5분만 더 | 누워볼게요"처럼 패턴이 갈라져 검사를 빠져나가므로,
                # 같은 문장의 앞부분과 이어 붙여 문장 단위로 검사한다
                sentence = f"{sentence_so_far} {phrase}".strip()
                if not sentence_so_far:
                    sentence_start = len(spoken)
                # 반복 검사는 이전 답변과 이번 답변의 앞 문장들과만 (같은 문장의 앞부분과는 비교하지 않는다)
                issues = [i for i in reply_issues(sentence, user_msg, stage, first, previous + spoken[:sentence_start])
                          if i in SENTENCE_ISSUES]
                # 문장 중간에서 끊은 구절은 "씨가", "같이"처럼 반말 어미로 끝나 보일 수 있으므로 반말 검사는
                # 문장 끝 구절에서 한다
                if mid_sentence and "반말" in issues:
                    issues.remove("반말")
                log.debug("구절 검사 %s | %s", issues or "통과", sentence)
                if any(i in STOP_ISSUES for i in issues):
                    if not spoken:
                        # 아직 아무 말도 안 했으면 문제를 알려 주고 다시 생성
                        retry_feedback = f"\n(주의: 방금 시작한 \"{phrase}\"은(는) {', '.join(issues)} 문제가 " \
                                         "있습니다. 다른 표현으로 새로 말하세요.)"
                    stopped = True
                    break  # 포기 발언·반말·외국어 등은 거기서 말을 멈춘다
                if issues and not sentence_so_far:
                    # 앞에서 한 말을 되풀이하거나 시각을 지어내거나 미리 끊겠다는 문장은 건너뛰고 다음 문장으로 넘어간다
                    skipping = not ends_sentence
                    continue
                # (이미 말하기 시작한 문장이 반복이면 끊긴 문장을 남기지 않도록 끝까지 말한다)
                spoken.append(phrase)
                yield phrase
                sentence_so_far = "" if ends_sentence else sentence
                if ends_sentence:
                    sentence_count += 1
                    if sentence_count >= MAX_SENTENCES:
                        break
        finally:
            phrases.close()
            # 남은 생성을 멈춘다 (서버: Ollama 연결 종료, 폰: 생성 취소). 폰의 스트림은 저절로 멈추지 않는다
            close = getattr(tokens, "close", None)
            if close:
                close()
        if not retry_feedback or attempt == MAX_RETRIES:
            break
        attempt_messages = messages[:-1] + [{**messages[-1], "content": messages[-1]["content"] + retry_feedback}]

    if stage == LAST_STAGE:
        if not CLOSING_RE.search(" ".join(spoken)):
            yield CLOSING_LINE if spoken else FALLBACK_LINES["end"]
    elif sentence_so_far or hangul_count(" ".join(spoken)) < 6 or \
            (stopped and not PRESENCE_RE.search(" ".join(spoken))):
        # 문장 중간에서 멈췄거나 "에이," 정도만 말했으면, 끊긴 채로 두지 않고 깨우는 말로 마무리한다.
        # 문제가 있어 멈췄는데 앞 문장이 공감뿐이면("휴강일 수도 있겠네요?") 깨우는 말 없이 끝나지 않게 한다
        yield random.choice(FALLBACK_LINES["wake"]).format(name=user_name or "")


def asks_current_time(message: str) -> bool:
    # "지금 몇 시야"처럼 현재 시각을 물을 때만. "회의 몇 시였지?"에 현재 시각을 주면 모델이 둘을 섞어 틀리게 답한다
    if re.search(r"회의|약속|예약|수업|일정|출근|미팅", message):
        return False
    # "진짜 아홉시 30분이야?", "벌써 4시야?"처럼 AI가 말한 시각을 되묻는 것도 시각 질문이다
    return bool(re.search(r"몇\s?시|시간|시계|시(\s?\d+\s?분)?\s?(이야|야|이에요|예요|맞아|인가|라고)?\s?\?", message))


class TurnRequest:
    """화면이 보내는 대화 한 턴 (서버는 pydantic으로 검사한 뒤, 폰은 JSON 그대로 넘긴다)"""

    def __init__(self, message: str, history: list = (), user_name: str = "사용자", user_schedule: str = "",
                 weather_info: str = "", conversation_level: int = 0, step_goal: int = 0,
                 weather: Optional[dict] = None, quiz_attempts: int = 0, **_ignored):
        self.message = message
        self.history = [{"role": m["role"], "content": m["content"]} for m in history]
        self.user_name = user_name or "사용자"
        self.user_schedule = user_schedule or ""
        self.weather_info = weather_info or ""
        self.conversation_level = int(conversation_level)
        self.step_goal = int(step_goal or 0)
        self.weather = weather_quiz.Weather.from_dict(weather) if weather else None
        self.quiz_attempts = int(quiz_attempts or 0)


def prepare_turn(request: TurnRequest, profile: str = ""):
    """사용자 대답을 분류해 이번 단계를 정하고, 모델에 보낼 시스템 프롬프트와 메시지를 만든다.
    profile: 지난 아침 기록으로 만든 문단 (build_profile_context). 통화 중에는 첫 인사 때의 것을 그대로 쓴다"""
    last_ai = next(
        (m["content"] for m in reversed(request.history) if m["role"] == "assistant"), ""
    )
    state = classify_user_state(request.message)
    log.debug("분류 %s | %s", state, request.message)
    stage = next_stage(max(0, min(request.conversation_level, LAST_STAGE)), state)
    if stage == QUIZ_STAGE:  # 날씨 정보가 없어 퀴즈를 낼 수 없다 (있으면 quiz_turn이 먼저 처리한다)
        stage = LAST_STAGE

    system_prompt = build_system_prompt(
        request.user_name, request.user_schedule, request.weather_info, profile, request.step_goal,
    )
    directive = build_turn_directive(
        request.user_schedule, request.weather_info, stage, state, last_ai,
        turn=len(request.history) // 2,
        ask_time=asks_current_time(request.message),
        # 대답 없음은 인용할 말이 없다
        user_message="" if state == "silent" else request.message,
        step_goal=request.step_goal,
    )

    messages = list(request.history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": request.message + directive})
    return stage, system_prompt, messages


def quiz_turn(request: TurnRequest):
    """날씨 퀴즈를 시작하거나 대답을 채점하는 턴이면 (단계, 대답, 틀린 횟수), 아니면 None.
    퀴즈는 숫자가 정확해야 해서 AI 대신 코드가 말한다(AI를 부르지 않아 바로 대답한다)"""
    if request.weather is None:
        return None
    quiz = weather_quiz.make_quiz(request.weather)
    current = max(0, min(request.conversation_level, LAST_STAGE))
    state = classify_user_state(request.message)
    if current == QUIZ_STAGE:
        turn = weather_quiz.answer_turn(quiz, request.message, request.quiz_attempts, request.user_name or "",
                                        silent=state == "silent")
        log.debug("퀴즈 %s | %s → %s", quiz.question, request.message, turn.text)
        return (LAST_STAGE if turn.passed else QUIZ_STAGE), turn.text, turn.attempts
    if next_stage(current, state) == QUIZ_STAGE:
        return QUIZ_STAGE, weather_quiz.enter_text(quiz, state), 0
    return None


def quiz_events(request: TurnRequest, stage: int, reply: str, attempts: int):
    """퀴즈 대답을 스트리밍과 같은 형식으로 보낸다. 문장 단위로 나눠 화면이 바로 읽기 시작하게 한다"""
    yield {"type": "meta", "conversation_level": stage, "alarm_can_end": stage == LAST_STAGE,
           "user_content": request.message, "quiz_attempts": attempts}
    for sentence in re.split(r"(?<=[.!?])\s+", reply):
        if sentence:
            yield {"type": "sentence", "text": sentence}
    yield {"type": "done", "stats": {}}


def chat_events(request: TurnRequest, stream: Stream, profile: str = "", stats: Optional[dict] = None,
                error_detail: Callable[[Exception], str] = str):
    """대화 한 턴을 이벤트로 낸다: meta(단계) → sentence(구절)… → done, 실패하면 error.
    서버는 한 줄에 JSON 하나(NDJSON)로, 폰은 화면에 바로 넘긴다. stats는 스트림이 채운 생성 시간 기록"""
    quiz = quiz_turn(request)
    if quiz:
        yield from quiz_events(request, *quiz)
        return
    stage, system_prompt, messages = prepare_turn(request, profile)
    # 단계는 생성 전에 정해지므로 먼저 보낸다.
    # user_content: 모델에 실제로 보낸 사용자 메시지(지시문 포함). 화면이 이걸 대화 기록에 저장해 다음 턴에
    # 그대로 보내면, 앞부분이 이번 프롬프트와 똑같아져 모델이 처리 결과를 재사용한다 (첫 소리 약 0.5초 단축)
    yield {"type": "meta", "conversation_level": stage, "alarm_can_end": stage == LAST_STAGE,
           "user_content": messages[-1]["content"], "quiz_attempts": 0}
    try:
        for phrase in stream_reply(stream, system_prompt, messages, request.message, stage,
                                   user_name=request.user_name):
            yield {"type": "sentence", "text": phrase}
        yield {"type": "done", "stats": stats or {}}
    except Exception as e:
        log.exception("대답 생성 실패")
        yield {"type": "error", "detail": error_detail(e)}


def initial_turn(user_name: str, user_schedule: str, weather_info: str, profile: str = "", step_goal: int = 0):
    """알람이 울렸을 때 첫 인사용 (시스템 프롬프트, 메시지). profile은 이 통화가 끝날 때까지 그대로 쓴다"""
    system_prompt = build_system_prompt(user_name, user_schedule, weather_info, profile, step_goal)
    prompt = "(알람이 울려 방금 전화를 걸었습니다. 첫 인사를 해 주세요.)" + build_turn_directive(
        user_schedule, weather_info, stage=0, ask_time=True
    )
    return system_prompt, [{"role": "user", "content": prompt}]


def initial_reply(stream: Stream, user_name: str, user_schedule: str, weather_info: str, profile: str = "",
                  step_goal: int = 0) -> dict:
    """첫 인사. user_content를 대화 기록 맨 앞에 두면 첫 대답부터 프롬프트 캐시가 재사용된다"""
    system_prompt, messages = initial_turn(user_name, user_schedule, weather_info, profile, step_goal)
    reply = generate_reply(stream, system_prompt, messages, "", stage=0, user_name=user_name, first=True)
    return {"message": reply, "user_content": messages[0]["content"], "system": system_prompt}
