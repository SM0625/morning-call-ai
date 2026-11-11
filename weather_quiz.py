"""날씨 퀴즈: 통화를 끝내기 전 마지막으로 정말 깼는지 확인한다.

대화로 일어났다고 판단되면 바로 끝내지 않고, 그날 날씨 조건에 맞는 질문을 하나 낸다.
비 소식이 있으면 강수확률, 추우면 최저 기온, 더우면 최고 기온, 일교차가 크면 최고 기온, 그 밖에는 지금 기온을 묻는다.
틀리면 "더 높아요/더 낮아요"로 힌트를 주는 업다운 게임으로 이어 가고, 맞히면 그날 날씨에 맞는 한마디와 함께 통화를 끝낸다.

질문과 채점은 AI가 아니라 코드가 한다. 소형 모델은 숫자를 자주 지어내고(예: "지금 열한 시 정분") 업다운을 틀리게
말할 수 있기 때문이다. 대신 문장을 여러 개 준비해 돌려 써서 같은 말만 반복하지 않게 한다.
"""
import random
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

MAX_ATTEMPTS = 4  # 이만큼 틀리면 정답을 알려 주고 끝낸다 (계속 못 맞혀 통화가 끝나지 않는 일이 없게)

RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
SNOW_CODES = {71, 73, 75, 77, 85, 86}


class Weather(BaseModel):
    """화면(static/index.html의 S.weather)이 보내는 날씨"""
    place: str = ""
    temp: int
    max: int
    min: int
    rain: int = 0  # 오늘 강수확률(%)
    code: int = 0  # WMO 날씨 코드
    desc: str = ""


@dataclass
class Quiz:
    subject: str     # "오늘 최고 기온"
    answer: int
    unit: str        # "도" 또는 "%"
    tolerance: int   # 이만큼 차이 나도 정답
    reason: str      # 왜 이 질문인지 (날씨 조건)
    advice: str      # 맞힌 뒤 날씨에 맞는 한마디

    @property
    def question(self) -> str:
        return f"{self.subject}은 {'몇 도' if self.unit == '도' else '몇 퍼센트'}일까요?"

    def say(self, n: int) -> str:
        """음성으로 읽기 좋게: -3도 → 영하 3도, 30% → 30퍼센트"""
        if self.unit == "%":
            return f"{n}퍼센트"
        return f"영하 {-n}도" if n < 0 else f"{n}도"


def make_quiz(w: Weather) -> Quiz:
    """날씨 조건에 맞는 질문을 고른다"""
    if w.rain >= 50 or w.code in RAIN_CODES | SNOW_CODES:
        kind = "눈" if w.code in SNOW_CODES else "비"
        return Quiz(f"오늘 {kind} 올 확률", w.rain, "%", 10, f"오늘 {kind} 소식이 있어요.",
                    "우산 꼭 챙기세요!" if kind == "비" else "길 미끄러우니 조심하세요!")
    if w.min <= 3:
        return Quiz("오늘 최저 기온", w.min, "도", 2, "오늘 꽤 쌀쌀해요.", "옷 따뜻하게 입고 나가세요!")
    if w.max >= 28:
        return Quiz("오늘 최고 기온", w.max, "도", 2, "오늘 꽤 더워요.", "물 자주 챙겨 드세요!")
    if w.max - w.min >= 10:
        return Quiz("오늘 최고 기온", w.max, "도", 2, f"오늘 일교차가 {w.max - w.min}도나 나요.", "겉옷 하나 챙기세요!")
    return Quiz("지금 기온", w.temp, "도", 2, "", "오늘도 상쾌하게 시작해요!")


# ─── 대답에서 숫자 읽기 ──────────────────────────────────────────────────────────
SINO = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
NATIVE_UNITS = {"하나": 1, "한": 1, "둘": 2, "두": 2, "셋": 3, "세": 3, "넷": 4, "네": 4, "다섯": 5, "여섯": 6,
                "일곱": 7, "여덟": 8, "아홉": 9}
NATIVE_TENS = {"열": 10, "스물": 20, "스무": 20, "서른": 30, "마흔": 40}


def sino_number(s: str) -> Optional[int]:
    """십이 → 12, 이십삼 → 23, 칠 → 7"""
    if not s or any(ch not in "영공일이삼사오육륙칠팔구십백" for ch in s):
        return None
    if s in ("영", "공"):
        return 0
    total, digit = 0, 0
    for ch in s:
        if ch in "십백":
            total += (digit or 1) * (10 if ch == "십" else 100)
            digit = 0
        else:
            digit = SINO[ch]
    return total + digit


def native_number(s: str) -> Optional[int]:
    """열두 → 12, 일곱 → 7, 스물 → 20"""
    tens = 0
    for word, value in NATIVE_TENS.items():
        if s.startswith(word):
            tens, s = value, s[len(word):]
            break
    if not s:
        return tens or None
    unit = NATIVE_UNITS.get(s)
    return None if unit is None else tens + unit


def parse_number(text: str) -> Optional[int]:
    """"17도", "영하 3도", "열두 도", "삼십 퍼센트", "십이"처럼 말한 숫자. 숫자가 없으면 None"""
    t = re.sub(r"\s", "", text)
    negative = bool(re.search(r"영하|마이너스", t))
    m = re.search(r"(-?)(\d{1,3})", t)
    if m:
        n = int(m.group(2))
        return -n if (negative or m.group(1)) else n
    # 한글 숫자는 단위(도·퍼센트·프로) 바로 앞이나, 대답 전체가 숫자일 때만 읽는다 ("일어났어"의 '일'을 숫자로 읽지 않게)
    t = re.sub(r"영하|마이너스|정도|쯤|아마|음+|어+|글쎄", "", t)
    m = re.search(r"([가-힣]+?)(?:도|퍼센트|프로|퍼)", t) or re.fullmatch(r"([가-힣]+?)(?:요|이요|예요|이에요|인가요|인가|일까요)?[?!.~]*", t)
    if not m:
        return None
    word = m.group(1)
    n = sino_number(word)
    if n is None:
        n = native_number(word)
    if n is None:
        return None
    return -n if negative else n


# ─── 대사 ────────────────────────────────────────────────────────────────────────
ENTER_LEADS = {
    "awake": ["좋아요, 목소리가 완전히 깼네요!", "드디어 일어나셨군요!", "오, 이제 정신이 좀 드셨나 봐요!"],
    "up": ["벌써 움직이셨군요, 멋져요!", "와, 몸까지 움직이셨네요!"],
    "hangup": ["끊기 전에 딱 하나만 물어볼게요!", "알겠어요, 대신 마지막으로 하나만요!"],
}
HIGHER = ["땡! 그보다 더 높아요.", "아쉬워요, 더 높아요!", "아니에요, 조금 더 높아요."]
LOWER = ["땡! 그보다 더 낮아요.", "아쉬워요, 더 낮아요!", "아니에요, 조금 더 낮아요."]
CLOSING = "그럼 이제 전화 끊을게요, 좋은 하루 보내세요!"


def enter_text(quiz: Quiz, state: str) -> str:
    lead = random.choice(ENTER_LEADS.get(state, ENTER_LEADS["awake"]))
    reason = f" {quiz.reason}" if quiz.reason else ""
    return f"{lead} 정말 깼는지 퀴즈 하나 낼게요.{reason} {quiz.question}"


def hint_range(quiz: Quiz) -> str:
    step = 10 if quiz.unit == "%" else 5
    low = (quiz.answer // step) * step
    return f"힌트를 드릴게요, {quiz.say(low)}에서 {quiz.say(low + step)} 사이예요."


@dataclass
class QuizTurn:
    text: str
    passed: bool    # 맞혔거나 정답을 알려 주고 끝냄 → 통화 종료
    attempts: int   # 지금까지 틀린 횟수


def answer_turn(quiz: Quiz, said: str, attempts: int, user_name: str, silent: bool) -> QuizTurn:
    """퀴즈 단계에서 사용자의 대답 하나를 채점한다"""
    if silent:
        return QuizTurn(f"{user_name}님? 이 문제만 맞히면 끊어 드릴게요. {quiz.question}", False, attempts)
    n = parse_number(said)
    if n is None:
        hint = f" {hint_range(quiz)}" if attempts >= 1 else ""
        return QuizTurn(f"숫자로 맞혀 보세요!{hint} {quiz.question}", False, attempts)
    if abs(n - quiz.answer) <= quiz.tolerance:
        verdict = "정답이에요!" if n == quiz.answer else "거의 맞았어요!"
        return QuizTurn(f"{verdict} {quiz.subject}은 {quiz.say(quiz.answer)}예요. {quiz.advice} {CLOSING}", True, attempts)
    attempts += 1
    if attempts >= MAX_ATTEMPTS:
        return QuizTurn(f"아깝네요! 정답은 {quiz.say(quiz.answer)}예요. 이만큼 맞혀 보셨으면 확실히 깨셨네요. "
                        f"{quiz.advice} {CLOSING}", True, attempts)
    direction = random.choice(HIGHER if quiz.answer > n else LOWER)
    hint = f" {hint_range(quiz)}" if attempts == MAX_ATTEMPTS - 1 else " 다시 맞혀 보세요."
    return QuizTurn(f"{direction}{hint}", False, attempts)
