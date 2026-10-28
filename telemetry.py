"""초개인화 피드백 루프: 텔레메트리 수신 → 기상 성공 판정 → 사용자 프로파일 갱신.

화면(static/index.html)이 통화가 끝나면 대화 기록(call_end)을, 5분 뒤 통화 후 행동(post_wake)을 보낸다.
둘은 session_id로 묶여 wake_sessions 한 줄이 된다. 받을 때마다 백그라운드에서 그날 기상 성공 여부를
규칙으로 판정하고, 최근 기록을 Ollama에 보내 '효과적인 말투'·'핑계 키워드'·'장기 기억'을 JSON으로 뽑아
user_profiles에 UPSERT한다. 알람 때는 load_profile()로 읽어 시스템 프롬프트에 넣는다.

main.py는 init()으로 Ollama 호출 함수를 넘기고 router를 붙인다 (분류·생성과 같은 GPU 설정을 써야
모델이 다시 로드되지 않는다).
"""
import json
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated, Callable, List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints, model_validator

log = logging.getLogger("uvicorn.error")

DB_PATH = os.getenv("MORNINGCALL_DB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "morningcall.db")

router = APIRouter(prefix="/api/v1")

_chat: Optional[Callable] = None
_default_model = ""
# 프로파일 갱신(LLM 호출)은 한 번에 하나씩. 겹치면 GPU를 나눠 써서 둘 다 느려진다
_profile_lock = threading.Lock()
# 통화 중에 프로파일 추출(약 8초)이 돌면 AI 대답이 그만큼 늦어지므로, 통화가 끝날 때까지 미룬다.
# 알람이 앱을 열 때 어제 못 보낸 post_wake가 함께 오므로 실제로 겹친다
CALL_IDLE_SEC = 60       # 마지막 대화 요청 뒤 이만큼 조용하면 통화가 끝난 것으로 본다
MAX_DEFER_SEC = 15 * 60  # 그래도 이 이상은 미루지 않는다
_last_call_activity = 0.0


def note_call_activity():
    """main.py가 예열·첫 인사·대화 요청마다 부른다"""
    global _last_call_activity
    _last_call_activity = time.time()


def wait_until_call_idle():
    deadline = time.time() + MAX_DEFER_SEC
    while time.time() - _last_call_activity < CALL_IDLE_SEC and time.time() < deadline:
        time.sleep(5)


def init(chat: Callable, default_model: str):
    """chat(model, messages, options, **kwargs)는 main.ollama_chat"""
    global _chat, _default_model
    _chat, _default_model = chat, default_model
    with db() as conn:
        conn.executescript(SCHEMA)


# ─── 받는 데이터 ─────────────────────────────────────────────────────────────────
# 화면이 만드는 ID (UUID 또는 시각+난수). 다른 모양은 받지 않는다
Id = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9-]{1,64}$")]


class Utterance(BaseModel):
    text: str = Field("", max_length=500)
    input: Literal["voice", "text", "silent"] = "voice"
    latency_sec: Optional[float] = Field(None, ge=0, le=600)  # AI 말이 끝나고 말하기 시작할 때까지 (음성일 때만)
    ai_before: str = Field("", max_length=500)                # 이 대답 직전에 AI가 한 말
    stage_before: int = Field(0, ge=0, le=3)
    at_sec: Optional[float] = None


class CallData(BaseModel):
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_sec: Optional[float] = None
    ended_by: Literal["button", "closed"] = "button"
    reached_end: bool = False  # AI가 다 일어났다고 판단했는지
    final_stage: int = Field(0, ge=0, le=3)
    snooze_count: int = Field(0, ge=0, le=100)
    turn_count: int = 0
    silent_count: int = 0
    latency_avg_sec: Optional[float] = None
    latency_max_sec: Optional[float] = None
    utterances: List[Utterance] = Field(default_factory=list, max_length=100)


class Behavior(BaseModel):
    """통화 후 행동. 브라우저는 걸음 수·화면 항목을 잴 수 없어 null로 보낸다"""
    window_sec: float = 0
    complete: bool = False
    step_sensor: bool = False
    steps: Optional[int] = Field(None, ge=0)
    screen_on_sec: Optional[float] = None
    screen_off_count: Optional[int] = None
    first_screen_off_sec: Optional[float] = None
    unlock_count: Optional[int] = None
    foreground_sec: Optional[float] = None
    app_left_count: Optional[int] = None
    first_app_left_sec: Optional[float] = None


class TelemetryIn(BaseModel):
    event: Literal["call_end", "post_wake"]
    user_id: Id
    session_id: Id
    client: Literal["web", "android"] = "web"
    sent_at: Optional[str] = None
    tz_offset_min: Optional[int] = None
    user_name: Optional[str] = Field(None, max_length=50)
    model: Optional[str] = Field(None, max_length=100)
    alarm_time: Optional[str] = Field(None, max_length=10)
    call: Optional[CallData] = None
    behavior: Optional[Behavior] = None

    @model_validator(mode="after")
    def check_body(self):
        if self.event == "call_end" and self.call is None:
            raise ValueError("call_end에는 call이 있어야 합니다")
        if self.event == "post_wake" and self.behavior is None:
            raise ValueError("post_wake에는 behavior가 있어야 합니다")
        return self


# ─── DB ──────────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    event       TEXT NOT NULL,
    payload     TEXT NOT NULL,
    received_at TEXT NOT NULL,
    UNIQUE (session_id, event)  -- 화면이 실패한 전송을 다시 보내도 한 번만 처리한다
);
CREATE TABLE IF NOT EXISTS wake_sessions (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    user_name    TEXT,
    alarm_time   TEXT,
    client       TEXT,
    model        TEXT,
    call         TEXT,     -- CallData JSON
    behavior     TEXT,     -- Behavior JSON
    wake_success INTEGER,  -- 1/0, 아직 판정 전이면 NULL
    wake_reason  TEXT,
    judged_at    TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wake_sessions_user ON wake_sessions (user_id, created_at);
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id           TEXT PRIMARY KEY,
    profile           TEXT NOT NULL,  -- 프로파일 JSON (build_profile 참고)
    sessions_analyzed INTEGER NOT NULL,
    updated_at        TEXT NOT NULL
);
"""


@contextmanager
def db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:  # 끝나면 커밋, 예외면 롤백
            yield conn
    finally:
        conn.close()


def now_local() -> str:
    # 서버와 폰이 같은 집에 있으므로 서버 시각을 그대로 쓴다 (날짜를 알람 기준으로 묶을 때 쓴다)
    return datetime.now().isoformat(timespec="seconds")


def store_event(p: TelemetryIn) -> bool:
    """원본을 남기고 wake_sessions에 합친다. 이미 받은 이벤트면 False"""
    now = now_local()
    with db() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO telemetry_events (user_id, session_id, event, payload, received_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (p.user_id, p.session_id, p.event, p.model_dump_json(), now),
        )
        if cur.rowcount == 0:
            return False
        column = "call" if p.event == "call_end" else "behavior"
        body = (p.call if p.event == "call_end" else p.behavior).model_dump_json()
        # 다른 사용자의 session_id로 덮어쓰지 못하게 user_id가 같을 때만 합친다
        conn.execute(
            f"""INSERT INTO wake_sessions (session_id, user_id, user_name, alarm_time, client, model, {column}, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_id) DO UPDATE SET
                    {column}   = excluded.{column},
                    user_name  = COALESCE(excluded.user_name, wake_sessions.user_name),
                    alarm_time = COALESCE(excluded.alarm_time, wake_sessions.alarm_time),
                    model      = COALESCE(excluded.model, wake_sessions.model)
                WHERE wake_sessions.user_id = excluded.user_id""",
            (p.session_id, p.user_id, p.user_name, p.alarm_time, p.client, p.model, body, now),
        )
    return True


def load_profile(user_id: str) -> Optional[dict]:
    if not user_id:
        return None
    with db() as conn:
        row = conn.execute("SELECT profile FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    return json.loads(row["profile"]) if row else None


# ─── API ─────────────────────────────────────────────────────────────────────────
# 동기 SQLite·Ollama 호출이 이벤트 루프를 막지 않도록 def(스레드풀)로 둔다
@router.post("/telemetry", status_code=202)
def receive_telemetry(payload: TelemetryIn, background: BackgroundTasks):
    is_new = store_event(payload)
    if is_new:
        # 응답을 먼저 돌려주고 판정·프로파일 갱신은 뒤에서 한다 (LLM 호출에 몇 초 걸린다)
        background.add_task(process_session, payload.session_id, payload.user_id)
    return {"ok": True, "duplicate": not is_new}


@router.get("/profile/{user_id}")
def get_profile(user_id: str):
    profile = load_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="아직 프로파일이 없습니다")
    return profile


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def require_local(request: Request):
    # 서버는 폰이 접속하도록 0.0.0.0으로 열려 있다. 대화 원문은 같은 와이파이의 다른 사람이 보지 못하게 이 PC에서만
    if not request.client or request.client.host not in LOCAL_HOSTS:
        raise HTTPException(status_code=403, detail="기록은 서버를 켠 PC에서만 볼 수 있습니다")


@router.get("/records")
def get_records(request: Request):
    """기록 보기 화면(records.html)용. 사용자별 프로파일과 통화 기록 전체"""
    require_local(request)
    with db() as conn:
        profiles = {r["user_id"]: json.loads(r["profile"]) for r in conn.execute("SELECT user_id, profile FROM user_profiles")}
        rows = conn.execute("SELECT * FROM wake_sessions ORDER BY created_at DESC").fetchall()
    users = {}
    for r in rows:
        user = users.setdefault(r["user_id"], {"user_id": r["user_id"], "user_name": r["user_name"],
                                               "profile": profiles.get(r["user_id"]), "sessions": []})
        user["sessions"].append({
            "session_id": r["session_id"], "created_at": r["created_at"], "alarm_time": r["alarm_time"],
            "client": r["client"], "model": r["model"],
            "wake_success": None if r["wake_success"] is None else bool(r["wake_success"]),
            "wake_reason": r["wake_reason"],
            "call": json.loads(r["call"]) if r["call"] else None,
            "behavior": json.loads(r["behavior"]) if r["behavior"] else None,
        })
    return {"db_path": DB_PATH, "users": list(users.values())}


# ─── 백그라운드: 판정 → 프로파일 ─────────────────────────────────────────────────
def process_session(session_id: str, user_id: str):
    try:
        judge_session(session_id)  # 판정은 DB만 쓰므로 바로 한다
        wait_until_call_idle()
        with _profile_lock:
            update_profile(user_id)
    except Exception:
        # 프로파일이 못 만들어져도 알람은 기본 프롬프트로 동작한다
        log.exception("프로파일 갱신 실패 (session %s)", session_id)


# 기상 판정 기준. 통화 후 5분 동안의 행동으로 본다
STEPS_AWAKE = 30        # 이만큼 걸었으면 일어나 움직인 것
STEPS_STILL = 10        # 이보다 적으면 거의 움직이지 않은 것
RESLEEP_SEC = 120       # 통화 후 이 안에 화면을 끄고 움직이지 않으면 다시 잠든 것
SCREEN_AWAKE_SEC = 90   # 화면을 이만큼 켜 두었으면 폰을 쓰며 깨어 있는 것


def judge_wake(call: Optional[dict], behavior: Optional[dict]) -> tuple:
    """(성공 여부, 이유). 통화 결과를 기본으로 하고, 통화 후 행동이 있으면 그걸로 뒤집을 수 있다"""
    if call and not call.get("reached_end"):
        return False, "AI가 일어났다고 판단하기 전에 통화를 끊음"
    if not behavior:
        return bool(call), "통화는 끝까지 함 (통화 후 행동 기록 전)"

    steps = behavior.get("steps")
    screen_on = behavior.get("screen_on_sec")
    off_at = behavior.get("first_screen_off_sec")
    if steps is not None and steps >= STEPS_AWAKE:
        return True, f"통화 후 5분 동안 {steps}걸음 움직임"
    still = steps is None or steps < STEPS_STILL
    if off_at is not None and off_at < RESLEEP_SEC and still and (screen_on or 0) < RESLEEP_SEC:
        return False, f"통화 후 {off_at:.0f}초 만에 화면을 끄고 거의 움직이지 않음 (다시 잠든 것으로 봄)"
    # 화면 항목은 안드로이드에서만 잰다. 브라우저의 '탭 가려짐'은 화면을 끈 것과 구분되지 않아 판정에 쓰지 않는다
    if screen_on is not None:
        if screen_on >= SCREEN_AWAKE_SEC or behavior.get("unlock_count") or behavior.get("app_left_count"):
            return True, "통화 후 폰을 쓰며 깨어 있음"
        if steps is not None and still:
            return False, "통화 후 움직임도 폰 사용도 거의 없음"
    if not call:
        return False, "통화 기록이 없고 통화 후 행동으로도 판단할 수 없음"
    return True, "통화는 끝까지 했고, 통화 후 행동으로는 판단할 수 없음"


def judge_session(session_id: str):
    with db() as conn:
        row = conn.execute("SELECT call, behavior FROM wake_sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return
        call = json.loads(row["call"]) if row["call"] else None
        behavior = json.loads(row["behavior"]) if row["behavior"] else None
        success, reason = judge_wake(call, behavior)
        conn.execute(
            "UPDATE wake_sessions SET wake_success = ?, wake_reason = ?, judged_at = ? WHERE session_id = ?",
            (int(success), reason, now_local(), session_id),
        )
    log.info("기상 판정 %s | %s | %s", session_id, "성공" if success else "실패", reason)


PROFILE_SESSIONS = 10        # 최근 몇 번의 아침으로 프로파일을 만들지
TRANSCRIPT_MAX_CHARS = 3500  # LLM에 보낼 대화 기록 길이 (길면 소형 모델이 느려지고 앞부분을 놓친다)
MAX_KEYWORDS = 5
MAX_MEMORIES = 5

# 말투 선택지. 3단계에서 말투마다 정해 둔 지시문을 시스템 프롬프트에 넣으므로 자유 문장이 아니라 고르게 한다
TONES = ["다정하게 달래기", "장난스럽게 놀리기", "단호하게 재촉하기", "일정으로 현실 일깨우기", "칭찬으로 북돋기"]

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "effective_tone": {"type": "string", "enum": TONES},
        "ineffective_tone": {"type": "string", "enum": TONES},
        "tone_evidence": {"type": "string"},
        "excuse_keywords": {"type": "array", "items": {"type": "string"}},
        "long_term_memory": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["effective_tone", "ineffective_tone", "tone_evidence", "excuse_keywords", "long_term_memory"],
}

EXTRACT_SYSTEM = "당신은 모닝콜 통화 기록을 분석해 사용자 프로파일을 만드는 분석가입니다. 기록에 있는 사실만 쓰고 JSON으로만 답합니다."


def recent_sessions(user_id: str) -> list:
    with db() as conn:
        return conn.execute(
            "SELECT * FROM wake_sessions WHERE user_id = ? AND wake_success IS NOT NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, PROFILE_SESSIONS),
        ).fetchall()


def session_call(row) -> dict:
    return json.loads(row["call"]) if row["call"] else {}


def compute_stats(rows: list) -> dict:
    """숫자로 셀 수 있는 것은 LLM에 맡기지 않는다 (소형 모델은 세기를 자주 틀린다)"""
    successes = sum(r["wake_success"] for r in rows)
    streak_type = "success" if rows[0]["wake_success"] else "failure"
    streak = 0
    for r in rows:
        if bool(r["wake_success"]) != (streak_type == "success"):
            break
        streak += 1
    calls = [session_call(r) for r in rows]
    latencies = [c["latency_avg_sec"] for c in calls if c.get("latency_avg_sec") is not None]
    return {
        "sessions": len(rows),
        "successes": successes,
        "success_rate": round(successes / len(rows), 2),
        "streak": {"type": streak_type, "count": streak},
        "avg_latency_sec": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "avg_snooze": round(sum(c.get("snooze_count", 0) for c in calls) / len(rows), 2),
        # 화면이 보낸 집계값 대신 발화 기록에서 직접 센다
        "avg_silent": round(sum(1 for c in calls for u in c.get("utterances", []) if u.get("input") == "silent")
                            / len(rows), 2),
    }


def recent_failures(rows: list, limit: int = 3) -> list:
    failures = []
    for r in rows:
        if r["wake_success"]:
            continue
        said = [u["text"] for u in session_call(r).get("utterances", []) if u.get("text")]
        failures.append({"date": r["created_at"][:10], "reason": r["wake_reason"], "said": said[:3]})
        if len(failures) >= limit:
            break
    return failures


def transcript(row) -> str:
    call = session_call(row)
    lines = [f"[{row['created_at'][:10]} · 기상 {'성공' if row['wake_success'] else '실패'}: {row['wake_reason']}]"]
    if call.get("snooze_count"):
        lines.append(f"(알람을 {call['snooze_count']}번 미룸)")
    for u in call.get("utterances", []):
        if u.get("ai_before"):
            lines.append(f"AI: {u['ai_before']}")
        said = "(대답 없음)" if u.get("input") == "silent" else u.get("text", "")
        latency = f" ({u['latency_sec']}초 뒤 대답)" if u.get("latency_sec") is not None else ""
        lines.append(f"나: {said}{latency}")
    return "\n".join(lines)


def build_extract_prompt(user_name: str, rows: list, previous: Optional[dict]) -> str:
    # 최근 아침부터 길이 제한까지 넣는다
    parts, used = [], 0
    for r in rows:
        t = transcript(r)
        if parts and used + len(t) > TRANSCRIPT_MAX_CHARS:
            break
        parts.append(t[:TRANSCRIPT_MAX_CHARS])
        used += len(t)
    prev = {k: previous.get(k) for k in ("effective_tone", "excuse_keywords", "long_term_memory")} if previous else None
    return f"""아래는 {user_name}님의 최근 모닝콜 통화 기록입니다(최근 것부터). 'AI'는 깨우는 쪽이 한 말, '나'는 {user_name}님의 대답입니다.

{chr(10).join(parts)}

이전 프로파일: {json.dumps(prev, ensure_ascii=False) if prev else "없음"}

다음을 뽑으세요.
- effective_tone: {user_name}님이 빨리 대답하거나 "일어났어"처럼 반응한 직전에, 그리고 기상에 성공한 날 AI가 주로 쓴 말투.
- ineffective_tone: {user_name}님이 핑계를 대거나 대답이 없던 직전에, 그리고 기상에 실패한 날 AI가 주로 쓴 말투.
- 말투는 다음 중에서만 고릅니다: {", ".join(TONES)}
- tone_evidence: 그렇게 고른 근거를 기록에서 들어 한 문장으로.
- excuse_keywords: {user_name}님이 더 자려고 댄 핑계. '나'의 대답에 나온 말을 2~8글자로 줄여서, 자주 나온 것부터 최대 {MAX_KEYWORDS}개.
- long_term_memory: 다음 아침에 깨울 때 기억해 두면 좋은 {user_name}님의 요즘 사정·일정·습관. '나'의 대답에 직접 나온 것만 짧은 문장으로 최대 {MAX_MEMORIES}개, 없으면 빈 배열. 기록에 없는 내용은 지어내지 마세요.
"""


def hangul(text: str) -> str:
    return re.sub(r"[^가-힣0-9]", "", text)


# 어느 문장에나 나오는 시간 표현. 이것만 겹쳐서는 대답에 근거했다고 보지 않는다
COMMON_BIGRAMS = {"어제", "오늘", "내일", "요즘", "지금", "아침", "매일", "항상", "자주", "새벽"}


def grounded(item: str, corpus: str) -> bool:
    # 소형 모델은 대답에 없던 핑계나 사실을 지어내고, 프롬프트 예시를 베끼기도 한다.
    # "피곤함"↔"피곤해", "요즘 시험 기간"↔"시험공부"처럼 형태가 달라도, 시간 표현이 아닌 두 글자 조각이
    # 실제 대답에 하나라도 있으면 근거가 있는 것으로 본다
    core = hangul(item)
    return any(core[i:i + 2] in corpus for i in range(len(core) - 1) if core[i:i + 2] not in COMMON_BIGRAMS)


def clean_list(items, limit: int, max_len: int = 40) -> list:
    out = []
    for item in items or []:
        item = str(item).strip()
        if item and len(item) <= max_len and item not in out:
            out.append(item)
    return out[:limit]


def extract_with_llm(user_name: str, rows: list, previous: Optional[dict], model: str) -> Optional[dict]:
    voiced = [u for r in rows for u in session_call(r).get("utterances", []) if u.get("text")]
    if not voiced or _chat is None:
        return None  # 분석할 대답이 없다
    response = _chat(
        model,
        [{"role": "system", "content": EXTRACT_SYSTEM},
         {"role": "user", "content": build_extract_prompt(user_name, rows, previous)}],
        {"temperature": 0.2, "num_predict": 400},
        format=PROFILE_SCHEMA,  # 스키마에 맞는 JSON만 생성하게 한다
    )
    data = json.loads(response["message"]["content"])
    corpus = hangul(" ".join(u["text"] for u in voiced))
    tone_ok = lambda t: t if t in TONES else None
    return {
        "effective_tone": tone_ok(data.get("effective_tone")),
        "ineffective_tone": tone_ok(data.get("ineffective_tone")),
        "tone_evidence": str(data.get("tone_evidence", "")).strip()[:120],
        "excuse_keywords": [k for k in clean_list(data.get("excuse_keywords"), MAX_KEYWORDS, max_len=15)
                            if grounded(k, corpus)],
        "long_term_memory": [m for m in clean_list(data.get("long_term_memory"), MAX_MEMORIES, max_len=60)
                             if grounded(m, corpus)],
    }


def merge_recent(new: list, old: list, limit: int) -> list:
    # 새로 뽑힌 것을 앞에 두고, 이번에 빠진 예전 것도 자리가 남으면 남긴다 (하루 기록이 짧아도 기억이 사라지지 않게)
    return clean_list(list(new) + list(old), limit, max_len=60)


def build_profile(user_id: str) -> Optional[dict]:
    rows = recent_sessions(user_id)
    if not rows:
        return None
    previous = load_profile(user_id) or {}
    user_name = next((r["user_name"] for r in rows if r["user_name"]), "사용자")
    model = rows[0]["model"] or _default_model  # 방금 통화에 쓴 모델이 이미 메모리에 올라와 있다
    try:
        llm = extract_with_llm(user_name, rows, previous, model)
    except Exception:
        log.exception("프로파일 추출 실패 — 이전 분석 결과를 유지합니다")
        llm = None
    llm = llm or {}
    return {
        "version": 1,
        "user_name": user_name,
        "stats": compute_stats(rows),
        "recent_failures": recent_failures(rows),
        "effective_tone": llm.get("effective_tone") or previous.get("effective_tone"),
        "ineffective_tone": llm.get("ineffective_tone") or previous.get("ineffective_tone"),
        "tone_evidence": llm.get("tone_evidence") or previous.get("tone_evidence", ""),
        "excuse_keywords": merge_recent(llm.get("excuse_keywords", []), previous.get("excuse_keywords", []), MAX_KEYWORDS),
        "long_term_memory": merge_recent(llm.get("long_term_memory", []), previous.get("long_term_memory", []), MAX_MEMORIES),
        "updated_at": now_local(),
    }


def update_profile(user_id: str):
    profile = build_profile(user_id)
    if profile is None:
        return
    with db() as conn:
        conn.execute(
            """INSERT INTO user_profiles (user_id, profile, sessions_analyzed, updated_at) VALUES (?, ?, ?, ?)
               ON CONFLICT (user_id) DO UPDATE SET
                   profile = excluded.profile,
                   sessions_analyzed = excluded.sessions_analyzed,
                   updated_at = excluded.updated_at""",
            (user_id, json.dumps(profile, ensure_ascii=False), profile["stats"]["sessions"], profile["updated_at"]),
        )
    log.info("프로파일 갱신 %s | %s", user_id, json.dumps(profile, ensure_ascii=False))
