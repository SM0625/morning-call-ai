"""구글 캘린더 연동: 비공개 iCal 주소에서 오늘 일정을 읽는다.

구글 캘린더 설정의 '비공개 주소(iCal 형식)'는 로그인 없이 일정을 읽을 수 있는 주소다. 화면이 이 주소를 보내면
서버가 받아 와서(브라우저는 구글 주소를 직접 읽지 못한다) 오늘 일정만 골라 "오전 10시 팀 미팅"처럼 돌려준다.
매주 반복하는 수업 같은 반복 일정도 오늘 날짜로 펼친다.

이 주소를 아는 사람은 누구나 일정을 볼 수 있으므로 로그에 남기지 않는다.
"""
import logging
import time
from datetime import date, datetime, timedelta, timezone

import httpx
import icalendar
import recurring_ical_events
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("uvicorn.error")
router = APIRouter(prefix="/api/v1")

# 서버가 아무 주소나 대신 요청하지 않도록 구글 캘린더 iCal 주소만 받는다
ALLOWED_HOST = "calendar.google.com"
ICAL_PATH_PREFIX = "/calendar/ical/"
FETCH_TIMEOUT_SEC = 8
MAX_ICS_BYTES = 5 * 1024 * 1024
CACHE_SEC = 300       # 대기 화면과 알람 때 연달아 읽으므로 잠깐 재사용한다
MAX_EVENTS = 8        # AI 지시문에 넣을 일정 수 (너무 많으면 첫 인사가 길어진다)
MAX_TITLE_CHARS = 40

_cache: dict = {}     # 주소 → (받은 시각, 내용)


class CalendarRequest(BaseModel):
    ical_url: str = Field(max_length=600)
    tz_offset_min: int = Field(540, ge=-720, le=840)  # 화면의 시간대 (한국 +540분)


def normalize_url(url: str) -> str:
    url = url.strip()
    if url.startswith("webcal://"):
        url = "https://" + url[len("webcal://"):]
    try:
        parsed = httpx.URL(url)
    except Exception:
        raise ValueError("주소 형식이 아닙니다")
    if parsed.scheme != "https" or parsed.host != ALLOWED_HOST or not parsed.path.startswith(ICAL_PATH_PREFIX):
        raise ValueError("구글 캘린더의 '비공개 주소(iCal 형식)'가 아닙니다")
    return str(parsed)


def fetch_ics(url: str) -> bytes:
    cached = _cache.get(url)
    if cached and time.time() - cached[0] < CACHE_SEC:
        return cached[1]
    try:
        response = httpx.get(url, timeout=FETCH_TIMEOUT_SEC, follow_redirects=False)
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="구글 캘린더에 연결하지 못했습니다")
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="캘린더를 찾지 못했습니다. 비공개 주소를 다시 복사해 주세요")
    if response.status_code != 200 or len(response.content) > MAX_ICS_BYTES:
        raise HTTPException(status_code=502, detail=f"구글 캘린더 응답 오류 ({response.status_code})")
    _cache[url] = (time.time(), response.content)
    return response.content


def korean_time(dt: datetime) -> str:
    """09:30 → "오전 9시 30분", 14:00 → "오후 2시" (AI가 그대로 읽어도 자연스럽게)"""
    hour = dt.hour % 12 or 12
    text = f"{'오전' if dt.hour < 12 else '오후'} {hour}시"
    return text + (f" {dt.minute}분" if dt.minute else "")


def today_events(ics: bytes, tz: timezone, today: date = None) -> list:
    calendar = icalendar.Calendar.from_ical(ics)
    today = today or datetime.now(tz).date()
    start = datetime(today.year, today.month, today.day, tzinfo=tz)
    events = []
    for event in recurring_ical_events.of(calendar).between(start, start + timedelta(days=1)):
        if str(event.get("STATUS", "")).upper() == "CANCELLED":
            continue
        title = str(event.get("SUMMARY", "")).strip()[:MAX_TITLE_CHARS] or "제목 없는 일정"
        begin = event.decoded("DTSTART")
        if isinstance(begin, datetime):
            # 시간대가 없는 일정은 사용자의 시간대로 본다
            local = (begin if begin.tzinfo else begin.replace(tzinfo=tz)).astimezone(tz)
            events.append({"time": local.strftime("%H:%M"), "label": korean_time(local), "title": title,
                           "all_day": False, "_sort": local})
        else:
            events.append({"time": None, "label": "종일", "title": title, "all_day": True, "_sort": start})
    events.sort(key=lambda e: (not e["all_day"], e["_sort"]))  # 종일 일정을 먼저
    for e in events:
        del e["_sort"]
    return events[:MAX_EVENTS]


def summarize(events: list) -> str:
    return ", ".join(f"{e['label']} {e['title']}" for e in events)


# 동기 HTTP 요청이 이벤트 루프를 막지 않도록 def(스레드풀)로 둔다
@router.post("/calendar/today")
def calendar_today(request: CalendarRequest):
    try:
        url = normalize_url(request.ical_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    tz = timezone(timedelta(minutes=request.tz_offset_min))
    try:
        events = today_events(fetch_ics(url), tz)
    except HTTPException:
        raise
    except Exception:
        log.exception("캘린더 해석 실패")
        raise HTTPException(status_code=502, detail="캘린더 내용을 읽지 못했습니다")
    return {"events": events, "summary": summarize(events)}
