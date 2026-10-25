"""Deterministic time handling.

Message times and `now` MUST be timezone-aware. A relative phrase is resolved to
an explicit calendar date ONLY for a small set of literal day-shift words, using
the message's own timezone. Anything richer ("清明前后", "下个月") is never guessed
into a calendar date; the caller keeps the original phrase plus an explicit,
optional valid range.
"""

from __future__ import annotations

import datetime as dt
import re

SH = dt.timezone(dt.timedelta(hours=8), "Asia/Shanghai")


class TimeError(ValueError):
    """Missing, unparsable, or timezone-less timestamp."""


def parse_iso(value: object) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise TimeError("missing timestamp")
    text = value.strip()
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise TimeError(f"unparsable timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TimeError(f"timestamp has no timezone: {value!r}")
    return parsed


def norm_iso(value: object) -> str:
    return parse_iso(value).isoformat()


def parse_flexible(value: object) -> dt.datetime | None:
    """Lenient parse for model-proposed valid ranges.

    Accepts a tz-aware ISO datetime or a bare ISO date (assumed Asia/Shanghai
    midnight). Returns None when unusable. Used ONLY for expiry comparison, never
    to overwrite the verbatim phrase.
    """
    if not value or not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return parse_iso(text)
    except TimeError:
        pass
    try:
        day = dt.date.fromisoformat(text)
    except ValueError:
        return None
    return dt.datetime.combine(day, dt.time.min, SH)


_REL_DAYS = {
    "今天": 0, "今日": 0, "今晚": 0,
    "明天": 1, "明日": 1, "明晚": 1,
    "昨天": -1, "昨日": -1, "昨晚": -1,
    "后天": 2, "前天": -2,
}


def resolve_relative_phrase(phrase: object, anchor: dt.datetime) -> str | None:
    """Resolve ONLY a WHOLE simple day-shift or single-weekday phrase.

    Day-shift words (今天/明天/昨天/后天/…) and a weekday with an optional week
    offset (周三 / 下周三 / 下下周三 / 上周三) are resolved deterministically
    against the message's own date. The phrase must match IN FULL: a richer
    sentence such as "明天以后" is NOT resolved from its substring "明天" — the
    caller keeps the verbatim phrase and leaves the derived date unknown.
    """
    if not phrase or not isinstance(phrase, str):
        return None
    text = re.sub(r"\s+", "", phrase)
    if not text:
        return None
    if text in _REL_DAYS:
        return (anchor + dt.timedelta(days=_REL_DAYS[text])).date().isoformat()
    return resolve_weekday_date(text, anchor)


_WEEKDAY_ORD = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
# Optional week offset BEFORE the weekday: 上 = last week (-7), 下 = next week
# (+7), 下下 = the week after next (+14). Any other prefix stays unknown.
_WEEK_SHIFTS = {"上": -7, "下": 7, "下下": 14}
_WEEKDAY_RE = re.compile(r"(?:(下下|下|上))?(周|星期|礼拜)([一二三四五六日天])")
_DATE_RE = re.compile(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})")


def resolve_weekday_date(phrase: object, anchor: dt.datetime) -> str | None:
    """Resolve a WHOLE weekday phrase with an optional week offset, else None.

    Only "周三"-style tokens (optionally prefixed 上/下/下下) match, and the match
    must span the entire phrase. Calendar arithmetic uses the week containing the
    anchor: 下下周三 is one fortnight after this week's Wednesday, 上周三 one week
    before it.
    """
    if not phrase or not isinstance(phrase, str):
        return None
    text = re.sub(r"\s+", "", phrase)
    match = _WEEKDAY_RE.fullmatch(text)
    if not match:
        return None
    monday = anchor.date() - dt.timedelta(days=anchor.weekday())
    target = monday + dt.timedelta(days=_WEEKDAY_ORD[match.group(3)])
    target += dt.timedelta(days=_WEEK_SHIFTS.get(match.group(1), 0))
    return target.isoformat()


def dates_in_text(text: object) -> set[str]:
    """All calendar dates written in `text`, normalised to YYYY-MM-DD."""
    out: set[str] = set()
    for year, month, day in _DATE_RE.findall(str(text or "")):
        try:
            out.add(dt.date(int(year), int(month), int(day)).isoformat())
        except ValueError:
            continue
    return out


def weekday_dates_in_text(text: object, anchor: dt.datetime) -> set[str]:
    out: set[str] = set()
    for match in _WEEKDAY_RE.finditer(str(text or "")):
        resolved = resolve_weekday_date(match.group(0), anchor)
        if resolved:
            out.add(resolved)
    return out
    return None


# Generic period phrases whose END boundary is determinable. This is a RANGE
# upper bound (never an exact action day) and is resolved ONLY for a phrase whose
# whole meaning is that period/modifier pair — matched as an EXACT normalised
# phrase (plus an optional benign deadline suffix), so a substring can never
# swallow a direction, modifier or year ("月底以后", "下个月初", "明年年底").
#
# Direction / partial-period markers that make the phrase NOT an upper bound of
# this period are rejected outright (conservative: the caller keeps the verbatim
# phrase and treats the window as unknown instead of inventing one).
_BOUND_DISQUALIFIERS = ("以后", "之后", "初", "上半月", "下半月", "月头", "开始")
_BOUND_DEADLINE_SUFFIXES = ("之前", "以前", "之内", "以内", "前")

# ---------------------------------------------------------------------------
# Explicit calendar ranges: an optional year + a month (Chinese or numeric) +
# an optional period-of-month. These resolve to the LAST day of the indicated
# range — a range UPPER BOUND used only to judge whether a window has elapsed,
# never an exact action day. Grammar is strict and matched against the WHOLE
# normalised phrase (after one benign deadline suffix), so a direction or
# modifier cannot be swallowed by a substring.
# ---------------------------------------------------------------------------
_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
              "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}
_MONTH_TOKEN = r"(?:\d{1,2}|十[一二]?|[一二三四五六七八九])"
_CAL_YEAR = r"(?:(?P<year>\d{4})\s*年|(?P<rel>今|本|明|去)\s*年)?"
# 「上旬/中旬/下旬」 may be attached directly to the month ("三月中旬") or with 份.
_CAL_RANGE = re.compile(
    r"^" + _CAL_YEAR +
    r"(?P<month>" + _MONTH_TOKEN + r")\s*月(?:份)?"
    r"(?:(?P<xun>上旬|中旬|下旬)|(?P<monthend>底|末))?"
    r"$"
)
_XUN_LAST_DAY = {"上旬": 10, "中旬": 20}


def _month_number(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        value = int(token)
    else:
        value = _CN_DIGITS.get(token, 0)
    return value if 1 <= value <= 12 else None


def _resolve_calendar_range(text: str, anchor: dt.datetime) -> str | None:
    """Last day of an explicit year/month[/旬] phrase, or None when unmatchable.

    Year attribution is conservative: an explicit year (or 今年/明年/去年) is
    honoured; with no year the anchor's year is used ONLY when the month is not
    already behind the anchor month. A month behind the anchor month with no year
    is ambiguous (this year or next year) and stays unknown.
    """
    match = _CAL_RANGE.match(text)
    if not match:
        return None
    month = _month_number(match.group("month"))
    if month is None:
        return None
    if match.group("year"):
        year = int(match.group("year"))
    elif match.group("rel"):
        offset = {"今": 0, "本": 0, "明": 1, "去": -1}[match.group("rel")]
        year = anchor.year + offset
    else:
        if month < anchor.month:
            return None                     # ambiguous across the year boundary
        year = anchor.year
    try:
        last = _last_day_of_month(year, month)
    except ValueError:
        return None
    xun = match.group("xun")
    if xun:
        day = _XUN_LAST_DAY.get(xun)
        if day is None:                     # 下旬 -> whole month
            return last.isoformat()
        if day > last.day:                  # e.g. a very short month
            return last.isoformat()
        return dt.date(year, month, day).isoformat()
    return last.isoformat()                 # month end (or explicit 月底/月末)

# normalised phrase -> (unit, offset); unit in {"month","year","week"}
_BOUND_EXACT: dict[str, tuple[str, int]] = {
    # source month end
    "月底": ("month", 0), "月末": ("month", 0),
    "这个月底": ("month", 0), "本月月底": ("month", 0), "这月底": ("month", 0), "本月底": ("month", 0),
    "这个月末": ("month", 0), "本月末": ("month", 0), "这月末": ("month", 0),
    # next month end
    "下个月": ("month", 1), "下个月底": ("month", 1), "下月底": ("month", 1),
    "下个月月末": ("month", 1), "下月月末": ("month", 1), "下个月月底": ("month", 1),
    # source year end
    "年底": ("year", 0), "年末": ("year", 0),
    "今年年底": ("year", 0), "今年年末": ("year", 0), "本年底": ("year", 0), "本年年底": ("year", 0),
    "这年底": ("year", 0),
    # next / last year end
    "明年年底": ("year", 1), "明年底": ("year", 1), "明年年末": ("year", 1),
    "去年年底": ("year", -1), "去年底": ("year", -1), "去年年末": ("year", -1),
    # source week end
    "这周": ("week", 0), "本周": ("week", 0), "这周末": ("week", 0), "本周末": ("week", 0),
    "这星期": ("week", 0), "这个星期": ("week", 0),
    # next week end
    "下周": ("week", 1), "下星期": ("week", 1), "下个星期": ("week", 1), "下周末": ("week", 1),
}


def _last_day_of_month(year: int, month: int) -> dt.date:
    if month == 12:
        return dt.date(year, 12, 31)
    return dt.date(year, month + 1, 1) - dt.timedelta(days=1)


def _week_end(anchor: dt.datetime) -> dt.date:
    monday = anchor.date() - dt.timedelta(days=anchor.weekday())
    return monday + dt.timedelta(days=6)  # Sunday


def resolve_bound_end(phrase: object, anchor: dt.datetime) -> str | None:
    """Determinable END boundary (last calendar day) for a WHOLE period phrase.

    Returns an ISO date for the last day of the indicated period, or None. This is
    a *range upper bound* for judging whether a plan/temporary window has elapsed
    — never an exact action day. Only an exact normalised phrase (optionally with
    one benign deadline suffix such as 「之前」) resolves; any phrase carrying a
    direction or partial-period marker is left unknown. '月底' -> last day of the
    message's own month, '下个月' -> last day of next month, '明年年底' -> last day
    of next year. A sentence-like phrase ('月底以后再动手') never resolves.
    """
    if not phrase or not isinstance(phrase, str):
        return None
    text = re.sub(r"\s+", "", phrase)
    if not text:
        return None
    for suffix in _BOUND_DEADLINE_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    # Explicit calendar ranges are matched first, on the WHOLE phrase: their
    # grammar already excludes directions/modifiers, and 「上旬/中旬/下旬」 is a
    # legitimate part of the form.
    calendar = _resolve_calendar_range(text, anchor)
    if calendar:
        return calendar
    # Any remaining direction / partial-period marker disqualifies the phrase.
    if any(bad in text for bad in _BOUND_DISQUALIFIERS):
        return None
    entry = _BOUND_EXACT.get(text)
    if not entry:
        return None
    unit, offset = entry
    if unit == "month":
        base = anchor.year * 12 + (anchor.month - 1) + offset
        year, month = divmod(base, 12)
        return _last_day_of_month(year, month + 1).isoformat()
    if unit == "year":
        return dt.date(anchor.year + offset, 12, 31).isoformat()
    return (_week_end(anchor) + dt.timedelta(days=7 * offset)).isoformat()
