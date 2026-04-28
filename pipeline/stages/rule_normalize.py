# rule_normalize.py  — Arch 2: universal rule-first pass
#
# KEY CHANGE FROM ARCH 1:
#   Old: cross-dep expressions returned None → LLM guessed the anchor date.
#   New: cross-dep expressions return a structured AnchorTag dict so the
#        graph builder can build the DAG before any LLM call is made.
#        The LLM never sees an unresolved anchor date — it only does
#        coref + arithmetic on dates that are already in the registry.
#
# Return contract:
#   {"status": "resolved",   "value": "<TimeML>"}          → rule fully solved
#   {"status": "anchor_dep", "anchor_tag": "<str>",
#    "offset_unit": "<str>", "offset_n": <float>}          → cross-dep, tagged for graph
#   {"status": "vague"}                                      → truly unresolvable (rare)

import re
from datetime import date, timedelta
from functools import lru_cache

try:
    from dateutil.relativedelta import relativedelta
    def _rd(years=0, months=0, days=0):
        return relativedelta(years=years, months=months, days=days)
except ImportError:
    def _rd(years=0, months=0, days=0):
        return timedelta(days=int(days) + int(months) * 30 + int(years) * 365)


# ── Number word map ───────────────────────────────────────────
WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "couple": 2, "few": 3, "several": 3,
    "half": 0.5, "dozen": 12, "twenty": 20, "thirty": 30,
}

WEEKDAYS = {
    "monday": 0,  "tuesday": 1,  "wednesday": 2, "thursday": 3,
    "friday": 4,  "saturday": 5, "sunday": 6,
    "mon": 0,     "tue": 1,      "wed": 2,       "thu": 3,
    "fri": 4,     "sat": 5,      "sun": 6,
}
WEEKDAY_TIMEX = {
    0: "XXXX-WXX-1", 1: "XXXX-WXX-2", 2: "XXXX-WXX-3",
    3: "XXXX-WXX-4", 4: "XXXX-WXX-5", 5: "XXXX-WXX-6", 6: "XXXX-WXX-7",
}

MONTHS_MAP = {
    "january": "01",  "february": "02", "march": "03",    "april": "04",
    "may": "05",      "june": "06",     "july": "07",     "august": "08",
    "september": "09","october": "10",  "november": "11", "december": "12",
    "jan": "01",      "feb": "02",      "mar": "03",      "apr": "04",
    "jun": "06",      "jul": "07",      "aug": "08",      "sep": "09",
    "oct": "10",      "nov": "11",      "dec": "12",
}

SEASONS = {"spring": "SP", "summer": "SU", "fall": "FA", "autumn": "FA", "winter": "WI"}

TOD_SUFFIX = {
    "morning":   "TMO", "afternoon": "TAF", "evening": "TEV",
    "night":     "TNI", "midnight":  "T00:00", "noon": "T12:00",
}


# ── Helpers ───────────────────────────────────────────────────

def _n(w: str) -> float:
    w = w.lower().strip()
    if w in WORD_NUM:
        return WORD_NUM[w]
    try:
        return float(w)
    except ValueError:
        return 1.0

def _i(w: str) -> int:
    return max(1, int(round(_n(w))))

def _clean(entity: str) -> str:
    return entity.strip().rstrip('.,!?;:').lower()

def _next_weekday(today: date, wd: int) -> date:
    ahead = (wd - today.weekday() + 7) % 7
    return today + timedelta(days=ahead or 7)

def _last_weekday(today: date, wd: int) -> date:
    back = (today.weekday() - wd + 7) % 7
    return today - timedelta(days=back or 7)

def _resolved(value: str) -> dict:
    return {"status": "resolved", "value": value}

def _anchor_dep(anchor_tag: str, offset_n: float = 0.0, offset_unit: str = "") -> dict:
    return {
        "status":      "anchor_dep",
        "anchor_tag":  anchor_tag,
        "offset_n":    offset_n,
        "offset_unit": offset_unit,
    }

def _vague() -> dict:
    return {"status": "vague"}


# ── Cross-dependency pattern detector ────────────────────────
# Recognises the pattern AND extracts a canonical anchor_tag +
# offset so the graph builder has structured edges to work with.

# Patterns: "<n> <unit> after/later/following/since/of the <event>"
_AFTER_PAT = re.compile(
    r'(?P<n>a|an|one|two|three|four|five|six|seven|eight|nine|ten|'
    r'eleven|twelve|couple|few|several|\d+(?:\.\d+)?)\s+'
    r'(?P<unit>hour|day|week|month|year)s?\s+'
    r'(?:after|later|following|of|since)\b',
    re.IGNORECASE,
)

# "within <n> <unit>s of/after ..."
_WITHIN_PAT = re.compile(
    r'within\s+'
    r'(?P<n>a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+(?:\.\d+)?)\s+'
    r'(?P<unit>hour|day|week|month|year)s?',
    re.IGNORECASE,
)

# Vague cross-deps with no numeric offset
_VAGUE_CROSS = re.compile(
    r'\b(shortly after|soon after|immediately after|'
    r'the next day|the following day|the following morning|'
    r'that (morning|evening|night|afternoon)|'
    r'at that (point|time|moment)|'
    r'by the time|since then|since that|'
    r'the (morning|evening|night|afternoon) of|'
    r'later that|later in the)\b',
    re.IGNORECASE,
)

# Expressions that reference a prior anchor without "after"
_PRIOR_REF = re.compile(
    r'\b(before the|prior to the|at the time of|at the (start|end|beginning) of)\b',
    re.IGNORECASE,
)


def _extract_cross_dep(entity: str) -> dict | None:
    """
    If entity is cross-dep, return an anchor_dep dict with extracted offset.
    Returns None if entity is self-contained.
    """
    e = entity.lower()

    m = _AFTER_PAT.search(e)
    if m:
        return _anchor_dep(
            anchor_tag="event",         # generic; graph builder refines per sentence
            offset_n=_n(m.group("n")),
            offset_unit=m.group("unit"),
        )

    m = _WITHIN_PAT.search(e)
    if m:
        return _anchor_dep(
            anchor_tag="event",
            offset_n=_n(m.group("n")),
            offset_unit=m.group("unit"),
        )

    if _VAGUE_CROSS.search(e):
        return _anchor_dep(anchor_tag="event", offset_n=0.0, offset_unit="")

    if _PRIOR_REF.search(e):
        return _anchor_dep(anchor_tag="event", offset_n=0.0, offset_unit="")

    return None


# ── Main rule engine ──────────────────────────────────────────

@lru_cache(maxsize=2048)
def rule_normalize(entity: str, entity_type: str) -> dict:
    """
    Universal rule pass — every entity goes through this.

    Returns one of:
      {"status": "resolved",   "value": "<TimeML>"}
      {"status": "anchor_dep", "anchor_tag": ..., "offset_n": ..., "offset_unit": ...}
      {"status": "vague"}
    """
    # Cross-dep check FIRST — before any self-contained matching
    cross = _extract_cross_dep(entity)
    if cross:
        return cross

    e = _clean(entity)
    today = date.today()
    y = today.year

    # ── DATE ──────────────────────────────────────────────────
    if entity_type == "DATE":

        anchors = {
            "today":                    today.isoformat(),
            "now":                      today.isoformat(),
            "this day":                 today.isoformat(),
            "the present day":          today.isoformat(),
            "at present":               today.isoformat(),
            "currently":                today.isoformat(),
            "tomorrow":                 (today + timedelta(1)).isoformat(),
            "day after tomorrow":       (today + timedelta(2)).isoformat(),
            "the day after tomorrow":   (today + timedelta(2)).isoformat(),
            "yesterday":                (today - timedelta(1)).isoformat(),
            "day before yesterday":     (today - timedelta(2)).isoformat(),
            "the day before yesterday": (today - timedelta(2)).isoformat(),
            "this weekend":             _next_weekday(today, 5).isoformat(),
            "next weekend":             (_next_weekday(today, 5) + timedelta(7)).isoformat(),
            "last weekend":             (_last_weekday(today, 5)).isoformat(),
            "this week":                today.isoformat(),
            "this month":               today.strftime("%Y-%m"),
            "this year":                str(y),
            "last year":                str(y - 1),
            "next year":                str(y + 1),
        }
        if e in anchors:
            return _resolved(anchors[e])

        vague_past = {
            "long ago", "ages ago", "a long time ago", "way back",
            "back then", "once", "in the past", "in the old days",
            "in earlier times", "historically", "previously",
        }
        vague_future = {
            "soon", "shortly", "in a moment", "someday", "one day",
            "in the future", "eventually", "at some point",
            "in due course", "in due time", "down the road",
        }
        if e in vague_past:   return _resolved("PAST_REF")
        if e in vague_future: return _resolved("FUTURE_REF")

        m = re.match(r'(this|next|last)\s+(spring|summer|fall|autumn|winter)', e)
        if m:
            offset = {"this": 0, "next": 1, "last": -1}[m.group(1)]
            return _resolved(f"{y + offset}-{SEASONS[m.group(2)]}")

        m = re.match(r'the\s+(\d{4})s', e)
        if m: return _resolved(f"{m.group(1)[:3]}X")
        m = re.match(r'the\s+(\d{2})s', e)
        if m:
            prefix = "19" if int(m.group(1)) > 20 else "20"
            return _resolved(f"{prefix}{m.group(1)[0]}X")

        m = re.match(r'the\s+(\d+)(st|nd|rd|th)\s+century', e)
        if m:
            c = int(m.group(1)) - 1
            return _resolved(f"{c:02d}XX")

        m = re.match(
            r'(next|last|this|coming)\s+'
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
            r'mon|tue|wed|thu|fri|sat|sun)', e)
        if m:
            wd = WEEKDAYS[m.group(2)]
            mod = m.group(1)
            if mod in ("next", "coming"): return _resolved(_next_weekday(today, wd).isoformat())
            if mod == "last":             return _resolved(_last_weekday(today, wd).isoformat())
            d = _next_weekday(today, wd)
            if d - today >= timedelta(7): d -= timedelta(7)
            return _resolved(d.isoformat())

        m = re.match(r'(next|last)\s+(week|month|year)', e)
        if m:
            mod, unit = m.group(1), m.group(2)
            sign = 1 if mod == "next" else -1
            if unit == "week":  return _resolved((today + timedelta(sign * 7)).isoformat())
            if unit == "month": return _resolved((today + _rd(months=sign)).strftime("%Y-%m"))
            if unit == "year":  return _resolved(str(y + sign))

        m = re.match(
            r'(a|an|one|two|three|four|five|six|seven|eight|nine|ten|'
            r'eleven|twelve|couple|few|several|\d+(?:\.\d+)?)\s+'
            r'(day|week|month|year)s?\s+ago', e)
        if m:
            n = _i(m.group(1)); unit = m.group(2)
            if unit == "day":   return _resolved((today - timedelta(n)).isoformat())
            if unit == "week":  return _resolved((today - timedelta(n * 7)).isoformat())
            if unit == "month": return _resolved((today + _rd(months=-n)).isoformat())
            if unit == "year":  return _resolved((today + _rd(years=-n)).isoformat())

        m = re.match(
            r'in\s+(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+'
            r'(day|week|month|year)s?', e)
        if m:
            n = _i(m.group(1)); unit = m.group(2)
            if unit == "day":   return _resolved((today + timedelta(n)).isoformat())
            if unit == "week":  return _resolved((today + timedelta(n * 7)).isoformat())
            if unit == "month": return _resolved((today + _rd(months=n)).isoformat())
            if unit == "year":  return _resolved((today + _rd(years=n)).isoformat())

        m = re.match(
            r'(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+'
            r'(day|week|month|year)s?\s+(?:from now|from today|hence|from this point)', e)
        if m:
            n = _i(m.group(1)); unit = m.group(2)
            if unit == "day":   return _resolved((today + timedelta(n)).isoformat())
            if unit == "week":  return _resolved((today + timedelta(n * 7)).isoformat())
            if unit == "month": return _resolved((today + _rd(months=n)).isoformat())
            if unit == "year":  return _resolved((today + _rd(years=n)).isoformat())

        m = re.match(
            r'(?:on\s+)?(?:(\d{1,2})\s+)?(january|february|march|april|may|june|july|'
            r'august|september|october|november|december|'
            r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
            r'(?:\s+(\d{1,2}))?(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?', e)
        if m:
            day_pre  = m.group(1)
            month    = MONTHS_MAP[m.group(2)]
            day_post = m.group(3)
            yr_str   = m.group(4)
            day = day_pre or day_post
            yr  = yr_str or str(y)
            if day: return _resolved(f"{yr}-{month}-{int(day):02d}")
            return _resolved(f"{yr}-{month}")

        if re.search(r'\bchristmas\b', e):       return _resolved("XXXX-12-25")
        if re.search(r'\bnew year', e):          return _resolved("XXXX-01-01")
        if re.search(r'\bhalloween\b', e):       return _resolved("XXXX-10-31")
        if re.search(r'\bthanksgiving\b', e):    return _resolved("XXXX-11-XX")
        if re.search(r'\bvalentine', e):         return _resolved("XXXX-02-14")
        if re.search(r'\brepublic day\b', e):    return _resolved("XXXX-01-26")
        if re.search(r'\bindependence day\b', e): return _resolved("XXXX-08-15")

        m = re.match(
            r'(early|mid|late)\s+'
            r'(january|february|march|april|may|june|july|august|'
            r'september|october|november|december)', e)
        if m:
            return _resolved(f"{y}-{MONTHS_MAP[m.group(2)]}")

        m = re.match(
            r'(last|this|next)\s+'
            r'(january|february|march|april|may|june|july|august|'
            r'september|october|november|december)', e)
        if m:
            offset = {"last": -1, "this": 0, "next": 1}[m.group(1)]
            return _resolved(f"{y + offset}-{MONTHS_MAP[m.group(2)]}")

    # ── TIME ──────────────────────────────────────────────────
    if entity_type == "TIME":
        yesterday = (today - timedelta(1)).isoformat()
        tomorrow  = (today + timedelta(1)).isoformat()

        anchored_times = {
            "last night":            f"{yesterday}TNI",
            "yesterday night":       f"{yesterday}TNI",
            "yesterday evening":     f"{yesterday}TEV",
            "yesterday morning":     f"{yesterday}TMO",
            "yesterday afternoon":   f"{yesterday}TAF",
            "yesterday at midnight": f"{yesterday}T00:00",
            "yesterday at noon":     f"{yesterday}T12:00",
            "this morning":          f"{today.isoformat()}TMO",
            "this afternoon":        f"{today.isoformat()}TAF",
            "this evening":          f"{today.isoformat()}TEV",
            "tonight":               f"{today.isoformat()}TNI",
            "tomorrow morning":      f"{tomorrow}TMO",
            "tomorrow afternoon":    f"{tomorrow}TAF",
            "tomorrow evening":      f"{tomorrow}TEV",
            "tomorrow night":        f"{tomorrow}TNI",
        }
        if e in anchored_times:
            return _resolved(anchored_times[e])

        time_exact = {
            "morning":       "XXXX-XX-XXTMO",
            "early morning": "XXXX-XX-XXTMO",
            "dawn":          "XXXX-XX-XXTMO",
            "sunrise":       "XXXX-XX-XXTMO",
            "afternoon":     "XXXX-XX-XXTAF",
            "evening":       "XXXX-XX-XXTEV",
            "night":         "XXXX-XX-XXTNI",
            "at night":      "XXXX-XX-XXTNI",
            "overnight":     "XXXX-XX-XXTNI",
            "midnight":      "XXXX-XX-XXT00:00",
            "noon":          "XXXX-XX-XXT12:00",
            "midday":        "XXXX-XX-XXT12:00",
            "late night":    "XXXX-XX-XXTNI",
            "late evening":  "XXXX-XX-XXTEV",
        }
        if e in time_exact:
            return _resolved(time_exact[e])

        m = re.match(r'in the\s+(morning|afternoon|evening|night)', e)
        if m:
            return _resolved({"morning": "XXXX-XX-XXTMO", "afternoon": "XXXX-XX-XXTAF",
                               "evening": "XXXX-XX-XXTEV", "night": "XXXX-XX-XXTNI"}[m.group(1)])

        m = re.match(r'yesterday\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$', e)
        if m:
            hr = int(m.group(1)); mi = m.group(2) or "00"
            if m.group(3) == "pm" and hr < 12: hr += 12
            if m.group(3) == "am" and hr == 12: hr = 0
            return _resolved(f"{yesterday}T{hr:02d}:{mi}")

        m = re.match(r'tomorrow\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$', e)
        if m:
            hr = int(m.group(1)); mi = m.group(2) or "00"
            if m.group(3) == "pm" and hr < 12: hr += 12
            if m.group(3) == "am" and hr == 12: hr = 0
            return _resolved(f"{tomorrow}T{hr:02d}:{mi}")

        m = re.match(r'today\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$', e)
        if m:
            hr = int(m.group(1)); mi = m.group(2) or "00"
            if m.group(3) == "pm" and hr < 12: hr += 12
            if m.group(3) == "am" and hr == 12: hr = 0
            return _resolved(f"{today.isoformat()}T{hr:02d}:{mi}")

        m = re.match(r'at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$', e)
        if m:
            hr = int(m.group(1)); mi = m.group(2) or "00"
            if m.group(3) == "pm" and hr < 12: hr += 12
            if m.group(3) == "am" and hr == 12: hr = 0
            return _resolved(f"XXXX-XX-XXT{hr:02d}:{mi}")

        m = re.match(r'^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$', e)
        if m:
            hr = int(m.group(1)); mi = m.group(2) or "00"
            if m.group(3) == "pm" and hr < 12: hr += 12
            if m.group(3) == "am" and hr == 12: hr = 0
            return _resolved(f"XXXX-XX-XXT{hr:02d}:{mi}")

        m = re.match(r'(?:around|about|roughly|approximately)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', e)
        if m:
            hr = int(m.group(1)); mi = m.group(2) or "00"
            if m.group(3) == "pm" and hr < 12: hr += 12
            return _resolved(f"XXXX-XX-XXT{hr:02d}:{mi}")

    # ── DURATION ──────────────────────────────────────────────
    if entity_type == "DURATION":

        dur_exact = {
            "half a day": "PT12H", "half a year": "P6M", "half a week": "P3.5D",
            "half a month": "P15D", "half an hour": "PT30M",
            "a couple of days": "P2D", "a couple of weeks": "P2W", "a couple of months": "P2M",
            "a few days": "P3D", "a few weeks": "P3W", "a few months": "P3M", "a few hours": "PT3H",
            "several days": "P3D", "several weeks": "P3W", "several months": "P3M",
            "several years": "P3Y", "several hours": "PT3H",
            "a day": "P1D", "a week": "P1W", "a month": "P1M", "a year": "P1Y",
            "an hour": "PT1H", "a minute": "PT1M",
            "overnight": "PT8H", "all day": "PT24H", "all night": "PT8H",
        }
        if e in dur_exact:
            return _resolved(dur_exact[e])

        m = re.match(
            r'^(?:about|around|nearly|almost|roughly|approximately\s+)?'
            r'(a|an|one|two|three|four|five|six|seven|eight|nine|ten|'
            r'eleven|twelve|couple|few|several|\d+(?:\.\d+)?)\s+'
            r'(day|week|month|year|hour|minute)s?$', e)
        if m:
            n = _n(m.group(1)); unit = m.group(2)
            n_str = str(int(n)) if n == int(n) else str(n)
            if unit == "day":    return _resolved(f"P{n_str}D")
            if unit == "week":   return _resolved(f"P{n_str}W")
            if unit == "month":  return _resolved(f"P{n_str}M")
            if unit == "year":   return _resolved(f"P{n_str}Y")
            if unit == "hour":   return _resolved(f"PT{n_str}H")
            if unit == "minute": return _resolved(f"PT{n_str}M")

        m = re.match(r'(\d+)\s+hours?\s+and\s+(\d+)\s+minutes?', e)
        if m:
            return _resolved(f"PT{m.group(1)}H{m.group(2)}M")

        m = re.match(r'(\d+)\s+and\s+a\s+half\s+(day|week|month|year|hour)s?', e)
        if m:
            n = int(m.group(1)); unit = m.group(2); v = n + 0.5
            if unit == "day":   return _resolved(f"P{v}D")
            if unit == "week":  return _resolved(f"P{v}W")
            if unit == "month": return _resolved(f"P{v}M")
            if unit == "year":  return _resolved(f"P{v}Y")
            if unit == "hour":  return _resolved(f"PT{v}H")

    # ── SET ───────────────────────────────────────────────────
    if entity_type == "SET":
        set_exact = {
            "every day": "P1D", "daily": "P1D", "each day": "P1D", "nightly": "P1D",
            "every night": "XXXX-XX-XXTNI", "every morning": "XXXX-XX-XXTMO",
            "every evening": "XXXX-XX-XXTEV", "every week": "P1W", "weekly": "P1W",
            "each week": "P1W", "every month": "P1M", "monthly": "P1M",
            "each month": "P1M", "every year": "P1Y", "yearly": "P1Y",
            "annually": "P1Y", "annual": "P1Y", "each year": "P1Y",
            "quarterly": "P3M", "every quarter": "P3M",
            "bi-weekly": "P2W", "biweekly": "P2W", "every two weeks": "P2W", "fortnightly": "P2W",
            "bi-monthly": "P2M", "every two months": "P2M",
            "twice a week": "P3.5D", "twice weekly": "P3.5D",
            "twice a day": "PT12H", "twice daily": "PT12H",
            "three times a week": "P2.3D", "every other day": "P2D",
            "every other week": "P2W", "every other month": "P2M",
            "hourly": "PT1H", "every hour": "PT1H", "every minute": "PT1M",
        }
        if e in set_exact:
            return _resolved(set_exact[e])

        m = re.match(
            r'every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
            r'mon|tue|wed|thu|fri|sat|sun)\s+(morning|afternoon|evening|night)', e)
        if m:
            wd_timex = WEEKDAY_TIMEX[WEEKDAYS[m.group(1)]]
            tod = {"morning": "TMO", "afternoon": "TAF",
                   "evening": "TEV", "night": "TNI"}[m.group(2)]
            return _resolved(f"{wd_timex}{tod}")

        m = re.match(
            r'every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
            r'mon|tue|wed|thu|fri|sat|sun)', e)
        if m:
            return _resolved(WEEKDAY_TIMEX[WEEKDAYS[m.group(1)]])

        m = re.match(
            r'every\s+(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+'
            r'(day|week|month|year)s?', e)
        if m:
            n = _i(m.group(1)); unit = m.group(2)
            if unit == "day":   return _resolved(f"P{n}D")
            if unit == "week":  return _resolved(f"P{n}W")
            if unit == "month": return _resolved(f"P{n}M")
            if unit == "year":  return _resolved(f"P{n}Y")

        m = re.match(r'once\s+a\s+(day|week|month|year)', e)
        if m:
            return _resolved({"day": "P1D", "week": "P1W", "month": "P1M", "year": "P1Y"}[m.group(1)])

    return _vague()
