"""
Rule-based replacements, per INTACT S3.1 / S5.2.

Direct identifiers (PERSON, CODE) are never sent to the LLM: statistical
disclosure control replaces them with a label plus a running number, which both
avoids inventing a name that belongs to a real person and keeps co-reference
chains intact across mentions.

Dates get a fixed generalisation ladder rather than LLM candidates -- cheaper,
and guaranteed truth-preserving, since every rung genuinely subsumes the one
below it.
"""
import re
from datetime import datetime

DIRECT_LABELS = {"PERSON", "CODE"}

_MONTHS = ("january february march april may june july august september "
           "october november december").split()
_SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
           5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn",
           10: "autumn", 11: "autumn"}

_DATE_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d",
                 "%d/%m/%Y", "%d.%m.%Y", "%B %Y", "%b %Y", "%Y")


class LabelCounter:
    """Assigns 'PERSON 1', 'CODE 2', ... consistently per entity within a document."""

    def __init__(self):
        self._seen = {}
        self._next = {}

    def label_for(self, entity_type, entity_key):
        key = (entity_type, entity_key)
        if key not in self._seen:
            n = self._next.get(entity_type, 0) + 1
            self._next[entity_type] = n
            self._seen[key] = f"{entity_type} {n}"
        return self._seen[key]


def parse_date(text):
    t = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt), fmt
        except ValueError:
            continue
    return None, None


def date_ladder(text):
    """Generalisation candidates for a date, most specific first.

    Mirrors the paper's worked example for 'March 12, 1999':
    March 1999 -> spring 1999 -> the first half of 1999 -> the late 1990s
    -> the late XX century. The rung the ladder starts on depends on how
    specific the original already is, so a bare year is not "generalised"
    into something more precise.
    """
    d, fmt = parse_date(text)
    if d is None:
        m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", text)
        if not m:
            return []
        year = int(m.group(1))
        has_month = has_day = False
    else:
        year = d.year
        has_month = "%B" in fmt or "%b" in fmt or "%m" in fmt
        has_day = "%d" in fmt

    decade = (year // 10) * 10
    part = "early" if year % 10 < 4 else ("late" if year % 10 > 6 else "mid")
    century = (year // 100) + 1
    half = "first half" if (d and d.month <= 6) else "second half"

    rungs = []
    if has_day and has_month and d:
        rungs.append(f"{_MONTHS[d.month - 1].capitalize()} {year}")
    if has_month and d:
        rungs.append(f"{_SEASON[d.month]} {year}")
        rungs.append(f"the {half} of {year}")
    rungs.append(f"the {part} {decade}s")
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(
        century % 10 if century % 100 not in (11, 12, 13) else 0, "th")
    rungs.append(f"the {part} {century}{suffix} century")
    # de-duplicate while keeping order
    out, seen = [], set()
    for r in rungs:
        if r.lower() not in seen:
            seen.add(r.lower())
            out.append(r)
    return out
