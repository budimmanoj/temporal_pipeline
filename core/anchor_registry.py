# anchor_registry.py  — Arch 2
#
# The AnchorRegistry is the shared mutable state that makes the
# divide-and-conquer approach work.
#
# KEY GUARANTEE: only rule-resolved or LLM-validated dates are written
# here.  Every write goes through a schema + range check.  A bad LLM
# response sets confidence=0 and conflict_flag rather than poisoning
# the chain.
#
# The registry is passed (by reference) to each layer executor call.
# Each layer reads the anchors it needs, does its work, then writes
# back — so the next layer always builds on verified facts.

import re
from datetime import date, datetime


# ── Validation ────────────────────────────────────────────────

_DATE_RE    = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_TIMEX_DATE = re.compile(r'^(\d{4}-\d{2}-\d{2})')

def _valid_date(value: str) -> bool:
    """Return True if value is a plausible YYYY-MM-DD string."""
    if not _DATE_RE.match(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def _extract_date(value: str) -> str | None:
    """Extract YYYY-MM-DD from a TimeML value string."""
    if _DATE_RE.match(value):
        return value
    m = _TIMEX_DATE.match(value)
    if m:
        return m.group(1)
    return None


# ── Registry ──────────────────────────────────────────────────

class AnchorRegistry:
    """
    Maps entity_id → resolved absolute date (YYYY-MM-DD).

    Also maintains a reverse index  sentence_idx → [entity_ids]
    so the layer executor can quickly find "what is the anchor date
    for sentence N" without scanning the whole registry.
    """

    def __init__(self):
        self._dates:      dict[int, str]        = {}   # entity_id → YYYY-MM-DD
        self._by_sent:    dict[int, list[int]]  = {}   # sentence_idx → [entity_ids]
        self._confidence: dict[int, float]      = {}   # entity_id → 0.0..1.0
        self._conflicts:  set[int]              = set()

    # ── Write ─────────────────────────────────────────────────

    def write(self,
              entity_id:   int,
              value:       str,
              sentence_idx: int,
              confidence:  float = 1.0,
              source:      str   = "rule") -> bool:
        """
        Write a resolved date to the registry.

        Performs schema + range validation before writing.
        Returns True on success, False if validation failed.
        On failure: writes confidence=0 and sets conflict_flag.
        """
        abs_date = _extract_date(value)

        # Validation gate
        if abs_date is None or not _valid_date(abs_date):
            # Do NOT poison the registry — flag the entity instead
            self._confidence[entity_id] = 0.0
            self._conflicts.add(entity_id)
            return False

        # Conflict check: if already written with a different date
        if entity_id in self._dates and self._dates[entity_id] != abs_date:
            self._conflicts.add(entity_id)
            # Keep the higher-confidence entry
            if confidence <= self._confidence.get(entity_id, 0.0):
                return False

        self._dates[entity_id]      = abs_date
        self._confidence[entity_id] = confidence
        self._by_sent.setdefault(sentence_idx, [])
        if entity_id not in self._by_sent[sentence_idx]:
            self._by_sent[sentence_idx].append(entity_id)

        return True

    # ── Read ──────────────────────────────────────────────────

    def read(self, entity_id: int) -> str | None:
        """Return the resolved date for entity_id, or None."""
        return self._dates.get(entity_id)

    def anchor_date_for_sentence(self, sentence_idx: int) -> str | None:
        """
        Return the best resolved date for the anchor event in sentence_idx.
        Returns the first (earliest-entity-id) resolved date found.
        """
        for s in range(sentence_idx - 1, -1, -1):
            eids = self._by_sent.get(s, [])
            for eid in eids:
                d = self._dates.get(eid)
                if d:
                    return d
        return None

    def snapshot(self) -> dict[int, str]:
        """Return a read-only copy of all currently resolved dates."""
        return dict(self._dates)

    def is_conflict(self, entity_id: int) -> bool:
        return entity_id in self._conflicts

    def confidence(self, entity_id: int) -> float:
        return self._confidence.get(entity_id, 0.0)

    def all_resolved_ids(self) -> list[int]:
        return list(self._dates.keys())

    def __len__(self):
        return len(self._dates)

    def __repr__(self):
        return (f"AnchorRegistry({len(self._dates)} entries, "
                f"{len(self._conflicts)} conflicts)")
