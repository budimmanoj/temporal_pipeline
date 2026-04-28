# conflict_resolver.py  — Arch 2: Stage 6
#
# Merges entity list after all layers have run.
# - Deduplicates overlapping spans from the same sentence.
# - Flags conflicts where two entities resolve to different dates.
# - Prefers higher-confidence entry on conflict.

from collections import defaultdict


def resolve_conflicts(entities: list) -> list:
    """
    Merge and deduplicate the post-layer-executor entity list.

    Rules:
    1. Group by (sentence_idx, text.lower()) — same text in same sentence = same entity.
    2. Within each group, keep the highest-confidence entry.
    3. If two entries in the same group have different absolute_dates → conflict_flag=True.
    4. Return flat list sorted by absolute_date (RECURRING last).
    """
    groups: dict[tuple, list] = defaultdict(list)

    for e in entities:
        key = (e.get("sentence_idx", 0), e.get("text", "").lower().strip())
        groups[key].append(e)

    merged = []
    for key, group in groups.items():
        if len(group) == 1:
            group[0]["conflict_flag"] = False
            merged.append(group[0])
            continue

        # Sort by confidence descending
        group.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        winner = group[0]

        # Check if any other entry disagrees on absolute_date
        dates = {g.get("absolute_date") for g in group if g.get("absolute_date")}
        winner["conflict_flag"] = len(dates) > 1

        if winner["conflict_flag"]:
            winner["conflict_alts"] = [
                {"absolute_date": g.get("absolute_date"), "method": g.get("method")}
                for g in group[1:]
            ]

        merged.append(winner)

    # Sort: dated first (by absolute_date), RECURRING/UNKNOWN last
    def sort_key(e):
        d = e.get("absolute_date", "")
        if d in ("RECURRING", "UNKNOWN", "", None):
            return "9999-99-99"
        return d

    merged.sort(key=sort_key)

    n_conflicts = sum(1 for e in merged if e.get("conflict_flag"))
    print(f"\n[Conflict Resolver]  {len(merged)} entities, {n_conflicts} conflict(s)")
    for e in merged:
        flag = "⚠ CONFLICT" if e.get("conflict_flag") else ""
        print(f"   {e.get('absolute_date','?'):12s}  [{e.get('type','?'):8s}]  "
              f"{e.get('text',''):35s}  {flag}")

    return merged
