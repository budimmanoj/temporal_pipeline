# graph_builder.py  — Arch 2: Stage 3 + 4
#
# Builds a DependencyGraph from the rule pass results and returns
# topologically sorted layer assignments.
#
# An edge  A → B  means "B depends on A being resolved first."
# Layer 0 = entities with no dependencies (always rule-resolved).
# Layer N = entities whose deepest dependency is at layer N-1.

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class DependencyGraph:
    # entity_id → entity dict (after rule pass)
    nodes:     dict = field(default_factory=dict)
    # entity_id → list of entity_ids it depends on
    edges:     dict = field(default_factory=lambda: defaultdict(list))
    # entity_id → layer number
    layer_map: dict = field(default_factory=dict)
    # ordered list of layer numbers
    topo_order: list = field(default_factory=list)
    cycle_free: bool = True
    cycle_nodes: list = field(default_factory=list)


def build_graph(entities: list) -> DependencyGraph:
    """
    Build a DAG from the post-rule-pass entity list.

    Each entity has:
      entity_id      : unique int
      rule_result    : dict from rule_normalize()
                         {"status": "resolved", "value": ...}
                       | {"status": "anchor_dep", "anchor_tag": ..., ...}
                       | {"status": "vague"}
      sentence_idx   : which sentence the entity came from

    Edges: an anchor_dep entity at sentence S references the main
    event/anchor of sentence M (M < S).  We find the resolved entity
    in sentence M that is most likely the anchor and draw an edge.

    Returns a fully populated DependencyGraph ready for topo sort.
    """
    g = DependencyGraph()

    for e in entities:
        eid = e["entity_id"]
        g.nodes[eid] = e
        g.edges[eid]  # ensure key exists

    # ── Build edges ───────────────────────────────────────────
    # For each anchor_dep entity, find its anchor.
    # Strategy: look for a resolved entity in the sentence referenced
    # by anchor_tag.  anchor_tag is "event" (from rule_normalize); the
    # graph builder refines it using sentence_idx heuristic.

    # Group resolved entities by sentence for quick lookup
    resolved_by_sent: dict[int, list] = defaultdict(list)
    for e in entities:
        if e["rule_result"]["status"] == "resolved":
            resolved_by_sent[e["sentence_idx"]].append(e["entity_id"])

    for e in entities:
        eid = e["entity_id"]
        rr  = e["rule_result"]
        if rr["status"] != "anchor_dep":
            continue

        s = e["sentence_idx"]
        # Find the most recent sentence before s that has a resolved entity
        anchor_eid = _find_anchor(eid, s, entities, resolved_by_sent)
        if anchor_eid is not None:
            g.edges[eid].append(anchor_eid)
            # Update the anchor_tag on the entity with the concrete id
            e["rule_result"]["anchor_entity_id"] = anchor_eid

    # ── Topo sort (Kahn's algorithm) ──────────────────────────
    in_degree = {eid: 0 for eid in g.nodes}
    for eid, deps in g.edges.items():
        for dep in deps:
            in_degree[eid] += 1   # eid depends on dep

    queue = deque(eid for eid, deg in in_degree.items() if deg == 0)
    layer = {eid: 0 for eid in queue}
    processed = 0

    while queue:
        eid = queue.popleft()
        processed += 1
        # All entities that depend on eid
        for other_eid, deps in g.edges.items():
            if eid in deps:
                in_degree[other_eid] -= 1
                layer[other_eid] = max(layer.get(other_eid, 0), layer[eid] + 1)
                if in_degree[other_eid] == 0:
                    queue.append(other_eid)

    if processed < len(g.nodes):
        # Cycle detected — find the culprits
        g.cycle_free  = False
        g.cycle_nodes = [eid for eid, deg in in_degree.items() if deg > 0]
        # Assign all cycle nodes to a special layer so we can still run
        for eid in g.cycle_nodes:
            layer[eid] = 999

    g.layer_map   = layer
    g.topo_order  = sorted(set(layer.values()))

    # Write layer back onto each entity
    for e in entities:
        e["layer"] = layer.get(e["entity_id"], 0)

    return g


def _find_anchor(dep_eid: int,
                 dep_sent: int,
                 entities: list,
                 resolved_by_sent: dict) -> int | None:
    """
    Find the entity_id most likely to be the anchor for dep_eid.

    Heuristic (in priority order):
    1. Resolved entity in the immediately preceding sentence.
    2. Resolved entity in any earlier sentence (most recent wins).
    3. If none, return None (entity becomes layer 0 anyway).
    """
    for s in range(dep_sent - 1, -1, -1):
        candidates = resolved_by_sent.get(s, [])
        if candidates:
            # Prefer the FIRST resolved entity in that sentence (usually the anchor event)
            return candidates[0]
    return None


def layer_batches(g: DependencyGraph) -> list[list]:
    """
    Returns a list-of-lists: [[entity_ids in layer 0], [layer 1], ...]
    Ordered from lowest to highest layer number.
    """
    batches: dict[int, list] = defaultdict(list)
    for eid, lyr in g.layer_map.items():
        batches[lyr].append(eid)
    return [batches[lyr] for lyr in sorted(batches)]
