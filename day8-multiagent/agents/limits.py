"""Hard limits, enforced in code, same philosophy as day4-rag/rag/agent.py's
three hard limits -- just distributed across three roles instead of one
loop. Starting values, not final ones: propose calibrating these against
real eval_multiagent.py runs the same way day4-rag's
RELATED_SIMILARITY_THRESHOLD was measured (RESULTS.md), not guessed once
and left. Centralized here (not duplicated per-node) so graph.py's
conditional edges and researcher.py's own mid-turn check can never disagree
about what the limit actually is.
"""

# Total real LLM calls across ALL nodes for one question -- roughly double
# Day 7's DEFAULT_MAX_STEPS=8, since three roles doing comparable work to
# one loop plausibly need more, not because 15 is independently justified.
MAX_TOTAL_STEPS = 15

# Wider than Day 7's DEFAULT_MAX_COST_USD=0.25 for the same reason as
# MAX_TOTAL_STEPS -- more calls per question, checked with the exact same
# real-token-usage accounting (agents/llm.py's cost_usd), never an estimate.
MAX_COST_USD = 0.35

# How many full writer -> critic round trips beyond the first draft are
# allowed before finalize() takes the last draft as-is (honestly labeled,
# not "approved" -- see agents/graph.py). The single number this project
# is least confident about; genuinely new territory with no Day 7 precedent.
MAX_REVISION_CYCLES = 2

# The researcher's OWN cap on tool-loop iterations within its one turn --
# in addition to, not instead of, MAX_TOTAL_STEPS above. Stops one confused
# researcher turn (e.g. rephrasing the same query without ever reading an
# article -- an observed Day 7 failure mode, see day4-rag/RESULTS.md) from
# eating the entire global budget before the graph ever produces a draft.
RESEARCHER_MAX_STEPS_PER_TURN = 4

# LangGraph's own recursion backstop, independent of the in-state counters
# above -- generously larger than MAX_TOTAL_STEPS should ever allow, so it
# should never actually fire. If it ever does, that's a signal the counter
# logic itself has a bug, not a normal stop -- see graph.py's
# "graph_recursion_limit" stopped_reason.
GRAPH_RECURSION_LIMIT = 25
