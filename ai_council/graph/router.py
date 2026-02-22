from langgraph.graph import END
from ai_council.graph.states.general_state import GeneralState

MAX_TICKS = 10
AGENTS = ["Angela", "Eva", "Kevin"]
AGENT_NODES = [f"{name.lower()}_node" for name in AGENTS]


def route_next_agent(state: GeneralState) -> str:
    if state["tick"] >= MAX_TICKS:
        return END

    last_speaker = state.get("last_speaker", "")

    try:
        idx = AGENTS.index(last_speaker)
        next_idx = (idx + 1) % len(AGENTS)
    except ValueError:
        next_idx = 0

    return AGENT_NODES[next_idx]
