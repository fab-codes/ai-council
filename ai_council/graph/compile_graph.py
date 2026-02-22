from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from ai_council.graph.nodes.agent_node_factory import create_agent_node
from ai_council.graph.router import route_next_agent, AGENTS, AGENT_NODES
from ai_council.graph.states.general_state import GeneralState
from ai_council.utils.logger import get_logger

logger = get_logger(__name__)


def compile_graph():
    graph = StateGraph(GeneralState)
    graph_memory = InMemorySaver()

    for name, node_name in zip(AGENTS, AGENT_NODES):
        graph.add_node(node_name, create_agent_node(name))
        logger.info(f"Added node: {node_name}")

    graph.add_edge(START, AGENT_NODES[0])

    possible_targets = AGENT_NODES + [END]
    for node_name in AGENT_NODES:
        graph.add_conditional_edges(node_name, route_next_agent, possible_targets)

    return graph.compile(checkpointer=graph_memory)
