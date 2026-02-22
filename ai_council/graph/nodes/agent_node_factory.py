from ai_council.agents.primitive_agent import PrimitiveAgent
from ai_council.graph.states.general_state import GeneralState


def create_agent_node(name: str):
    """Create Langgraph node with passed name."""
    agent = PrimitiveAgent(name)

    async def node(state: GeneralState) -> dict:
        return await agent.process(state)

    node.__name__ = f"{name.lower()}_node"
    return node
