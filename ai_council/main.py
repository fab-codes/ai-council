from ai_council.graph.compile_graph import compile_graph
from ai_council.graph.states.general_state import GeneralState
from ai_council.utils.logger import get_logger

logger = get_logger(__name__)


async def start_network() -> dict:
    app = compile_graph()

    config = {"configurable": {"thread_id": "1"}}

    initial_state: GeneralState = {
        "tick": 0,
        "last_speaker": "Human",
        "last_message": "Ciao a tutti! Cosa pensate del libero arbitrio?",
    }

    result = await app.ainvoke(initial_state, config)

    return result


if __name__ == "__main__":
    import asyncio

    asyncio.run(start_network())
