from langgraph.graph import MessagesState


class GeneralState(MessagesState):
    tick: int
    last_speaker: str
    last_message: str

    pass
