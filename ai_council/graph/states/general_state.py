from typing import TypedDict


class GeneralState(TypedDict):
    tick: int
    last_speaker: str
    last_message: str
