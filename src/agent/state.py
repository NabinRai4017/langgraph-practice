from typing import Annotated
from langgraph.graph import MessagesState


class State(MessagesState):
    """Graph state. Extends MessagesState which includes a `messages` key
    with automatic list-append reducer."""
    pass
