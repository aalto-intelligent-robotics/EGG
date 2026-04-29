from enum import Enum


class ActionType(str, Enum):
    MOVE = "move"
    PICK = "pick"
    PLACE = "place"
    OPEN = "open"
    CLOSE = "close"
    TOGGLE_ON = "toggleon"
    TOGGLE_OFF = "toggleoff"
