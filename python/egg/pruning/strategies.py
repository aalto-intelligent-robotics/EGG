from enum import Enum


class RetrievalStrategy(Enum):
    PRUNING_UNIFIED = 0
    PRUNING_UNIFIED_NO_EDGE = 1
    SPATIAL_ONLY = 2
    EVENT_ONLY = 3
    NO_EDGE = 4
    FULL_UNIFIED = 5
