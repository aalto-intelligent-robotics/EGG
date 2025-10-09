from enum import Enum


class RetrievalStrategy(Enum):
    """
    Enumeration of different retrieval strategies for manipulating EGG using QueryProcessor.

    Each strategy determines how the data within EGG is processed and what components (spatial, event, edges) are included or excluded.
    """

    PRUNING_UNIFIED = 0
    """Combines graph pruning with full EGG."""

    PRUNING_UNIFIED_NO_EDGE = 1
    """Combines graph pruning with EGG without edges."""

    SPATIAL_ONLY = 2
    """No pruning, retrieves information focusing solely on spatial components."""

    EVENT_ONLY = 3
    """No pruning, retrieves information focusing solely on event components."""

    NO_EDGE = 4
    """No pruning, retrieves information with EGG without edges."""

    FULL_UNIFIED = 5
    """Use the full EGG without pruning."""
