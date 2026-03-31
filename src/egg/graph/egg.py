import logging
from pydantic import BaseModel

from egg.graph.event import EventComponents
from egg.graph.spatial import SpatialComponents
from egg.utils.logger import getLogger

logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="graph/egg.log",
)


class EGG(BaseModel):
    """
    EGG (Event-Grounding Graph) framework that grounds events semantic context to spatial geometrics.
    """

    spatial: SpatialComponents
    event: EventComponents
    use_gt_id: bool = True
    use_gt_caption: bool = True
    use_guided_auto_caption: bool = True
    device: str = "cuda:0"
    do_sample: bool = False

    def pretty_str(self) -> str:
        """
        Generates a human-readable string representation of the spatial, event,
        and edge components within EGG.

        :returns: String representation of the current graph's state.
        :rtype: str
        """
        egg_str = ""
        egg_str += self.spatial.pretty_str()
        egg_str += self.event.pretty_str()
        # edge_str = "\n🔗🔗🔗 EDGES 🔗🔗🔗\n"
        # for edge in self.event_edges:
        #     edge_str += edge.pretty_str()
        # egg_str += edge_str
        return egg_str
