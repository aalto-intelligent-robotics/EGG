from copy import deepcopy
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
    events: EventComponents
    use_gt_id: bool = True
    use_gt_caption: bool = True
    use_guided_auto_caption: bool = True
    device: str = "cuda:0"
    do_sample: bool = False

    def is_empty(self) -> bool:
        return self.spatial.is_empty() and self.events.is_empty()

    def set_spatial_components(self, spatial_components: SpatialComponents):
        """
        Updates the spatial component of EGG.

        :param spatial_components: New spatial components.
        :type spatial_components: SpatialComponents
        """
        self.spatial = spatial_components

    def set_event_components(self, event_components: EventComponents):
        """
        Updates the event component of EGG.

        :param event_components: New event configuration.
        :type event_components: EventComponents
        """
        self.events = event_components

    def get_spatial_components(self) -> SpatialComponents:
        """
        Retrieves a copy of the current spatial component.

        :returns: A copy of the spatial components.
        :rtype: SpatialComponents
        """
        return deepcopy(self.spatial)

    def get_event_components(self) -> EventComponents:
        """
        Retrieves a copy of the current event component.

        :returns: A copy of the event components.
        :rtype: EventComponents
        """
        return deepcopy(self.events)

    def serialize(self):
        """
        Serializes the entire EGG state including spatial components, event components,
        and event-object edges.

        :returns: Dictionary representation of the EGG's current state.
        :rtype: Dict
        """
        spatial_data = self.spatial.model_dump(mode="python")
        event_data = self.events.model_dump(mode="python")
        egg_data = {
            "nodes": {"spatial": spatial_data, "events": event_data},
        }
        return egg_data

    def pretty_str(self) -> str:
        """
        Generates a human-readable string representation of the spatial, event,
        and edge components within EGG.

        :returns: String representation of the current graph's state.
        :rtype: str
        """
        egg_str = ""
        egg_str += self.spatial.pretty_str()
        egg_str += self.events.pretty_str()
        # edge_str = "\n🔗🔗🔗 EDGES 🔗🔗🔗\n"
        # for edge in self.event_edges:
        #     edge_str += edge.pretty_str()
        # egg_str += edge_str
        return egg_str
