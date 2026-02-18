import logging

from egg.graph.node import ObjectNode
from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="perception/instance_matching.log",
)


def are_similar_objects_gt(object_node_0: ObjectNode, object_node_1: ObjectNode):
    if object_node_0.name == object_node_1.name:
        return True
    return False


def are_similar_objects_vision(object_node_0: ObjectNode, object_node_1: ObjectNode):
    # TODO: Implement this
    raise NotImplementedError


def are_similar_objects(
    object_node_0: ObjectNode, object_node_1: ObjectNode, use_gt: bool = True
):
    if use_gt:
        return are_similar_objects_gt(
            object_node_0=object_node_0, object_node_1=object_node_1
        )
    else:
        return are_similar_objects_vision(
            object_node_0=object_node_0, object_node_1=object_node_1
        )
