from typing import ClassVar, Literal
from pydantic import BaseModel, ConfigDict, Field
import re

from egg.utils.geometry import Position, Dimensions, AxisAlignedBoundingBox, Rotation

Ai2ThorTemperature = Literal["Cold", "Hot", "RoomTemp"]


class Ai2ThorAgentMetadata(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    name: str
    position: Position
    rotation: Rotation
    cameraHorizon: float
    isStanding: bool
    inHighFrictionArea: bool


class Ai2ThorRoomMetadata(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    roomType: str
    floorPolygon: list[dict[str, float]]

    def get_polygon_corners(self) -> list[Position]:
        return [
            Position(
                x=corner["x"],
                y=corner["y"],
                z=corner["z"],
            )
            for corner in self.floorPolygon
        ]


class Ai2ThorAABBMetadata(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    center: Position
    size: Dimensions


class Ai2ThorObjectMetadata(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    objectId: str
    objectType: str = Field(frozen=True)
    name: str

    visible: bool

    position: dict[str, float]
    rotation: dict[str, float]

    axisAlignedBoundingBox: Ai2ThorAABBMetadata

    parentReceptacles: list[str] | None
    receptacle: bool = Field(frozen=True)
    receptacleObjectIds: list[str] | None

    temperature: Ai2ThorTemperature
    isHeatSource: bool
    isColdSource: bool

    moveable: bool = Field(frozen=True)
    isMoving: bool

    pickupable: bool = Field(frozen=True)
    isPickedUp: bool

    dirtyable: bool = Field(frozen=True)
    isDirty: bool

    canBeUsedUp: bool = Field(frozen=True)
    isUsedUp: bool

    cookable: bool = Field(frozen=True)
    isCooked: bool

    sliceable: bool = Field(frozen=True)
    isSliced: bool

    toggleable: bool = Field(frozen=True)
    isToggled: bool

    breakable: bool = Field(frozen=True)
    isBroken: bool

    canFillWithLiquid: bool = Field(frozen=True)
    isFilledWithLiquid: bool

    openable: bool
    isOpen: bool
    openness: bool = Field(ge=0, le=1)

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        aabb_center = self.axisAlignedBoundingBox.center
        center = Position(x=aabb_center.x, y=aabb_center.y, z=aabb_center.z)
        aabb_size = self.axisAlignedBoundingBox.size
        dim = Dimensions(x=aabb_size.x, y=aabb_size.y, z=aabb_size.z)
        return AxisAlignedBoundingBox(center=center, size=dim)


ActionDict = dict[str, str | None]

ACTION_PATTERN = re.compile(r"^\s*(\w+)\s*\(\s*(.*?)\s*\)\s*$")
# Match either '...' or "..." string tokens inside the parentheses
ARG_TOKEN_PATTERN = re.compile(
    r"""
    (?:'([^']*)')        # group 1: single-quoted
  | (?:"([^"]*)")        # group 2: double-quoted
""",
    re.VERBOSE,
)


def parse_action(action_str: str) -> ActionDict:
    """
    Parse a single action string like:
      Move('Pen|surface|2|17')
      Place('CellPhone|surface|3|46', 'Box|surface|3|36')
      ToggleOn('Toaster|surface|2|1')
      Open('Fridge|2|1')
      Close('Fridge|2|1')
      Pick('Apple|surface|2|3')

    Returns a dict:
      {
        "action_type": "<Move|Place|ToggleOn|Open|Close|Pick>",
        "object": "<object_id or None>",
        "receptacle": "<receptacle_id or None>"
      }

    Notes:
    - action_type is the verb at the start (e.g., "Place" -> action_type "Place").
    - object is the first argument (if present).
    - receptacle is the second argument (if present), otherwise None.
    """
    m = ACTION_PATTERN.match(action_str)
    if not m:
        raise ValueError(f"Unrecognized action format: {action_str}")

    verb, args_str = m.group(1), m.group(2)

    # Extract args as quoted string tokens (supports single or double quotes)
    args: list[str] = []
    if args_str.strip():
        for sm in ARG_TOKEN_PATTERN.finditer(args_str):
            token = sm.group(1) if sm.group(1) is not None else sm.group(2)
            args.append(token)

    obj: str | None = args[0] if len(args) >= 1 else None
    receptacle: str | None = args[1] if len(args) >= 2 else None

    return {
        "action_type": verb,  # Use the actual verb (e.g., "Place", not "Move")
        "object": obj,
        "receptacle": receptacle,
    }


def parse_action_list(actions: list[str]) -> list[ActionDict]:
    """
    Parse a list of action strings.
    """
    return [parse_action(a) for a in actions]
