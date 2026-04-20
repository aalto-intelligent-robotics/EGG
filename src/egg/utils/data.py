from typing import ClassVar, Literal
from pydantic import BaseModel, ConfigDict, Field

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
