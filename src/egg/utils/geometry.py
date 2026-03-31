from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import ClassVar


class Position(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    x: float
    y: float
    z: float


class Dimensions(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    z: float = Field(ge=0)


class Rotation(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    roll: float = Field(ge=0, le=360)
    pitch: float = Field(ge=0, le=360)
    yaw: float = Field(ge=0, le=360)


class Odometry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    position: Position
    rotation: Rotation


class Polygon(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    corners: list[Position] = Field(default_factory=list, min_length=3)
