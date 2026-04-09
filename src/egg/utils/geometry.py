import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import ClassVar


class Position(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    x: float
    y: float
    z: float

    def as_numpy(self) -> NDArray[np.float32]:
        return np.array([self.x, self.y, self.z])

    def round(self, ndigits: int = 3) -> None:
        self.x = round(self.x, ndigits)
        self.y = round(self.y, ndigits)
        self.z = round(self.z, ndigits)


class Dimensions(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    z: float = Field(ge=0)

    def as_numpy(self) -> NDArray[np.float32]:
        return np.array([self.x, self.y, self.z])

    def round(self, ndigits: int = 3) -> None:
        self.x = round(self.x, ndigits)
        self.y = round(self.y, ndigits)
        self.z = round(self.z, ndigits)


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

class AxisAlignedBoundingBox(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    center: Position
    size: Dimensions

    def round(self, ndigits: int = 3):
        self.center.round(ndigits=ndigits)
        self.size.round(ndigits=ndigits)
