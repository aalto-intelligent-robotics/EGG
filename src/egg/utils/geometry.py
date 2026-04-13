import numpy as np
from typing import Literal, Self
from numpy.typing import NDArray, ArrayLike
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import ClassVar, override


class CoordXYZ(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    x: float
    y: float
    z: float

    def __add__(self, other: "CoordXYZ"):
        return CoordXYZ(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def as_numpy(self) -> NDArray[np.float32]:
        return np.array([self.x, self.y, self.z])

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def round(self, ndigits: int = 3) -> None:
        self.x = round(self.x, ndigits)
        self.y = round(self.y, ndigits)
        self.z = round(self.z, ndigits)

    def as_numpy_2d(
        self, omitted_axis: Literal["x", "y", "z"] = "y"
    ) -> NDArray[np.float32]:
        if omitted_axis == "x":
            return np.array([self.y, self.z])
        elif omitted_axis == "y":
            return np.array([self.x, self.z])
        elif omitted_axis == "z":
            return np.array([self.x, self.y])

    @classmethod
    def from_numpy(cls, arr: NDArray[np.float16 | np.float32 | np.float64]) -> Self:
        assert len(arr) == 3
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

    @classmethod
    def from_list(cls, lst: list[float]) -> Self:
        if len(lst) != 3:
            raise ValueError(f"Expected list of length 3, got {len(lst)}")
        return cls(x=float(lst[0]), y=float(lst[1]), z=float(lst[2]))

    @classmethod
    def from_tuple(cls, tup: tuple[float, float, float]) -> Self:
        return cls(x=float(tup[0]), y=float(tup[1]), z=float(tup[2]))


class Position(CoordXYZ):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    def euclidean(self, pos: "Position") -> float:
        return float(np.linalg.norm(self.as_numpy() - pos.as_numpy()))

    def euclidean2d(
        self,
        other: "Position",
        omitted_axis: Literal["x", "y", "z"] = "y",
    ) -> float:
        d = self.as_numpy_2d(omitted_axis=omitted_axis) - other.as_numpy_2d(
            omitted_axis
        )
        return float(np.linalg.norm(d))


class Dimensions(CoordXYZ):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    z: float = Field(ge=0)

    def get_max_dim(self) -> float:
        return max(self.x, self.y, self.z)


class Rotation(CoordXYZ):
    x: float = Field(ge=0, le=360)
    y: float = Field(ge=0, le=360)
    z: float = Field(ge=0, le=360)


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

    def min_max(self) -> tuple[Position, Position]:
        """
        Returns the min and max corners of the AABB as (bmin, bmax).
        Interprets 'size' as full lengths; change 'half = size' if your size is half-extents.
        """
        c = self.center.as_numpy().astype(float)
        half = self.size.as_numpy().astype(float) / 2.0
        bmin = c - half
        bmax = c + half
        return Position.from_numpy(bmin), Position.from_numpy(bmax)

    def clearance(self, p: Position) -> float:
        """
        Signed minimum distance from point p to any AABB face.
        Positive if p is inside (distance to nearest face), negative if outside (penetration).
        """
        bmin, bmax = self.min_max()
        dx_min = p.x - bmin.x
        dx_max = bmax.x - p.x
        dy_min = p.y - bmin.y
        dy_max = bmax.y - p.y
        dz_min = p.z - bmin.z
        dz_max = bmax.z - p.z
        return float(min(dx_min, dx_max, dy_min, dy_max, dz_min, dz_max))

    def contains(self, pos: Position) -> bool:
        """
        Convenience check: True if p is inside or on the boundary.
        """
        bmin, bmax = self.min_max()
        return bool(
            (pos.as_numpy() >= bmin.as_numpy()).all()
            and (pos.as_numpy() <= bmax.as_numpy()).all()
        )

    def contains_2d(
        self,
        pos: Position,
        omitted_axis: Literal["x", "y", "z"] = "y",
        offset: float = 0.0,
    ) -> bool:
        """
        Convenience check: True if p is inside or on the boundary.
        """
        bmin, bmax = self.min_max()
        pos_numpy_2d = pos.as_numpy_2d(omitted_axis=omitted_axis)
        bmin_numpy_2d = (
            bmin.as_numpy_2d(omitted_axis=omitted_axis)
            - np.ones_like(pos_numpy_2d) * offset
        )
        bmax_numpy_2d = (
            bmax.as_numpy_2d(omitted_axis=omitted_axis)
            + np.ones_like(pos_numpy_2d) * offset
        )
        return bool(
            (pos_numpy_2d >= bmin_numpy_2d).all()
            and (pos_numpy_2d <= bmax_numpy_2d).all()
        )
