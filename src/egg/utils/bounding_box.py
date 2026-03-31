from pydantic import BaseModel, ConfigDict
from typing_extensions import ClassVar

from egg.utils.geometry import Position, Dimensions

class AxisAlignedBoundingBox(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    center: Position
    size: Dimensions
