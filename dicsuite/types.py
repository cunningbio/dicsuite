from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Callable, Optional, List

@dataclass
class ImageData:
    image_path: Path
    source_path: Path
    loader: Callable[[Path], np.ndarray]
    _image: np.ndarray | None = field(default=None, init=False)

    def read(self):
        if self._image is None:
            self._image = self.loader(self.image_path)

    def get_output_path(self, output_root):
        try:
            rel = self.image_path.relative_to(self.source_path.parent)
        except ValueError:
            rel = self.image_path.name
        return output_root / rel

@dataclass
class GeneralConfig:
    run_cellpose: bool = False
    recursive: bool = False
    mode: str = "batch"
    infer_from_first: bool = True
    contrast_adj: bool = True
    use_gpu: bool = True
    write_collage: bool = True

@dataclass
class ReconstructionConfig:
    method: str = "inverse"
    smooth_in: float | List = field(default_factory=lambda: [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10])
    stabil_in: float | List = field(default_factory=lambda: [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10])
    shear_angle: Optional[float] = None

@dataclass
class SegmentationConfig:
    model_type: str = "cyto"
    diameter: Optional[float] = None

