from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Callable, Optional, List, Dict, Any

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
    lower_q: float = 0.25
    upper_q: float = 0.95
    use_gpu: bool = True
    write_collage: bool = True
    invert: bool = False

@dataclass
class ReconstructionConfig:
    method: str = "yin"
    shear_angle: Optional[float] = None
    method_params: dict[str, Any] = field(default_factory=dict)

@dataclass
class SegmentationConfig:
    model_type: str = "cyto"
    diameter: Optional[float] = None

