from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Callable

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
