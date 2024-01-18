from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PredictionDataset(Dataset):
    def __init__(self,
                 test_dir: Path,
                 transforms: Callable):
        super().__init__()

        self._images_paths: list[Path] = sorted([image_path for image_path in test_dir.iterdir()])
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._images_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self._images_paths[index]

        image = np.asarray(Image.open(image_path))
        image = self._transforms(image=image)['image']

        return image, image_path.stem
