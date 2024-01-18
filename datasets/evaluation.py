from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class EvaluationDataset(Dataset):
    def __init__(self,
                 places_dirs: list[Path],
                 number_of_images_per_place: int,
                 transforms: Callable,
                 return_indices: bool = False):
        super().__init__()

        self._places_images_paths: list[list[Path]] = [
            sorted([image_path for image_path in place_dir.iterdir()
                    if image_path.is_file()])[:number_of_images_per_place]
            for place_dir in places_dirs
        ]
        self._number_of_images_per_place = number_of_images_per_place
        self._transforms = transforms
        self._return_indices = return_indices

    def __len__(self) -> int:
        return len(self._places_images_paths) * self._number_of_images_per_place

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, int, int]:
        place_index = index // self._number_of_images_per_place
        image_index = index % self._number_of_images_per_place
        image_path = self._places_images_paths[place_index][image_index]

        image = np.asarray(Image.open(image_path))
        image = self._transforms(image=image)['image']

        if self._return_indices:
            return image, torch.tensor(place_index), place_index, image_index

        return image, torch.tensor(place_index)

    def get_path_by_index(self, place_id: int, image_index: int) -> Path:
        return self._places_images_paths[place_id][image_index]
