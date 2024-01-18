from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MetricLearningDataset(Dataset):
    def __init__(self,
                 places_dirs: list[Path],
                 number_of_places_per_batch: int,
                 number_of_images_per_place: int,
                 number_of_batches_per_epoch: int,
                 transforms: Callable):
        super().__init__()

        self._places_images_paths: list[list[Path]] = [
            sorted([image_path for image_path in place_dir.iterdir() if image_path.is_file()])
            for place_dir in places_dirs
        ]
        self._number_of_places = number_of_places_per_batch
        self._number_of_images_per_place = number_of_images_per_place
        self._number_of_samples_per_epoch = number_of_batches_per_epoch
        self._transforms = transforms

    def __len__(self) -> int:
        return self._number_of_samples_per_epoch

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        selected_places_indices = torch.randperm(len(self._places_images_paths))[:self._number_of_places]
        transformed_images = []
        selected_images_place_ids = []
        for place_index in selected_places_indices:
            place_images = self._places_images_paths[place_index]
            selected_image_indices = torch.randperm(len(place_images))[:self._number_of_images_per_place]
            selected_images = [np.asarray(Image.open(place_images[image_index]))
                               for image_index in selected_image_indices]
            for image in selected_images:
                image = self._transforms(image=image)['image']

                transformed_images.append(image)
                selected_images_place_ids.append(place_index)

        transformed_images, selected_images_place_ids = self._shuffle(transformed_images, selected_images_place_ids)

        return torch.stack(transformed_images), torch.tensor(selected_images_place_ids)

    @staticmethod
    def _shuffle(images: list[torch.Tensor], place_ids: list[int]) -> tuple[list[torch.Tensor], list[int]]:
        indices = torch.randperm(len(images))

        return [images[index] for index in indices], [place_ids[index] for index in indices]
