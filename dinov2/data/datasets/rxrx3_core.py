import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset

from datasets import load_dataset

from huggingface_hub import hf_hub_download
import pandas as pd


# Dummy decoders (replace with actual logic or imports)
class ImageDataDecoder:
    def __init__(self, data: bytes):
        self.data = data

    def decode(self):
        return Image.open(io.BytesIO(self.data)).convert("RGB")

class TargetDecoder:
    def __init__(self, target: Any):
        self.target = target

    def decode(self):
        return self.target  # or one-hot encode, etc.

# Actual dataset class
class MyBasicDataset(ExtendedVisionDataset):
    def __init__(self, root: str, transform=None, target_transform=None):
        super().__init__(root=root, transforms=transform)


        file_path_metadata = hf_hub_download("recursionpharma/rxrx3-core", filename="metadata_rxrx3_core.csv",repo_type="dataset")
        file_path_embs = hf_hub_download("recursionpharma/rxrx3-core", filename="OpenPhenom_rxrx3_core_embeddings.parquet",repo_type="dataset")
        open_phenom_embeddings = pd.read_parquet(file_path_embs)
        rxrx3_core_metadata = pd.read_csv(file_path_metadata)
        rxrx3_core_metadata["mapping"] = rxrx3_core_metadata.experiment_name + "/Plate" + rxrx3_core_metadata.plate.astype(str) + "/" + rxrx3_core_metadata.address.astype(str) 

        mapping = pd.DataFrame(rxrx3_core["__key__"]).reset_index().rename(columns={0: "paths", "index": "id"})

        self.image_set = load_dataset("recursionpharma/rxrx3-core", split='train')
        self.metadata = rxrx3_core_metadata
        self.mapping = mapping

    def __len__(self):
        return len(self.metadata)

    def get_image_data(self, index: int):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        id self.metadata["mapping"]
        with open(img_path, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        img_name = self.image_files[index]
        return self.labels.get(img_name, 0)  # default to 0 if not found

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target