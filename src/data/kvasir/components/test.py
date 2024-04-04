from typing import Any, Optional

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from data.kvasir.components.kvasir_seg_dataset import KvasirSegDataset

if __name__ == "__main__":
    # kvasir = Transform_KvasirSEGDataset()
    print("reading")
    