import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Any, Optional
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from src.data.kvasir.components.kvasir_dataset import KvasirDataset

class Transform_KvasirSEGDataset(Dataset):
    mean = None
    std = None

    def __init__(self, dataset: KvasirDataset, transform: Optional[Compose] = None) -> None:
        super().__init__()

        self.dataset = dataset

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        image, mask = self.dataset[index]  # (768, 768, 3), (768, 768)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            # img_size set in hydra config
            image = transformed["image"]  # (3, img_size, img_size)
            mask = transformed["mask"]  # (img_size, img_size), uint8
            mask = mask.unsqueeze(0).float()  # (1, img_size, img_size)

        return image, mask
    
if __name__ == "__main__":
    kvasir = Transform_KvasirSEGDataset(KvasirDataset())
    image, mask = kvasir[100]
    print(mask.max())    