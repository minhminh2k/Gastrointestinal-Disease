import shutil
import zipfile
import gdown
import os 
import numpy as np
import pandas as pd

from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import torch

class KvasirDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data",
    ) -> None:
        super().__init__()
        
        self.data_dir = data_dir
        
        self.prepare_data()
        
        image_folder_dir = os.path.join(self.data_dir, "kvasir-seg/Kvasir-SEG/images") 
        mask_folder_dir = image_folder_dir.replace("images", "masks")

        self.image_path_list = []
        self.masks_path_list = []
                
        # Read all the image paths
        for file in os.listdir(image_folder_dir):
            if file.endswith(".jpg"):
                image_path = os.path.join(image_folder_dir, file)
                self.image_path_list.append(image_path)
                
                mask_path = image_path.replace("images", "masks")
                self.masks_path_list.append(mask_path)
        
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, index):
        image = np.array(Image.open(self.image_path_list[index]).convert("RGB"))
        mask = Image.open(self.masks_path_list[index])
        mask = self.grayscale_mask(mask)
        
        return np.array(image, dtype=np.uint8), mask
    
    def prepare_data(self):
        data_path = os.path.join(self.data_dir, "kvasir-seg")
        if os.path.exists(data_path):
            print("Data is downloaded")
            return
        
        file_id = "1VQrfSqG2HPy5y1FykcsrE0x1d6iyxENC"
        output = "kvasir-seg.zip"
        print("Downloading data")
        
        gdown.download(id=file_id, output=output, quiet=False)

        os.makedirs(data_path, exist_ok=True)

        shutil.move("./kvasir-seg.zip", data_path)

        downloaded_file = os.path.join(data_path, "kvasir-seg.zip")

        print("Extracting ...")
        with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
            zip_ref.extractall(data_path)

        print("Removing unnecessary file")
        os.remove(downloaded_file)

        print("Done")
        
    def grayscale_mask(self, mask):
        mask_gray = mask.convert('L')

        mask_gray_np = np.array(mask_gray)

        # Threshold
        threshold = 10  
        mask_gray_img = np.where(mask_gray_np > threshold, 1, 0).astype(np.uint8)
        
        return mask_gray_img
        
if __name__ == "__main__":
    kvasir = KvasirDataset()
    print(len(kvasir.image_path_list))
    print(len(kvasir.masks_path_list))
    
    print(kvasir.image_path_list[100])
    image, mask = kvasir[100]
    print(mask.dtype)
    print(mask.max())
    print(mask.min())
    
    
    