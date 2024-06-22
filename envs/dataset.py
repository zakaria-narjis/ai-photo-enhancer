
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.io import read_image

IMG_SIZE = 64 #training image size

default_aug = test_augmentation = transforms.Compose([
            transforms.Resize(size = IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
   
        ])
# torch.inference_mode():
class PhotoEnhancementDataset(Dataset):
    def __init__(self, mode="train", transform = default_aug):
       self.SOURCE_IMGS_PATH = "./fivek_dataset/trainSource.txt"
       self.TARGET_IMGS_PATH = "./fivek_dataset/trainTarget.txt"
       self.transform = transform 
       with open(self.SOURCE_IMGS_PATH ) as file:
           content = file.read() 
           self.source_files = content.split('\n')

       with open(self.TARGET_IMGS_PATH) as file:
           content = file.read()
           self.target_files = content.split('\n')
    
    def __len__(self):
        return len(self.source_files)
    
    def __getitem__(self,idx):
        source_path = self.source_files[idx]
        target_path = self.target_files[idx]
        img = self.transform(read_image(source_path))/
        target = self.transform(read_image(target_path))

        return img, target 

    
