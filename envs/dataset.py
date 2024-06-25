
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.io import read_image

IMG_SIZE = 64 #training image size
ORIGINAL_FOLDER = './dataset/original/'
EXPERTC_FOLDER  = './dataset/expertC/'
default_aug = transforms.Compose([
            transforms.Resize(size = IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        ])
# torch.inference_mode():
class FiveKDataset(Dataset):
    """
    A dataset that reads Adobe5K dataset images
    output : tensor of unprocessed images
    """
    def __init__(self, mode="train", transform = default_aug):
        if mode =='train':
            self.IMGS_PATH = "./dataset/trainSource_jpg.txt"
        else:
            self.IMGS_PATH = "./dataset/test_jpg.txt"

        self.transform = transform 
        with open(self.IMGS_PATH ) as file:
            content = file.read() 
            self.img_files = content.split('\n')


    def __len__(self):
        return len(self.source_files)
    
    def __getitem__(self,idx):
        
        source_path = self.img_files[idx]
        source = read_image(ORIGINAL_FOLDER+source_path)
        target = read_image(EXPERTC_FOLDER+source_path)

        return source, target 

    
