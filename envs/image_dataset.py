
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.transforms import v2
from PIL import Image 
 
IMG_SIZE = 64 #training image size
ORIGINAL_FOLDER = './dataset/original/'
EXPERTC_FOLDER  = './dataset/expertC/'

default_aug = transforms.Compose([
            v2.Resize(size = (IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor(),
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
        return len(self.img_files)
    
    def __getitem__(self,idx):
        
        source_path = self.img_files[idx]
        # source = self.transform(read_image(ORIGINAL_FOLDER+source_path))

        source = self.transform(Image.open(ORIGINAL_FOLDER+source_path))

        # target = self.transform(read_image(EXPERTC_FOLDER+source_path))
        target = source
        return source, target 

    
