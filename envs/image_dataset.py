

from torch.utils.data import Dataset

import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.transforms import v2

import os



# torch.inference_mode():
class FiveKDataset(Dataset):
    """
    A dataset that reads Adobe5K dataset images
    output : tensor of unprocessed images
    """
    def __init__(self, image_size,mode="train", ):
        if mode =='train':
            self.IMGS_PATH = "./dataset/FiveK/train/"
        else:
            self.IMGS_PATH = "./dataset/FiveK/test/"

        self.transform = transforms.Compose([
            v2.Resize(size = (image_size,image_size), interpolation= transforms.InterpolationMode.BICUBIC),
        ])


        self.img_files = [filename for filename in os.listdir(self.IMGS_PATH+'input/')]

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self,idx):
        source_path = self.img_files[idx]
        source = self.transform(read_image(self.IMGS_PATH+'input/'+source_path))
        target = self.transform(read_image(self.IMGS_PATH+'target/'+source_path))
        # source = self.transform(Image.open(ORIGINAL_FOLDER+source_path))

        return source, target 

    
