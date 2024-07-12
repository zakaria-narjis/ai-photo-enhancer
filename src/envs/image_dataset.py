

from torch.utils.data import Dataset

import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.transforms import v2
import os
from pathlib import Path



# torch.inference_mode():
class FiveKDataset(Dataset):
    """
    A dataset that reads Adobe5K dataset images
    output : tensor of unprocessed images
    """
    def __init__(self, image_size,mode="train", resize= True,augment_data=True):
        current_dir = Path(__file__).parent.absolute()
        if mode =='train':
            self.IMGS_PATH = os.path.join(current_dir, "..", "..", "dataset", "FiveK", "train")
        else:
            self.IMGS_PATH = os.path.join(current_dir, "..", "..", "dataset", "FiveK", "test")
        self.resize= resize
        if augment_data:
            self.transform = transforms.Compose([
                v2.Resize(size = (image_size,image_size), interpolation= transforms.InterpolationMode.BICUBIC),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5)
            ])
        else:
            self.transform = transforms.Compose([
                v2.Resize(size = (image_size,image_size), interpolation= transforms.InterpolationMode.BICUBIC),
            ])

        self.img_files = [filename for filename in os.listdir(self.IMGS_PATH+'/input/')]

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self,idx):
        source_path = self.img_files[idx]

        source = read_image(self.IMGS_PATH+'/input/'+source_path)
        target = read_image(self.IMGS_PATH+'/target/'+source_path)
        if self.resize:
            source,target = self.transform(source,target)
        # source = self.transform(Image.open(ORIGINAL_FOLDER+source_path))

        return source, target 

    
