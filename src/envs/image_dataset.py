
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.v2.functional as F
import random
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
        self.image_size =image_size
        self.augment_data = augment_data
        self.img_files = [filename for filename in os.listdir(self.IMGS_PATH+'/input/')]

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self,idx):
        source_path = self.img_files[idx]

        source = read_image(self.IMGS_PATH+'/input/'+source_path)
        target = read_image(self.IMGS_PATH+'/target/'+source_path)
        if self.resize:
            source =  F.resize(source,(self.image_size,self.image_size), interpolation= F.InterpolationMode.BICUBIC)
            target =  F.resize(target,(self.image_size,self.image_size), interpolation= F.InterpolationMode.BICUBIC)
            
        if self.augment_data:
            if random.random() > 0.5:
                source = F.hflip(source)
                target = F.hflip(target)
            if random.random() > 0.5:
                source = F.vflip(source)
                target = F.vflip(target)
            
        # source = self.transform(Image.open(ORIGINAL_FOLDER+source_path))

        return source, target

    
