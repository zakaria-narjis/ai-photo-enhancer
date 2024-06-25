from dataset import FiveKDataset
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
BATCH_SIZE = 64
ENCODING_BATCH_SIZE = 128
IMG_SIZE = 64 #training image size

default_aug = transforms.Compose([
            transforms.Resize(size = IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        ])


class PhotoEnhancement(Dataset):
    """
        Encode dataset images
        output : torch.Tensors of encoded images and source images (3,H,W)
    """
    def __init__(self,mode = 'train',transform =default_aug) -> None:
        super().__init__()
        self.img_dataset = FiveKDataset(mode=mode)
        self.img_dataloader = DataLoader(self.img_dataset , batch_size=ENCODING_BATCH_SIZE, shuffle=False)
        self.transform = transform
        #Encoding imgs
        self.encoded_source = []
        self.encoded_target  = []
        for source,target in self.img_dataloader:
            self.encoded_source.append(self.transform(source))
            self.encoded_target.append(self.transform(target))
        self.encoded_source = torch.cat(self.encoded_source)
        self.encoded_target = torch.cat(self.encoded_target)


    def __len__(self,):
        return self.encoded_source.shape[0]

    def __getitem__(self, index):
        encoded_source  =   self.encoded_source[index]
        encoded_target  =   self.encoded_target[index]
        source_image,target_image = self.img_dataset[index]

        return source_image,target_image,encoded_source,encoded_target
    

def create_dataloaders():
    train_dataset = PhotoEnhancement(mode='train')
    test_dataset = PhotoEnhancement(mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = True)