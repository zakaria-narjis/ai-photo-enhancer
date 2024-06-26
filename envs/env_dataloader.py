from .image_dataset import FiveKDataset
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
from .features_extractor import ResnetEncoder

BATCH_SIZE = 64
ENCODING_BATCH_SIZE = 128
IMG_SIZE = 64 #training image size

# default_aug = transforms.Compose([
#             transforms.Resize(size = (IMG_SIZE,IMG_SIZE) , interpolation=transforms.InterpolationMode.BICUBIC),
#         ])


class PhotoEnhancement(Dataset):
    """
        Encode dataset images
        output : torch.Tensors of encoded and raw source/target images (3,H,W)
    """
    def __init__(self,mode = 'train', pre_encode = True) -> None:
        super().__init__()
        self.img_dataset = FiveKDataset(mode=mode)
        self.img_dataloader = DataLoader(self.img_dataset , batch_size=ENCODING_BATCH_SIZE, shuffle=False)
        self.encoder = ResnetEncoder()
        self.pre_encode = pre_encode
        if self.pre_encode == True:
        #Encoding imgs
            self.encoded_source = []
            self.encoded_target  = []
            print(f'Encoding {mode}ing data ...')
            for source,target in tqdm(self.img_dataloader):
                self.encoded_source.append(self.encoder.encode(source).cpu())
                self.encoded_target.append(self.encoder.encode(target).cpu())
            print('finished...')   
            self.encoded_source = torch.cat(self.encoded_source)
            self.encoded_target = torch.cat(self.encoded_target)


    def __len__(self,):
        return self.encoded_source.shape[0]

    def __getitem__(self, index):
        source_image,target_image = self.img_dataset[index]# raw images
        if self.pre_encode == True:

            encoded_source  =   self.encoded_source[index]
            encoded_target  =   self.encoded_target[index]
            return source_image,target_image,encoded_source,encoded_target
        
        else:
            return source_image,target_image
    

def create_dataloaders(pre_encode,batch_size=BATCH_SIZE,shuffle=True):
    test_dataset = PhotoEnhancement(mode='test', pre_encode = pre_encode)
    train_dataset = PhotoEnhancement(mode='train',pre_encode=pre_encode) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle = shuffle)

    return train_dataloader,test_dataloader