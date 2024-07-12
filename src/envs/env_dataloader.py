from .image_dataset import FiveKDataset
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from .features_extractor import ResnetEncoder


ENCODING_BATCH_SIZE = 128

class PhotoEnhancement(Dataset):
    """
        Encode dataset images
        output : torch.Tensors of encoded and raw source/target images (3,H,W)
    """
    def __init__(self,image_size,mode = 'train', pre_encode = True,resize=True) -> None:
        super().__init__()
        self.img_dataset = FiveKDataset(mode=mode,image_size=image_size,resize=resize,augment_data=True)
        self.img_dataloader = DataLoader(self.img_dataset , batch_size=ENCODING_BATCH_SIZE, shuffle=False)
        self.pre_encode = pre_encode
        if self.pre_encode == True:
            image_encoder = ResnetEncoder()
        #Encoding imgs
            self.encoded_source = []
            self.encoded_target  = []
            print(f'Encoding {mode}ing data ...')
            for source,target in tqdm(self.img_dataloader, position=0, leave=True):
                self.encoded_source.append(image_encoder.encode(source/255.0).cpu())
                self.encoded_target.append(image_encoder.encode(target/255.0).cpu())
            print('finished...')   
            self.encoded_source = torch.cat(self.encoded_source)
            self.encoded_target = torch.cat(self.encoded_target)


    def __len__(self,):
        return len(self.img_dataset)

    def __getitem__(self, index):
        source_image,target_image = self.img_dataset[index]# raw images
        if self.pre_encode == True:

            encoded_source  =   self.encoded_source[index]
            encoded_target  =   self.encoded_target[index]
            return source_image,target_image,encoded_source,encoded_target
        
        else:
            return source_image,target_image
    

def create_dataloaders(batch_size,image_size,train=True,pre_encode= True,shuffle=True,resize=True):
    if train:    
        train_dataset = PhotoEnhancement(image_size=image_size,mode='train',pre_encode=pre_encode,resize=resize)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle)
    else: 
        test_dataset = PhotoEnhancement(image_size=image_size,mode='test', pre_encode = pre_encode,resize=resize)
        dataloader = DataLoader(test_dataset, batch_size=batch_size , shuffle = shuffle)

    return dataloader