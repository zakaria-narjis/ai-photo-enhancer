from .image_dataset import FiveKDataset
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from .features_extractor import ResnetEncoder


class PhotoEnhancement:
    """
        Encode dataset images
        output : torch.Tensors of encoded and raw source/target images (3,H,W)
    """
    def __init__(self,image_size,
                 mode = 'train', 
                 resize=True,
                 augment_data=False,
                 use_txt_features=False) -> None:
        return FiveKDataset(image_size, mode=mode, resize=resize, 
                 augment_data=augment_data, 
                 use_txt_features=use_txt_features,
                 device='cuda')
        
    

def create_dataloaders(batch_size,image_size,use_txt_features=False,
                       train=True,augment_data=False,shuffle=True,resize=True,pre_encoding_device='cuda'):
    if train:    
        train_dataset = PhotoEnhancement(image_size, mode='train', resize=resize, 
                 augment_data=augment_data, 
                 use_txt_features=use_txt_features,
                 device=pre_encoding_device)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle)
    else: 
        test_dataset = PhotoEnhancement(image_size, mode='test', resize=resize, 
                 augment_data=augment_data, 
                 use_txt_features=use_txt_features,
                 device=pre_encoding_device)
        dataloader = DataLoader(test_dataset, batch_size=batch_size , shuffle = shuffle)

    return dataloader