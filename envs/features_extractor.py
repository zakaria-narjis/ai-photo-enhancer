import torch
from torchvision import transforms
from torchvision import models

DEVICE = 'cuda'

class Extractor:
    """
        Features extractor boilerplate
    """
    
    
    def __init__(self,):
        self.name=''
        self.input_shape=None
        self.output_shape=None
        self.model = None
        self.preprocess = None
        self.device = None

class ResnetEncoder(Extractor):
    def __init__(self,):
        super().__init__()
        self.device = DEVICE 
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1])).to(self.device) #remove classifier
        self.model.eval()
        self.preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
        

        
    def encode(self,images:torch.Tensor)->torch.Tensor:

        assert images.dim()==4
        assert images.shape[1]==3
        with torch.inference_mode():
            output = images.clone().to(self.device)/255.0
            output = self.preprocess(output)
            output = self.model(output)
            output = torch.flatten(output,start_dim=-3,end_dim=-1)
        return output
