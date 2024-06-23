import torch
from torch import sigmoid
import numpy as np
import torch.nn.functional as F
try:
    from .dehaze.src import dehaze
except:
    from dehaze.src import dehaze

def sigmoid_inverse(y):
    epsilon = 10**(-3)
    y = F.relu(y-epsilon)+epsilon
    y = 1-epsilon-F.relu((1-epsilon)-y)
    y = (1/y)-1
    output = -np.log(y.numpy())
    return torch.tensor(output)


class AdjustContrast():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["contrast"]

    def __call__(self, images:torch.Tensor, parameters:torch.Tensor):
        batch_size = parameters.shape[0]
        mean = images.view(batch_size,-1).mean(1)
        mean = mean.view(batch_size, 1, 1, 1)
        parameters = parameters.view(batch_size, 1, 1, 1)
        editted = (images-mean)*(parameters+1)+mean
        editted = F.relu(editted)
        editted = 1-F.relu(1-editted)
        return editted
    

class AdjustDehaze():

    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["dehaze"]

    def __call__(self, images, parameters):
        """
        Takes a batch of images where B (the last dim) is the batch size
        args:
            images: torch.Tensor # B H W C 
            parameters :torch.Tensor # N
        return:
            output: torch.Tensor #  B H W C 
        """
        assert images.dim()==4
        batch_size = parameters.shape[0]
        output = []
        for image_index in range(batch_size):
            image = images[image_index].numpy()
            scale = max((image.shape[:2])) / 512.0
            omega = float(parameters[image_index])
            editted= dehaze.DarkPriorChannelDehaze(
                wsize=int(15*scale), radius=int(80*scale), omega=omega,
                t_min=0.25, refine=True)(image * 255.0) / 255.0
            editted = F.relu(editted)
            editted= 1-F.relu(1-editted)
            output.append(torch.tensor(editted))
        output = torch.stack(output)
        return output
    
    class AdjustClarity():
        def __init__(self):
            self.num_parameters = 1
            self.window_names = ["parameter"]
            self.slider_names = ["clarity"]

        def __call__(self, images, parameters):
            """
            Takes a batch of images where B (the last dim) is the batch size
            args:
                images: torch.Tensor # B H W C 
                parameters :torch.Tensor # N
            return:
                output: torch.Tensor #  B H W C 
            """
            assert images.dim()==4
            batch_size = parameters.shape[0]
            output = [] 
            clarity = parameters.view(batch_size, 1, 1, 1)
            for image in images: 
                input = image.numpy()      
                scale = max((input.shape[:2])) / 512.0
                unsharped = cv2.bilateralFilter((input*255.0).astype(np.uint8),
                                                    int(32*scale), 50, 10*scale)/255.0
                output.append(torch.tensor(unsharped))
            output = torch.stack(output) 
            editted_images = images + (images-output) * clarity
            
            return editted_images