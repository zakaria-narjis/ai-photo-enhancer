import torch
from torch import sigmoid
import numpy as np
import torch.nn.functional as F
try:
    from .dehaze.src import dehaze
except:
    from dehaze.src import dehaze

def numpy_sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_inverse(y):
    epsilon = 10**(-3)
    y = F.relu(y-epsilon)+epsilon
    y = 1-epsilon-F.relu((1-epsilon)-y)
    y = (1/y)-1
    output = -np.log(y.numpy())
    return torch.tensor(output)

class SigmoidInverse():

    def __init__(self):
        self.num_parameters = 0

    def __call__(self, images):
        return sigmoid_inverse(images)
    

new_sig_inv = SigmoidInverse()

class AdjustContrast():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["contrast"]

    def __call__(self, images:torch.Tensor, parameters:torch.Tensor):

        assert images.dim()==4
        assert images.shape[0]==parameters.shape[0]

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
class AdjustExposure():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["exposure"]

    def __call__(self, images, parameters):
        batch_size = parameters.shape[0]
        exposure = parameters.view(batch_size, 1, 1, 1)
        output = images+exposure*5
        return output

class AdjustTemp():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["temp"]

    def __call__(self, images, parameters):
        batch_size = parameters.shape[0]
        temp = parameters.view(batch_size, 1, 1, 1)
        editted = torch.clone(images)  

        index_high = (temp>0).view(-1)
        index_low = (temp<=0).view(-1)

        editted[index_high,:,:,1] += temp[index_high,:,:,0]*1.6
        editted[index_high,:,:,2] += temp[index_high,:,:,0]*2   
        editted[index_low,:,:,0] -= temp[index_low,:,:,0]*2.0
        editted[index_low,:,:,1] -= temp[index_low,:,:,0]*1.0          

        return editted
class AdjustShadows:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["shadows"]
    
    def __call__(self, list_hsv, parameters):
        batch_size = parameters.shape[0]
        shadows = parameters.view(batch_size, 1, 1).numpy()

        v = list_hsv[2].numpy()
        
        # Calculate shadows mask

        shadows_mask = 1 - numpy_sigmoid((v - 0.0) * 5.0)
        # Adjust v channel based on shadows mask
        adjusted_v = v * (1 + shadows_mask * shadows * 5.0)
        adjusted_v = torch.tensor(adjusted_v)

        return [list_hsv[0], list_hsv[1], adjusted_v]

class AdjustHighlights: # I should change the sigmoid to torch.sigmoid
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["highlights"]

    def custom_sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def __call__(self, list_hsv, parameters):
        batch_size = parameters.shape[0]
        highlights = parameters.view(batch_size, 1, 1).numpy()
   
        v = list_hsv[2].numpy()
        
        # Calculate highlights mask using custom sigmoid function
        highlights_mask = numpy_sigmoid((v - 1) * 5)
        
        # Adjust v channel based on highlights mask
        adjusted_v = 1 - (1 - v) * (1 - highlights_mask * highlights * 5)
        adjusted_v = torch.tensor(adjusted_v)
        
        return [list_hsv[0], list_hsv[1], adjusted_v]
    

class AdjustBlacks:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["blacks"]

    def __call__(self, list_hsv, parameters):
        batch_size = parameters.shape[0]
        blacks = parameters.view(batch_size, 1, 1)
        blacks = blacks + 1
        v = list_hsv[2]
        
        # Calculate the adjustment factor
        adjustment_factor = (torch.sqrt(blacks) - 1) * 0.2
        
        # Adjust the v channel
        adjusted_v = v + (1 - v) * adjustment_factor

        return [list_hsv[0], list_hsv[1], adjusted_v]

class AdjustWhites:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["whites"]

    def __call__(self, list_hsv, parameters):
        batch_size = parameters.shape[0]
        whites= parameters.view(batch_size, 1, 1)
        whites= whites=+ 1
        v = list_hsv[2]
        
        # Calculate the adjustment factor
        adjustment_factor = (torch.sqrt(whites) - 1) * 0.2
        
        # Adjust the v channel
        adjusted_v = v + v * adjustment_factor

        return [list_hsv[0], list_hsv[1], adjusted_v]