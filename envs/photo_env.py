import gymnasium as gym
import logging
from .new_edit_photo import PhotoEditor
from .env_dataloader import create_dataloaders
import torch
from .features_extractor import ResnetEncoder
from typing import Sequence
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  

# PRE_ENCODE = False
# IMSIZE = 64
# PSNR_REWARD = True
# if PSNR_REWARD:
#     THRESHOLD = -25
# else:
#     THRESHOLD = -0.01

# TEST_BATCH_SIZE = 500
# TRAIN_BATCH_SIZE = 32
# FEATURES_SIZE = 512
# [Srgb2Photopro(), AdjustDehaze(), AdjustClarity(), AdjustContrast(),
#                 SigmoidInverse(), AdjustExposure(), AdjustTemp(), AdjustTint(),
#                 Sigmoid(), Bgr2Hsv(), AdjustWhites(), AdjustBlacks(), AdjustHighlights(),
#                 AdjustShadows(), AdjustVibrance(), AdjustSaturation(), Hsv2Bgr(), Photopro2Srgb()]

# SLIDERS_TO_USE = ["contrast","exposure","shadows","highlights","whites"]

logging.basicConfig(
    filename='photo_enhancement.log',  # Name of the log file
    filemode='w',                      # Mode to write (w) and overwrite if it exists
    level=logging.DEBUG,               # Set level to DEBUG to capture all messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
    datefmt='%Y-%m-%d %H:%M:%S'        # Date format
)
logging.disable(logging.CRITICAL)


def sample_near_values_batch(tensor, batch_size, std_dev=0.05, clip_min=0.0, clip_max=1.0):
    """
    Generate a batch of sampled values near the given tensor.
    
    Args:
    tensor (torch.Tensor): The input tensor to sample near.
    batch_size (int): The number of samples to generate.
    std_dev (float): Standard deviation for the normal distribution.
    clip_min (float): Minimum value to clip the result.
    clip_max (float): Maximum value to clip the result.
    
    Returns:
    torch.Tensor: A batch of tensors with sampled values.
    """
    # Expand the input tensor to the desired batch size
    batched_tensor = tensor.unsqueeze(0).expand(batch_size, -1)
    
    # Create a noise tensor with the same shape as the batched tensor
    noise = torch.randn_like(batched_tensor) * std_dev
    
    # Add the noise to the batched tensor
    sampled = batched_tensor + noise
    
    # Clip the values to ensure they're within the specified range
    sampled = torch.clamp(sampled, clip_min, clip_max)
    
    return sampled
class Observation_Space:
    
    def __init__(self,
                 shape:Sequence[int] | None = None,
                 dtype: torch.dtype | None = None,
                 ):

        self._shape = None if shape is None else tuple(shape)
        self.dtype = dtype

    def shape(self,):
        return self._shape
    

class Action_Space:

    def __init__(self,
                 high:float,
                 low:float,
                 shape:Sequence[int] | None = None,
                 dtype: torch.dtype | None = None,
                 ) -> None:
        self._shape = None if shape is None else tuple(shape)
        self.dtype = dtype
        self.high = high
        self.low = low

    def shape(self,):
        return self._shape
    
    def sample(self,batch_size):
        original_tensor = torch.tensor([0.125, 0.125, 0.375, 0.125, 0., 0.0625, 0.9375, 0.375, 0.0625, 0., 0.125, 0.125])
        # return sample_near_values_batch(original_tensor, batch_size) torch.rand()
        return torch.rand(batch_size,self._shape[1])
class PhotoEnhancementEnv(gym.Env):
    metadata = {
            'render.modes': ['human', 'rgb_array'],
        }
    def __init__(self,
                    batch_size,
                    imsize,
                    training_mode,
                    done_threshold ,
                    pre_encode ,
                    edit_sliders ,
                    features_size ,
                    logger=None
                    ):
            super().__init__()
            """
                Args:
                    batch_size(int): number of sub environements (batch of images) to be enhanced
                    logger(logger) : logger used
                    imsize(int) : resized image size used for training and
                    training_mode(bool): train mode
                    done_threshold (bool): minimum threshold to considered an image enhanced 
                    dataloader (torch.utils.data.Dataloder): images data loader
                    pre_encode (bool): whether to encode the images or not
            """
            self.logger = logger or logging.getLogger(__name__)
            self.imsize = imsize
            self.batch_size = batch_size
            self.training_mode = training_mode
            self.pre_encode = pre_encode
            self.dataloader = create_dataloaders(batch_size,train=training_mode,image_size=imsize,pre_encode=pre_encode)
            self.edit_sliders = edit_sliders 
            self.photo_editor = PhotoEditor(sliders=edit_sliders)
            self.num_parameters = self.photo_editor.num_parameters
            self.features_size = features_size
            self.iter_dataloader_count = 0 #counts number of batch of samples seen by the agent
            self.iter_dataloader = iter(self.dataloader) #iterator over the dataloader
            if self.pre_encode:
                self.observation_space= Observation_Space(
                        shape = self.dataloader.dataset.encoded_source.shape,
                        dtype = torch.float32)
                                                          
                self.action_space = Action_Space(
                    high = 1,
                    low = -1,
                    shape = (self.batch_size, self.num_parameters),
                    dtype = torch.float32
                )
            else:
                self.observation_space= Observation_Space(
                        shape = (self.batch_size, self.features_size),
                        dtype = torch.float32)
                                                          
                self.action_space = Action_Space(
                    high = 1,
                    low = -1,
                    shape = (self.batch_size, self.num_parameters),
                    dtype = torch.float32
                )
            if self.pre_encode :
                self.image_encoder = ResnetEncoder()
                

            self.done_threshold = done_threshold 
            self.target_images = None # Batch of images (B,3,H,W) of target images (ground_truth)
            self.encoded_target = None 
            self.state = None #Batch of images (B,3,H,W) that correspond to the agent state each image can be seen as a sub state in a sub env   
            # self.action = None # Batch of actions  (B,N_params)
            # self.done = None # Batch of Bool that state wether the the sub env (images) reached the best enhacement
            self.sub_env_running = None
       

    def reset_data_iterator(self,):
        """
            Reset dataloader when the agent went through the whole samples
        """
        self.logger.debug('reset dataloader')
        self.iter_dataloader = iter(self.dataloader)

    def reset (self):
        self.logger.debug('reset the episode')
        if self.iter_dataloader_count == len(self.iter_dataloader):
            self.reset_data_iterator()
            self.iter_dataloader_count = 0

        if self.pre_encode:    
            source_image,target_image,encoded_source,encoded_target = next(self.iter_dataloader) 
            # self.target_images = target_images/255.0
            # self.encoded_target = encoded_target
            self.sub_env_running = torch.Tensor([index for index in range(source_image.shape[0])]).to(torch.int32)
            self.state = { 
                'encoded_enhanced_image':encoded_source,              
                'encoded_source':encoded_source,  
                'enanced_image': source_image/255.0,      
                'source_image':source_image/255.0,   
                'target_image':target_image/255.0,
            }
            self.iter_dataloader_count += 1
            encoded_source_images = self.state['encoded_source']
            encoded_enhanced_images = self.state['encoded_enhanced_image']   
            # batch_observation = torch.cat((encoded_source_images,encoded_enhanced_images),dim=1)   
            batch_observation =  encoded_source_images

        else:
            source_image,target_image = next(self.iter_dataloader) 
            self.iter_dataloader_count += 1
        #  self.sub_env_running = torch.Tensor([index for index in range(source_image.shape[0])]).to(torch.int32)
            self.state = {
                'enhanced_image':source_image/255.0,                     
                'source_image':source_image/255.0, 
                'target_image':target_image/255.0,
            }
            batch_observation= self.state['source_image']
        return batch_observation
    

    def compute_rewards(self,enhanced_image,target_image):
        """
            args:
                enhanced_image: (Next_State) batch of enhanced images using parameters generated by the agent
                target: batch of target images
        """
        enhanced =torch.flatten(enhanced_image.clone(),start_dim=1, end_dim=-1)
        target = torch.flatten(target_image.clone(),start_dim=1, end_dim=-1)

        rmse = enhanced-target
        rmse = torch.pow(rmse,2).mean(1)
        rmse = torch.sqrt(rmse)
        if (rmse==0).all():
            return torch.zeros(enhanced[0])
        else:
            psnr = ((20 * torch.log10(1/ rmse))-50)        
            rewards = psnr
            return rewards
        # rewards = -rmse
        # return rewards

    def check_done(self,rewards:torch.Tensor,threshold:float):
        """
        Function that check if the enhanced image reached a certain minium threshold of enhancement. Mainly used for marking the end of enhancement of the image
            args: 
                rewards: tensor of batch rewards
                threshold: minimum threshold of enhaancement should be a value<0 for the case of rmse
            return:
                tensor of bool 
        """
        return (rewards>threshold)
    

    def step(self,batch_actions:torch.Tensor):
        """
        args:
            batch_action: torch.Tensor with shape (B,N_params) where N_paramas is the number of photo enhancing paramters (N_params = self.num_parameters)

        return:
            next_state(dict): {
                source_image(Tensor): Batch of images with shape(B,3,H,W),
                encoded_enhanced_image : Batch of encoded enhanced images (B,features_size)
                encoded_source : Batch of encoded source images (B,features_size)
            reward : Batch reward tensor with shape (B,)
            done: list of Bool True if the agent reached acceptable performance False not yet
            info : dict information 
        """
        if self.pre_encode:
            #update state
            self.state['source_image'] = torch.index_select(self.state['source_image'],0,self.sub_env_running)
            # encoded_enhanced_image = torch.index_select(self.state['encoded_enhanced_image'],0,self.sub_env_running)
            self.state['encoded_source']  =torch.index_select(self.state['encoded_source'],0,self.sub_env_running)

            self.state['target_image'] = torch.index_select(self.state['target_image'],0,self.sub_env_running)
            # self.encoded_target = torch.index_select(self.encoded_target,0,self.sub_env_running)
            self.sub_env_running = torch.Tensor([index for index in range(self.sub_env_running.shape[0])]).to(torch.int32)
            source_images = self.state['source_image']# batch of images that have to be enhanced
            target_images = self.state['target_image']
            encoded_source_images = self.state['encoded_source']
            # actions =  torch.index_select(batch_actions,0,self.sub_env_running)

            enhanced_image = self.photo_editor(source_images.permute(0,2,3,1),batch_actions) #(B,H,W,3)
            enhanced_image = enhanced_image.permute(0,3,1,2) 
            encoded_enhanced_images = self.image_encoder.encode(enhanced_image).cpu()
            rewards = self.compute_rewards(enhanced_image,target_images)

            done = self.check_done(rewards,self.done_threshold)

            # self.state['encoded_enhanced_image'] = encoded_enhanced_images
            rewards[done]+=1 

            self.state['enhanced_image'] = enhanced_image
            running_sub_env_index = [not sub_env_state for sub_env_state in done]
            self.sub_env_running = self.sub_env_running[running_sub_env_index] # tensor of indicies of running sub_envs(images that didn't reach the threshold in self.check_done)
            
            info = {} #not used

            # encoded_source_image = self.state['encoded_source']
            # batch_observation = torch.cat((encoded_source_images,encoded_enhanced_images),dim=1)   
            batch_observation =  encoded_enhanced_images
            # the whole episode should end when (done==True).all()
        else:
            source_images = self.state['source_image']
            target_images = self.state['target_image']

            enhanced_image = self.photo_editor(source_images.permute(0,2,3,1),batch_actions)
            enhanced_image = enhanced_image.permute(0,3,1,2) 

            rewards = self.compute_rewards(enhanced_image,target_images)
            done = self.check_done(rewards,self.done_threshold)      
            rewards[done]+=10
            self.state['enhanced_image'] = enhanced_image
            batch_observation =  enhanced_image
        return batch_observation, rewards, done


class PhotoEnhancementEnvTest(PhotoEnhancementEnv):
    def __init__(self, batch_size, imsize, done_threshold, pre_encode, edit_sliders, features_size, training_mode=False, logger=None):
        super(PhotoEnhancementEnvTest, self).__init__(
            batch_size=batch_size,
            imsize=imsize,
            training_mode=training_mode,
            done_threshold=done_threshold,
            pre_encode=pre_encode,
            edit_sliders=edit_sliders,
            features_size=features_size,
            logger=logger
        )