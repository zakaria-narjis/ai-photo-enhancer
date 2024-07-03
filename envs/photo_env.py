import gymnasium as gym
import logging
from .new_edit_photo import PhotoEditor
from .env_dataloader import create_dataloaders
import torch
from .features_extractor import ResnetEncoder
from typing import Sequence


PRE_ENCODE = True
IMSIZE = 64
THRESHOLD = -70
TEST_BATCH_SIZE = 500
TRAIN_BATCH_SIZE = 128

logging.basicConfig(
    filename='photo_enhancement.log',  # Name of the log file
    filemode='w',                      # Mode to write (w) and overwrite if it exists
    level=logging.DEBUG,               # Set level to DEBUG to capture all messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
    datefmt='%Y-%m-%d %H:%M:%S'        # Date format
)

image_encoder = ResnetEncoder()
train_dataloader,test_dataloader = create_dataloaders(TRAIN_BATCH_SIZE,TEST_BATCH_SIZE,image_size=IMSIZE,pre_encode=PRE_ENCODE)

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
        return torch.rand(batch_size,self._shape[1])

class PhotoEnhancementEnv(gym.Env):
    metadata = {
            'render.modes': ['human', 'rgb_array'],
        }
    def __init__(self,
                    batch_size=TRAIN_BATCH_SIZE,
                    logger=None,
                    imsize=IMSIZE,
                    training_mode=True,
                    done_threshold = THRESHOLD,
                    dataloader = train_dataloader,
                    pre_encode = PRE_ENCODE
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
            self.dataloader = dataloader

            self.photo_editor = PhotoEditor()
            self.num_parameters = self.photo_editor.num_parameters

            self.iter_dataloader_count = 0 #counts number of batch of samples seen by the agent
            self.iter_dataloader = iter(self.dataloader) #iterator over the dataloader 
            if self.pre_encode:
                self.observation_space= Observation_Space(
                        shape = self.dataloader.dataset.encoded_source.shape,
                        dtype = torch.float32)
                                                          
                self.action_space = Action_Space(
                    high = 1,
                    low = -1,
                    shape = (self.batch_size,self.num_parameters),
                    dtype = torch.float32
                )
            else:
                """
                    Not implemented yet
                """
                pass

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
            source_image,target_images,encoded_source,encoded_target = next(self.iter_dataloader) 
            self.target_images = target_images/255.0
            self.encoded_target = encoded_target
            self.sub_env_running = torch.Tensor([index for index in range(source_image.shape[0])]).to(torch.int32)
            self.state = { 
                'encoded_enhanced_image':encoded_source,              
                'encoded_source':encoded_source,          
                'source_image':source_image,   
            }
            self.iter_dataloader_count += 1
            encoded_source_image = self.state['encoded_source']
            encoded_enhanced_image = self.state['encoded_enhanced_image']   
            batch_observation = torch.cat((encoded_source_image,encoded_enhanced_image),dim=1)   

        else:
            source_image,target_images = next(self.iter_dataloader) 
            self.target_images = target_images
            batch_observation = {
                'enhanced_image':source_image,                     
                'source_image':source_image, 
            }

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
        psnr = (20 * torch.log10(1/ rmse))-100 
        
        rewards = psnr

        return rewards

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

        #update state
        self.state['source_image'] = torch.index_select(self.state['source_image'],0,self.sub_env_running)
        self.state['encoded_enhanced_image'] = torch.index_select(self.state['encoded_enhanced_image'],0,self.sub_env_running)
        self.state['encoded_source'] =torch.index_select(self.state['encoded_source'],0,self.sub_env_running)

        self.target_images = torch.index_select(self.target_images,0,self.sub_env_running)
        self.encoded_target = torch.index_select(self.encoded_target,0,self.sub_env_running)

        source_images = self.state['source_image']/255.0 # batch of images that have to be enhanced
        # actions =  torch.index_select(batch_actions,0,self.sub_env_running)

        enhanced_image = self.photo_editor(source_images.permute(0,2,3,1),batch_actions) #(B,H,W,3)
        enhanced_image = enhanced_image.permute(0,3,1,2) 
        encoded_enhanced_image = image_encoder.encode(enhanced_image).cpu()
        rewards = self.compute_rewards(enhanced_image,self.target_images)
        done = self.check_done(rewards,self.done_threshold)
        self.state['encoded_enhanced_image'] = encoded_enhanced_image

        running_sub_env_index = [not sub_env_state for sub_env_state in done]
        self.sub_env_running = self.sub_env_running[running_sub_env_index] # tensor of indicies of running sub_envs(images that didn't reach the threshold in self.check_done)
        
        info ={} #not used

        encoded_source_image = self.state['encoded_source']
        batch_observation = torch.cat((encoded_source_image,encoded_enhanced_image),dim=1)   

        # the whole episode should end when (done==True).all()
        return batch_observation, rewards, done


class PhotoEnhancementEnvTest(PhotoEnhancementEnv):

    def __init__(self,
                 batch_size=TEST_BATCH_SIZE,
                 logger=None,
                 dataloader = test_dataloader,
                 ):
        super(PhotoEnhancementEnvTest,self).__init__(batch_size = batch_size,
                                                 logger = logger,
                                                 dataloader = dataloader,
                                                 )
    
