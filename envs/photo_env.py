import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import os
from .new_edit_photo import PhotoEditor
from .env_dataloader import create_dataloaders
import torch
from .features_extractor import ResnetEncoder

# DATASET_DIR = "./dataset/"
# TARGET_DIR = "expertC/"
# ORIGINAL_DIR = "original/"
PRE_ENCODE = True
IMSIZE = 64
THRESHOLD = -0.01
TEST_BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 64


image_encoder = ResnetEncoder()
train_dataloader,test_dataloader = create_dataloaders(pre_encode=PRE_ENCODE)

class PhotoEnhancementEnv(gym.Env):
    metadata = {
            'render.modes': ['human', 'rgb_array'],
        }
    def __init__(self,
                    batch_size,
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
            self.train_dataloader = dataloader

            self.photo_editor = PhotoEditor()
            self.num_parameters = self.photo_editor.num_parameters

            self.iter_dataloader_count = 0 #counts number of batch of samples seen by the agent
            self.iter_dataloader = iter(self.train_dataloader) #iterator over the dataloader 

        #     self.action_space = spaces.Dict({
        #     'parameters':
        #     spaces.Box(low=-1.0, high=1.0,
        #                     shape=(self.batch_size, self.num_parameters), dtype=torch.float32),
        # })
            # if self.pre_encode == True:
            #     self.observation_space = spaces.Dict({
                     
            #     'enhanced_encoded_source':
            #     spaces.Box(low=-torch.inf,
            #             high=+torch.inf,
            #             shape=(-1, self.train.dataset.encoded_source.shape[1]),
            #             dtype=torch.float32),

            #     'encoded_image':spaces.Box(low=-torch.inf,
            #             high=+torch.inf,
            #             shape=(-1, self.train.dataset.encoded_source.shape[1]),
            #             dtype=torch.float32),
                
            #     'source_image':spaces.Box(low=0,
            #                 high=255,
            #                 shape=(-1, 3, self.imsize, self.imsize),
            #                 dtype=torch.uint8), 
            # }
            # )
                
            # else:

            #     self.observation_space = spaces.Dict({
            #     'image':
            #     spaces.Box(low=0,
            #             high=255,
            #             shape=(self.batch_size, 3, self.imsize, self.imsize),
            #             dtype=torch.uint8),

            #     'enhanced_image':spaces.Box(low=0,
            #             high=255,
            #             shape=(self.batch_size, 3, self.imsize, self.imsize),
            #             dtype=torch.uint8)
            # }
            # )
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
        self.iter_dataloader = iter(self.train_dataloader)

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
            observation = { 
                'enhanced_encoded_image':encoded_source,              
                'encoded_source':encoded_source,          
                'source_image':source_image,   
            }
            self.iter_dataloader_count += 1

        else:
            source_image,target_images = next(self.iter_dataloader) 
            self.target_images = target_images
            observation = {
                'enhanced_image':source_image,                     
                'source_image':source_image, 
            }

        self.state = observation

        return observation
    

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
        
        rewards = -rmse

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
            observation: Batch observation tensor with shape(B,3,H,W)
            reward : Batch reward tensor with shape (B,)
            done: list of Bool True if the agent reached acceptable performance False not yet
            info : dict information 
        """
        source_images = self.state['source_image']/255.0 # batch of images that have to be enhanced
        actions =  torch.index_select(batch_actions,0,self.sub_env_running)

        enhanced_image = self.photo_editor(source_images.permute(0,2,3,1),actions) #(B,H,W,3)
        enhanced_image = enhanced_image.permute(0,3,1,2) 
        encoded_enhanced_image = image_encoder.encode(enhanced_image).cpu()
        rewards = self.compute_rewards(enhanced_image,self.target_images)
        done  = self.check_done(rewards,self.done_threshold)

        running_sub_env_index = [not sub_env_state for sub_env_state in done]
        self.sub_env_running = self.sub_env_running[running_sub_env_index] # tensor of indicies of running sub_envs(images that didn't reach the threshold in self.check_done)
        

        #update state
        self.state['source_image'] = torch.index_select(source_images,0,self.sub_env_running)
        self.state['encoded_enhanced_image'] = torch.index_select(encoded_enhanced_image,0,self.sub_env_running)
        self.state['encoded_source'] =torch.index_select(encoded_enhanced_image,0,self.sub_env_running)

        self.target_images = torch.index_select(self.target_images,0,self.sub_env_running)
        self.encoded_target = torch.index_select(self.encoded_target,0,self.sub_env_running)

        info ={} #not used
        next_state = self.state
        # the whole episode should end when (done==True).all()
        return next_state, rewards, done


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
    
