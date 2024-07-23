import gymnasium as gym
import logging
from .new_edit_photo import PhotoEditor
from .env_dataloader import create_dataloaders
import torch
from typing import Sequence
from tensordict import TensorDict
import os
import yaml
from pathlib import Path
from sac.sac_inference import InferenceAgent
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
    
class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

class PhotoEnhancementEnv(gym.Env):
    """
        Custom gym environment for photo enhancement task
    """
    def __init__(self,
                    batch_size,
                    imsize,
                    training_mode,
                    done_threshold ,
                    edit_sliders ,
                    features_size ,
                    discretize,
                    discretize_step,
                    use_txt_features="embedded",
                    augment_data=False,
                    pre_encoding_device='cuda:0',
                    pre_load_images=True, 
                    preprocessor_agent_path=None,            
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
                    use_txt_features (bool): whether to encode the images or not
            """
            self.logger = logger or logging.getLogger(__name__)
            self.imsize = imsize
            self.batch_size = batch_size
            self.training_mode = training_mode
            self.use_txt_features = use_txt_features
            self.preprocessor_agent_path = preprocessor_agent_path
            self.pre_encoding_device = pre_encoding_device

            self.dataloader = create_dataloaders(
                batch_size=batch_size,image_size=imsize,use_txt_features=use_txt_features,
                train=training_mode,augment_data=augment_data,shuffle=True,
                resize=True,pre_encoding_device=pre_encoding_device,pre_load_images=pre_load_images)
            
            self.edit_sliders = edit_sliders 
            self.photo_editor = PhotoEditor(sliders=edit_sliders)
            self.num_parameters = self.photo_editor.num_parameters
            self.features_size = features_size
            self.iter_dataloader_count = 0 #counts number of batch of samples seen by the agent
            self.iter_dataloader = iter(self.dataloader) #iterator over the dataloader
            if self.use_txt_features:
                self.observation_space= Observation_Space(
                        shape = (self.batch_size, self.features_size*3),
                        dtype = torch.float32)
                                                          
                self.action_space = Action_Space(
                    high = 1,
                    low = -1,
                    shape = (self.batch_size, self.num_parameters),
                    dtype = torch.float32
                )
            if self.use_txt_features=="one_hot":
                self.observation_space= Observation_Space(
                        shape = (self.batch_size, self.features_size+16),
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
            self.discretize=discretize
            self.discretize_step = discretize_step    

            self.done_threshold = done_threshold 
            self.target_images = None # Batch of images (B,3,H,W) of target images (ground_truth)
            self.encoded_target = None 
            self.state = None #Batch of images (B,3,H,W) that correspond to the agent state each image can be seen as a sub state in a sub env   
            self.sub_env_running = None

            if self.preprocessor_agent_path!=None:
                self.load_preprocessor_agent()

    def load_preprocessor_agent(self,):
        current_dir = Path(__file__).parent.absolute()
        with open(os.path.join(self.preprocessor_agent_path,"configs/sac_config.yaml")) as f:
            sac_config_dict =yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(self.preprocessor_agent_path,"configs/env_config.yaml")) as f:
            env_config_dict =yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(current_dir,"../configs/inference_config.yaml")) as f:
            inf_config_dict =yaml.load(f, Loader=yaml.FullLoader)    
        inference_config = Config(inf_config_dict)
        sac_config = Config(sac_config_dict)
        env_config = Config(env_config_dict)              
        inference_env = PhotoEnhancementEnvTest(
                            batch_size=env_config.train_batch_size,
                            imsize=env_config.imsize,
                            training_mode=False,
                            done_threshold=env_config.threshold_psnr,
                            edit_sliders=env_config.sliders_to_use,
                            features_size=env_config.features_size,
                            discretize=env_config.discretize,
                            discretize_step= env_config.discretize_step,
                            use_txt_features=env_config.use_txt_features,
                            augment_data=env_config.augment_data,
                            pre_encoding_device=self.pre_encoding_device,
                            pre_load_images=False,   
                            logger=None)# useless just to get the action space size for the Networks and whether to use txt features or not
        self.photo_editor = PhotoEditor(env_config.sliders_to_use)
        inference_config.device = self.pre_encoding_device
        self.preprocessor_agent = InferenceAgent(inference_env, inference_config)
        self.preprocessor_agent.device = self.pre_encoding_device
        os.path.join(self.preprocessor_agent_path,'models','backbone.pth')
        self.preprocessor_agent.load_backbone(os.path.join(self.preprocessor_agent_path,'models','backbone.pth'))
        self.preprocessor_agent.load_actor_weights(os.path.join(self.preprocessor_agent_path,'models','actor_head.pth'))
        self.preprocessor_agent.load_critics_weights(os.path.join(self.preprocessor_agent_path,'models','qf1_head.pth'),
                                    os.path.join(self.preprocessor_agent_path,'models','qf2_head.pth'))
        
    def compute_preprocessor_threshold(self,improvement_threshold=5):
        with torch.no_grad():
            pre_batch_actions = self.preprocessor_agent.act(self.state['source_image'],deterministic=False,n_samples=0) #sampled actions
            pre_enhanced_image = self.photo_editor(self.state['source_image'].permute(0,2,3,1),pre_batch_actions)
            pre_enhanced_image = pre_enhanced_image.permute(0,3,1,2)
            pre_rewards = self.compute_rewards(pre_enhanced_image,self.state['target_image'])
            self.state['source_image'] = pre_enhanced_image
            self.state['enhanced_image'] = pre_enhanced_image
            done_threshold = pre_rewards+improvement_threshold
        return done_threshold

    def reset_data_iterator(self,):
        """
            Reset dataloader when the agent went through the whole samples
        """
        # self.logger.debug('reset dataloader')
        self.iter_dataloader = iter(self.dataloader)

    def reset (self):
        # self.logger.debug('reset the episode')
        if self.iter_dataloader_count == len(self.iter_dataloader):
            self.reset_data_iterator()
            self.iter_dataloader_count = 0

        if self.use_txt_features=="embedded":    
            source_image, txt_semantic_features, img_semantic_features, target_image= next(self.iter_dataloader) 
            self.state = {
                'source_image':source_image/255.0,  
                'enhanced_image':source_image/255.0,                     
                'ts_features':txt_semantic_features,
                'ims_features':img_semantic_features,
                'target_image':target_image/255.0,
            }
            self.iter_dataloader_count += 1
            if self.preprocessor_agent_path!=None:
                self.done_threshold = self.compute_preprocessor_threshold()
            batch_observation= TensorDict(
                        {
                            "batch_images":self.state['source_image'],
                            "ts_features":self.state['ts_features'],
                            "ims_features":self.state['ims_features'],
                        },
                        batch_size = [self.state['source_image'].shape[0]],
                    )
        elif self.use_txt_features=="one_hot":
            source_image, txt_features, target_image= next(self.iter_dataloader) 
            self.state = {
                'source_image':source_image/255.0,  
                'enhanced_image':source_image/255.0,                     
                'ts_features':txt_features,
                'target_image':target_image/255.0,
            }
            self.iter_dataloader_count += 1
            if self.preprocessor_agent_path!=None:
                self.done_threshold = self.compute_preprocessor_threshold()
            batch_observation= TensorDict(
                        {
                            "batch_images":self.state['source_image'],
                            "ts_features":self.state['ts_features'],
                        },
                        batch_size = [self.state['source_image'].shape[0]],
                    )
        else:
            source_image,target_image = next(self.iter_dataloader) 
            self.iter_dataloader_count += 1
            self.state = {
                'enhanced_image':source_image/255.0,                     
                'source_image':source_image/255.0, 
                'target_image':target_image/255.0,
            }
            if self.preprocessor_agent_path!=None:
                self.done_threshold = self.compute_preprocessor_threshold()
            batch_observation = TensorDict(
                        {
                            "batch_images":self.state['source_image'],
                        },
                        batch_size = [self.state['source_image'].shape[0]],
                    )
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
        
        if self.discretize :
            batch_actions = torch.round((batch_actions+1)/self.discretize_step)*self.discretize_step-1
         
        if self.use_txt_features=="embedded":
            source_images = self.state['source_image']
            target_images = self.state['target_image']
            ts_features = self.state['ts_features']
            ims_features = self.state['ims_features']

            enhanced_image = self.photo_editor(source_images.permute(0,2,3,1),batch_actions)
            enhanced_image = enhanced_image.permute(0,3,1,2) 
            rewards = self.compute_rewards(enhanced_image,target_images)
            done = self.check_done(rewards,self.done_threshold)      
            rewards[done]+=10
            self.state['enhanced_image'] = enhanced_image
            batch_observation= TensorDict(
                        {
                            "batch_images":enhanced_image,
                            "ts_features":ts_features,
                            "ims_features":ims_features,
                        },
                        batch_size = [enhanced_image.shape[0]],
                    )
        elif self.use_txt_features=="one_hot":
            source_images = self.state['source_image']
            target_images = self.state['target_image']
            ts_features = self.state['ts_features']
            enhanced_image = self.photo_editor(source_images.permute(0,2,3,1),batch_actions)
            enhanced_image = enhanced_image.permute(0,3,1,2) 
            rewards = self.compute_rewards(enhanced_image,target_images)
            done = self.check_done(rewards,self.done_threshold)      
            rewards[done]+=10
            self.state['enhanced_image'] = enhanced_image
            batch_observation= TensorDict(
                        {
                            "batch_images":enhanced_image,
                            "ts_features":ts_features,
                        },
                        batch_size = [enhanced_image.shape[0]],)
        else:
            source_images = self.state['source_image']
            target_images = self.state['target_image']

            enhanced_image = self.photo_editor(source_images.permute(0,2,3,1),batch_actions)
            enhanced_image = enhanced_image.permute(0,3,1,2) 

            rewards = self.compute_rewards(enhanced_image,target_images)
            done = self.check_done(rewards,self.done_threshold)      
            rewards[done]+=10
            self.state['enhanced_image'] = enhanced_image
            batch_observation= TensorDict(
                        {
                            "batch_images":enhanced_image,
                        },
                        batch_size = [enhanced_image.shape[0]],
                    )
        return batch_observation, rewards, done


class PhotoEnhancementEnvTest(PhotoEnhancementEnv):
    def __init__(self, batch_size, imsize, done_threshold, edit_sliders, features_size, 
    discretize, discretize_step, use_txt_features=False, augment_data=False, 
    pre_encoding_device='cuda:0', training_mode=False,pre_load_images=True, preprocessor_agent_path=None, logger=None):
        super(PhotoEnhancementEnvTest, self).__init__(
            batch_size=batch_size,
            imsize=imsize,
            training_mode=training_mode,
            done_threshold=done_threshold,
            edit_sliders=edit_sliders,
            features_size=features_size,
            discretize=discretize,
            discretize_step=discretize_step,
            use_txt_features=use_txt_features,
            augment_data=augment_data,
            pre_encoding_device=pre_encoding_device,
            pre_load_images=pre_load_images,
            preprocessor_agent_path=preprocessor_agent_path,
            logger=logger
        )