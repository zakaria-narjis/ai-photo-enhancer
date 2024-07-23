import yaml
import time
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from envs.photo_env import PhotoEnhancementEnv
from envs.photo_env import PhotoEnhancementEnvTest
from sac.sac_algorithm import SAC
import multiprocessing as mp
import argparse
import logging
from sac.utils import *
from tqdm.auto import tqdm

from datetime import datetime
import os
from pathlib import Path
import re 


def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name)

def getdatetime():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def make_dirs_and_open(file_path, mode):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return open(file_path, mode)


def main():
    current_dir = Path(__file__).parent.absolute()
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_tag', help='experiment tag')
    parser.add_argument('sac_config', help='YAML sac config file')
    parser.add_argument('env_config', help='YAML env config file')
    parser.add_argument('outdir', nargs='?', type=str, help='directory to put experiment results',default=os.path.join(current_dir.parent, 'experiments/runs'))
    parser.add_argument('save_model', nargs='?',type=bool, default=True)
    parser.add_argument('--logger_level', type=int, default=logging.INFO)

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    
    # Configure logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.logger_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(args.logger_level)

    with open(args.sac_config) as f:
        config_dict =yaml.load(f, Loader=yaml.FullLoader)

    with open(args.env_config) as f:
        env_config_dict =yaml.load(f, Loader=yaml.FullLoader)

    sac_config = Config(config_dict)
    env_config = Config(env_config_dict)

    exp_name = sanitize_filename(sac_config.exp_name)
    exp_tag = sanitize_filename(args.experiment_tag)
    run_name = f"{exp_name}__{exp_tag}__{getdatetime()}"
    run_name = run_name[:255]  # Truncate to 255 characters to avoid potential issues with very long paths
    run_dir = os.path.join(args.outdir, run_name)  


    with make_dirs_and_open(os.path.join(run_dir, 'configs/sac_config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, indent=4, default_flow_style=False)
   
    with make_dirs_and_open(os.path.join(run_dir, 'configs/env_config.yaml'), 'w') as f:
        yaml.dump(env_config_dict, f, indent=4, default_flow_style=False)


    SEED = sac_config.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = sac_config.torch_deterministic
    torch.autograd.set_detect_anomaly(True)
    print()
    env = PhotoEnhancementEnv(
                        batch_size=env_config.train_batch_size,
                        imsize=env_config.imsize,
                        training_mode=True,
                        done_threshold=env_config.threshold_psnr,
                        edit_sliders=env_config.sliders_to_use,
                        features_size=env_config.features_size,
                        discretize=env_config.discretize,
                        discretize_step= env_config.discretize_step,
                        use_txt_features=env_config.use_txt_features,
                        augment_data=env_config.augment_data,
                        pre_encoding_device=env_config.pre_encoding_device,   
                        pre_load_images = env_config.pre_load_images,
                        preprocessor_agent_path=env_config.preprocessor_agent_path, 
                        logger=None
    )
    test_env = PhotoEnhancementEnvTest(
                        batch_size=env_config.test_batch_size,
                        imsize=env_config.imsize,
                        training_mode=False,
                        done_threshold=env_config.threshold_psnr,
                        edit_sliders=env_config.sliders_to_use,
                        features_size=env_config.features_size,
                        discretize=env_config.discretize,
                        discretize_step = env_config.discretize_step,
                        use_txt_features=env_config.use_txt_features,
                        augment_data=env_config.augment_data,
                        pre_encoding_device=env_config.pre_encoding_device,
                        pre_load_images = env_config.pre_load_images,
                        preprocessor_agent_path=env_config.preprocessor_agent_path,    
                        logger=None
    )

    logger.info(f'Sliders used {env.edit_sliders}')
    logger.info(f'Number of sliders used { env.num_parameters}')
    logger.info(f'Sliders used {test_env .edit_sliders}')
    logger.info(f'Number of sliders used {test_env .num_parameters}')

    writer = SummaryWriter(run_dir)
    writer.add_text(
        "SAC_hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(sac_config).items()])),
    )
    writer.add_text(
        "env_parameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(env_config).items()])),
    )
    try:    
        agent = SAC(env,sac_config,writer)

        if env_config.preprocessor_agent_path!=None: #Double agent mode
            test_env.preprocessor_agent = env.preprocessor_agent # share the same preprocessor agent
            agent.backbone.load_state_dict(env.preprocessor_agent.backbone.state_dict())
            agent.backbone.eval().requires_grad_(False)
            
        agent.start_time = time.time()
        logger.info(f'Start Training at {getdatetime()}')
        for i in tqdm(range(sac_config.total_timesteps), position=0, leave=True):
            episode_count = 0 
            agent.reset_env()
            envs_mean_rewards =[]
            if agent.global_step>env_config.backbone_warmup:
                agent.backbone.train().requires_grad_(True)
            while True:     
                episode_count+=1
                agent.global_step+=1
                rewards,batch_dones = agent.train()
                envs_mean_rewards.append(rewards.mean().item())
                if(batch_dones==True).any():
                    num_env_done = int(batch_dones.sum().item())
                    agent.writer.add_scalar("charts/num_env_done", num_env_done , agent.global_step)
                if agent.global_step % 100 == 0:
                    ens_mean_episodic_return = sum(envs_mean_rewards)
                    agent.writer.add_scalar("charts/mean_episodic_return", ens_mean_episodic_return, agent.global_step)

                if (batch_dones==True).all()==True or episode_count==sac_config.max_episode_timesteps:
                    episode_count=0           
                    break 
            if agent.global_step%200==0:
                agent.backbone.eval().requires_grad_(False)
                agent.actor.eval().requires_grad_(False)
                agent.qf1.eval().requires_grad_(False)
                agent.qf2.eval().requires_grad_(False)
                with torch.no_grad():
                    n_images = 5
                    obs = test_env.reset() 
                    actions = agent.actor.get_action(**obs.to(sac_config.device))
                    _,rewards,dones = test_env.step(actions[0])
                    agent.writer.add_scalar("charts/test_mean_episodic_return", rewards.mean().item(), agent.global_step)
                    agent.writer.add_images("test_images",test_env.state['source_image'][:n_images],0)
                    if env_config.preprocessor_agent_path!=None:           
                        agent.writer.add_images("test_images",test_env.original_image[:n_images],1)
                        agent.writer.add_images("test_images",test_env.state['enhanced_image'][:n_images],2)
                        agent.writer.add_images("test_images",test_env.state['target_image'][:n_images],3)
                    else:
                        agent.writer.add_images("test_images",test_env.state['enhanced_image'][:n_images],1)
                        agent.writer.add_images("test_images",test_env.state['target_image'][:n_images],2)
                agent.backbone.train().requires_grad_(True)
                agent.actor.train().requires_grad_(True)
                agent.qf1.train().requires_grad_(True)
                agent.qf2.train().requires_grad_(True)
                
        logger.info(f'Ended training at {getdatetime()}')
        if args.save_model:
                models_dir = os.path.join(run_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                logger.info(f"Saving models in {models_dir}")
                torch.save(agent.backbone.state_dict(), run_dir+'/models/backbone.pth')
                save_actor_head(agent.actor, run_dir+'/models/actor_head.pth')
                save_critic_head(agent.qf1, run_dir+'/models/qf1_head.pth')
                save_critic_head(agent.qf2, run_dir+'/models/qf2_head.pth')
        writer.close()
    except Exception as e:
        
        logger.exception("An error occurred during training")
        if agent.global_step>1000:
            if args.save_model:
                models_dir = os.path.join(run_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)
                logger.info(f"Saving models after exception in {models_dir}")
                torch.save(agent.backbone.state_dict(), run_dir+'/models/backbone.pth')
                save_actor_head(agent.actor, run_dir+'/models/actor_head.pth')
                save_critic_head(agent.qf1, run_dir+'/models/qf1_head.pth')
                save_critic_head(agent.qf2, run_dir+'/models/qf2_head.pth')
        writer.close()

if __name__=="__main__":

    main()