import yaml
import time
import random
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import time
from envs.photo_env import PhotoEnhancementEnv
from envs.photo_env import PhotoEnhancementEnvTest
from sac.sac_algorithm import SAC
from envs.new_edit_photo import PhotoEditor
import multiprocessing as mp
import argparse
import logging
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  

from datetime import datetime

def getdatetime():
    c = datetime.now()
    rounded_seconds = round(c.second + c.microsecond / 1e6, 2)
    formatted_time = c.strftime('%Y-%m-%d_%H:%M:') + f'{rounded_seconds:05.2f}'
    return formatted_time


class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='YAML config file')
    parser.add_argument('outdir', type=str, help='directory to put training log',default='experiments/')
    parser.add_argument('save_actor',type=bool, default=True)
    parser.add_argument('save_critics',type=bool, default=False)
    parser.add_argument('--logger_level', type=int, default=logging.INFO)

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args.logger_level)
    logger.debug('reset dataloader')

    with open("configs/hyperparameters.yaml") as f:
        config_dict =yaml.load(f, Loader=yaml.FullLoader)

    with open("configs/config.yaml") as f:
        env_config_dict =yaml.load(f, Loader=yaml.FullLoader)
    run_name = f"{sac_config.exp_name}__{sac_config.seed}__{getdatetime()}"  
    sac_config = Config(config_dict)
    env_config = Config(env_config_dict)

    with open(os.path.join(args.outdir, f'configs/{run_name}/sac_config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, indent=4, default_flow_style=False)
   
    with open(os.path.join(args.outdir, f'configs/{run_name}/env_config.yaml'), 'w') as f:
        yaml.dump(env_config_dict, f, indent=4, default_flow_style=False)


    SEED = sac_config.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = sac_config.torch_deterministic
    torch.autograd.set_detect_anomaly(True)

    env = PhotoEnhancementEnv(
                        batch_size=env_config.train_batch_size,
                        imsize=env_config.imsize,
                        training_mode=True,
                        done_threshold=env_config.threshold_psnr,
                        pre_encode=False,
                        edit_sliders=env_config.sliders_to_use,
                        features_size=env_config.features_size,
                        logger=None
    )
    test_env = PhotoEnhancementEnvTest(
                        batch_size=env_config.test_batch_size,
                        imsize=env_config.imsize,
                        training_mode=False,
                        done_threshold=env_config.threshold_psnr,
                        pre_encode=False,
                        edit_sliders=env_config.sliders_to_use,
                        features_size=env_config.features_size,
                        logger=None
    )

    logger.debug(f'Sliders used {env.edit_sliders}')
    logger.debug(f'Number of sliders used { env.num_parameters}')
    logger.debug(f'Sliders used {test_env .edit_sliders}')
    logger.debug(f'Number of sliders used {test_env .num_parameters}')

    writer = SummaryWriter(f"runs/{run_name}")
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
        agent.start_time = time.time()
        for i in range(sac_config.total_timesteps):
            episode_count = 0 
            agent.reset_env()
            envs_mean_rewards =[]
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
                agent.backbone.eval()
                with torch.no_grad():
                    n_images = 5
                    obs = test_env.reset() 
                    actions = agent.actor.get_action(obs.to(sac_config.device))
                    _,rewards,dones = test_env.step(actions[0].cpu())
                    agent.writer.add_scalar("charts/test_mean_episodic_return", rewards.mean().item(), agent.global_step)
                    agent.writer.add_images("test_images",test_env.state['source_image'][:n_images],0)
                    agent.writer.add_images("test_images",test_env.state['enhanced_image'][:n_images],1)
                    agent.writer.add_images("test_images",test_env.state['target_image'][:n_images],2)
                agent.backbone.train()
        if args.save_actor:
            torch.save(agent.actor.state_dict(), args.outdir+f'{run_name}actor_model.pth')
        if args.save_critics:
            torch.save(agent.qf1.state_dict(), args.outdir+f'{run_name}critic_model.pth')
            torch.save(agent.qf1.state_dict(), args.outdir+f'{run_name}critic_model.pth')
    except:
        if agent.global_step>1000:
            if args.save_actor:
                torch.save(agent.actor.state_dict(), args.outdir+f'{run_name}actor_model.pth')
            if args.save_critics:
                torch.save(agent.qf1.state_dict(), args.outdir+f'{run_name}critic_model.pth')
                torch.save(agent.qf1.state_dict(), args.outdir+f'{run_name}critic_model.pth')
