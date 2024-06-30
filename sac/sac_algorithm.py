from .sac_networks import Actor, SoftQNetwork
from envs.photo_env import PhotoEnhancementEnv

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import yaml
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer,LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.envs.transforms import (
    Compose,
    CatTensors
)

SEED = args.seed
DEVICE= 'CUDA'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = args.torch_deterministic

class SAC:

    def __init__(self,
                 env,
                 args,
                 writer):
        self.env = env #train env
        self.device = args.device
        self.writer = writer

        #networks
        self.actor = Actor(env).to(self.device)
        self.qf1 = SoftQNetwork(env).to(self.device)
        self.qf2 = SoftQNetwork(env).to(self.device)
        self.qf1_target = SoftQNetwork(env).to(self.device)
        self.qf2_target = SoftQNetwork(env).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)
    
        #Training related
        self.global_step = 0
        self.start_time = None
    
        # entropy
        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

        #ReplayBuffer
        self.rb = TensorDictReplayBuffer(
                        storage=LazyMemmapStorage(args.buffer_size,), sampler=SamplerWithoutReplacement()
                                                        )
        self.state = self.env.reset() # observation tensor (B,N_Featuresx2)

    def train(self,):
        """
            perform one global step of training
        """
        # encoded_source_image = self.obs[0]['encoded_source']
        # encoded_enhanced_image = self.obs[0]['encoded_enhanced_image']   
        # batch_observation = torch.cat((encoded_source_image,encoded_enhanced_image),dim=1)   

        # ALGO LOGIC: put action logic here
        runing_envs = self.env.sub_env_running # get running sub envs (images to be enhanced)
        
        batch_obs= torch.index_select(self.state,0,runing_envs).to(self.device)

        if self.global_step < self.args.learning_starts:
            actions = self.env.action_space.sample(len(runing_envs))
        else:
            actions, _, _ = self.actor.get_action(batch_obs)
            actions = actions.detach().cpu()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_batch_obs, rewards, dones = self.env.step(actions)

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         print(f"global_step={self.global_step}, episodic_return={info['episode']['r']}")
        #         self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
        #         self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
        #         break

        # # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]

        batch_transition = TensorDict(
            {
                "observations":batch_obs,
                "next_observations":next_batch_obs,
                "rewards":rewards,
                "dones":dones,
            },
            batch_size = [batch_obs.shape[0]]

        )
        self.rb.extand(batch_transition)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        self.state =  next_batch_obs

        # ALGO LOGIC: training.
        if self.global_step > self.args.learning_starts:
            data = self.rb.sample(self.args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(data["next_observations"])
                qf1_next_target = self.qf1_target(data["next_observations"], next_state_actions)
                qf2_next_target = self.qf2_target(data["next_observations"], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data["rewards"].flatten() + (1 - data["dones"].flatten()) * self.args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = self.qf1(data["observations"], data["actions"]).view(-1)
            qf2_a_values = self.qf2(data["observations"], data["actions"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            if self.global_step % self.args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    self.args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = self.actor.get_action(data["observations"])
                    qf1_pi = self.qf1(data["observations"], pi)
                    qf2_pi = self.qf2(data["observations"], pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(data["observations"])
                        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        alpha = self.log_alpha.exp().item()

            # update the target networks
            if self.global_step % self.args.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            if self.global_step % 100 == 0:
                self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), self.global_step)
                self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), self.global_step)
                self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), self.global_step)
                self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), self.global_step)
                self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, self.global_step)
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.global_step)
                self.writer.add_scalar("losses/alpha", alpha, self.global_step)
                self.print("SPS:", int(self.global_step / (time.time() - self.start_time)))
                self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
                if self.args.autotune:
                    self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), self.global_step)
        
        return rewards.mean()