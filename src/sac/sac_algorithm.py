
from .sac_networks import Actor, SoftQNetwork, ResNETBackbone, SemanticBackbone,SemanticBackboneOC

import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer,LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement



class SAC:

    def __init__(self,
                 env,
                 args,
                 writer):
        self.env = env #train env
        self.device = args.device
        self.writer = writer
        self.args = args
        #networks
        if self.env.use_txt_features=="embedded":
            self.backbone = SemanticBackbone().to(self.device)
        elif self.env.use_txt_features=="one_hot":
            self.backbone = SemanticBackboneOC().to(self.device)
        elif self.env.use_txt_features==False:
            self.backbone = ResNETBackbone().to(self.device)      
            
        self.actor = Actor(env,self.backbone).to(self.device)
        self.qf1 = SoftQNetwork(env,self.backbone).to(self.device)
        self.qf2 = SoftQNetwork(env,self.backbone).to(self.device)
        self.qf1_target = SoftQNetwork(env,self.backbone).to(self.device)
        self.qf2_target = SoftQNetwork(env,self.backbone).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())


        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)
    
        #Training related
        self.global_step = 0
        self.start_time = None
    
        # entropy
        if args.autotune:
            # self.target_entropy = -torch.prod(torch.Tensor(env.action_space._shape[1]).to(self.device)).item()
            self.target_entropy = - env.action_space._shape[1]
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

        #ReplayBuffer
        self.rb = TensorDictReplayBuffer(
                        storage=LazyMemmapStorage(args.buffer_size,), sampler=SamplerWithoutReplacement()
                                                        )

    def reset_env(self,):
        self.state= self.env.reset()# observation tensor (B,N_Featuresx2)

    def train(self,):
        """
            perform one global step of training
        """

        # # ALGO LOGIC: put action logic here
        # runing_envs = self.env.sub_env_running # get running sub envs (images to be enhanced)
        # # if len(runing_envs)<self.env.batch_size:
        # #     print('d',self.state,runing_envs)
        # #     print(self.state.shape,runing_envs.shape)
        # # batch_obs= torch.index_select(self.state,0,runing_envs).to(self.device)

        batch_obs = self.state.to(self.device)

        if self.global_step < self.args.learning_starts:
            actions = self.env.action_space.sample(batch_obs.shape[0]).to(self.device)
        else:
            actions, _, _ = self.actor.get_action(**batch_obs)
            actions = actions.detach()
        next_batch_obs, rewards, dones = self.env.step(actions)
        batch_transition = TensorDict(
            {
                "observations":batch_obs.clone(),
                "next_observations":next_batch_obs.clone(),
                "actions":actions.clone(),
                "rewards":rewards.clone(),
                "dones":dones.clone(),
            },
            batch_size = [batch_obs.shape[0]],
        )
        self.rb.extend(batch_transition)
        self.update()
        # runing_envs = self.env.sub_env_running 
        # self.state =  torch.index_select(next_batch_obs,0,runing_envs).to(self.device)
        return rewards,dones
    
    def act_eval(self,obs):
        self.backbone.eval()
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor.get_action(**obs.to(self.device))
        self.backbone.train()
        self.actor.train()    
        return actions
    
    def update(self,):
        # ALGO LOGIC: training.
        if self.global_step > self.args.learning_starts:
            data = self.rb.sample(self.args.batch_size).to(self.device)
            with torch.no_grad():
                if self.args.gamma!=0:
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(**data["next_observations"])
                    qf1_next_target = self.qf1_target(**data["next_observations"], actions=next_state_actions)
                    qf2_next_target = self.qf2_target(**data["next_observations"], actions=next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value = data["rewards"].flatten() + (1 - data["dones"].to(torch.float32).flatten()) * self.args.gamma * (min_qf_next_target).view(-1)
                else:
                    next_q_value = data["rewards"].flatten()

            qf1_a_values = self.qf1(**data["observations"], actions = data["actions"]).view(-1)
            qf2_a_values = self.qf2(**data["observations"], actions = data["actions"]).view(-1)
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
                    pi, log_pi, _ = self.actor.get_action(**data["observations"])
                    qf1_pi = self.qf1(**data["observations"], actions=pi)
                    qf2_pi = self.qf2(**data["observations"], actions=pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.args.autotune:
                        # with torch.no_grad():
                        #     _, log_pi, _ = self.actor.get_action(data["observations"])
                        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

            # update the target networks
            if self.args.gamma!=0:
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
                self.writer.add_scalar("losses/alpha", self.alpha, self.global_step)
                # print("SPS:", int(self.global_step / (time.time() - self.start_time)))
                self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
                if self.args.autotune:
                    self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), self.global_step)
        
        