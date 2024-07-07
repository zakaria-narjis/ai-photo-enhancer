from sac.sac_networks import Actor, SoftQNetwork, Backbone
import torch
from utils import *

class InferenceAgent:

    def __init__(self,inference_env, inference_args):
        self.args =inference_args
        self.device = inference_args.device
        self.backbone = Backbone().to(self.device)
        self.env = inference_env
        
    def load_backbone (self,backbone_path):
        self.backbone.load_state_dict(torch.load(backbone_path))
        self.actor = Actor(self.env,self.backbone).to(self.device)
        self.qf1 = SoftQNetwork(self.env,self.backbone).to(self.device)
        self.qf2 = SoftQNetwork(self.env,self.backbone).to(self.device)
        self.backbone.eval()

    def load_actor_weights(self,actor_path):
        load_actor_head(self.actor, actor_path)
        self.actor.eval()
        
    def load_critics_weights(self,qf1_path,qf2_path):  
        load_critic_head(self.qf1, qf1_path)
        load_critic_head(self.qf2, qf2_path)
        self.qf1.eval()
        self.qf2.eval()

    def act(self,obs):
        with torch.inference_mode():
            actions = self.actor.get_action(obs.to(self.device))   
        return actions
    
    def critic(self,obs,actions):
        with torch.inference_mode():
            qf1_pi = self.qf1(obs.to(self.device), actions.to(self.device))
            qf2_pi = self.qf2(obs.to(self.device), actions.to(self.device))
            value = torch.min(qf1_pi, qf2_pi)

        return value