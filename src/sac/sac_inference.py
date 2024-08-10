from .sac_networks import (
    Actor,
    SoftQNetwork,
    ResNETBackbone,
    SemanticBackbone,
    SemanticBackboneOC,
    ResNETHistBackbone,
)
import torch
from .utils import load_actor_head, load_critic_head


class InferenceAgent:

    def __init__(self, inference_env, inference_args):
        self.args = inference_args
        self.device = inference_args.device
        if inference_env.use_txt_features == "embedded":
            self.backbone = SemanticBackbone().to(self.device)
        elif inference_env.use_txt_features == "one_hot":
            self.backbone = SemanticBackboneOC().to(self.device)
        elif inference_env.use_txt_features == "histogram":
            self.backbone = ResNETHistBackbone().to(self.device)
        else:
            self.backbone = ResNETBackbone().to(self.device)
        self.env = inference_env
        self.discretize = self.env.discretize
        self.discretize_step = self.env.discretize_step

    def discretize_actions(self, batch_actions):
        return (
            torch.round((batch_actions + 1) / self.discretize_step)
            * self.discretize_step
            - 1
        )

    def load_backbone(self, backbone_path):
        self.backbone.load_state_dict(
            torch.load(backbone_path, map_location=self.device)
        )
        self.actor = Actor(self.env, self.backbone).to(self.device)
        self.qf1 = SoftQNetwork(self.env, self.backbone).to(self.device)
        self.qf2 = SoftQNetwork(self.env, self.backbone).to(self.device)
        self.backbone.eval().requires_grad_(False)

    def load_actor_weights(self, actor_path):
        load_actor_head(self.actor, actor_path, self.device)
        self.actor.eval()

    def load_critics_weights(self, qf1_path, qf2_path):
        load_critic_head(self.qf1, qf1_path, self.device)
        load_critic_head(self.qf2, qf2_path, self.device)
        self.qf1.eval().requires_grad_(False)
        self.qf2.eval().requires_grad_(False)

    def act(self, obs, deterministic=True, n_samples=None):
        if n_samples is None:
            n_samples = self.args.n_actions_samples
        best_actions = None
        with torch.inference_mode():

            if deterministic:

                actions = self.actor.get_action(**obs.to(self.device))[
                    2
                ]  # mean action
                best_actions = actions

            else:
                best_actions = self.actor.get_action(**obs.to(self.device))[
                    0
                ]  # mean action
                best_values = self.critic(obs, best_actions).view(-1)

                for sample in range(n_samples):
                    if sample == 0:
                        actions = self.actor.get_action(**obs.to(self.device))[
                            2
                        ]  # sampled action
                        values = self.critic(obs, actions).view(-1)
                    else:
                        actions = self.actor.get_action(**obs.to(self.device))[
                            0
                        ]  # sampled action
                        values = self.critic(obs, actions).view(-1)

                    if (values > best_values).any():
                        best_actions[values > best_values] = actions[
                            values > best_values
                        ]
                        best_values[values > best_values] = values[
                            values > best_values
                        ]

        if self.discretize:
            best_actions = self.discretize_actions(best_actions)
        return best_actions

    def critic(self, obs, actions):
        with torch.inference_mode():
            qf1_pi = self.qf1(
                **obs.to(self.device), actions=actions.to(self.device)
            )
            qf2_pi = self.qf2(
                **obs.to(self.device), actions=actions.to(self.device)
            )
            value = torch.min(qf1_pi, qf2_pi)

        return value
