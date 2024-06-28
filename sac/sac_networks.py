import torch
from torch import nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -5



class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        if env.pre_encode:
            input_shape = env.observation_space[1]*2 #ENCODED SOURCE+ ENCODED ENHANCED IMAGE
            output_shape = env.action_space[1]
        else:
            """
                Not implemented Yet
            """
            pass
        self.fc1 = nn.Linear(input_shape , 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, output_shape)
        self.fc_logstd = nn.Linear(256, output_shape)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        if env.pre_encode:
            input_shape = env.observation_space[1]*2 #ENCODED SOURCE+ ENCODED ENHANCED IMAGE
            output_shape = env.action_space[1]
        else:
            """
                Not implemented Yet
            """
            pass    
        self.fc1 = nn.Linear(input_shape+output_shape , 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
