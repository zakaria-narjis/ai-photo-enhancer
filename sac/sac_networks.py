import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
import torch.nn.init as init
LOG_STD_MAX = 3
LOG_STD_MIN = -5


class Backbone(nn.Module):
    def __init__(self,):
        super().__init__()
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])#remove classifier
    def forward(self,batch_images):
        x = self.preprocess (batch_images)
        features = self.model(x)
        features=torch.flatten(features,start_dim=-3,end_dim=-1)
        return features

class Actor(nn.Module):
    def __init__(self, env, features_extractor=None,use_xavier = True):
        super().__init__()
        input_shape = env.observation_space._shape[1]*1 
        output_shape = env.action_space._shape[1]
        if env.pre_encode:
            self.features_extractor = nn.Identity()
        else:
            assert features_extractor!=None
            self.features_extractor = features_extractor
            
        self.fc1 = nn.Linear(input_shape , 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, output_shape)
        self.fc_logstd = nn.Linear(128, output_shape)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        if use_xavier:
            self._initialize_weights()
            
    def forward(self, x):
        x = self.features_extractor(x)
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
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

class SoftQNetwork(nn.Module):
    def __init__(self, env,features_extractor=None,use_xavier = True):
        super().__init__()
        input_shape = env.observation_space._shape[1]*1 
        output_shape = env.action_space._shape[1]
        if env.pre_encode:
            self.features_extractor = nn.Identity()
        else:
            assert features_extractor!=None
            self.features_extractor = features_extractor

        self.fc1 = nn.Linear(input_shape+output_shape , 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128, 1)

        if use_xavier:
            self._initialize_weights()

    def forward(self, x, a):
        x = self.features_extractor(x)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    
