import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
import torch.nn.init as init
LOG_STD_MAX = 3
LOG_STD_MIN = -5


class ResNETBackbone(nn.Module):
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

class CrossModalAttention(nn.Module):
    def __init__(self, bert_dim=768, clip_dim=512, resnet_dim=512, common_dim=512):
        super().__init__()
        self.bert_projection = nn.Linear(bert_dim, common_dim)       
        self.attention = nn.MultiheadAttention(embed_dim=common_dim*2, num_heads=8)
        
    def forward(self, bert_features, clip_features, resnet_features):
        # Project all features to a common dimension
        b = self.bert_projection(bert_features)
        c = clip_features
        r = resnet_features
        # features = torch.cat([b, r], dim=1)
        # q = features
        # k = features
        # v = features
        
        # # Reshape tensors to (seq_len, batch_size, common_dim)
        # q = q.unsqueeze(0)  # (1, batch_size, common_dim)
        # k = k.unsqueeze(0)  # (1, batch_size, common_dim)
        # v = v.unsqueeze(0)  # (1, batch_size, common_dim)
        
        # # Compute attention
        # attn_output, _ = self.attention(query=q, key=k, value=v)
        
        # output = attn_output.squeeze(0)+features
        output = torch.cat([b, r], dim=1)
        return  output   

class SemanticBackbone(nn.Module):
    def __init__(self,):
        super().__init__()
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.resnet  = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])#remove classifier
        self.attention = CrossModalAttention()

    def forward(self, batch_images, ts_features, ims_features):
        res_f = self.preprocess(batch_images)
        res_f = self.resnet(res_f)
        res_f = torch.flatten(res_f,start_dim=-3,end_dim=-1)
        features = self.attention(ts_features, ims_features, res_f)

        return features

class SemanticBackboneOC(nn.Module):
    """
    Semantic Backbone for onehot encoded text features
    
    """
    def __init__(self,):
        super().__init__()
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.resnet  = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    def forward(self,batch_images,ts_features):
        res_f = self.preprocess(batch_images)
        res_f = self.resnet(res_f)
        res_f = torch.flatten(res_f,start_dim=-3,end_dim=-1)
        res_f = torch.cat([res_f, ts_features], dim=1)
        return res_f


class Actor(nn.Module):
    def __init__(self, env, features_extractor,use_xavier = True):
        super().__init__()
        input_shape = env.observation_space._shape[1]
        output_shape = env.action_space._shape[1]
        self.features_extractor = features_extractor
            
            
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
        if use_xavier:
            self._initialize_weights()
            
    def forward(self,**kwargs):     
        x = self.features_extractor(**kwargs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, **kwargs):
        mean, log_std = self(**kwargs)
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
    def __init__(self, env,features_extractor,use_xavier = True):
        super().__init__()
        input_shape = env.observation_space._shape[1] 
        output_shape = env.action_space._shape[1]
        self.features_extractor = features_extractor

        self.fc1 = nn.Linear(input_shape+output_shape , 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256, 1)

        if use_xavier:
            self._initialize_weights()

    def forward(self, **kwargs):
        actions = kwargs.pop('actions')
        x = self.features_extractor(**kwargs)
        x = torch.cat([x, actions], 1)
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
    
