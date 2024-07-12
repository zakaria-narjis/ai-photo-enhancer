import torch

def save_actor_head(actor_model, file_path):
    # Extract the state dictionaries of the relevant layers
    actor_head_state_dict = {
        'fc1': actor_model.fc1.state_dict(),
        'fc2': actor_model.fc2.state_dict(),
        'fc_mean': actor_model.fc_mean.state_dict(),
        'fc_logstd': actor_model.fc_logstd.state_dict()
    }
    # Save the state dictionaries to a file
    torch.save(actor_head_state_dict, file_path)

def load_actor_head(actor_model, file_path,device):
    # Load the state dictionaries from the file
    actor_head_state_dict = torch.load(file_path, map_location=device)
    # Load the state dictionaries into the model
    actor_model.fc1.load_state_dict(actor_head_state_dict['fc1'])
    actor_model.fc2.load_state_dict(actor_head_state_dict['fc2'])
    actor_model.fc_mean.load_state_dict(actor_head_state_dict['fc_mean'])
    actor_model.fc_logstd.load_state_dict(actor_head_state_dict['fc_logstd'])

def save_critic_head(critic_model, file_path):
    # Extract the state dictionaries of the relevant layers
    critic_head_state_dict = {
        'fc1': critic_model.fc1.state_dict(),
        'fc2': critic_model.fc2.state_dict(),
        'fc3': critic_model.fc3.state_dict()
    }
    # Save the state dictionaries to a file
    torch.save(critic_head_state_dict, file_path)

def load_critic_head(critic_model, file_path,device):
    # Load the state dictionaries from the file
    critic_head_state_dict = torch.load(file_path, map_location=device)
    # Load the state dictionaries into the model
    critic_model.fc1.load_state_dict(critic_head_state_dict['fc1'])
    critic_model.fc2.load_state_dict(critic_head_state_dict['fc2'])
    critic_model.fc3.load_state_dict(critic_head_state_dict['fc3'])