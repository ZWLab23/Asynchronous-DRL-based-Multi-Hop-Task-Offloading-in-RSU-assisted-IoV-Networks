import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch
import os
bianx
class ActorCritic(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim,  hidden_dim)
        self.fc_actor = nn.Linear( hidden_dim, output_dim)
        self.fc_critic = nn.Linear( hidden_dim, 1)

    def actor(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_actor(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def critic(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_critic(x)
        return v




    def save(self, path):
        checkpoint = os.path.join(path, 'a3c.pt')
        torch.save(self.state_dict(), checkpoint)


    def load(self, path):
        checkpoint = os.path.join(path, 'a3c.pt')
        self.aload_state_dict(torch.load(checkpoint))


# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical
# import torch
# import os
#
# class ActorCritic(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim, device):
#         super(ActorCritic, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
#         self.fc_actor = nn.Linear(hidden_dim, output_dim).to(device)
#         self.fc_critic = nn.Linear(hidden_dim, 1).to(device)
#
#     def actor(self, x, softmax_dim=0):
#         x = F.relu(self.fc1(x))
#         x = self.fc_actor(x)
#         prob = F.softmax(x, dim=softmax_dim)
#         return prob
#
#     def critic(self, x):
#         x = F.relu(self.fc1(x))
#         v = self.fc_critic(x)
#         return v
#
#     def save(self, path):
#         checkpoint = os.path.join(path, 'a3c.pt')
#         state_dict_on_cpu = {key: val.cpu() for key, val in self.state_dict().items()}
#         torch.save(state_dict_on_cpu, checkpoint)
#
#     def load(self, path, device):
#         checkpoint = os.path.join(path, 'a3c.pt')
#         state_dict = torch.load(checkpoint, map_location=device)
#         self.load_state_dict(state_dict)
