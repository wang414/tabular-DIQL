import torch.nn as nn
import torch.nn.functional as F
import torch
'''
输入当前的状态、当前agent的obs、其他agent执行的动作、当前agent的编号对应的one-hot向量、所有agent上一个timestep执行的动作
输出当前agent的所有可执行动作对应的联合Q值——一个n_actions维向量
'''


class ComaCritic(nn.Module):
    def __init__(self, input_shape, args,output_dim = None):
        super(ComaCritic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        if output_dim is None:
            self.fc3 = nn.Linear(args.critic_dim, self.args.n_actions)
        else:
            self.fc3 = nn.Linear(args.critic_dim, output_dim)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class SAC_Actor(nn.Module):
    def __init__(self, input_shape,args, min_log_std=-10, max_log_std=2):
        super(SAC_Actor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.actor_hidden_dim)
        self.fc2 = nn.Linear(args.actor_hidden_dim, args.actor_hidden_dim)
        self.mu_head = nn.Linear(args.actor_hidden_dim, self.args.n_actions)
        self.log_std_head = nn.Linear(args.actor_hidden_dim, self.args.n_actions)


        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head