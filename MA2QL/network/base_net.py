import torch.nn as nn
import torch.nn.functional as f


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.rnn_dim = self.args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.rnn_dim)
        self.rnn = nn.GRUCell(self.rnn_dim, self.rnn_dim)
        self.fc2 = nn.Linear(self.rnn_dim, args.n_actions)
        self.no_rnn_fc = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def forward(self, obs, hidden_state):
        # print('obs = {}'.format(obs.shape))
        # print('hidden_state = {}'.format(hidden_state.shape))
        x_before = self.fc1(obs)
        # print('x_before = {}'.format(x_before.shape))
        x = f.relu(x_before)
        # print('x = {}'.format(x.shape))
        h_in = hidden_state.reshape(-1, self.rnn_dim)
        # print('h_in = {}'.format(h_in.shape))
        h = self.rnn(x, h_in)
        # print('h = {}'.format(h))
        if self.args.no_rnn:
            u = f.relu(self.no_rnn_fc(x))
            q = self.fc2(u)
        else:
            q = self.fc2(h)
        # print('q = {}'.format(q))
        return q, h





# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q
