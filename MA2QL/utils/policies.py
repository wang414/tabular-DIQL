import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=False, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        # print('inp_shape = {}'.format(inp.shape))
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        # print('out_shape = {}'.format(out.shape))
        # ret = F.softmax(out, dim=-1)
        # print('ret_shape = {}'.format(ret.shape))
        return out

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=False, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, X, hidden_state):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        # print('inp_shape = {}'.format(inp.shape))
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        # print('out_shape = {}'.format(out.shape))
        # ret = F.softmax(out, dim=-1)
        # print('ret_shape = {}'.format(ret.shape))
        return out


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        if self.args.check_alg and self.args.alg == 'qmix':
            self.rnn_dim = self.args.rnn_hidden_dim_for_qmix
        else:
            self.rnn_dim = self.args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.rnn_dim)
        self.rnn = nn.GRUCell(self.rnn_dim, self.rnn_dim)
        self.fc2 = nn.Linear(self.rnn_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        # print('obs = {}'.format(obs))
        # print('hidden_state = {}'.format(hidden_state))
        x_before = self.fc1(obs)
        # print('x_before = {}'.format(x_before))
        x = f.relu(x_before)
        # print('x = {}'.format(x))
        h_in = hidden_state.reshape(-1, self.rnn_dim)
        # print('h_in = {}'.format(h_in))
        h = self.rnn(x, h_in)
        # print('h = {}'.format(h))
        q = self.fc2(h)
        # print('q = {}'.format(q))
        return q, h

class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False,return_all_log_pi = False ):
        probs = super(DiscretePolicy, self).forward(obs)

        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            # log_probs = F.log_softmax(out, dim=1)
            pass
        if return_all_probs:
            rets.append(probs)
        if return_all_log_pi:
            # rets.append(F.log_softmax(out, dim=1))
            pass
        if return_log_pi:
            # return log probability of selected action
            # rets.append(log_probs.gather(1, int_act))
            pass
        if regularize:
            # rets.append([(out**2).mean()])
            pass
        if return_entropy:
            # rets.append(-(log_probs * probs).sum(1).mean())
            pass
        if len(rets) == 1:
            return rets[0]
        return rets
