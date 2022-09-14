import pickle
import random
from collections import deque
import time
import torch
import Env
from torch import nn
from torch.nn import functional as F
import argparse
import numpy as np
import os
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="test c51 for the independence multi-agent environment")
parser.add_argument("--train", action="store_true", default=True, help="train the model")
parser.add_argument("--test", action="store_true", default=False, help="test the model")
parser.add_argument("--path", type=str, default='test', help="save folder path or the test model path")
parser.add_argument("--method1", action='store_true', default=False)
parser.add_argument("--method2", action='store_true', default=False)
parser.add_argument("--method3", action='store_true', default=False)
parser.add_argument("--method4", action='store_true', default=False)
parser.add_argument("--modelname", type=str, default='hardtaskc51', help="saving model name")
parser.add_argument("--dataset", type=int, default=0, help="choose the model")
parser.add_argument("--vmax", type=int, default=5, help="set the vmax")
parser.add_argument("--vmin", type=int, default=-5, help="set the vmin")
parser.add_argument("--N", type=int, default=51, help="set the numbers of the atoms")
parser.add_argument("--eps", type=float, default=0.33, help="set the epsilon")
parser.add_argument("--gamma", type=float, default=0.99, help="set the gamma")
parser.add_argument("--Lr", type=float, default=0.5, help="set the learning rate")
parser.add_argument("--cap", type=int, default=20000, help="the capability of the memory buffer")
parser.add_argument("--step", type=int, default=100, help="the frequency of training")
parser.add_argument("--freq", type=int, default=100, help="the frequency of update the model")
parser.add_argument("--episode", type=int, default=10000, help="set episode rounds")
parser.add_argument("--ucb", type=float, default=0.85, help="set the upper confidence bound")
parser.add_argument("--verbose", action='store_true', default=False, help="print verbose test process")
parser.add_argument("--GPU", action="store_true", default=False, help="use cuda core")
parser.add_argument("--batchsize", type=int, default=100, help="learning batchsize")
parser.add_argument("--randstart", action='store_true', default=False, help="random start from any state")
parser.add_argument("--iql", action='store_true', default=False)
parser.add_argument("--network", action='store_true', default=False)
parser.add_argument("--weight", type=float, default=0.5)
parser.add_argument("--samplenum", type=int, default=10)
parser.add_argument("--overlap", action='store_true', default=False)


args = parser.parse_args()
# in tabular case state=30, actions=5, agents=3
tabel_lr = args.Lr
network_lr = args.Lr

test_flg = False


class Z_table(nn.Module):
    """
        input should be one hot vector which stands for the states
    """

    def __init__(self, n_states, n_actions, N):
        super(Z_table, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.N = N
        self.Linear = nn.Linear(n_states, N * n_actions, bias=False)
        nn.init.constant(self.Linear.weight, 0.0)

    def forward(self, state):
        par = self.Linear(torch.tensor(state, dtype=torch.float32))
        par = par.reshape(-1, self.n_actions, self.N)
        return F.softmax(par, dim=2)


class Q_table(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Q_table, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.Linear = nn.Linear(n_states, n_actions, bias=False)
        nn.init.constant(self.Linear.weight, 0.0)

    def forward(self, state):
        par = self.Linear(torch.tensor(state, dtype=torch.float32))
        return par


class C51agent:
    def __init__(self, n_states, n_actions, N, v_min, v_max, eps, gamma, alpha, idx, ucb):
        self.n_states = n_states
        self.n_actions = n_actions
        self.N = N
        self.v_min = v_min
        self.v_max = v_max
        self.model = Z_table(n_states, n_actions, N)
        self.target_model = Z_table(n_states, n_actions, N)
        self.eps = eps
        self.deltaZ = (v_max - v_min) / float(N - 1)
        self.Z = [v_min + i * self.deltaZ for i in range(N)]
        self.gamma = gamma
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=network_lr)
        self.idx = idx
        self.ucb = ucb
        self.Q = Q_table(n_states, n_actions)
        self.target_Q = Q_table(n_states, n_actions)
        self.optimizer_Q = torch.optim.SGD(self.Q.parameters(), lr=tabel_lr)

    def save_checkpoint(self, folder):
        torch.save(self.model.state_dict(), folder + '/c51_agent{}_run{}.pkl'.format(self.idx, run_num))
        torch.save(self.Q.state_dict(), folder + '/iql_agent{}_run{}.pkl'.format(self.idx, run_num))

    def get_iql_opt_action(self, state):
        with torch.no_grad():
            Q = self.target_Q(state)
            Q.squeeze(dim=0)
            # print("place1")
            # print(Q)
            ans = rand_argmax(Q)
            # print(ans)
            return ans

    def get_opt_action(self, state):
        with torch.no_grad():
            """E = []
            for i in range(self.n_actions):
                E.append(self.target_model(state)[i] * torch.tensor(self.Z, dtype=torch.float32))

            E = torch.vstack(E)
            E = E.sum(dim=1)"""
            Q = self.target_model(state)
            # print(Q.shape)
            Q = Q * torch.tensor(self.Z, dtype=torch.float32)
            Q = torch.squeeze(Q, 0)
            Q = Q.sum(dim=1)
        return Q.argmax()

    def get_iql_action(self, state):
        rand = torch.rand(1)
        if rand <= self.eps:
            return random.randrange(0, self.n_actions)
        else:
            return self.get_iql_opt_action(state)

    def test_iql_opt_action(self, state):
        with torch.no_grad():
            Q = self.Q(state)
            Q.squeeze(dim=0)
            return Q.argmax()

    def get_action(self, state):
        rand = torch.rand(1)
        if rand <= self.eps:
            return random.randrange(0, self.n_actions)
        else:
            return self.get_opt_action(state)

    def get_ucb_action(self, state):
        rand = torch.rand(1)
        if rand <= self.eps:
            return random.randrange(0, self.n_actions)
        else:
            return self.get_opt_ucb_action(state)

    def get_method3_action(self, state):
        rand = torch.rand(1)
        if rand <= self.eps:
            return random.randrange(0, self.n_actions)
        else:
            return self.get_opt_method3_action(state)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_Q.load_state_dict(self.Q.state_dict())

    def train_replay(self, memory, batch_size):
        # print("enter here")
        num_samples = min(batch_size, len(memory))
        replay_samples = random.sample(memory, num_samples)
        # Project Next State Value Distribution (of optimal action) to Current State
        b_s = [sample['s'] for sample in replay_samples]
        b_r = [sample['r'] for sample in replay_samples]
        b_a = [sample['a'] for sample in replay_samples]
        b_s_ = [sample['s_'] for sample in replay_samples]
        b_d_ = [sample['done'] for sample in replay_samples]

        b_s = np.array(b_s)
        b_r = np.array(b_r)
        b_s_ = np.array(b_s_)
        b_a = torch.LongTensor(b_a)

        z_eval = self.model(b_s)  # (batch-size * n_actions * N)
        mb_size = z_eval.size(0)
        # print("b_a shape:{}".format(b_a.shape))
        z_eval = torch.stack([z_eval[i].index_select(dim=0, index=b_a[i, self.idx]) for i in range(mb_size)]).squeeze(1)
        # (batch-size * N)
        z_next = self.target_model(b_s_).detach()  # (m, N_ACTIONS, N_ATOM)
        z_next = z_next.numpy()
        range_value = np.array(self.Z, dtype=float)
        z_range = np.array(self.Z, dtype=float)
        z_range = z_range * self.gamma
        q_next_mean = np.sum(z_next * range_value, axis=2)  # (m, N_ACTIONS)
        opt_act = np.argmax(q_next_mean, axis=1)  # (batch_size)
        opt_act = opt_act.astype(int)
        m_prob = np.zeros([num_samples, self.N])
        global test_flg
        if test_flg:
            print("prev z: {}".format(z_eval))
            print("prev q: {}".format((z_eval * torch.tensor(self.Z)).sum()))
            print("transition: [s:{}, r:{}, a:{}, s_{}, done:{}]".format(b_s[0].argmax(), b_r[0], b_a[0].argmax(),
                                                                         b_s_[0].argmax(), b_d_[0]))
        for i in range(num_samples):
            # Get Optimal Actions for the next states (from distribution z)
            # z = self.model(replay_samples[i]['s_'])  # should be updated model
            # zd = [i.detach() for i in z]  # detach version of model should be updated
            if b_d_[i]:  # Terminal State
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, b_r[i]))
                bj = (Tz - self.v_min) / self.deltaZ
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[i][int(m_l)] += (m_u - bj)
                m_prob[i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.N):
                    # print("{} {} {}".format(i, opt_act[i], j))
                    Tz = min(self.v_max,
                             max(self.v_min, b_r[i] + z_range[j]))

                    bj = (Tz - self.v_min) / self.deltaZ
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[i][int(m_l)] += z_next[i, opt_act[i], j] * (m_u - bj)
                    m_prob[i][int(m_u)] += z_next[i, opt_act[i], j] * (bj - m_l)

        m_prob = torch.FloatTensor(m_prob)
        # print("{} {}".format(m_prob.shape, (-torch.log(z_eval + 1e-8)).shape))
        loss = m_prob * (-torch.log(z_eval + 1e-6))
        loss = torch.sum(loss)
        self.optimizer.zero_grad()
        #  loss.backward(retain_graph=True)  # 误差反向传播
        loss.backward()
        matrix = self.model.Linear.weight.clone().detach()
        self.optimizer.step()
        matrix_ = self.model.Linear.weight.clone().detach()
        with torch.no_grad():
            return F.l1_loss(matrix_, matrix)
        # print('finish')

    def train_replay_iql(self, memory, batch_size):
        num_samples = min(batch_size, len(memory))
        replay_samples = random.sample(memory, num_samples)
        # Project Next State Value Distribution (of optimal action) to Current State
        b_s = [sample['s'] for sample in replay_samples]
        b_r = [sample['r'] for sample in replay_samples]
        b_a = [sample['a'] for sample in replay_samples]
        b_s_ = [sample['s_'] for sample in replay_samples]
        b_s = np.array(b_s)
        b_r = np.array(b_r)
        b_s_ = np.array(b_s_)
        b_a = torch.LongTensor(b_a)
        b_a = b_a[:, self.idx].unsqueeze(1)
        # print('{} {}'.format(self.Q(b_s).shape, b_a[:, self.idx].unsqueeze(1).shape))
        q_eval = self.Q(b_s).gather(1, b_a)  # shape (batch, 1)

        # print('original:\n{}\n chosen:\n{}\n index:\n{}'.format(self.Q(b_s), q_eval, b_a[:, self.idx]))
        # print("q_eval shape {}".format(q_eval.dtype))
        q_next = self.target_Q(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        b_r = torch.from_numpy(b_r).type(torch.float32)
        q_target = b_r + self.gamma * q_next.max(1)[0]  # shape (batch)
        q_target = q_target[:, None]
        # print("q_target:{}".format(q_target))
        """
        print(q_target)
        print(q_next)
        """
        loss = F.mse_loss(q_eval, q_target)
        # print(loss)
        # 计算, 更新 eval net
        Q_prev = self.Q.Linear.weight.clone().detach()
        self.optimizer_Q.zero_grad()
        loss.backward()  # 误差反向传播
        self.optimizer_Q.step()
        Q_new = self.Q.Linear.weight.clone().detach()
        return F.l1_loss(Q_prev, Q_new)

    def train_replay_iql_target_ucb(self, memory, batch_size):
        with torch.no_grad():
            num_samples = min(batch_size, len(memory))
            replay_samples = random.sample(memory, num_samples)
            # Project Next State Value Distribution (of optimal action) to Current State
            b_s = [sample['s'] for sample in replay_samples]
            b_r = [sample['r'] for sample in replay_samples]
            b_a = [sample['a'] for sample in replay_samples]
            b_s_ = [sample['s_'] for sample in replay_samples]
            b_s = np.array(b_s)
            b_r = np.array(b_r)
            b_s_ = np.array(b_s_)
            b_a = torch.LongTensor(b_a)
            b_a = b_a[:, self.idx].unsqueeze(1)
            # print('{} {}'.format(self.Q(b_s).shape, b_a[:, self.idx].unsqueeze(1).shape))
            q_eval = self.Q(b_s).gather(1, b_a)  # shape (batch, 1)

            # print('original:\n{}\n chosen:\n{}\n index:\n{}'.format(self.Q(b_s), q_eval, b_a[:, self.idx]))
            # print("q_eval shape {}".format(q_eval.dtype))
            q_next1 = self.target_Q(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
            cdf = self.target_model(b_s_).detach()  # [num_samples * actions * N]
            for i in range(self.N - 1, 0, -1):
                cdf[:, :, i - 1] += cdf[:, :, i]
            q_next2 = torch.zeros([num_samples, self.n_actions])
            for idx in range(num_samples):
                for action in range(self.n_actions):
                    for i in range(self.N - 1, 0, -1):
                        if cdf[idx, action, i] >= 1 - self.ucb:
                            q_next2[idx, action] = self.Z[i]
                            break

            b_r = torch.from_numpy(b_r).type(torch.float32)
            q_target = b_r + self.gamma * \
                       ((1.0 - args.weight) * q_next1.max(1)[0] + args.weight * q_next2.max(1)[0])  # shape (batch)
            q_target = q_target[:, None]
            tabel = self.Q.Linear.weight
            Q_prev = self.Q.Linear.weight.clone().detach()
            # print("Q_prev:{}".format(Q_prev))
            # print("Q_target:{}".format(q_target))
            # print("Q_tabel:{}".format(tabel.shape))
            b_s = b_s.argmax(axis=1)
            # print("b_s : {}".format(b_s))
            # print("b_a : {}".format(b_a))
            for idx in range(num_samples):
                tabel[b_a[idx], b_s[idx]] = (1 - tabel_lr) * tabel[b_a[idx], b_s[idx]] + tabel_lr * q_target[idx]
            # loss = F.mse_loss(q_eval, q_target)
            # 计算, 更新 eval net

            Q_new = self.Q.Linear.weight.clone().detach()
            return F.l1_loss(Q_prev, Q_new)

    def train_replay_iql_qtabel(self, memory, batch_size):
        with torch.no_grad():
            num_samples = min(batch_size, len(memory))
            replay_samples = random.sample(memory, num_samples)
            # Project Next State Value Distribution (of optimal action) to Current State
            b_s = [sample['s'] for sample in replay_samples]
            b_r = [sample['r'] for sample in replay_samples]
            b_a = [sample['a'] for sample in replay_samples]
            b_s_ = [sample['s_'] for sample in replay_samples]
            b_s = np.array(b_s)
            b_r = np.array(b_r)
            b_s_ = np.array(b_s_)
            b_a = torch.LongTensor(b_a)
            b_a = b_a[:, self.idx].unsqueeze(1)
            # print(b_a)
            # print('{} {}'.format(self.Q(b_s).shape, b_a[:, self.idx].unsqueeze(1).shape))
            # q_eval = self.Q(b_s).gather(1, b_a)  # shape (batch, 1)

            # print('original:\n{}\n chosen:\n{}\n index:\n{}'.format(self.Q(b_s), q_eval, b_a[:, self.idx]))
            # print("q_eval shape {}".format(q_eval.dtype))
            q_next = self.Q(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
            b_r = torch.from_numpy(b_r).type(torch.float32)
            q_target = b_r + self.gamma * q_next.max(1)[0]  # shape (batch)
            q_target = q_target[:, None]  # [batch_size * 1]
            # print("q_target:{}".format(q_target))
            tabel = self.Q.Linear.weight
            Q_prev = self.Q.Linear.weight.clone().detach()
            # print("Q_prev:{}".format(Q_prev))
            # print("Q_target:{}".format(q_target))
            # print("Q_tabel:{}".format(tabel.shape))
            b_s = b_s.argmax(axis=1)
            # print("b_s : {}".format(b_s))
            # print("b_a : {}".format(b_a))
            for idx in range(num_samples):
                tabel[b_a[idx], b_s[idx]] = (1 - tabel_lr) * tabel[b_a[idx], b_s[idx]] + tabel_lr * q_target[idx]
            # loss = F.mse_loss(q_eval, q_target)
            # 计算, 更新 eval net

            Q_new = self.Q.Linear.weight.clone().detach()
            return F.l1_loss(Q_prev, Q_new)

    def rand_peek(self):
        x = np.zeros([self.n_states])
        state = np.random.randint(0, self.n_states)
        x[state] = 1
        x = torch.FloatTensor(x).reshape(1, -1)
        y = self.model(x).squeeze()
        return "for state {},\nQ is {}\n".format(state, torch.sum(y * torch.FloatTensor(self.Z), dim=1))

    def test_opt_action(self, state, verbose):
        with torch.no_grad():
            Q = self.model(state)
            # print(Q.shape)
            Q = torch.squeeze(Q, 0)
            Q = Q * torch.tensor(self.Z, dtype=torch.float32)
            Q = Q.sum(dim=1)
            action = Q.argmax()
            if verbose:
                print("agent {} q-table is {}, choose action {}".format(self.idx, Q, action))
        return action

    def get_opt_ucb_action(self, state):
        with torch.no_grad():
            Q = self.target_model(state)
            # print(Q.shape)
            Q = torch.squeeze(Q, 0)
            # print("Q:{}".format(Q))
            cdf = Q
            # print("Q is : {}".format((Q * torch.tensor(self.Z)).sum(dim=1)))
            for i in range(self.N - 1, 0, -1):
                cdf[:, i - 1] += cdf[:, i]
            act_val = np.zeros([self.n_actions], dtype=float)
            for action in range(self.n_actions):
                for i in range(self.N - 1, 0, -1):
                    if cdf[action][i] >= 1 - self.ucb:
                        act_val[action] = self.Z[i]
                        break
            action = act_val.argmax()
        return action

    def get_opt_method3_action(self, state):
        with torch.no_grad():
            Q = self.target_model(state)
            # print(Q.shape)
            Q = torch.squeeze(Q, 0)
            # print("Q:{}".format(Q))
            cdf = Q.clone()
            # print("Q is : {}".format((Q * torch.tensor(self.Z)).sum(dim=1)))
            for i in range(self.N - 1, 0, -1):
                cdf[:, i - 1] += cdf[:, i]
            act_val = np.zeros([self.n_actions], dtype=float)
            for action in range(self.n_actions):
                for i in range(self.N - 1, 0, -1):
                    if cdf[action][i] >= 1 - self.ucb:
                        act_val[action] = self.Z[i]
                        break
            # print(Q.shape)
            Q = Q * torch.tensor(self.Z, dtype=torch.float32)
            Q = Q.sum(dim=1)
            Q = Q.numpy()
            act_val += Q
            action = act_val.argmax()
        return action

    def test_ucb_opt_action(self, state, verbose, ucb):
        tmp = self.ucb
        if ucb is not None:
            self.ucb = ucb
        with torch.no_grad():
            Q = self.model(state)
            # print(Q.shape)
            Q = torch.squeeze(Q, 0)
            # print("Q:{}".format(Q))
            cdf = Q
            # print("Q is : {}".format((Q * torch.tensor(self.Z)).sum(dim=1)))
            for i in range(self.N - 1, 0, -1):
                cdf[:, i - 1] += cdf[:, i]
            act_val = np.zeros([self.n_actions], dtype=float)
            for action in range(self.n_actions):
                for i in range(self.N - 1, 0, -1):
                    if cdf[action][i] >= 1 - self.ucb:
                        act_val[action] = self.Z[i]
                        break
            action = act_val.argmax()
            # print(cdf)
            if verbose:
                print("agent {} q-table(ucb) is {}, choose action {}".format(self.idx, act_val, action))
        self.ucb = tmp
        return action

    def load(self, folder):
        self.model.load_state_dict(torch.load(folder + '/agent{}.pkl'.format(self.idx)))
        self.target_model.load_state_dict(torch.load(folder + '/agent{}.pkl'.format(self.idx)))

    def generate_pi_dis(self):
        with torch.no_grad():
            s = torch.eye(self.n_states)
            Z = self.model(s) * torch.tensor(self.Z, dtype=torch.float32)
            Q = Z.sum(dim=2)
            pi = Q.argmax(dim=1)
            return pi

    def generate_pi_iql(self):
        with torch.no_grad():
            s = torch.eye(self.n_states)
            Q = self.Q(s)
            pi = Q.argmax(dim=1)
            return pi


start_time = time.time()
training_time = 0


class Multi_C51:
    """
        multi, independent, C51
    """
    c51agents = []
    memory = deque()

    def __init__(self, n_agents, ucb, n_states, n_actions, N, v_min, v_max, utf, eps, gamma, batch_size=32,
                 alpha=0.001, max_memory=50000, model_name='multi_c51'):
        self.n_agents = n_agents
        self.n_actions = n_agents
        self.n_states = n_agents
        self.N = N
        self.v_min = v_min
        self.v_max = v_max
        self.batch_size = batch_size
        for i in range(n_agents):
            self.c51agents.append(C51agent(n_states, n_actions, N, v_min, v_max, eps, gamma, alpha, i, ucb))
        self.ucb = ucb
        self.max_memory = max_memory
        self.update_target_freq = utf
        self.model_name = model_name

    def get_joint_iql_action(self, state):
        actions = [agent.get_iql_action(state) for agent in self.c51agents]
        return actions

    def get_joint_action(self, state):
        actions = [agent.get_action(state) for agent in self.c51agents]
        return actions

    def get_joint_ucb_action(self, state):
        actions = [agent.get_ucb_action(state) for agent in self.c51agents]
        return actions

    def get_joint_method3_action(self, state):
        actions = [agent.get_method3_action(state) for agent in self.c51agents]
        return actions

    def store_transition(self, s, a, r, s_, done):
        self.memory.append({'s': s, 'a': a, 'r': r, 's_': s_, 'done': done})
        if len(self.memory) > self.max_memory:
            self.memory.popleft()

    def update_target_models(self):
        # print("updating")
        for agent in self.c51agents:
            agent.update_target_model()

    def save_checkpoint(self, folder_name):
        Folder = 'logs/' + folder_name
        if not os.path.exists(Folder):  # 是否存在这个文件夹
            os.makedirs(Folder)
        Folder += '/' + str(self.model_name)
        if not os.path.exists(Folder):
            os.makedirs(Folder)
        for agent in self.c51agents:
            agent.save_checkpoint(Folder)

    def load_agent(self, folder_name):
        for agent in self.c51agents:
            agent.load(folder_name)

    def train_replay_iql(self):
        st_time = time.time()
        q_judge = 0
        for agent in self.c51agents:
            # print("enter agent" + str(agent.idx) + " !!!")
            # agent.train_replay(self.memory, self.batch_size)
            if args.network:
                q_judge += agent.train_replay_iql(self.memory, self.batch_size)
            else:
                q_judge += agent.train_replay_iql_qtabel(self.memory, self.batch_size)

        global training_time
        training_time += time.time() - st_time
        return q_judge / self.n_agents

    def train_replay_iql_target_ucb(self):
        st_time = time.time()
        q_judge = 0
        for agent in self.c51agents:
            # print("enter agent" + str(agent.idx) + " !!!")
            # agent.train_replay(self.memory, self.batch_size)
            q_judge += agent.train_replay_iql_target_ucb(self.memory, self.batch_size)
        global training_time
        training_time += time.time() - st_time
        return q_judge / self.n_agents

    def train_replay(self):
        st_time = time.time()
        q_judge = 0
        for agent in self.c51agents:
            # print("enter agent" + str(agent.idx) + " !!!")
            # agent.train_replay(self.memory, self.batch_size)
            q_judge += agent.train_replay(self.memory, self.batch_size)
        global training_time
        training_time += time.time() - st_time
        return q_judge / self.n_agents

    def test_opt_action(self, state, verbose):
        actions = [agent.test_opt_action(state, verbose) for agent in self.c51agents]
        return actions

    def test_ucb_opt_action(self, state, verbose, ucb):
        actions = [agent.test_ucb_opt_action(state, verbose, ucb) for agent in self.c51agents]
        return actions

    def test_iql_opt_action(self, state):
        actions = [agent.test_iql_opt_action(state) for agent in self.c51agents]
        return actions

    def generate_pi_dis(self):
        return [agent.generate_pi_dis() for agent in self.c51agents]

    def generate_pi_iql(self):
        return [agent.generate_pi_iql() for agent in self.c51agents]


def test(multi_c51, verbose):
    env = Env.chooce_the_game(args.dataset, args.randstart)
    return_list = []
    R1 = []
    if verbose:
        print("verbose test process: ")

    sample_num = args.samplenum
    print("for Q based c51")
    st = "for Q based c51\n"
    for i in range(sample_num):
        ep_r = 0
        s = env.reset()
        if verbose:
            print("episode {}".format(i + 1))
        while True:
            a = multi_c51.test_opt_action(s, verbose)  # 根据dqn来接受现在的状态，得到一个行为
            actions_v = []
            for j in range(env.agent_num):
                v = np.zeros(env.action_num)
                v[a[j]] = 1
                actions_v.append(v)
            s_, r, done = env.step(actions_v)  # 根据环境的行为，给出一个反馈

            if verbose:
                print("transition(s:{},a:{},r:{},s_:{},dom:{})".format(s.argmax(), a, r, s_.argmax(), done))

            ep_r += r

            if done:
                break
            s = s_  # 现在的状态赋值到下一个状态上去
        R1.append(ep_r)
    ep_r1 = 0
    for reward in R1:
        ep_r1 += reward
    ep_r1 /= sample_num
    return_list.append(ep_r1)
    print("total mean reward {}".format(ep_r1))
    st += "total mean reward {}\n".format(ep_r1)

    ucb_range = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95]

    for ucb in ucb_range:
        R2 = []
        print("for UCB based c51, ucb is {}".format(ucb))
        st += "for ucb based c51, ucb is {}\n".format(ucb)
        for i in range(sample_num):
            ep_r = 0
            s = env.reset()
            if verbose:
                print("episode {}".format(i + 1))
            while True:
                a = multi_c51.test_ucb_opt_action(s, verbose, ucb)  # 根据dqn来接受现在的状态，得到一个行为
                actions_v = []
                for j in range(env.agent_num):
                    v = np.zeros(env.action_num)
                    v[a[j]] = 1
                    actions_v.append(v)
                s_, r, done = env.step(actions_v)  # 根据环境的行为，给出一个反馈

                if verbose:
                    print("transition(s:{},a:{},r:{},s_:{},dom:{})".format(s.argmax(), a, r, s_.argmax(), done))

                ep_r += r

                if done:
                    break
                s = s_  # 现在的状态赋值到下一个状态上去
            R2.append(ep_r)
        ep_r2 = 0
        for reward in R2:
            ep_r2 += reward
        ep_r2 /= sample_num
        print("total mean reward {}".format(ep_r2))
        st += "total mean reward {}\n".format(ep_r2)
        return_list.append(ep_r2)
    # print("peek the pi")
    # for i in range(env.agent_num):
    #    print("peeking agent {}".format(i))
    #    multi_c51.c51agents[i].rand_peek()

    s1 = "totol time is %f" % (time.time() - start_time)
    print(s1)
    st += s1 + '\n'
    s2 = "total training time is %f" % training_time
    print(s2)
    st += s2 + '\n'
    return st, return_list, ucb_range


def rand_argmax(tens):
    max_idxs, = torch.where(tens == tens.max())
    return np.random.choice(max_idxs)


run_num = 0

def train():
    Folder = 'logs/' + args.path
    if not os.path.exists(Folder):  # 是否存在这个文件夹
        os.makedirs(Folder)
    Folder += '/' + args.modelname
    if not os.path.exists(Folder):  # 是否存在这个文件夹
        os.makedirs(Folder)
    global run_num
    while os.path.exists(os.path.join(Folder, 'run{}'.format(run_num))):
        run_num += 1
    os.makedirs(os.path.join(Folder, 'run{}'.format(run_num)))
    env = Env.chooce_the_game(args.dataset, args.randstart)
    multi_c51 = Multi_C51(n_agents=env.agent_num, ucb=args.ucb, n_states=env.state_num, n_actions=env.action_num,
                          N=args.N, v_min=args.vmin, v_max=args.vmax, utf=args.freq, eps=args.eps, gamma=args.gamma,
                          max_memory=args.cap, alpha=args.Lr, batch_size=args.batchsize, model_name=args.modelname)
    with open(Folder + '/result_run{}.txt'.format(run_num), 'w') as f:
        f.write('{}\n'.format(args))
    t = 0
    time_step = args.step
    max_episode = args.episode
    if test_flg:
        max_episode = 1
    test_num = args.samplenum
    iter_list = []
    val_list = []
    q_judge_list = []
    pi_judge_list = []
    flag = False
    q_judge = 0
    global tabel_lr
    if args.iql:
        pi_prev = multi_c51.generate_pi_iql()
        for i in range(max_episode):
            s = env.reset()
            if i > 2000 and val_list[-1] >= 17.5:
                tabel_lr *= 0.99
            while True:
                a = multi_c51.get_joint_iql_action(s)  # 根据dqn来接受现在的状态，得到一个行为
                actions_v = []
                for j in range(env.agent_num):
                    v = np.zeros(env.action_num)
                    v[a[j]] = 1
                    actions_v.append(v)
                s_, r, done = env.step(actions_v)  # 根据环境的行为，给出一个反馈
                t += 1
                multi_c51.store_transition(s, a, r, s_, False)  # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态
                # print((s, a, r, s_, done, t))

                if t % time_step == 0:
                    q_judge += multi_c51.train_replay_iql()
                if t % args.freq == 0:
                    multi_c51.update_target_models()

                if done:
                    break
                s = s_  # 现在的状态赋值到下一个状态上去

            if i % 100 == 0:
                file = open("{}/result_run{}.txt".format(Folder, run_num), 'a')
                print("at episode %d" % i)
                file.write("at episode %d\n" % i)
                multi_c51.save_checkpoint(args.path)
                ep_r = 0
                print("q_judge:{}".format(q_judge))
                q_judge_list.append(q_judge)
                q_judge = 0
                r_list = []
                pi_new = multi_c51.generate_pi_iql()
                pi_judge = 0
                for old, new in zip(pi_prev, pi_new):
                    # print("{}, {}".format(old, new))
                    pi_judge += (old != new).sum()
                pi_prev = pi_new
                for _ in range(test_num):
                    total = 0
                    s = env.reset()
                    while True:
                        a = multi_c51.test_iql_opt_action(s)  # 根据dqn来接受现在的状态，得到一个行为
                        actions_v = []
                        for j in range(env.agent_num):
                            v = np.zeros(env.action_num)
                            v[a[j]] = 1
                            actions_v.append(v)
                        s_, r, done = env.step(actions_v)  # 根据环境的行为，给出一个反馈
                        total += r
                        t += 1
                        # print((s, a, r, s_, done, t))
                        if done:
                            break
                        s = s_  # 现在的状态赋值到下一个状态上去
                    ep_r += total
                    r_list.append(total)
                ep_r /= test_num
                print('iql mean reward is {}\nreward:{}'.format(ep_r, r_list))
                s = 'iql mean reward is {}\n'.format(ep_r)
                print('pi_judge is :{}'.format(pi_judge))
                s += 'pi_judge is :{}\n'.format(pi_judge)
                pi_judge_list.append(pi_judge)
                file.write(s)
                val_list.append(ep_r)
                iter_list.append(i)
                print('-' * 50)
                file.write('-' * 50 + '\n')
                # test(multi_c51, args.verbose)
                with open('{}/q_judge_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(q_judge_list, f)
                with open('{}/iql_iter_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(iter_list, f)
                with open('{}/iql_val_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(val_list, f)
                with open('{}/pi_judge_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(pi_judge_list, f)
                plt.figure(figsize=(16, 16))
                axes = plt.subplot(2, 1, 1)
                plt.plot(iter_list, val_list, label='iql')
                axes.set_title('iql')
                axes = plt.subplot(2, 2, 3)
                axes.set_title('q_judge')
                plt.plot(iter_list, q_judge_list, label='q_judge')
                axes = plt.subplot(2, 2, 4)
                axes.set_title('pi_judge')
                plt.plot(iter_list, pi_judge_list, label='pi_judge')
                plt.savefig(Folder + '/result')
                plt.close()
    elif args.method4:
        pi_prev = multi_c51.generate_pi_iql()
        val_list1 = []
        for i in range(max_episode):
            s = env.reset()
            if i > 2000 and val_list1[-1] >= 17.5:
                tabel_lr *= 0.99
            while True:
                if args.overlap:
                    if i < 2500 or i > 10000:
                        a = multi_c51.get_joint_action(s)  # 根据dqn来接受现在的状态，得到一个行为\
                    else:
                        a = multi_c51.get_joint_iql_action(s)
                else:
                    if i < 2500:
                        a = multi_c51.get_joint_action(s)  # 根据dqn来接受现在的状态，得到一个行为\
                    else:
                        a = multi_c51.get_joint_iql_action(s)
                    
                actions_v = []
                for j in range(env.agent_num):
                    v = np.zeros(env.action_num)
                    v[a[j]] = 1
                    actions_v.append(v)
                s_, r, done = env.step(actions_v)  # 根据环境的行为，给出一个反馈
                t += 1
                multi_c51.store_transition(s, a, r, s_, False)  # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态
                # print((s, a, r, s_, done, t))

                if t % time_step == 0:
                    multi_c51.train_replay()
                    q_judge += multi_c51.train_replay_iql_target_ucb()

                if t % args.freq == 0:
                    multi_c51.update_target_models()

                if done:
                    break
                s = s_  # 现在的状态赋值到下一个状态上去

            if i % 100 == 0:
                file = open("{}/result_run{}.txt".format(Folder, run_num), 'a')
                print("at episode %d" % i)
                file.write("at episode %d\n" % i)
                multi_c51.save_checkpoint(args.path)
                ep_r = 0
                print("q_judge:{}".format(q_judge))
                q_judge_list.append(q_judge)
                q_judge = 0
                r_list1 = []
                pi_new = multi_c51.generate_pi_iql()
                pi_judge = 0
                for old, new in zip(pi_prev, pi_new):
                    # print("{}, {}".format(old, new))
                    pi_judge += (old != new).sum()
                pi_prev = pi_new
                s, r_list, ucb_list = test(multi_c51, args.verbose)
                file.write(s)
                if not flag:
                    for _ in range(len(r_list)):
                        val_list.append([])
                    flag = True
                assert len(r_list) == len(ucb_list) + 1 and len(r_list) == len(val_list)
                for index in range(len(r_list)):
                    val_list[index].append(r_list[index])
                for _ in range(test_num):
                    total = 0
                    s = env.reset()
                    while True:
                        a = multi_c51.test_iql_opt_action(s)  # 根据dqn来接受现在的状态，得到一个行为
                        actions_v = []
                        for j in range(env.agent_num):
                            v = np.zeros(env.action_num)
                            v[a[j]] = 1
                            actions_v.append(v)
                        s_, r, done = env.step(actions_v)  # 根据环境的行为，给出一个反馈
                        total += r
                        t += 1
                        # print((s, a, r, s_, done, t))
                        if done:
                            break
                        s = s_  # 现在的状态赋值到下一个状态上去
                    ep_r += total
                    r_list1.append(total)
                ep_r /= test_num
                print('iql mean reward is {}\nreward:{}'.format(ep_r, r_list1))
                s = 'iql mean reward is {}\n'.format(ep_r)
                print('pi_judge is :{}'.format(pi_judge))
                s += 'pi_judge is :{}\n'.format(pi_judge)
                pi_judge_list.append(pi_judge)
                file.write(s)
                val_list1.append(ep_r)
                iter_list.append(i)
                print('-' * 50)
                file.write('-' * 50 + '\n')
                # test(multi_c51, args.verbose)
                with open('{}/q_judge_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(q_judge_list, f)
                with open('{}/iql_iter_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(iter_list, f)
                with open('{}/iql_val_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(val_list1, f)
                with open('{}/pi_judge_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(pi_judge_list, f)
                plt.figure(figsize=(16, 16))
                axes = plt.subplot(2, 1, 1)
                plt.plot(iter_list, val_list1, label='iql')
                axes.set_title('iql')
                axes = plt.subplot(2, 2, 3)
                axes.set_title('q_judge')
                plt.plot(iter_list, q_judge_list, label='q_judge')
                axes = plt.subplot(2, 2, 4)
                axes.set_title('pi_judge')
                plt.plot(iter_list, pi_judge_list, label='pi_judge')
                plt.savefig(Folder + '/result_run{}'.format(run_num))
                plt.close()
    else:
        pi_prev = multi_c51.generate_pi_dis()
        # print(pi_prev)
        for i in range(max_episode):
            s = env.reset()
            while True:
                if args.method2:
                    a = multi_c51.get_joint_ucb_action(s)
                elif args.method3:
                    a = multi_c51.get_joint_method3_action(s)
                else:
                    a = multi_c51.get_joint_action(s)  # 根据dqn来接受现在的状态，得到一个行为
                actions_v = []
                for j in range(env.agent_num):
                    v = np.zeros(env.action_num)
                    v[a[j]] = 1
                    actions_v.append(v)
                s_, r, done = env.step(actions_v)  # 根据环境的行为，给出一个反馈
                t += 1
                multi_c51.store_transition(s, a, r, s_, False)  # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态
                # print((s, a, r, s_, done, t))

                if t % time_step == 0:
                    q_judge += multi_c51.train_replay()
                if t % args.freq == 0:
                    multi_c51.update_target_models()

                if done:
                    break
                s = s_  # 现在的状态赋值到下一个状态上去

            if i % 100 == 0:
                file = open("{}/result_run{}.txt".format(Folder, run_num), 'a')
                print("at episode %d" % i)
                file.write("at episode %d\n" % i)
                print("rand peek start")
                file.write("rand peek start\n")
                agent = random.randint(0, multi_c51.n_agents - 1)
                print('peek agent {}'.format(agent))
                file.write('peek agent {}\n'.format(agent))
                print('q_judge:{}'.format(q_judge))
                q_judge_list.append(q_judge)
                q_judge = 0
                s = multi_c51.c51agents[agent].rand_peek()
                print(s)
                file.write(s)
                print("rand peek end")
                file.write("rand peek end\n")
                multi_c51.save_checkpoint(args.path)
                s, r_list, ucb_list = test(multi_c51, args.verbose)
                file.write(s)
                if not flag:
                    for _ in range(len(r_list)):
                        val_list.append([])
                    flag = True
                assert len(r_list) == len(ucb_list) + 1 and len(r_list) == len(val_list)
                for index in range(len(r_list)):
                    val_list[index].append(r_list[index])
                iter_list.append(i)
                pi_new = multi_c51.generate_pi_dis()
                pi_judge = 0
                for old, new in zip(pi_prev, pi_new):
                    # print("{}, {}".format(old, new))
                    pi_judge += (old != new).sum()
                pi_prev = pi_new
                print("pi_judge:{}".format(pi_judge))
                pi_judge_list.append(pi_judge)
                print('-' * 50)
                file.write('-' * 50 + '\n')
                # test(multi_c51, args.verbose)
                with open('{}/q_judge_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(q_judge_list, f)
                with open('{}/c51iter_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(iter_list, f)
                with open('{}/c51_Q_val_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(val_list[0], f)
                with open('{}/pi_judge_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                    pickle.dump(pi_judge_list, f)
                for j in range(len(r_list)):
                    if j == 0:
                        continue
                    with open('{}/c51_ucb{}_val_run{}.pkl'.format(Folder, ucb_list[j - 1]), 'wb') as f:
                        pickle.dump(val_list[j], f)
                plt.figure(figsize=(16, 16))
                axes = plt.subplot(2, 2, 3)
                plt.plot(iter_list, q_judge_list, label='q_judge')
                axes.set_title('q_judge')
                axes = plt.subplot(2, 2, 4)
                plt.plot(iter_list, pi_judge_list, label='pi_judge')
                axes.set_title('pi_judge')
                axes = plt.subplot(2, 1, 1)
                for idx in range(len(r_list)):
                    if idx == 0:
                        plt.plot(iter_list, val_list[idx], label='Q')
                    else:
                        plt.plot(iter_list, val_list[idx], label='ucb{}'.format(ucb_list[idx - 1]))
                axes.set_title('total mean reward')
                plt.legend()
                plt.savefig('{}/result'.format(Folder))


def test_agent():
    env = Env.chooce_the_game(args.dataset, args.randstart)
    multi_c51 = Multi_C51(n_agents=env.agent_num, ucb=args.ucb, n_states=env.state_num, n_actions=env.action_num,
                          N=args.N, v_min=args.vmin, v_max=args.vmax, utf=args.freq, eps=args.eps, gamma=args.gamma,
                          max_memory=args.cap, alpha=args.Lr, batch_size=args.batchsize, model_name=args.modelname)
    multi_c51.load_agent('test_agent')
    # test(multi_c51, False)


if __name__ == "__main__":
    print(args)
    if args.train:
        train()
    # os.system('shutdown -s -t 0')
