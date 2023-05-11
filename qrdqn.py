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
parser.add_argument("--modelname", type=str, default='qrdqn', help="saving model name")
parser.add_argument("--dataset", type=int, default=0, help="choose the model")
parser.add_argument("--N", type=int, default=200, help="set the numbers of the atoms")
parser.add_argument("--eps", type=float, default=0.33, help="set the epsilon")
parser.add_argument("--gamma", type=float, default=0.99, help="set the gamma")
parser.add_argument("--Lr", type=float, default=0.001, help="set the learning rate")
parser.add_argument("--cap", type=int, default=20000, help="the capability of the memory buffer")
parser.add_argument("--step", type=int, default=100, help="the frequency of training")
parser.add_argument("--iqrdqn", action='store_true', default=False)
parser.add_argument("--method4", action='store_true', default=False)
parser.add_argument("--freq", type=int, default=100, help="the frequency of update the model")
parser.add_argument("--episode", type=int, default=10000, help="set episode rounds")
parser.add_argument("--ucb", type=int, default=60, help="set the upper confidence bound")
parser.add_argument("--verbose", action='store_true', default=False, help="print verbose test process")
parser.add_argument("--GPU", action="store_true", default=False, help="use cuda core")
parser.add_argument("--batchsize", type=int, default=100, help="learning batchsize")
parser.add_argument("--randstart", action='store_false', default=True, help="random start from any state")
parser.add_argument("--iql", action='store_true', default=False)
parser.add_argument("--network", action='store_true', default=False)
parser.add_argument("--weight", type=float, default=0.8)
parser.add_argument("--samplenum", type=int, default=10)
parser.add_argument("--overlap", action='store_true', default=False)
parser.add_argument("--determine", default=False, action='store_true')


args = parser.parse_args()
# in tabular case state=30, actions=5, agents=3
tl_init = tabel_lr = args.Lr
nl_init = network_lr = args.Lr

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

    def forward(self, state):
        par = self.Linear(torch.tensor(state, dtype=torch.float32))
        par = par.reshape(-1, self.n_actions, self.N)
        return par


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


class qrdqnagent:
    def __init__(self, n_states, n_actions, N, eps, gamma, alpha, idx, ucb, weight):
        self.n_states = n_states
        self.n_actions = n_actions
        self.N = N
        self.model = Z_table(n_states, n_actions, N)
        self.target_model = Z_table(n_states, n_actions, N)
        self.eps = eps
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=network_lr)
        self.idx = idx
        self.ucb = ucb
        self.weight = weight
        self.taus = torch.arange(
            0, N+1, dtype=torch.float32) / N
        self.tau_hats = ((self.taus[1:] + self.taus[:-1]) / 2.0).view(1, N)
        self.selected_idxs = torch.ones([1, self.n_actions, 1], dtype=torch.int32) * (torch.arange(9).reshape(1, -1) - 4 + self.ucb*2).clip(0, 199)

    def save_checkpoint(self, folder):
        torch.save(self.model.state_dict(), folder + '/qrdqn_agent{}_run{}.pkl'.format(self.idx, run_num))

    def get_opt_action(self, state):
        with torch.no_grad():
            Q = self.target_model(state).squeeze(dim=0)
            # print(Q.shape)
            Q = Q.mean(dim=-1)
        return rand_argmax(Q)

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

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_replay_method4(self, memory, batch_size):
        # print("enter here")
        num_samples = min(batch_size, len(memory))
        replay_samples = random.sample(memory, num_samples)
        # Project Next State Value Distribution (of optimal action) to Current State
        b_s = [sample['s'] for sample in replay_samples]
        b_r = [sample['r'] for sample in replay_samples]
        b_a = [sample['a'] for sample in replay_samples]
        b_s_ = [sample['s_'] for sample in replay_samples]
        b_d = [sample['done'] for sample in replay_samples]

        b_s = torch.tensor(b_s)
        b_r = torch.tensor(b_r)
        b_s_ = torch.tensor(b_s_)
        b_a = torch.LongTensor(b_a)[:,self.idx]
        b_a = b_a.unsqueeze(-1)
        b_d = torch.tensor(b_d, dtype=torch.float32)

        # Calculate quantile values of current states and actions at taus.
        current_sa_quantiles = self.model(b_s)  # bs, action, N
        current_sa_quantiles = evaluate_quantile_at_action(current_sa_quantiles, b_a).transpose(1, 2)
    
        assert current_sa_quantiles.shape == (batch_size, self.N, 1)
        

        selected_idxs = self.selected_idxs.expand(batch_size, -1, -1)
        with torch.no_grad():
            # Calculate Q values of next states.
                # Sample the noise of online network to decorrelate between
                # the action selection and the quantile calculation.
            next_q = self.model(b_s_)
            next_q = next_q.gather(2, selected_idxs).mean(dim=-1) * self.weight + (1-self.weight) * next_q.mean(dim=-1)
            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (batch_size, 1)

            # Calculate quantile values of next states and actions at tau_hats.
            next_sa_quantiles = evaluate_quantile_at_action(
                self.target_model(b_s_),
                next_actions)
            assert next_sa_quantiles.shape == (batch_size, 1, self.N)

            # Calculate target quantile values.
            target_sa_quantiles = b_r[..., None, None] + (
                1.0 - b_d[..., None, None]) * self.gamma * next_sa_quantiles
            assert target_sa_quantiles.shape == (batch_size, 1, self.N)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (batch_size, self.N, self.N)

        quantile_huber_loss = calculate_quantile_huber_loss(td_errors, self.tau_hats)
        self.optimizer.zero_grad()
        quantile_huber_loss.backward()
        self.optimizer.step()

    def train_replay_iqrdqn(self, memory, batch_size):
        # print("enter here")
        num_samples = min(batch_size, len(memory))
        replay_samples = random.sample(memory, num_samples)
        # Project Next State Value Distribution (of optimal action) to Current State
        b_s = [sample['s'] for sample in replay_samples]
        b_r = [sample['r'] for sample in replay_samples]
        b_a = [sample['a'] for sample in replay_samples]
        b_s_ = [sample['s_'] for sample in replay_samples]
        b_d = [sample['done'] for sample in replay_samples]

        b_s = torch.tensor(b_s)
        b_r = torch.tensor(b_r)
        b_s_ = torch.tensor(b_s_)
        b_a = torch.LongTensor(b_a)[:,self.idx]
        b_a = b_a.unsqueeze(-1)
        b_d = torch.tensor(b_d, dtype=torch.float32)

        # Calculate quantile values of current states and actions at taus.
        current_sa_quantiles = self.model(b_s)  # bs, action, N
        current_sa_quantiles = evaluate_quantile_at_action(current_sa_quantiles, b_a).transpose(1, 2)
    
        assert current_sa_quantiles.shape == (batch_size, self.N, 1)
        with torch.no_grad():
            # Calculate Q values of next states.
                # Sample the noise of online network to decorrelate between
                # the action selection and the quantile calculation.
            next_q = self.model(b_s_).mean(dim=-1)
            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (batch_size, 1)

            # Calculate quantile values of next states and actions at tau_hats.
            next_sa_quantiles = evaluate_quantile_at_action(
                self.target_model(b_s_),
                next_actions)
            assert next_sa_quantiles.shape == (batch_size, 1, self.N)

            # Calculate target quantile values.
            target_sa_quantiles = b_r[..., None, None] + (
                1.0 - b_d[..., None, None]) * self.gamma * next_sa_quantiles
            assert target_sa_quantiles.shape == (batch_size, 1, self.N)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (batch_size, self.N, self.N)

        quantile_huber_loss = calculate_quantile_huber_loss(td_errors, self.tau_hats)
        self.optimizer.zero_grad()
        quantile_huber_loss.backward()
        self.optimizer.step()


    def rand_peek(self):
        x = np.zeros([self.n_states])
        state = np.random.randint(0, self.n_states)
        x[state] = 1
        x = torch.FloatTensor(x).reshape(1, -1)
        y = self.model(x).squeeze()
        return "for state {},\nQ is {}\n".format(state, torch.sum(y * torch.FloatTensor(self.Z), dim=1))

    def test_opt_action(self, state):
        with torch.no_grad():
            Q = self.model(state)
            # print(Q.shape)
            Q = torch.squeeze(Q, 0)
            Q = Q.mean(dim=1)
            action = rand_argmax(Q)
        return action

    def get_opt_ucb_action(self, state):
        with torch.no_grad():
            z = self.target_model(state)
            z = z.squeeze(0)
            act_val = z[:,self.ucb*2]
            action = rand_argmax(act_val)
        return action

    def test_ucb_opt_action(self, state, ucb):
        selected_idxs = torch.ones([self.n_actions, 1], dtype=torch.int32) * (torch.arange(9).reshape(1, -1) - 4 + self.ucb*2).clip(0, 199)
        
        z = self.model(state)
        z = z.squeeze(0)
        act_val = z.gather(1, selected_idxs).mean(dim=-1) * self.weight + (1-self.weight) * z.mean(dim=-1)
        action = rand_argmax(act_val)
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


class Multi_qrdqn:
    """
        multi, independent, C51
    """
    qrdqnagents = []
    memory = deque()

    def __init__(self, n_agents, ucb, weight, n_states, n_actions, N, utf, eps, gamma, batch_size=32,
                 alpha=0.001, max_memory=50000, model_name='multi_qrdqn'):
        self.n_agents = n_agents
        self.n_actions = n_agents
        self.n_states = n_agents
        self.N = N
        self.batch_size = batch_size
        for i in range(n_agents):
            self.qrdqnagents.append(qrdqnagent(n_states, n_actions, N, eps, gamma, alpha, i, ucb, weight))
        self.ucb = ucb
        self.max_memory = max_memory
        self.update_target_freq = utf
        self.model_name = model_name

    def get_joint_iql_action(self, state):
        actions = [agent.get_iql_action(state) for agent in self.qrdqnagents]
        return actions

    def get_joint_action(self, state):
        actions = [agent.get_action(state) for agent in self.qrdqnagents]
        return actions

    def get_joint_ucb_action(self, state):
        actions = [agent.get_ucb_action(state) for agent in self.qrdqnagents]
        return actions

    def get_joint_method3_action(self, state):
        actions = [agent.get_method3_action(state) for agent in self.qrdqnagents]
        return actions

    def test_ucb_opt_action(self, state, ucb):
        action = [agent.test_ucb_opt_action(state, ucb) for agent in self.qrdqnagents]
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory.append({'s': s, 'a': a, 'r': r, 's_': s_, 'done': done})
        if len(self.memory) > self.max_memory:
            self.memory.popleft()

    def update_target_models(self):
        # print("updating")
        for agent in self.qrdqnagents:
            agent.update_target_model()

    def save_checkpoint(self, folder_name):
        Folder = 'logs/' + folder_name
        if not os.path.exists(Folder):  # 是否存在这个文件夹
            os.makedirs(Folder)
        Folder += '/' + str(self.model_name)
        if not os.path.exists(Folder):
            os.makedirs(Folder)
        for agent in self.qrdqnagents:
            agent.save_checkpoint(Folder)

    def load_agent(self, folder_name):
        for agent in self.qrdqnagents:
            agent.load(folder_name)


    def train_replay_method4(self):
        st_time = time.time()
        for agent in self.qrdqnagents:
            agent.train_replay_method4(self.memory, self.batch_size)
        global training_time
        training_time += time.time() - st_time

    def train_replay_iqrdqn(self):
        st_time = time.time()
        for agent in self.qrdqnagents:
            agent.train_replay_iqrdqn(self.memory, self.batch_size)
        global training_time
        training_time += time.time() - st_time

    def test_opt_action(self, state):
        actions = [agent.test_opt_action(state) for agent in self.qrdqnagents]
        return actions

    def test_iql_opt_action(self, state):
        actions = [agent.test_iql_opt_action(state) for agent in self.qrdqnagents]
        return actions

    def generate_pi_dis(self):
        return [agent.generate_pi_dis() for agent in self.qrdqnagents]

    def generate_pi_iql(self):
        return [agent.generate_pi_iql() for agent in self.qrdqnagents]


def test(multi_qrdqn, verbose, mean_reward_list):
    env = Env.chooce_the_game(args.dataset, args.randstart, args.determine)
    return_list = []
    R1 = []
    if verbose:
        print("verbose test process: ")

    sample_num = args.samplenum
    print("for mean value")
    st = "for mean value\n"
    for i in range(sample_num):
        ep_r = 0
        s = env.reset()
        if verbose:
            print("episode {}".format(i + 1))
        while True:
            a = multi_qrdqn.test_opt_action(s)  # 根据dqn来接受现在的状态，得到一个行为
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
    mean_reward_list.append(ep_r1)
    return_list.append(ep_r1)
    print("total mean reward {}".format(ep_r1))
    st += "total mean reward {}\n".format(ep_r1)

    ucb_range = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

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
                a = multi_qrdqn.test_ucb_opt_action(s, ucb)  # 根据dqn来接受现在的状态，得到一个行为
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
    max_idxs = torch.where(tens == tens.max())
    return np.random.choice(max_idxs)

run_num = 0

def train():
    Folder = 'newlogs/' + args.path
    if not os.path.exists(Folder):  # 是否存在这个文件夹
        os.makedirs(Folder)
    Folder += '/' + args.modelname
    if not os.path.exists(Folder):  # 是否存在这个文件夹
        os.makedirs(Folder)
    global run_num
    while os.path.exists(os.path.join(Folder, 'run{}'.format(run_num))):
        run_num += 1
    os.makedirs(os.path.join(Folder, 'run{}'.format(run_num)))
    env = Env.chooce_the_game(args.dataset, args.randstart, args.determine)
    print("nstate:{}, nagent:{}, nactions:{}".format(env.state_num,env.agent_num,env.action_num))
    agents = Multi_qrdqn(n_agents=env.agent_num, ucb=args.ucb, n_states=env.state_num, n_actions=env.action_num,
                          N=args.N, utf=args.freq, eps=args.eps, gamma=args.gamma, weight=args.weight,
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
    flag = False
    file = open("{}/result_run{}.txt".format(Folder, run_num), 'a')
    val_list1 = []
    for i in range(max_episode):
        s = env.reset()
        while True:
            a = agents.get_joint_action(s)  # 根据dqn来接受现在的状态，得到一个行为\
            actions_v = []
            for j in range(env.agent_num):
                v = np.zeros(env.action_num)
                v[a[j]] = 1
                actions_v.append(v)
            s_, r, done = env.step(actions_v)  # 根据环境的行为，给出一个反馈
            t += 1
            agents.store_transition(s, a, r, s_, False)  # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态
            # print((s, a, r, s_, done, t))

            if t % time_step == 0:
                if args.method4:
                    agents.train_replay_method4()
                else:
                    agents.train_replay_iqrdqn()

            if t % args.freq == 0:
                agents.update_target_models()

            if done:
                break
            s = s_  # 现在的状态赋值到下一个状态上去

        if i % 100 == 0:
            print("at episode %d" % i)
            file.write("at episode %d\n" % i)
            agents.save_checkpoint(args.path)
            s, r_list, ucb_list = test(agents, args.verbose, val_list1)
            file.write(s)
            if not flag:
                for _ in range(len(r_list)):
                    val_list.append([])
                flag = True
            assert len(r_list) == len(ucb_list) + 1 and len(r_list) == len(val_list)
            for index in range(len(r_list)):
                val_list[index].append(r_list[index])
            iter_list.append(i)
            print('-' * 50)
            file.write('-' * 50 + '\n')
            # test(multi_c51, args.verbose)
            with open('{}/iql_iter_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                pickle.dump(iter_list, f)
            with open('{}/iql_val_run{}.pkl'.format(Folder, run_num), 'wb') as f:
                pickle.dump(val_list1, f)
            plt.figure(figsize=(16, 16))
            axes = plt.subplot(2, 1, 1)
            plt.plot(iter_list, val_list1, label='iql')
            plt.savefig(Folder + '/result_run{}'.format(run_num))
            plt.close()
    file.close()


# def test_agent():
#     env = Env.chooce_the_game(args.dataset, args.randstart, args.determine)
#     multi_c51 = Multi_C51(n_agents=env.agent_num, ucb=args.ucb, n_states=env.state_num, n_actions=env.action_num,
#                           N=args.N, v_min=args.vmin, v_max=args.vmax, utf=args.freq, eps=args.eps, gamma=args.gamma,
#                           max_memory=args.cap, alpha=args.Lr, batch_size=args.batchsize, model_name=args.modelname)
#     multi_c51.load_agent('test_agent')
    # test(multi_c51, False)

def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))


def calculate_quantile_huber_loss(td_errors, taus, weights=None, kappa=1.0):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (
        batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
        dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    if weights is not None:
        quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
    else:
        quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss


def evaluate_quantile_at_action(s_quantiles, actions):
        assert s_quantiles.shape[0] == actions.shape[0]

        batch_size = s_quantiles.shape[0]
        N = s_quantiles.shape[2]

        # Expand actions into (batch_size, 1, N).
        action_index = actions[:,:, None].expand(batch_size, 1, N)

        # Calculate quantile values at specified actions.
        sa_quantiles = s_quantiles.gather(dim=1, index=action_index)

        return sa_quantiles

if __name__ == "__main__":
    print(args)
    if args.train:
        train()
    # os.system('shutdown -s -t 0')
