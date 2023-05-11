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
parser.add_argument("--modelname", type=str, default='hardtaskc51', help="saving model name")
parser.add_argument("--dataset", type=int, default=0, help="choose the model")
parser.add_argument("--eps", type=float, default=0.33, help="set the epsilon")
parser.add_argument("--gamma", type=float, default=0.99, help="set the gamma")
parser.add_argument("--Lr", type=float, default=0.1, help="set the learning rate")
parser.add_argument("--cap", type=int, default=20000, help="the capability of the memory buffer")
parser.add_argument("--step", type=int, default=100, help="the frequency of training")
parser.add_argument("--freq", type=int, default=100, help="the frequency of update the model")
parser.add_argument("--episode", type=int, default=10000, help="set episode rounds")
parser.add_argument("--verbose", action='store_true', default=False, help="print verbose test process")
parser.add_argument("--GPU", action="store_true", default=False, help="use cuda core")
parser.add_argument("--batchsize", type=int, default=100, help="learning batchsize")
parser.add_argument("--randstart", action='store_false', default=True, help="random start from any state")
parser.add_argument("--network", action='store_true', default=False)
parser.add_argument("--samplenum", type=int, default=10)


args = parser.parse_args()
# in tabular case state=30, actions=5, agents=3
tl_init = tabel_lr = args.Lr
nl_init = network_lr = args.Lr

test_flg = False



class Q_table(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Q_table, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.Linear = nn.Linear(n_states, n_actions, bias=False)

    def forward(self, state):
        par = self.Linear(torch.tensor(state, dtype=torch.float32))
        return par


class Agent:
    def __init__(self, n_states, n_actions, eps, gamma, idx):
        self.n_states = n_states
        self.n_actions = n_actions
        self.eps = eps
        self.gamma = gamma
        self.idx = idx
        self.Q = Q_table(n_states, n_actions)
        self.target_Q = Q_table(n_states, n_actions)
        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr=tabel_lr)

    def save_checkpoint(self, folder):
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
            return rand_argmax(Q)

    def update_target_model(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

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
        q_next = self.Q(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        a_next = q_next.argmax(dim=-1,keepdim=True)
        q_next = self.target_Q(b_s_).detach().gather(1, a_next)

        b_r = torch.from_numpy(b_r).type(torch.float32)
        q_target = b_r + self.gamma * q_next  # shape (batch)
        # q_target = q_target[:, None]
        # print("q_target:{}".format(q_target))
        loss = F.mse_loss(q_eval, q_target)
        # print(loss)
        # 计算, 更新 eval net
        Q_prev = self.Q.Linear.weight.clone().detach()
        self.optimizer_Q.zero_grad()
        loss.backward()  # 误差反向传播
        self.optimizer_Q.step()
        Q_new = self.Q.Linear.weight.clone().detach()
        return F.l1_loss(Q_prev, Q_new)
    # def train_replay_iql(self, memory, batch_size):
    #     num_samples = min(batch_size, len(memory))
    #     replay_samples = random.sample(memory, num_samples)
    #     # Project Next State Value Distribution (of optimal action) to Current State
    #     b_s = [sample['s'] for sample in replay_samples]
    #     b_r = [sample['r'] for sample in replay_samples]
    #     b_a = [sample['a'] for sample in replay_samples]
    #     b_s_ = [sample['s_'] for sample in replay_samples]
    #     b_s = np.array(b_s)
    #     b_r = np.array(b_r)
    #     b_s_ = np.array(b_s_)
    #     b_a = torch.LongTensor(b_a)
    #     b_a = b_a[:, self.idx].unsqueeze(1)
    #     # print('{} {}'.format(self.Q(b_s).shape, b_a[:, self.idx].unsqueeze(1).shape))
    #     q_eval = self.Q(b_s).gather(1, b_a)  # shape (batch, 1)

    #     # print('original:\n{}\n chosen:\n{}\n index:\n{}'.format(self.Q(b_s), q_eval, b_a[:, self.idx]))
    #     # print("q_eval shape {}".format(q_eval.dtype))
    #     q_next = self.target_Q(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
    #     b_r = torch.from_numpy(b_r).type(torch.float32)
    #     q_target = b_r + self.gamma * q_next.max(1)[0]  # shape (batch)
    #     q_target = q_target[:, None]
    #     # print("q_target:{}".format(q_target))
    #     """
    #     print(q_target)
    #     print(q_next)
    #     """
    #     loss = F.mse_loss(q_eval, q_target)
    #     # print(loss)
    #     # 计算, 更新 eval net
    #     Q_prev = self.Q.Linear.weight.clone().detach()
    #     self.optimizer_Q.zero_grad()
    #     loss.backward()  # 误差反向传播
    #     self.optimizer_Q.step()
    #     Q_new = self.Q.Linear.weight.clone().detach()
    #     return F.l1_loss(Q_prev, Q_new)

    def rand_peek(self):
        x = np.zeros([self.n_states])
        state = np.random.randint(0, self.n_states)
        x[state] = 1
        x = torch.FloatTensor(x).reshape(1, -1)
        y = self.model(x).squeeze()
        return "for state {},\nQ is {}\n".format(state, torch.sum(y * torch.FloatTensor(self.Z), dim=1))


    def load(self, folder):
        self.Q.load_state_dict(torch.load(folder + '/iql_agent{}_run{}.pkl'.format(self.idx, run_num)))

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
    agents = []
    memory = deque()

    def __init__(self, n_agents, n_states, n_actions, utf, eps, gamma, batch_size=32,
                 alpha=0.001, max_memory=50000, model_name='multi_c51'):
        self.n_agents = n_agents
        self.n_actions = n_agents
        self.n_states = n_agents
        self.batch_size = batch_size
        for i in range(n_agents):
            self.agents.append(Agent(n_states, n_actions, eps, gamma, i))
        self.max_memory = max_memory
        self.update_target_freq = utf
        self.model_name = model_name

    def get_joint_iql_action(self, state):
        actions = [agent.get_iql_action(state) for agent in self.agents]
        return actions


    def store_transition(self, s, a, r, s_, done):
        self.memory.append({'s': s, 'a': a, 'r': r, 's_': s_, 'done': done})
        if len(self.memory) > self.max_memory:
            self.memory.popleft()

    def update_target_models(self):
        # print("updating")
        for agent in self.agents:
            agent.update_target_model()

    def save_checkpoint(self, folder_name):
        Folder = 'logs/' + folder_name
        if not os.path.exists(Folder):  # 是否存在这个文件夹
            os.makedirs(Folder)
        Folder += '/' + str(self.model_name)
        if not os.path.exists(Folder):
            os.makedirs(Folder)
        for agent in self.agents:
            agent.save_checkpoint(Folder)

    def load_agent(self, folder_name):
        for agent in self.agents:
            agent.load(folder_name)

    def train_replay_iql(self):
        st_time = time.time()
        q_judge = 0
        for agent in self.agents:
            # print("enter agent" + str(agent.idx) + " !!!")
            # agent.train_replay(self.memory, self.batch_size)
            q_judge += agent.train_replay_iql(self.memory, self.batch_size)

        global training_time
        training_time += time.time() - st_time
        return q_judge / self.n_agents


    def test_iql_opt_action(self, state):
        actions = [agent.test_iql_opt_action(state) for agent in self.agents]
        return actions

    def generate_pi_iql(self):
        return [agent.generate_pi_iql() for agent in self.agents]


def rand_argmax(tens):
    max_idxs, = torch.where(tens == tens.max())
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
    env = Env.chooce_the_game(args.dataset, True, False)
    multi_c51 = Multi_C51(n_agents=env.agent_num, n_states=env.state_num, n_actions=env.action_num,
                          utf=args.freq, eps=args.eps, gamma=args.gamma,
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
    q_judge = 0
    global tabel_lr
    file = open("{}/result_run{}.txt".format(Folder, run_num), 'a')
    pi_prev = multi_c51.generate_pi_iql()
    for i in range(max_episode):
        s = env.reset()
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
            s = 'iql mean reward is {}\nr_list:{}\n'.format(ep_r, r_list)
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
            plt.savefig(Folder + '/result_run{}'.format(run_num))
            plt.close()
    file.close()


def test_agent():
    env = Env.chooce_the_game(args.dataset, args.randstart, args.determine)
    multi_c51 = Multi_C51(n_agents=env.agent_num, n_states=env.state_num, n_actions=env.action_num,
                          utf=args.freq, eps=args.eps, gamma=args.gamma,
                          max_memory=args.cap, alpha=args.Lr, batch_size=args.batchsize, model_name=args.modelname)
    multi_c51.load_agent('test_agent')
    # test(multi_c51, False)


if __name__ == "__main__":
    print(args)
    if args.train:
        train()
    # os.system('shutdown -s -t 0')
