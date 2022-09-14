import yaml
import torch
import pickle
import random
from multiprocessing import Process, Queue
import numpy as np
import os
from datetime import datetime
# from tqdm.auto import trange
import argparse

# import gym

# from make_env import make_env

# from utils.set_random_seed import set_random_seed
# from utils.builder import build_lmba,build_mba,build_mfa,build_cmba
# from agent import PolicyAgent
parser = argparse.ArgumentParser(description='Put in config file path')
parser.add_argument('--path', default='./config/initial7.yaml', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--use_model', action='store_true')
parser.add_argument('--use_latent', action='store_true')
parser.add_argument('--use_cvae', action='store_true')
parser.add_argument('--device', default='cuda', type=str)

args = parser.parse_args()
#build_agent = build_cmba if args.use_cvae else (
#    build_lmba if args.use_latent else (build_mba if args.use_model else build_mfa))

with open(args.path, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

config['device'] = args.device


def get_element(array, index):
    ret = array
    # print('array_shape = {}'.format(np.shape(array)))
    # print('index = {}'.format(index))
    for x in index:
        ret = ret[x]
        # print('x = {}, ret_shape = {}'.format(x,np.shape(ret)))
    return ret


def make_mat_game_from_file(filename):
    with open(filename, 'rb') as f:
        matrix_para = pickle.load(f)
        r_mat = matrix_para['reward']
        trans_mat = matrix_para['trans_mat']
        end_state = np.zeros(np.shape(r_mat)[0])
        state_num = np.shape(r_mat)[0]
        max_episode_length = 40
        init_state = 0
        env = MatrixGame(r_mat, trans_mat, init_state, end_state, max_episode_length, evaluate_mat=True)
        return env


class MatrixGame:
    def __init__(self, r_mat, trans_mat, init_state, end_state, max_episode_length, evaluate_mat=False):
        r_shape = np.shape(r_mat)
        self.r_mat = r_mat
        self.trans_mat = trans_mat
        self.state_num = r_shape[0]

        self.agent_num = len(r_shape) - 1
        self.action_num = r_shape[1]
        self.share_observation_space = []
        self.observation_space = []
        self.action_space = []

        self.init_state = init_state
        self.now_state = init_state
        self.step_count = 0
        self.end_state = end_state
        self.max_episode_length = max_episode_length

    def reset(self):
        self.now_state = self.init_state
        self.step_count = 0
        s = self.get_state()
        return s

    def get_ac_idx(self, action):
        idx = 0
        for a in action:
            idx = self.action_num * idx + a
            # print('idx = {} a = {}'.format(idx,a))
        return idx

    def get_state(self):
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        return state

    def step(self, action, evaluate=False):
        # print('step = {} action  = {}'.format(self.step_count))
        sa_index = []
        sa_index.append(self.now_state)
        # sa_index += action
        # action = np.array(action)
        for a in action:
            sa_index.append(np.argmax(a))
        # print('sa_index = {},action = {}'.format(sa_index, action))
        r = get_element(self.r_mat, sa_index)
        next_s_prob = get_element(self.trans_mat, sa_index)
        # print('sa_index = {} next_s_prob = {}'.format(sa_index,next_s_prob))
        next_state = np.random.choice(range(self.state_num), size=1, p=next_s_prob)[0]
        self.now_state = next_state
        self.step_count += 1

        return self.get_state(), r, self.step_count >= self.max_episode_length


def interact(queue1, queue2, idx):
    set_random_seed(args.seed + idx * 10)
    env = make_mat_game_from_file('a_mat_model/random_matrix_game_3.pkl')
    # env = gym.make(config['env_name'])
    # env.seed(args.seed+idx*10)
    queue1.put(env.reset())

    while True:
        action = queue2.get()
        if isinstance(action, bool):
            observation = env.reset()
            queue1.put(observation)
            action = queue2.get()
        observation, reward, done = env.step(action)
        back_observation = env.reset() if done else observation
        queue1.put((observation, reward, done, back_observation))


# def agent_interact(queue1,queue2,idx):
#     set_random_seed(args.seed+idx*10)
#     agent = build_agent(config,idx)

#     while True:
#         data = queue1.get()
#         if isinstance(data,np.ndarray):
#             action = agent.interact(data)
#             queue2.put(action)
#         elif isinstance(data, tuple):
#             agent.result(*data)
#         else:
#             agent.round_done()

def eff_to_full_action(action):
    if len(action.shape) == 1:
        ans = np.zeros(config['full_action_dim'])
        for i, j in enumerate(config['action_mask']):
            ans[j] = action[i]
    elif len(action.shape) == 2:
        ans = np.zeros((action.shape[0], config['full_action_dim']))
        for i, j in enumerate(config['action_mask']):
            ans[:, j] = action[:, i]
    return ans


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


"""
if __name__ == '__main__':
    set_random_seed(args.seed)
    time = datetime.now()
    save_path = './result/{}_{}_{}_{}_{}_{}/'.format(time.month,time.day,time.hour,time.minute,time.second,args.seed)
    record_path = './train_log/{}_{}_{}_{}_{}_{}/'.format(time.month,time.day,time.hour,time.minute,time.second,args.seed)
    
    num_stable_agent = config['stable_agent']
    num_learning_agent = config['learning_agent']
    num_worker = config['worker']
    config['save_path'] = save_path
    config['record_path'] = record_path

    episode_length = config['episode_length']
    num_episode = config['episode']
    num_round = config['round']

    print('preserved at dir' + save_path +' and log file is record in '+ record_path)
    if not os.path.isdir(record_path):
            os.makedirs(record_path)
    
    with open(record_path+'config.yaml','w') as f:
        yaml.dump(config,f)
    

    agent = [PolicyAgent() for _ in range(num_stable_agent)]+[build_agent(config,i) for i in range(num_learning_agent)]
    # agent[0].policy.load('result/1_9_22_50_7_1/ppo0/round=4901')
    env_process = []
    env_queue = []
    
    observations = []
    for i in range(num_worker):
        env_queue.append([Queue(),Queue()])
        env_process.append(Process(target=interact,args=(env_queue[-1][0],env_queue[-1][1],i)))
        env_process[-1].start()
        observation = env_queue[-1][0].get()
        observations.append(observation)
        
    observations = np.array(observations)
    back_observations = observations
    
    # agent_process = []
    # agent_queue = []

    # for i in range(num_learning_agent):
    #     agent_queue.append([Queue(),Queue()])
    #     agent_process.append(Process(target=agent_interact,args=(agent_queue[-1][0],agent_queue[-1][1],i)))
    #     agent_process[-1].start()
        
    for i in range(num_round):
        print('\n\nRound {}'.format((i)))    
        
        with trange(200,desc='collecting env data') as steps:
            
            total_reward = 0
            observations = np.array(back_observations)

            for step in steps:

                actions = []
                next_observations = []
                rewards = []
                dones = []
                
                for j in range(len(agent)):
                    action = agent[j].interact(observations)
                    actions.append(action)
                
                # action1 = np.concatenate([a[:,:1] for a in actions],axis=-1)
                # action2 = np.concatenate([a[:,1:] for a in actions],axis=-1)
                # actions = np.concatenate((action1,action2.mean(-1)[:,np.newaxis]),axis=-1)               
                actions = np.stack(actions,axis=-2)
                observations = []
                for j in range(num_worker):
                    env_queue[j][1].put(actions[j])
                    next_observation,reward,done,back_observation = env_queue[j][0].get()
                    
                    observations.append(back_observation)
                    next_observations.append(next_observation)
                    rewards.append([reward])
                    dones.append([done])
                
                next_observations = np.array(next_observations)
                observations = np.array(observations)
                rewards = np.array(rewards)
                dones = np.array(dones)
                total_reward += rewards.mean()
                
                for j in range(num_stable_agent,len(agent)):
                    agent[j].result(next_observations,rewards,dones)
                
                steps.set_postfix(rewards = np.mean(rewards) ,avg_episode_reward = total_reward/(step+1))
                
        for j in range(num_stable_agent,len(agent)):
            agent[j].round_done()

        with open(record_path+'main.log','a') as f:
            f.write('{}\n'.format(total_reward/200))
        
    for i in range(config['worker']):
        env_process[i].terminate()
    
    for i in range(num_learning_agent):
        agent[i].final()
"""
