from runner import Runner
# from smac.env import StarCraft2Env
from common.arguments import  get_ippo_args,get_aga_args,get_maac_args, get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
# from utils.env_wrappers import SubprocVecEnv
# from utils.make_env import make_sc2_env
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from pathlib import Path
import pickle
import os
import copy
from collections import deque
# def make_parallel_env(args, n_rollout_threads):
#     seed = args.seed
#     def get_env_fn(rank):
#         def init_env():
#             env = make_sc2_env(args,rank)
#             np.random.seed(seed + rank * 1000)
#             return env
#         return init_env
#     if n_rollout_threads == 1:
#         return DummyVecEnv([get_env_fn(0)])
#     else:
#         return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
def get_element(array,index):
    ret = array
    # print('array_shape = {}'.format(np.shape(array)))
    # print('index = {}'.format(index))
    for x in index:
        ret = ret[x]
        # print('x = {}, ret_shape = {}'.format(x,np.shape(ret)))
    return ret
def make_not_exist_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class MatrixGame:
    def __init__(self,r_mat,trans_mat,init_state,end_state,max_episode_length,evaluate_mat=False):
        r_shape = np.shape(r_mat)
        self.r_mat = r_mat
        self.trans_mat = trans_mat
        self.state_num = r_shape[0]
        self.agent_num = len(r_shape) - 1
        self.action_num = r_shape[1]
        self.init_state = init_state
        self.now_state  = init_state
        self.step_count = 0
        self.end_state = end_state
        self.max_episode_length = max_episode_length
        self.state_action_count = np.zeros_like(r_mat).reshape([self.state_num,-1])
        self.long_step_count = 0
        self.evaluate_mat = evaluate_mat
        self.traverse = 0
        self.abs_traverse = 0
        self.relative_traverse = 0
    def eval_traverse(self):
        # print('state_action_count = {}'.format(self.state_action_count))
        print('long_step_count = {}'.format(self.long_step_count))
        covered_count = (self.state_action_count > 0).sum()
        all_state_action = self.state_action_count.shape[0] * self.state_action_count.shape[1]
        traverse = covered_count / all_state_action
        max_traverse = min(self.long_step_count / all_state_action, 1)
        relative_traverse = covered_count / self.long_step_count
        print('abs_traverse = {} max_traverse = {} relative_traverse = {}'.format(traverse, max_traverse,
                                                                                  relative_traverse))
        freq_mat = self.state_action_count / self.long_step_count
        freq_mat = freq_mat.reshape(self.r_mat.shape)
        static_return = (freq_mat * self.r_mat).sum()
        print('static_return = {}'.format(static_return))
        self.traverse = traverse
        self.abs_traverse = max_traverse
        self.relative_traverse = relative_traverse
        return self.traverse, self.abs_traverse,self.relative_traverse,static_return
    def reset(self):
            self.now_state = self.init_state
            self.step_count = 0
            ret = np.zeros(self.state_num)
            ret[self.now_state] = 1
            return ret
    def reset_evaluate(self):
        self.long_step_count = 0
        self.state_action_count = np.zeros_like(self.state_action_count)
    def get_obs(self):
        obs = []
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        for i in range(self.agent_num):
            obs.append(state)
        return obs
    def get_ac_idx(self,action):
        idx = 0
        for a in action:
            idx = self.action_num * idx + a
            # print('idx = {} a = {}'.format(idx,a))
        return idx
    def get_state(self):
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        return state
    def step(self,action,evaluate=False):
        # print('step = {} action  = {}'.format(self.step_count))
        sa_index = []
        sa_index.append(self.now_state)
        # sa_index += action
        action = np.array(action)
        for a in action:
            sa_index.append(a)
        # print('sa_index = {},action = {}'.format(sa_index, action))
        if not evaluate:
            ac_idx = self.get_ac_idx(action)
            self.state_action_count[self.now_state,ac_idx] += 1
            self.long_step_count += 1

        r = get_element(self.r_mat,sa_index)
        next_s_prob = get_element(self.trans_mat,sa_index)
        # print('sa_index = {} next_s_prob = {}'.format(sa_index,next_s_prob))
        next_state = np.random.choice(range(self.state_num),size = 1, p = next_s_prob)[0]
        self.now_state = next_state
        self.step_count += 1

        done = self.end_state[self.now_state]
        if self.step_count >= self.max_episode_length:
            done = 1
        return r,done,None
    def get_env_info(self):
        env_info = {}
        env_info["n_actions"] = self.action_num
        env_info["n_agents"] = self.agent_num
        env_info["state_shape"] = self.state_num
        env_info["obs_shape"] = self.state_num
        env_info["episode_limit"] = self.max_episode_length
        return env_info
    def get_avail_agent_actions(self,agent_id):
        return np.ones(self.action_num)
    def get_model_info(self,state,action):
        sa_index = []
        sa_index.append(state)
        action = np.array(action)
        # print('action = {}'.format(action))
        for a in action:
            sa_index.append(a)
        r = get_element(self.r_mat, sa_index)
        next_s_prob = get_element(self.trans_mat, sa_index)
        # print('action = {} sa_index = {} self.trans_mat = {} next_s_prob = {}'.format(action,sa_index,self.trans_mat.shape, next_s_prob.shape  ))
        return r,next_s_prob
    def close(self):
        return
def make_mat_game_one():
    r_mat = np.zeros([1,3,3])
    # r_mat[0] = np.array([8,-12,-12,-12,0,0,-12,0,0]).reshape([3,3])
    r_mat[0] = np.array([-12, 8, -12, 0, -12, 0, 0, -12, 0]).reshape([3, 3])
    # r_mat[0] = np.array([5, 5, 5, 5, 5, -100, -100, -100, 8]).reshape([3, 3])
    trans_mat = np.ones([1,3,3,1])
    end_state = np.zeros([1])
    max_episode_length = 1
    init_state = 0
    env = MatrixGame(r_mat,trans_mat,init_state,end_state,max_episode_length)
    return env
def make_mat_game_two():
    r_mat = np.zeros([2, 2, 2])
    r_mat[0] = np.array([1,0,0,0]).reshape([2, 2])
    trans_mat = np.zeros([2, 2, 2, 2])
    trans_mat[0, :, :, 0] = np.array([1, 1, 1, 0]).reshape([2, 2])
    trans_mat[0, :, :, 1] = np.array([0, 0, 0, 1]).reshape([2, 2])
    trans_mat[1, :, :, 0] = np.array([0, 0, 0, 0]).reshape([2, 2])
    trans_mat[1, :, :, 1] = np.array([1, 1, 1, 1]).reshape([2, 2])
    end_state = np.zeros([2])
    max_episode_length = 100
    init_state = 0
    env = MatrixGame(r_mat, trans_mat, init_state, end_state, max_episode_length)
    return env
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  #为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed) #为当前GPU设置随机种子；  　
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def make_mat_game_from_file(filename,longer=False):
    with open(filename,'rb') as f:
        matrix_para = pickle.load(f)
        r_mat = matrix_para['reward']

        trans_mat = matrix_para['trans_mat']
        end_state = np.zeros(np.shape(r_mat)[0])
        state_num = np.shape(r_mat)[0]
        if longer:
            if filename.startswith('random_matrix_game_3'):
                max_episode_length = 40
            else:
                max_episode_length = int(state_num *1.34)
        else:
            max_episode_length = state_num
        init_state = 0
        env = MatrixGame(r_mat, trans_mat, init_state, end_state, max_episode_length,evaluate_mat=True)
        return env

def save_and_plot(args,plot_dir,save_prefix,iter_list,check_val_list,Q_check_list,pi_check_list):
    make_not_exist_dir(plot_dir)
    plt.title('{}'.format(args.map))
    plt.xlabel('iteration')
    plt.ylabel('mean episode rewards')
    plt.plot(iter_list, check_val_list)
    plt.savefig('{}_meap_episode_rewards.png'.format(save_prefix), format='png')
    plt.clf()

    plt.title('{}'.format(args.map))
    plt.xlabel('iteration')
    plt.ylabel('Q gap')
    plt.plot(iter_list, Q_check_list)
    plt.savefig('{}_Q_gap.png'.format(save_prefix), format='png')
    plt.clf()

    plt.title('{}'.format(args.map))
    plt.xlabel('iteration')
    plt.ylabel('pi gap')
    plt.plot(iter_list, pi_check_list)
    plt.savefig('{}_pi_gap.png'.format(save_prefix), format='png')
    plt.clf()

    with open('{}_iter.pkl'.format(save_prefix), 'wb') as f:
        pickle.dump(iter_list, f)
    with open('{}_check_val.pkl'.format(save_prefix), 'wb') as f:
        pickle.dump(check_val_list, f)
    with open('{}_Q_gap.pkl'.format(save_prefix), 'wb') as f:
        pickle.dump(Q_check_list, f)
    with open('{}_pi_gap.pkl'.format(save_prefix), 'wb') as f:
        pickle.dump(pi_check_list, f)

if __name__ == '__main__':

    args = get_common_args()


    # args = get_coma_args(args)
    # print('args = {}'.format(args))

    args.map = 'random_matrix_game_3'
    prefix = 'converged'
    data_file_name = '{}_matrix_data.txt'.format(prefix)
    save_file_name = '{}_true_A.txt'.format(prefix)
    if args.map == 'random_matrix_game_1':
        env = make_mat_game_one()
    else:
        env = make_mat_game_from_file('{}.pkl'.format(args.map))
    env_info = env.get_env_info()
    print('env_info = {}'.format(env_info))
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    if args.alg.find('coma') > -1:
        print('COMA??????????????????????????')
        args = get_coma_args(args)
    elif args.alg.find('maac') > -1:
        print('MAAC??????????????????????????')
        args = get_g2anet_args(args)
        args.hard = False
    elif args.alg.find('ippo') > -1:
        args = get_ippo_args(args)
    elif args.alg.find('central_v') > -1:
        print('central_v??????????????????????????')
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        print('reinforce??????????????????????????')
        args = get_reinforce_args(args)
    elif args.alg.find('aga') > -1:
        args = get_aga_args(args)
    else:
        print('mixer_args !!!!!!!!!!!!!!!!!!!!!!!')
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        print('commnet !!!!!!!!!!!!!!!!!!!!!!!')
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1 or args.alg.find('gaac')> -1:
        print('gaac !!!!!!!!!!!!!!!!!!!!!!!')
        args = get_g2anet_args(args)
    print('after_args = \n{}'.format(args))







    rnn_dim_list = [64]
    qmix_dim_list = [32]


    n_actions = args.n_actions
    n_agents = args.n_agents
    n_states = args.state_shape
    gamma = args.gamma
    dp_eps = 1e-2
    Q = np.zeros([n_agents, n_states, n_actions])
    pi = np.zeros([n_agents, n_states,n_actions])
    A_from_data = np.zeros([n_agents, n_states, n_actions])
    with open(data_file_name,'r') as f:
        all_data = f.read()
        start = 0
        for a in range(n_agents):
            for s in range(n_states):
                i1 = all_data.find('[',start)
                i2 = all_data.find(']',start)
                # print('all_data = {}'.format(all_data))
                # print('start = {} i1 = {} , i2 = {} , str = {}'.format(start,i1,i2,all_data[i1:i2+1]))
                A_str = all_data[i1:i2+1]

                start = i2 + 1

                i1 = all_data.find('[', start)
                i2 = all_data.find(']', start)
                pi_str = all_data[i1:i2 + 1]
                # print('all_data = {}'.format(all_data))
                # print('start = {} i1 = {} , i2 = {} , str = {}'.format(start, i1, i2, all_data[i1:i2 + 1]))
                start = i2 + 1
                print('state {} agent {}: A = {} pi = {}'.format(s,a, A_str,pi_str))
                A_from_data[a,s] = np.array(eval(A_str))
                pi[a, s] = np.array(eval(pi_str))
        print('all_data = {}'.format(all_data))
    maxloop = 1
    check_interval = 1
    check_episode = 20
    iter_list = []
    check_val_list = []
    Q_check_list = []
    pi_check_list = []

    for iter in range(maxloop):
        print('iteration {}'.format(iter))
        Q_for_check = copy.deepcopy(Q)
        pi_for_check = copy.deepcopy(pi)
        print('update for policy in aga')
        dp_judge = 100000000000000
        s_idx = np.arange(n_states)
        # print('s_idx = {}'.format(s_idx))
        all_ac_count = n_actions ** (n_agents - 1)
        all_base_ac = np.zeros([all_ac_count, n_agents - 1],dtype = np.int32)
        for idx in range(all_ac_count):
            cnt = n_agents - 2
            all_a = idx
            while all_a > 0:
                # print('cnt = {}, all_a = {}'.format(cnt,all_a))
                all_base_ac[idx,cnt] = (all_a % n_actions)
                all_a = all_a // n_actions
                cnt -= 1
            # print('all_base_ac[{}] = {}'.format(idx,all_base_ac[idx]))

        while dp_judge >= dp_eps:
            new_Q = np.zeros([n_agents, n_states, n_actions])
            for i in range(n_agents):
                next_V = np.zeros(n_states)
                for s in range(n_states):
                    next_V[s] = np.dot(Q[i,s],pi[i,s])
                for s in range(n_states):
                    for a in range(n_actions):
                        for all_a in range(all_ac_count):
                            pi_other = 1

                            base_ac_i = all_base_ac[all_a]
                            # print('before insert ac = {}'.format(base_ac_i))
                            base_ac_i = np.insert(base_ac_i,i,a)
                            # print('after insert ac = {}'.format(base_ac_i))
                            for j in range(n_agents):
                                if j == i:
                                    continue

                                pi_other *= pi[j,s,base_ac_i[j] ]
                            r, p = env.get_model_info(s, base_ac_i)
                            # print('next_Q = {} pi[i] = {} p = {} np.dot(p, next_Q) = {}'.format(next_Q,pi[i],p,np.dot(p, next_Q)))
                            new_Q[i, s, a] += pi_other * (r + gamma * (np.dot(p, next_V)))
            dp_judge = np.linalg.norm(Q - new_Q)
            Q = new_Q
            print('dp_judge_policy = {}'.format(dp_judge))


    with open(save_file_name,'w') as f:
        baseline = (Q * pi).sum(axis = -1,keepdims=True)
        true_A = Q - baseline
        flag = 0
        for s in range(n_states):
            for a in range(n_agents):
                output_str = "state {} agent {}::\ntrue A = {}, Q = {}, baseline = {}\nargmax = {}\npi = {}\nargmax = {}\ndata A = {}\nargmax = {}\ndist = {}\n\n".format(s,a,true_A[a,s],Q[a,s],baseline[a,s], np.argmax(true_A[a,s]),pi[a,s],np.argmax(pi[a,s]),A_from_data[a,s],np.argmax(A_from_data[a,s]),np.linalg.norm(true_A[a,s]-A_from_data[a,s]))
                f.write(output_str)
                flag += (np.argmax(true_A[a,s]) == np.argmax(pi[a,s]) ) & (np.argmax(A_from_data[a,s]) == np.argmax(pi[a,s]) )
                # print('dp_judge = {}'.format(dp_judge))
        f.write('all_dist = {}\n'.format(np.linalg.norm(true_A-A_from_data)))
        f.write('consist number = {}'.format(flag))