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

    init_seed = args.seed
    for i in range(args.total_run_num):
        args.seed = init_seed + i
        set_seed(args.seed)
        args.mat_test = True
        if not args.draw_plot_for_dmac:
            args.no_rnn = True
            args.single_exp = True
        # args = get_coma_args(args)
        # print('args = {}'.format(args))
        if args.map == 'MMM2':
            args.map = 'random_matrix_game_4'
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

        rnn_dim_list = [64]
        qmix_dim_list = [32]

        if args.dp_test == 'dp':
            n_actions = args.n_actions
            n_agents = args.n_agents
            n_states = args.state_shape
            gamma = args.gamma
            dp_eps = 1e-2
            Q = np.zeros([n_agents, n_states, n_actions])
            pi = np.zeros([n_agents, n_states], dtype=np.int32)
            maxloop = 500
            check_interval = 1
            check_episode = 20
            iter_list = []
            check_val_list = []
            Q_check_list = []
            pi_check_list = []
            plot_dir = 'mat_test_plot'
            if args.aga_tag == True:
                prefix = 'aga_Q'
            else:
                prefix = 'iql'

            special_name = 'none'

            save_prefix = plot_dir + '/{}_{}_{}_{}'.format(args.dp_test, prefix, args.map, special_name)

            for iter in range(maxloop):
                print('iteration {}'.format(iter))
                Q_for_check = copy.deepcopy(Q)
                pi_for_check = copy.deepcopy(pi)

                if prefix == 'aga_Q' and False:

                    print('update for policy in aga')
                    dp_judge = 100000000000000
                    s_idx = np.arange(n_states)
                    # print('s_idx = {}'.format(s_idx))
                    while dp_judge >= dp_eps:
                        new_Q = np.zeros([n_agents, n_states, n_actions])
                        for i in range(n_agents):
                            for s in range(n_states):
                                base_ac_i = pi[:, s]
                                for a in range(n_actions):
                                    base_ac_i[i] = a
                                    r, p = env.get_model_info(s, base_ac_i)
                                    next_Q = Q[i][s_idx,pi[i]]
                                    # print('next_Q = {} pi[i] = {} p = {} np.dot(p, next_Q) = {}'.format(next_Q,pi[i],p,np.dot(p, next_Q)))
                                    new_Q[i, s, a] = r + gamma * (np.dot(p, next_Q))
                        dp_judge = np.linalg.norm(Q - new_Q)
                        Q = new_Q
                        print('dp_judge_policy = {}'.format(dp_judge))



                for i in range(n_agents):
                    print('update agent {}'.format(i))
                    dp_judge = 100000000000000
                    if prefix == 'aga_Q':
                        new_Q_i = np.zeros([n_states, n_actions])
                    elif prefix == 'iql':
                        new_Q = np.zeros([n_agents, n_states, n_actions])
                    while dp_judge >= dp_eps:
                        for s in range(n_states):
                            base_ac_i = []
                            for j in range(n_agents):
                                base_ac_i.append(pi[j, s])
                            for a in range(n_actions):
                                base_ac_i[i] = a
                                r, p = env.get_model_info(s, base_ac_i)
                                next_Q_max = np.max(Q[i, :, :], axis=-1)
                                if prefix == 'aga_Q':
                                    new_Q_i[s, a] = r + gamma * (np.dot(p, next_Q_max))
                                elif prefix == 'iql':
                                    new_Q[i, s, a] = r + gamma * (np.dot(p, next_Q_max))
                        if prefix == 'aga_Q':
                            dp_judge = np.linalg.norm(Q[i] - new_Q_i)
                            Q[i] = new_Q_i
                            pi[i] = np.argmax(Q[i, :, :], axis=-1)
                        elif prefix == 'iql':
                            dp_judge = np.linalg.norm(Q[i] - new_Q[i])
                            Q[i] = new_Q[i]

                        # print('dp_judge = {}'.format(dp_judge))
                if prefix == 'iql':
                    pi = np.argmax(Q, axis=-1)

                aga_Q_judge = np.linalg.norm(Q_for_check - Q)
                aga_pi_judge = (pi_for_check != pi).sum()
                print('Q_judge = {}, pi_judge = {}'.format(aga_Q_judge, aga_pi_judge))
                # if aga_pi_judge == 0:
                #     break
                if iter % check_interval == 0:
                    check_val = 0
                    for e in range(check_episode):
                        env.reset()
                        done = False
                        r_list = []
                        while not done:
                            s = env.get_state()
                            s = np.argmax(s)
                            action = pi[:, s]
                            # print('action = {}'.format(action))
                            r, done, _ = env.step(action)
                            r_list.append(r)
                        check_val += np.array(r_list).sum()
                    check_val /= check_episode

                    iter_list.append(iter)
                    check_val_list.append(check_val)
                    pi_check_list.append(aga_pi_judge)
                    Q_check_list.append(aga_Q_judge)

                    save_and_plot(args,plot_dir,save_prefix,iter_list,check_val_list,Q_check_list,pi_check_list)

        elif args.dp_test == 'dp_optimal':
            n_actions = args.n_actions
            n_agents = args.n_agents
            n_states = args.state_shape
            gamma = args.gamma
            dp_eps = 1e-2
            n_joint_actions = n_actions ** n_agents
            Q = np.zeros([ n_states, n_joint_actions ])
            pi = np.zeros([n_states], dtype=np.int32)
            maxloop = 500
            check_interval = 1
            check_episode = 20
            iter_list = []
            check_val_list = []
            Q_check_list = []
            pi_check_list = []
            plot_dir = 'mat_test_plot'
            prefix = 'aga_Q'

            def joint_to_idv(joint_a,n_actions = n_actions,n_agents = n_agents):
                res = []
                while joint_a > 0:
                    r = joint_a % n_actions
                    joint_a = joint_a // n_actions
                    res.append(r)
                if len(res) < n_agents:
                    for i in range(len(res),n_agents):
                        res.append(0)
                reversed(res)
                return res

            special_name = 'none'

            save_prefix = plot_dir + '/{}_{}_{}_{}'.format(args.dp_test, prefix, args.map, special_name)

            for iter in range(maxloop):
                print('iteration {}'.format(iter))
                Q_for_check = copy.deepcopy(Q)
                pi_for_check = copy.deepcopy(pi)


                dp_judge = 100000000000000
                new_Q = np.zeros_like(Q)
                value_iter = 0
                while dp_judge >= dp_eps:
                    for s in range(n_states):
                        for joint_a in range(n_joint_actions):
                            idv_a = joint_to_idv(joint_a)
                            r, p = env.get_model_info(s, idv_a)
                            next_Q_max = np.max(Q, axis=-1)
                            new_Q[s, joint_a] = r + gamma * (np.dot(p, next_Q_max))
                    dp_judge = np.linalg.norm(Q - new_Q)
                    Q = new_Q

                    print('value_iter = {} dp_judge = {}'.format(value_iter,dp_judge))
                    value_iter += 1
                pi = np.argmax(Q, axis=-1)

                aga_Q_judge = np.linalg.norm(Q_for_check - Q)
                aga_pi_judge = (pi_for_check != pi).sum()
                print('Q_judge = {}, pi_judge = {}'.format(aga_Q_judge, aga_pi_judge))
                # if aga_pi_judge == 0:
                #     break
                if iter % check_interval == 0:
                    check_val = 0
                    for e in range(check_episode):
                        env.reset()
                        done = False
                        r_list = []
                        while not done:
                            s = env.get_state()
                            s = np.argmax(s)
                            joint_a = pi[s]
                            action = joint_to_idv(joint_a)
                            # print('action = {}'.format(action))
                            r, done, _ = env.step(action)
                            r_list.append(r)
                        check_val += np.array(r_list).sum()
                    check_val /= check_episode

                    iter_list.append(iter)
                    check_val_list.append(check_val)
                    pi_check_list.append(aga_pi_judge)
                    Q_check_list.append(aga_Q_judge)

                    save_and_plot(args,plot_dir,save_prefix,iter_list,check_val_list,Q_check_list,pi_check_list)

        elif args.dp_test == 'tabular':
            n_actions = args.n_actions
            n_agents = args.n_agents
            n_states = args.state_shape
            gamma = args.gamma
            dp_eps = 1e-2
            Q = np.zeros([n_agents, n_states, n_actions])
            pi = np.zeros([n_agents, n_states], dtype=np.int32)


            check_interval = 1
            check_episode = 20
            iter_list = []
            check_val_list = []
            Q_check_list = []
            pi_check_list = []
            plot_dir = 'mat_test_plot'
            if args.aga_tag == True:
                prefix = 'aga_Q'
            else:
                prefix = 'iql'



            epsilon_reset = True
            lr_reset = True
            special_name = 'er_and_lr_seed_{}'.format(args.seed)

            save_prefix = plot_dir + '/{}_{}_{}_{}'.format(args.dp_test,prefix,args.map,special_name)

            if prefix == 'iql':
                buffer_size = 100000
            elif prefix == 'aga_Q':
                buffer_size = 5000

            sample_steps = 5000
            mini_batch_num = 10
            batch_size = sample_steps // mini_batch_num

            buffer = {}
            key_list = ['s','a','r','s_next','done']
            # print('buffer_size = {}'.format(buffer_size))
            for key in key_list:
                buffer[key] = deque(maxlen=buffer_size)

            maxloop = 500

            epsilon = args.epsilon
            anneal_epsilon = args.anneal_epsilon
            min_epsilon = args.min_epsilon

            count_table = np.zeros_like(Q)

            for iter in range(maxloop):
                env.reset()
                s = env.get_state()
                s = np.argmax(s)

                if prefix == 'aga_Q':
                    if epsilon_reset:
                        epsilon = args.epsilon
                        min_epsilon = args.min_epsilon
                        anneal_step = sample_steps
                        anneal_epsilon = (epsilon - min_epsilon) / anneal_step
                    if lr_reset:
                        count_table = np.zeros_like(Q)
                print('iter {} sampling ...... epsilon = {}'.format(iter,epsilon))
                for _ in range(sample_steps):
                    action = []
                    update_index = iter % n_agents
                    for i in range(n_agents):
                        if prefix == 'iql':
                            if np.random.random() < epsilon:
                                a_i = np.random.choice(np.arange(n_actions))
                            else:
                                a_i = pi[i,s]
                        elif prefix == 'aga_Q':
                            if i == update_index:
                                if np.random.random() < epsilon:
                                    a_i = np.random.choice(np.arange(n_actions))
                                else:
                                    a_i = pi[i, s]
                            else:
                                a_i = pi[i, s]
                        action.append(a_i)

                    epsilon = np.maximum(min_epsilon,epsilon - anneal_epsilon)

                    action = np.array(action,dtype=np.int32)
                    r,done,_ = env.step(action)
                    next_s = env.get_state()
                    next_s = np.argmax(next_s)

                    experience = [s,action,r,next_s,done]
                    for k,v in zip(key_list,experience):
                        buffer[k].append(v)
                    s = next_s
                print('iter {} sample has ended epsilon = {}'.format(iter,epsilon))
                for b in range(mini_batch_num):
                    print('iter {} training for batch {}'.format(iter,b))
                    curr_buffer_size = len(buffer['s'])
                    idx = np.random.choice(np.arange(curr_buffer_size,dtype = np.int32),size = batch_size,replace=False)
                    # print('idx_dtype = {}'.format(idx.dtype))
                    batch = {}
                    for key in key_list:
                        # print('key = {}, buffer_shape = {}'.format(key, np.array(buffer[key]).shape ))
                        batch[key] = np.array(buffer[key])[idx]
                    if prefix == 'iql':
                        new_Q = copy.deepcopy(Q)
                        for s,a,r,s_next,done in zip(batch['s'],batch['a'],batch['r'],batch['s_next'],batch['done']):
                            for i in range(n_agents):
                                a_i = a[i]
                                count_table[i,s,a_i] += 1
                                delta = r + gamma * np.max(new_Q[i,s_next,:]) - new_Q[i,s,a_i]
                                alpha = 1 / count_table[i,s,a_i]
                                new_Q[i,s,a_i] += alpha * delta
                        new_pi = np.argmax(new_Q,axis = -1)
                        aga_Q_judge = np.linalg.norm(new_Q - Q)
                        aga_pi_judge = (new_pi != pi).sum()

                        pi = new_pi
                        Q = new_Q

                    elif prefix == 'aga_Q':
                        new_Q_i = copy.deepcopy(Q[update_index])
                        for s, a, r, s_next, done in zip(batch['s'], batch['a'], batch['r'], batch['s_next'],
                                                         batch['done']):
                            a_i = a[update_index]
                            count_table[update_index, s, a_i] += 1
                            delta = r + gamma * np.max(new_Q_i[s_next, :]) - new_Q_i[s, a_i]
                            alpha = 1 / count_table[update_index, s, a_i]
                            new_Q_i[s, a_i] += alpha * delta
                        new_pi_i = np.argmax(new_Q_i, axis=-1)
                        aga_Q_judge = np.linalg.norm(new_Q_i - Q[update_index])
                        aga_pi_judge = (new_pi_i != pi[update_index]).sum()

                        pi[update_index] = new_pi_i
                        Q[update_index] = new_Q_i
                print('iter = {} Q_judge = {}, pi_judge = {}'.format(iter, aga_Q_judge, aga_pi_judge))
                if iter % check_interval == 0:
                    check_val = 0
                    for e in range(check_episode):
                        env.reset()
                        done = False
                        r_list = []
                        while not done:
                            s = env.get_state()
                            s = np.argmax(s)
                            action = pi[:, s]
                            # print('action = {}'.format(action))
                            r, done, _ = env.step(action)
                            r_list.append(r)
                        check_val += np.array(r_list).sum()
                    check_val /= check_episode

                    iter_list.append(iter)
                    check_val_list.append(check_val)
                    pi_check_list.append(aga_pi_judge)
                    Q_check_list.append(aga_Q_judge)

                    save_and_plot(args,plot_dir,save_prefix,iter_list,check_val_list,Q_check_list,pi_check_list)

        elif args.dp_test == 'network':
            # args.t_max = 600000
            if args.step_log_interval is None:
                args.step_log_interval = args.batch_size * args.episode_limit
            if args.aga_tag:
                args.buffer_size = args.batch_size

            for rnn_dim,qmix_dim in zip(rnn_dim_list,qmix_dim_list):
                args.rnn_hidden_dim = rnn_dim
                args.qmix_hidden_dim = qmix_dim
                runner = Runner(env, args)
                if args.learn:
                    run_i = args.seed
                    runner.run(run_i)
                else:
                    win_rate = runner.evaluate_sparse()
                    print('The win rate of {} is  {}'.format(args.alg, win_rate))
                    break
                env.close()
