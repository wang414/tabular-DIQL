from runner import Runner
# from smac.env import StarCraft2Env
from common.arguments import get_aga_args,get_maac_args, get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
# from utils.env_wrappers import SubprocVecEnv
# from utils.make_env import make_sc2_env
import torch
import numpy as np
import sys
import random
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
        self.evaluate_mat = evaluate_mat
    def reset(self):
        self.now_state = self.init_state
        self.step_count = 0
        ret = np.zeros(self.state_num)
        ret[self.now_state] = 1
        return ret
    def get_obs(self):
        obs = []
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        for i in range(self.agent_num):
            obs.append(state)
        return obs
    def get_state(self):
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        return state
    def step(self,action,evaluate=False):
        # print('step = {} action  = {}'.format(self.step_count))
        sa_index = []
        sa_index.append(self.now_state)
        sa_index += action
        r = get_element(self.r_mat,sa_index)
        next_s_prob = get_element(self.trans_mat,sa_index)
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
    def close(self):
        return
def make_mat_game_one():
    r_mat = np.zeros([1,3,3])
    # r_mat[0] = np.array([8,-12,-12,-12,0,0,-12,0,0]).reshape([3,3])
    # r_mat[0] = np.array([-12, 8, -12, 0, -12, 0, 0, -12, 0]).reshape([3, 3])
    r_mat[0] = np.array([5, 5, 5, 5, 5, -100, -100, -100, 8]).reshape([3, 3])
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


if __name__ == '__main__':
    with open('100K_target_dec_idv_aga_F_2_period_1_single_stop_other_exp_value_clip_4_steps.txt','w') as f:
        sys.stdout = f
        args = get_common_args()
        for i in range(10):
            print('################current run {}##################'.format(i +1 ))
            set_seed(i)
            args = get_aga_args(args)
            args.t_max = 100000
            args.mat_test = True
            args.mat_map = 1
            if args.mat_map == 1:
                args.map = 'matrix_game_one'
                env = make_mat_game_one()
            elif args.mat_map == 2:
                args.map = 'matrix_game_two'
                env = make_mat_game_two()
            env_info = env.get_env_info()
            args.n_actions = env_info["n_actions"]
            args.n_agents = env_info["n_agents"]
            args.state_shape = env_info["state_shape"]
            args.obs_shape = env_info["obs_shape"]
            args.episode_limit = env_info["episode_limit"]


            rnn_dim_list = [64]
            qmix_dim_list = [32]

            # if args.aga_tag:
            #     args.buffer_size = args.batch_size

            for rnn_dim,qmix_dim in zip(rnn_dim_list,qmix_dim_list):
                args.rnn_hidden_dim = rnn_dim
                args.qmix_hidden_dim = qmix_dim
                runner = Runner(env, args)
                if args.learn:
                    runner.run(i)
                else:
                    win_rate = runner.evaluate_sparse()
                    print('The win rate of {} is  {}'.format(args.alg, win_rate))
                    break
                env.close()
