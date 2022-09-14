from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_ippo_args,get_aga_args, get_maac_args, get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
from utils.env_wrappers import ParallelEnvSC2
from utils.make_env import make_sc2_env
from gym.spaces import Box, Discrete
import numpy as np
import sys
import torch
from utils.make_env import make_env
import pickle
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import argparse



def make_mpe_parallel_env(env_id, n_rollout_threads, seed,collision_penal = 0,vision = 1):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True,collision_penal = collision_penal,vision = vision)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])



def make_parallel_env(args, n_rollout_threads):
    seed = args.seed
    def get_env_fn(rank):
        def init_env():
            env = make_sc2_env(args,rank)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env

    return ParallelEnvSC2([get_env_fn(i) for i in range(n_rollout_threads)])


def main_mine():

    print('argv = {}'.format(sys.argv))

    args = get_common_args()
    args.seed = args.seed + 1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


    if args.env_type == 'mpe':
        env = make_mpe_parallel_env(args.map,args.n_episodes,args.seed)
    else:
        if args.n_episodes > 1:
            env = make_parallel_env(args,args.n_episodes)
        else:
            env = StarCraft2Env(map_name=args.map,
                                step_mul=args.step_mul,
                                difficulty=args.difficulty,
                                game_version=args.game_version,
                                replay_dir=args.replay_dir)
    if args.env_type == 'mpe':
        env_info = {}
        env_info["n_agents"] = len(env.observation_space)
        env_info["n_actions"] = env.action_space[0][0] if isinstance(env.action_space[0], Box) else env.action_space[0].n
        env_info["obs_shape"] = env.observation_space[0].shape[0]
        env_info["state_shape"] = 1
        env_info["episode_limit"] = 50
    else:
        env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    print('prev_args = \n{}'.format(args))
    if args.alg.find('coma') > -1:
        print('COMA??????????????????????????')
        args = get_coma_args(args)
    elif args.alg.find('maac') > -1:
        print('MAAC??????????????????????????')
        args = get_g2anet_args(args)
        args.hard = False
    elif args.alg.find('central_v') > -1:
        print('central_v??????????????????????????')
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        print('reinforce??????????????????????????')
        args = get_reinforce_args(args)
    elif args.alg.find('aga') > -1:
        args = get_aga_args(args)
    elif args.alg == 'ippo':
        args = get_ippo_args(args)
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

    grid_search_idx = eval(args.grid_search_idx)
    with open('grids_key.pkl', 'rb') as f:
        grid_search_key = pickle.load(f)
    print('grid_search_para = {}'.format(grid_search_key))
    with open('grids_para.pkl', 'rb') as f:
        grid_search_para = pickle.load(f)
    print('grid_search_para = {}'.format(grid_search_para))
    for id in grid_search_idx:
        paras = grid_search_para[id]
        print('paras = {}'.format(paras))
        args = vars(args)
        for key in grid_search_key:
            args[key] = paras[key]
        args = argparse.Namespace(**args)
        args.aga_sample_size = args.batch_size
        args.special_name = 'grid_search_{}'.format(id)
        print('args = {}'.format(args))


        if args.cut_episode:
            args.episode_limit = min(100,args.episode_limit)
        if args.alg.startswith('maac'):
            print(args.obs_shape)
            print(args.n_actions)
            args.sa_size = [ [args.obs_shape,args.n_actions] for i in range(args.n_agents)]


        rnn_dim_list = [64]
        qmix_dim_list = [32]
        if args.set_bf is not None:
            args.buffer_size = args.set_bf
            print('set_bf = {}'.format(args.set_bf))
        if args.online:
            args.buffer_size = args.batch_size

        print('args.epsilon = {},args.min_epsilon = {}, args.anneal_epsilon = {},args.epsilon_anneal_scale = {}'.format(args.epsilon ,args.min_epsilon , args.anneal_epsilon ,args.epsilon_anneal_scale))

        if args.alg == 'aga':
            if args.aga_tag:
                args.buffer_size = args.aga_sample_size

        for rnn_dim,qmix_dim in zip(rnn_dim_list,qmix_dim_list):
            args.rnn_hidden_dim = rnn_dim
            args.qmix_hidden_dim = qmix_dim
            runner = Runner(env, args)
            if args.learn:
                runner.run(id)
            else:
                win_rate = runner.evaluate_sparse()
                print('The win rate of {} is  {}'.format(args.alg, win_rate))
                break
            env.close()
            del runner

if __name__ == '__main__':
    main_mine()