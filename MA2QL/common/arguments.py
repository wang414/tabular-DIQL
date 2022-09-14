import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idv_test_actor', type=bool, default=False)
    parser.add_argument('--idv_test_critic', type=bool, default=False)
    parser.add_argument('--idv_common_lr', type=bool, default=False)
    parser.add_argument('--logger_check_output', type=bool, default=False)
    parser.add_argument('--idv_comparing', type=bool, default=False)
    parser.add_argument('--test_no_use_td_lambda', type=bool, default=False)
    parser.add_argument('--classic_sac', type=bool, default=False)
    parser.add_argument('--idv_off_policy', type=bool, default=False)
    parser.add_argument('--idv_use_gae', type=bool, default=False)
    parser.add_argument('--aga_update_index_interval', type=int, default=1)
    parser.add_argument('--aga_check_interval', type=int, default=1)
    parser.add_argument('--aga_dp_update_alpha', type=float, default=1)
    parser.add_argument('--aga_dp_step', type=int, default=1)
    parser.add_argument('--q_update_alpha', type=float, default=0.5)
    parser.add_argument('--const_alpha', type=bool, default=False)
    parser.add_argument('--mat_epsilon_step', type=int, default=None)
    parser.add_argument('--maxloop', type=int, default=500)
    parser.add_argument('--use_sample_done', type=bool, default=False)
    parser.add_argument('--iteration_epsilon_step', type=int, default=None)
    parser.add_argument('--iteration_epsilon_decay', type=bool, default=False)
    parser.add_argument('--aga_tabular_sample_steps', type=int, default=5000)
    parser.add_argument('--aga_one_step_update', type=bool, default=False)
    parser.add_argument('--shuffle_check', type=bool, default=False)
    parser.add_argument('--shuffle_num', type=int, default=-1)
    parser.add_argument('--dmac_zero_init', type=bool, default=False)
    parser.add_argument('--on_policy_multi_batch', type=bool, default=False)
    parser.add_argument('--mix_off_batch_size', type=int, default=8)
    parser.add_argument('--mix_on_buffer_size', type=int, default=32)
    parser.add_argument('--mix_on_batch_size', type=int, default=4)
    parser.add_argument('--mix_sample', type=bool, default=False)
    parser.add_argument('--buffer_type',type=str,default='episode')
    parser.add_argument('--test_greedy', type=bool, default=False)
    parser.add_argument('--dop_dmac_tag', type=bool, default=False)
    parser.add_argument('--alpha_min', type=float, default=0.05)
    parser.add_argument('--alpha_max', type=float, default=0.5)
    parser.add_argument('--alpha_step', type=int, default=20000)
    parser.add_argument('--alpha_decay', type=bool, default=False)
    parser.add_argument('--mixer_module', type=str, default='qmix')
    parser.add_argument('--policy_train_mode', type=str, default='qmix')
    parser.add_argument('--critic_no_rnn', type=bool, default=False)
    parser.add_argument('--double_min', type=bool, default=False)
    parser.add_argument('--no_dmac', type=bool, default=False)
    parser.add_argument('--dmac_tau', type=float, default=0.001)
    parser.add_argument('--coma_zero_eps', type=bool, default=False)
    parser.add_argument('--draw_plot_for_dmac', type=bool, default=False)
    parser.add_argument('--grid_search_idx', type=str, default=None)
    parser.add_argument('--stage_lr', type=bool, default=False)
    parser.add_argument('--aga_sample_size', type=int, default=128)
    parser.add_argument('--reset_epsilon_min', type=float, default=None)
    parser.add_argument('--epsilon_reset_factor', type=float, default=1)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--sample_mode', type=str, default='episode')
    parser.add_argument('--q_table_output', type=bool, default=False)
    parser.add_argument('--check_mat_policy', type=bool, default=False)
    parser.add_argument('--no_rnn', type=bool, default=False)
    parser.add_argument('--dp_test', type=str, default='network')
    parser.add_argument('--behavior_check_output', type=bool, default=False)
    parser.add_argument('--env_type', type=str, default='sc2')
    parser.add_argument('--new_period', type=bool, default=False)
    parser.add_argument('--soft_target', type=bool, default=False)
    parser.add_argument('--correct_dec', type=bool, default=False)
    parser.add_argument('--target_dec', type=bool, default=False)
    parser.add_argument('--aga_first', type=bool, default=False)
    parser.add_argument('--period', type=int, default=1)
    parser.add_argument('--min_epsilon', type=float, default=None)
    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--anneal_steps', type=int, default=None)
    parser.add_argument('--use_minibatch', type=bool, default=False)
    parser.add_argument('--aga_minibatch', type=int, default=32)
    parser.add_argument('--stop_other_exp', type=bool, default=False)
    parser.add_argument('--value_eps', type=float, default=0.2)
    parser.add_argument('--value_clip', type=bool, default=False)
    parser.add_argument('--aga_F', type=float, default=1)
    parser.add_argument('--optim_reset', type=bool, default=False)
    parser.add_argument('--epsilon_reset', type=bool, default=False)
    parser.add_argument('--single_exp', type=bool, default=False)
    parser.add_argument('--idv_para', type=bool, default=False)
    parser.add_argument('--aga_tag', type=bool, default=False)
    parser.add_argument('--special_name', type=str, default=None)
    parser.add_argument('--total_run_num', type=int, default=1)
    parser.add_argument('--online', type=bool, default=False)
    parser.add_argument('--train_steps', type=int, default=1)
    parser.add_argument('--n_episodes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--two_hyper_layers', type=bool, default=True)
    parser.add_argument('--step_log_interval',type = int ,default= 5000)
    parser.add_argument('--t_max',type=int ,default= 1005000)
    parser.add_argument('--log_mode',type=str,default = 'step')
    parser.add_argument('--auto_draw', type=bool, default=True)
    parser.add_argument('--double_q', type=bool, default=True)
    parser.add_argument('--optim_eps', type=float, default=0.99)
    parser.add_argument('--mat_map', type=int, default=1)
    parser.add_argument('--mat_test', type=bool, default=False)
    parser.add_argument('--cut_episode', type=bool, default=False)
    parser.add_argument('--set_bf',type = int,default= None)
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--target_update_cycle', type=int, default=None)
    parser.add_argument('--check_qmix', type=bool, default=False)
    parser.add_argument('--check_alg', type=bool, default=False)
    parser.add_argument('--use_td_lambda', type=bool, default=False)
    parser.add_argument('--td_lambda', type=float, default=0.8)
    parser.add_argument("--reward_scale", default=10., type=float)
    # the environment setting

    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='MMM2', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='aga', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--threshold', type=float, default=19.9, help='threshold to judge whether win')
    args = parser.parse_args()
    print('from_parser_args = {}'.format(args))
    return args

def get_maac_args(args):
    args.save_cycle = 5000
    args.pol_hidden_dim = 128
    args.critic_hidden_dim = 128
    args.attend_heads = 4
    args.pi_lr = 0.001
    args.q_lr = 0.001
    args.tau = 0.001
    args.gamma = 0.99
    args.reward_scale = 1
    args.use_gpu = True

    args.n_epoch = 20000

    # the number of the episodes in one epoch


    # the number of the train steps in one epoch

    # # how often to evaluate
    args.evaluate_cycle = 100

    # experience replay
    args.buffer_size = int(5e3)
    args.grad_norm_clip = 10
    return args
# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    if args.mat_test:
        if args.coma_zero_eps:
            args.epsilon = 0
            args.min_epsilon = 0
        else:
            args.epsilon = 0.5
            args.min_epsilon = 0.02
        anneal_step = 5000
        args.anneal_epsilon = (args.epsilon - args.min_epsilon ) / anneal_step
        args.epsilon_anneal_scale = 'step'


    else:
        args.epsilon = 0.5
        args.anneal_epsilon = 0.00064
        args.min_epsilon = 0.02
        args.epsilon_anneal_scale = 'epoch'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10
    args.buffer_size = int(5e3)
    return args


# arguments of vnd、 qmix、 qtran

def get_ippo_args(args):
    args.gae_lambda = 0.95
    args.policy_clip = 0.2
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.rnn_hidden_dim_for_qmix = 64
    args.qmix_hidden_dim_for_qmix = 32
    args.critic_dim = 64

    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3
    args.lr = 5e-4



    # epsilon greedy
    if args.mat_test:
        args.epsilon = 1.0
        args.min_epsilon = 0.1
        if args.anneal_steps is None:
            anneal_steps = 100000
        else:
            anneal_steps = args.anneal_steps
    else:
        args.epsilon = 1.0
        args.min_epsilon = 0.02
        if args.anneal_steps is None:
            anneal_steps = 100000
        else:
            anneal_steps = args.anneal_steps
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    if args.epsilon_reset:
        args.epsilon_anneal_scale = 'epoch_step'
    else:
        args.epsilon_anneal_scale = 'epoch_step'

    args.reset_epsilon = 1.0
    args.reset_min_epsilon = 0.002
    reset_anneal_steps = 128
    args.reset_anneal_epsilon = (args.reset_epsilon  - args.reset_min_epsilon) / reset_anneal_steps

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch


    # the number of the train steps in one epoch
    args.train_steps_for_qmix = 1

    # # how often to evaluate
    args.evaluate_cycle = 100

    # experience replay
    args.batch_size_for_qmix = 32

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args
def get_aga_args(args):
    args.attend_head_num = 4
    # parser.add_argument("--check_tag", default=False, type=bool)
    args.eps_tag = False
    # parser.add_argument("--eps_tag", default=False, type=bool)
    args.abs_tag = False
    # parser.add_argument("--abs_tag", default=False, type=bool)
    args.state_info = 2
    # parser.add_argument("--state_info", default=2, type=int)
    args.policy_log_tag = False
    # parser.add_argument("--policy_log_tag", default=True, type=bool)
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.rnn_hidden_dim_for_qmix = 64
    args.qmix_hidden_dim_for_qmix = 32


    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    if args.lr is None:
        args.lr = 5e-4



    # epsilon greedy
    if args.mat_test:
        if args.epsilon is None:
            args.epsilon = 1.0
        if args.min_epsilon is None:
            args.min_epsilon = 0.02
        if args.anneal_steps is None:
            anneal_steps = 100000
        else:
            anneal_steps = args.anneal_steps
    else:
        args.epsilon = 1.0
        args.min_epsilon = 0.02
        if args.anneal_steps is None:
            anneal_steps = 100000
        else:
            anneal_steps = args.anneal_steps
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    if args.epsilon_reset:
        args.epsilon_anneal_scale = 'epoch_step'
    else:
        args.epsilon_anneal_scale = 'epoch_step'

    args.reset_epsilon = 1.0
    args.reset_min_epsilon = 0.002
    reset_anneal_steps = int(args.batch_size * args.episode_limit * args.epsilon_reset_factor)
    print('args.batch_size = {} args.episode_limit = {}  args.epsilon_reset_factor = {} reset_anneal_steps = {}'.format(args.batch_size,args.episode_limit,args.epsilon_reset_factor,reset_anneal_steps))
    args.reset_anneal_epsilon = (args.reset_epsilon  - args.reset_min_epsilon) / reset_anneal_steps

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch


    # the number of the train steps in one epoch
    args.train_steps_for_qmix = 1

    # # how often to evaluate
    args.evaluate_cycle = 100

    # experience replay
    args.batch_size_for_qmix = 32

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args



def get_mixer_args(args):
    args.attend_head_num = 4
    args.n_head = 4
    args.attend_reg_coef= 0.001
    # parser.add_argument("--check_tag", default=False, type=bool)
    args.eps_tag = False
    # parser.add_argument("--eps_tag", default=False, type=bool)
    args.abs_tag = False
    # parser.add_argument("--abs_tag", default=False, type=bool)
    args.state_info = 2
    # parser.add_argument("--state_info", default=2, type=int)
    args.policy_log_tag = False
    # parser.add_argument("--policy_log_tag", default=True, type=bool)
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.rnn_hidden_dim_for_qmix = 64
    args.qmix_hidden_dim_for_qmix = 32


    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64

    args.lr = 5e-4



    # epsilon greedy
    args.epsilon = 0.5
    args.min_epsilon = 0.002
    anneal_steps = 10000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'epoch_step'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch


    # the number of the train steps in one epoch
    args.train_steps_for_qmix = 1

    # # how often to evaluate
    args.evaluate_cycle = 100

    # experience replay
    args.batch_size_for_qmix = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch


    # how often to evaluate
    args.evaluate_cycle = 100

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch


    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch


    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10
    args.buffer_size = int(5e3)


    args.attention_dim = 32
    args.hard = True
    return args

