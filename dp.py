import argparse
import copy
import pickle
from collections import deque
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from Env import chooce_the_game
import os


def make_not_exist_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_and_plot(plot_dir,save_prefix,iter_list,check_val_list,Q_check_list,pi_check_list,run_num):
    make_not_exist_dir(plot_dir)
    make_not_exist_dir(save_prefix)
    matplotlib.use('TKAgg')
    plt.title('{}'.format("title"))
    plt.xlabel('iteration')
    plt.ylabel('mean episode rewards')
    plt.plot(iter_list, check_val_list)
    plt.savefig('{}/mean_episode_rewards_{}.png'.format(save_prefix,run_num), format='png')
    plt.clf()

    plt.title('{}'.format("title"))
    plt.xlabel('iteration')
    plt.ylabel('Q gap')
    plt.plot(iter_list, Q_check_list)
    plt.savefig('{}/Q_gap_{}.png'.format(save_prefix,run_num), format='png')
    plt.clf()

    plt.title('{}'.format("title"))
    plt.xlabel('iteration')
    plt.ylabel('pi gap')
    plt.plot(iter_list, pi_check_list)
    plt.savefig('{}/pi_gap_{}.png'.format(save_prefix,run_num), format='png')
    plt.clf()

    with open('{}/iter_{}.pkl'.format(save_prefix,run_num), 'wb') as f:
        pickle.dump(iter_list, f)
    with open('{}/mean_episode_rewards_{}.pkl'.format(save_prefix,run_num), 'wb') as f:
        pickle.dump(check_val_list, f)
    with open('{}/Q_gap_{}.pkl'.format(save_prefix,run_num), 'wb') as f:
        pickle.dump(Q_check_list, f)
    with open('{}/pi_gap_{}.pkl'.format(save_prefix,run_num), 'wb') as f:
        pickle.dump(pi_check_list, f)


env = chooce_the_game(0)

aga_tag = True
arg = "dp"

if arg == "dp":
    n_actions = env.action_num
    n_agents = env.agent_num
    n_states = env.state_num
    gamma = 0.75
    dp_eps = 1e-2
    Q = np.zeros([n_agents, n_states, n_actions])
    pi = np.zeros([n_agents, n_states], dtype=np.int32)
    maxloop = 10000
    check_interval = 100
    check_episode = 20
    iter_list = []
    check_val_list = []
    Q_check_list = []
    pi_check_list = []
    plot_dir = 'mat_test_plot'
    if aga_tag:
        prefix = 'aga_Q'
    else:
        prefix = 'iql'

    save_prefix = plot_dir + '/{}_{}_{}_{}'.format("dp", prefix, "", "test")

    p_save_prefix = Path(save_prefix)
    if not p_save_prefix.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         p_save_prefix.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    run_dir = save_prefix + "/run{}".format(run_num)


    make_not_exist_dir(run_dir)

    aga_pi_judge = 100
    aga_Q_judge = 1000
    for iter in range(maxloop):
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

            save_and_plot(plot_dir, save_prefix, iter_list, check_val_list, Q_check_list, pi_check_list, run_num)

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
                            next_Q = Q[i][s_idx, pi[i]]
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
            dp_iter_step = 0
            while True:
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
                alpha = 0.002
                if prefix == 'aga_Q':
                    dp_judge = np.linalg.norm(Q[i] - new_Q_i)
                    Q[i] = (1 - alpha) * Q[i] + alpha * new_Q_i
                    pi[i] = np.argmax(Q[i, :, :], axis=-1)
                elif prefix == 'iql':
                    dp_judge = np.linalg.norm(Q[i] - new_Q[i])
                    Q[i] = (1 - alpha) * Q[i] + alpha * new_Q[i]
                if dp_judge >= dp_eps:
                    break

                # print('dp_judge = {}'.format(dp_judge))
        if prefix == 'iql':
            pi = np.argmax(Q, axis=-1)

        aga_Q_judge = np.linalg.norm(Q_for_check - Q)
        aga_pi_judge = (pi_for_check != pi).sum()
        print('Q_judge = {}, pi_judge = {}'.format(aga_Q_judge, aga_pi_judge))
        # if aga_pi_judge == 0:
        #     break


elif arg == 'dp_optimal':
    n_actions = env.action_num
    n_agents = env.agent_num
    n_states = env.state_num
    gamma = 0.75
    dp_eps = 1e-2
    n_joint_actions = n_actions ** n_agents
    Q = np.zeros([n_states, n_joint_actions])
    pi = np.zeros([n_states], dtype=np.int32)
    maxloop = 10000
    check_interval = 1000
    check_episode = 20
    iter_list = []
    check_val_list = []
    Q_check_list = []
    pi_check_list = []
    plot_dir = 'mat_test_plot'
    prefix = 'aga_Q'


    def joint_to_idv(joint_a, n_actions=n_actions, n_agents=n_agents):
        res = []
        while joint_a > 0:
            r = joint_a % n_actions
            joint_a = joint_a // n_actions
            res.append(r)
        if len(res) < n_agents:
            for i in range(len(res), n_agents):
                res.append(0)
        reversed(res)
        return res

    save_prefix = plot_dir + '/{}_{}_{}_{}'.format("dp_optimal", prefix, "", "opt")

    p_save_prefix = Path(save_prefix)
    if not p_save_prefix.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         p_save_prefix.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    run_dir = save_prefix + "/run{}".format(run_num)
    make_not_exist_dir(run_dir)

    aga_pi_judge = 100
    aga_Q_judge = 1000
    for iter in range(maxloop):
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

            save_and_plot(plot_dir, save_prefix, iter_list, check_val_list, Q_check_list, pi_check_list, run_num)

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

            print('value_iter = {} dp_judge = {}'.format(value_iter, dp_judge))
            value_iter += 1
        pi = np.argmax(Q, axis=-1)

        aga_Q_judge = np.linalg.norm(Q_for_check - Q)
        aga_pi_judge = (pi_for_check != pi).sum()
        print('Q_judge = {}, pi_judge = {}'.format(aga_Q_judge, aga_pi_judge))
        # if aga_pi_judge == 0:
        #     break


"""
elif args.dp_test == 'tabular':
    n_actions = args.n_actions
    n_agents = args.n_agents
    n_states = args.state_shape
    gamma = args.gamma
    dp_eps = 1e-2
    Q = np.zeros([n_agents, n_states, n_actions])
    pi = np.zeros([n_agents, n_states], dtype=np.int32)

    check_interval = args.aga_check_interval
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

    special_name = args.special_name
    if special_name is None:
        special_name = 'er_and_lr_seed_{}'.format(args.seed)

    save_prefix = plot_dir + '/{}_{}_{}_{}'.format(args.dp_test, prefix, args.map, special_name)

    p_save_prefix = Path(save_prefix)
    if not p_save_prefix.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         p_save_prefix.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    run_dir = save_prefix + "/run{}".format(run_num)
    make_not_exist_dir(run_dir)

    if prefix == 'iql':
        buffer_size = args.buffer_size
    elif prefix == 'aga_Q':
        buffer_size = args.buffer_size

    sample_steps = args.aga_tabular_sample_steps
    mini_batch_num = args.train_steps
    batch_size = sample_steps // mini_batch_num

    buffer = {}
    key_list = ['s', 'a', 'r', 's_next', 'done']
    # print('buffer_size = {}'.format(buffer_size))
    for key in key_list:
        buffer[key] = deque(maxlen=buffer_size)

    maxloop = args.maxloop

    epsilon = args.epsilon
    min_epsilon = args.min_epsilon

    if args.mat_epsilon_step is None:
        mat_epsilon_step = sample_steps
    else:
        mat_epsilon_step = args.mat_epsilon_step
    anneal_step = mat_epsilon_step
    anneal_epsilon = (epsilon - min_epsilon) / anneal_step

    iter_epsilon = epsilon
    iter_min_epsilon = min_epsilon
    if args.iteration_epsilon_step is not None:
        iteration_epsilon_step = args.iteration_epsilon_step
    else:
        iteration_epsilon_step = maxloop
    iter_anneal_epsilon = (iter_epsilon - iter_min_epsilon) / iteration_epsilon_step

    count_table = np.zeros_like(Q)
    sample_done = False

    aga_pi_judge = 100
    aga_Q_judge = 1000
    for iter in range(maxloop):
        print('{}/{} iteration epsilon = {}'.format(iter, maxloop, epsilon))
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

            save_and_plot(args, plot_dir, save_prefix, iter_list, check_val_list, Q_check_list, pi_check_list, run_num)

        if args.use_sample_done:
            if sample_done:
                env.reset()
        else:
            env.reset()
        s = env.get_state()
        s = np.argmax(s)

        if prefix == 'aga_Q':
            if epsilon_reset:
                if args.iteration_epsilon_decay:
                    iter_epsilon = np.maximum(iter_min_epsilon, iter_epsilon - iter_anneal_epsilon)
                epsilon = iter_epsilon
                min_epsilon = args.min_epsilon
                anneal_step = sample_steps
                anneal_epsilon = (epsilon - min_epsilon) / anneal_step
            if lr_reset:
                count_table = np.zeros_like(Q)
        print('iter {} sampling ...... epsilon = {}'.format(iter, epsilon))
        for _ in range(sample_steps):
            action = []
            update_index = (iter // args.aga_update_index_interval) % n_agents
            for i in range(n_agents):
                if prefix == 'iql':
                    if np.random.random() < epsilon:
                        a_i = np.random.choice(np.arange(n_actions))
                    else:
                        a_i = pi[i, s]
                elif prefix == 'aga_Q':
                    if i == update_index:
                        if np.random.random() < epsilon:
                            a_i = np.random.choice(np.arange(n_actions))
                        else:
                            a_i = pi[i, s]
                    else:
                        a_i = pi[i, s]
                action.append(a_i)

            epsilon = np.maximum(min_epsilon, epsilon - anneal_epsilon)

            action = np.array(action, dtype=np.int32)
            r, done, _ = env.step(action)
            sample_done = done
            next_s = env.get_state()
            next_s = np.argmax(next_s)

            experience = [s, action, r, next_s, done]
            for k, v in zip(key_list, experience):
                buffer[k].append(v)
            s = next_s
        print('iter {} sample has ended epsilon = {}'.format(iter, epsilon))
        for b in range(mini_batch_num):
            print('iter {} training for batch {}'.format(iter, b))
            curr_buffer_size = len(buffer['s'])
            idx = np.random.choice(np.arange(curr_buffer_size, dtype=np.int32), size=batch_size, replace=False)
            # print('idx_dtype = {}'.format(idx.dtype))
            batch = {}
            for key in key_list:
                # print('key = {}, buffer_shape = {}'.format(key, np.array(buffer[key]).shape ))
                batch[key] = np.array(buffer[key])[idx]
            if prefix == 'iql':
                new_Q = copy.deepcopy(Q)
                for s, a, r, s_next, done in zip(batch['s'], batch['a'], batch['r'], batch['s_next'], batch['done']):
                    for i in range(n_agents):
                        a_i = a[i]
                        count_table[i, s, a_i] += 1
                        delta = r + gamma * np.max(new_Q[i, s_next, :]) - new_Q[i, s, a_i]
                        alpha = 1 / count_table[i, s, a_i]
                        new_Q[i, s, a_i] += alpha * delta
                new_pi = np.argmax(new_Q, axis=-1)
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
                    if args.const_alpha:
                        alpha = args.q_update_alpha
                    else:
                        alpha = 1 / count_table[update_index, s, a_i]
                    new_Q_i[s, a_i] += alpha * delta
                new_pi_i = np.argmax(new_Q_i, axis=-1)
                aga_Q_judge = np.linalg.norm(new_Q_i - Q[update_index])
                aga_pi_judge = (new_pi_i != pi[update_index]).sum()

                pi[update_index] = new_pi_i
                Q[update_index] = new_Q_i
        print('iter = {} Q_judge = {}, pi_judge = {}'.format(iter, aga_Q_judge, aga_pi_judge))
"""