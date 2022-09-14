import torch
import numpy as np
import pickle
import sys
import random

check_output = False
matrix_file_name = 'random_matrix_game_3'
with open('{}.pkl'.format(matrix_file_name),'rb') as f:
    matrix_game_para = pickle.load(f)
    transition = matrix_game_para['trans_mat']
    reward = matrix_game_para['reward']

matrix_shape = np.shape(transition)
state_num = matrix_shape[0]
action_num = matrix_shape[1]
agent_num = len(matrix_shape) - 2

def build_index(num_index,agent_num=agent_num,action_num=action_num):
    result = []
    for _ in range(agent_num):
        res = num_index % action_num
        num_index = num_index // action_num
        result.append(res)
    return result

def get_hash(joint_action,agent_num=agent_num,action_num=action_num):
    set_count = np.zeros(action_num,dtype=np.int32)
    for single_a in joint_action:
        set_count[single_a] += 1
    return set_count

total_action_num = action_num ** agent_num

for from_state in range(state_num):
    reward_s = {}
    transition_s = {}
    count_s = {}
    for all_a in range(total_action_num):
        all_action_index = build_index(all_a)
        set_count = get_hash(all_action_index)
        sa_index = [from_state] + all_action_index
        sa_index = tuple(sa_index)
        set_count = tuple(set_count)
        if check_output:
            print('action = {} hash = {}'.format(all_action_index,set_count))
        if set_count not in count_s:
            count_s[set_count] = 1
            reward_s[set_count] = np.array(reward[sa_index])
            transition_s[set_count] = np.array(transition[sa_index])
        else:
            count_s[set_count] += 1
            reward_s[set_count] += np.array(reward[sa_index])
            transition_s[set_count] += np.array(transition[sa_index])
    if check_output:
        print('state: {}'.format(from_state))

    first_flag = True

    for all_a in range(total_action_num):
        all_action_index = build_index(all_a)
        set_count = get_hash(all_action_index)
        sa_index = [from_state] + all_action_index
        sa_index = tuple(sa_index)
        set_count = tuple(set_count)
        flag = check_output
        if flag:
            if first_flag:
                print('hash_tag = {}, hash_reward = {}, hash_transition = {}, hash_count = {}'.format(set_count,reward_s[set_count],transition_s[set_count],count_s[set_count]))
                first_flag = False
            print('action = {}'.format(all_action_index))

            print('prev  transition = {}'.format(transition[sa_index]))
        reward[sa_index] = reward_s[set_count] / count_s[set_count]
        transition[sa_index] = transition_s[set_count] / count_s[set_count]
        if flag:
            print('after  transition = {}'.format( transition[sa_index]))
            print('prob_sum = {}'.format(transition[sa_index].sum()))

if check_output:
    joint_a_list = [[0,0,1],[1,0,0],[0,1,0]]

    for s in range(state_num):
        print('state: {}'.format(s))
        for joint_a in joint_a_list:
            sa_index = [s] + joint_a
            sa_index = tuple(sa_index)
            hash_tag = get_hash(joint_a)
            print('joint_a = {}, hash = {}'.format(joint_a,hash_tag))
            print('reward = {}'.format(reward[sa_index]))
            print('transition = {}'.format(transition[sa_index]))

sym_matrix_game_para = {}
sym_matrix_game_para['trans_mat'] = transition
sym_matrix_game_para['reward'] = reward

with open('{}_symmetric.pkl'.format(matrix_file_name),'wb') as f:
    pickle.dump(sym_matrix_game_para,f)