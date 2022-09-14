import torch
import numpy as np
import pickle
import sys
import random

type = 'random'


state_num = 50
action_num = 5
agent_num = 1

reward_scale = 100

trans_shape = []
trans_shape.append(state_num)
for i in range(agent_num):
    trans_shape.append(action_num)
trans_shape.append(state_num)

reward_shape = []
reward_shape.append(state_num)
for i in range(agent_num):
    reward_shape.append(action_num)
if type == 'dop':
    transition = 4 * np.random.random(trans_shape) - 2
elif type == 'random':
    transition = 4 * np.random.random(trans_shape) -2
print('transition_before = {}'.format(transition))
transition = torch.softmax(torch.tensor(transition),dim = -1)
transition = np.array(transition)
print('transition_after = {}'.format(transition))
if type == 'dop':
    reward = np.ones(reward_shape) * (-10)

    reward[0,1,5,9] = 10.0
elif type == 'random':
    reward = reward_scale*np.random.random(reward_shape) - (0.5 * reward_scale)
print('reward = {}'.format(reward))
if type == 'dop':

    print('r[1,5,9] = {}'.format(reward[0,1,5,9]))
# end_state = np.random.choice(np.arange(state_num),size=end_state_num,replace=False)
# print('end_state = {}'.format(end_state))

matrix_game_para = {}
matrix_game_para['trans_mat'] = transition
matrix_game_para['reward'] = reward

with open('random_matrix_game_1_agent_50x5.pkl','wb') as f:
    pickle.dump(matrix_game_para,f)