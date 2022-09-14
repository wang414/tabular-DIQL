import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time

def my_change_shape(x):
    x = np.array(x)
    x_len = len(np.shape(x))
    if x_len == 2:
        x = np.transpose(x,[1,0])
        x = np.expand_dims(x,2)
    elif x_len == 3:
        x = np.transpose(x,[1,0,2])
    elif x_len == 4:
        x = np.transpose(x, [2, 0, 1,3])
    return x
class RolloutWorker:
    def __init__(self, env, agents, args):
        self.mp_tag = args.n_episodes > 1
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        if not args.alg.startswith('maac'):
            self.epsilon = args.epsilon
            self.anneal_epsilon = args.anneal_epsilon
            self.min_epsilon = args.min_epsilon
        else:
            self.epsilon = 0
        print('Init RolloutWorker')
    def epsilon_reset(self):
        print('successfully reset epsilon')
        self.epsilon = self.args.reset_epsilon
        self.anneal_epsilon = self.args.reset_anneal_epsilon
        self.min_epsilon = self.args.reset_min_epsilon
    def generate_episode(self, episode_num=None, evaluate=False):
        # print('generate_episode_evaluate = {}'.format(evaluate))
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        # MULTIPROCESS
        if self.args.env_type == 'mpe':
            obs = self.env.reset()
            obs = np.transpose(obs,[1,0,2]).squeeze(1)
        else:
            self.env.reset()
        step = 0
        if self.mp_tag:
            episode_reward = np.zeros(self.args.n_episodes)
        else:
            episode_reward = 0  # cumulative rewards
        if self.mp_tag:
            already_terminated = [False for _ in range(self.args.n_episodes)]
            terminated = [False for _ in range(self.args.n_episodes)]
            last_action = np.zeros((self.args.n_agents, self.args.n_episodes, self.args.n_actions))
            info = [{'battle_won': 0} for _ in range(self.args.n_episodes)]
        else:
            terminated = False
            last_action = np.zeros((self.args.n_agents, self.args.n_actions))
            info = {'battle_won': 0}
        # MULTIPROCESS
        self.agents.policy.init_hidden(self.args.n_episodes)

    # epsilon #MULTIPROCESS
        epsilon = 0 if evaluate else self.epsilon
        if self.args.alg.startswith('maac'):
            epsilon = 0
        elif self.args.epsilon_anneal_scale == 'episode':
            for i in range(self.args.n_episodes):
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        elif self.args.epsilon_anneal_scale == 'epoch':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # print('now_epsilon = {} init_epsilon = {}, anneal_epsilon = {}, min_epsilon = {}'.format(epsilon,self.epsilon, self.anneal_epsilon,
        #                                                                    self.min_epsilon))

        #
        # if self.args.mat_test:
        #     epsilon = 1

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        if self.args.shuffle_num > -1:
            print('shuffle test for number {}'.format(self.args.shuffle_num))

        while True:
        # while step < 5:
            # time.sleep(0.2)
            # MULTIPROCESS
            # print('step = {}'.format(step))
            # print('terminated = {}'.format(terminated))
            if self.args.env_type == 'mpe':
                state = [0]
                # print('obs_shape = {}'.format(np.shape(obs)))
            else:
                obs = self.env.get_obs()
                state = self.env.get_state()
            # print('obs_shape = {}'.format(np.shape(obs)))
            # print('state_shape = {}'.format(np.shape(state)))
            actions, avail_actions, actions_onehot,actions_for_mpe = [], [], [],[]
            for agent_id in range(self.n_agents):
                # MULTIPROCESS
                if self.args.env_type == 'mpe':
                    avail_action = np.ones(self.args.n_actions)
                else:
                    avail_action = self.env.get_avail_agent_actions(agent_id)

                if self.args.alg == 'maven':
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, maven_z, evaluate = evaluate)
                else:
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate = evaluate)
                # print('agent_id = {}'.format(agent_id))
                # generate onehot vector of th action
                # MULTIPROCESS
                if self.mp_tag:
                    action = np.array(action)
                # actions_for_mpe.append([action])
                eye = np.eye(self.n_actions)
                action_onehot = eye[action]
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            # print('actions = {}'.format(actions))
            # MULTIPROCESS

            if self.mp_tag:
                if self.args.mat_test:
                    actions_input = np.array(actions)
                    reward, terminated, info = self.env.step(actions_input,evaluate = evaluate)
                elif self.args.env_type == 'mpe':
                    actions_input = np.transpose(actions_onehot,[1,0,2])
                    next_obs,reward, terminated, info = self.env.step(actions)
                    next_obs = np.transpose(next_obs, [1, 0, 2]).squeeze(1)

                    if step == self.episode_limit - 1:
                        terminated = np.ones(self.args.n_episodes)
                    else:
                        terminated = np.zeros(self.args.n_episodes)
                    reward = np.mean(reward, axis=-1)
                else:
                    reward, terminated, info = self.env.step(actions,terminated,info)
            else:
                if self.args.mat_test:
                    actions_input = np.array(actions)
                    if self.args.shuffle_num == 0:
                        actions_input[[0,1,2]] = actions_input[[1,0,2]]
                    elif self.args.shuffle_num == 1:
                        actions_input[ [0, 1, 2]] = actions_input[ [2, 0, 1]]
                    elif self.args.shuffle_num == 2:
                        actions_input[ [0, 1, 2]] = actions_input[ [1, 2, 0]]
                    curr_state = np.argmax(self.env.get_obs()[0])
                    reward, terminated, info = self.env.step(actions_input, evaluate=evaluate)
                    next_state = np.argmax(self.env.get_obs()[0])
                    if self.args.shuffle_check:
                        print('step {}: curr_state = {}, action_input = {}, action_init = {}, reward = {}, next_state = {}'.format(step,curr_state,actions_input,actions,reward,next_state))
                elif self.args.env_type == 'mpe':
                    # print('actions = {}'.format(actions))
                    action_input = np.array([actions_onehot])
                    # print('action_input_shape = {}'.format(np.shape(action_input)))
                    next_obs,reward, terminated, info = self.env.step(action_input)
                    if step == self.episode_limit - 1:
                        terminated = 1
                    else:
                        terminated = 0
                    # print('terminated = {}'.format(terminated))
                    next_obs = np.transpose(next_obs, [1, 0, 2]).squeeze(1)
                    reward = np.mean(reward,axis = -1).squeeze(0)
                    # print('reward = {}'.format(reward))
                else:
                    reward, terminated, info = self.env.step(actions)
            # if self.args.mat_test:
            #     print('terminated = {}, actions = {}, reward = {}'.format(terminated,actions,reward))
            # MULTIPROCESS
            if self.mp_tag:
                win_tag = []
                for t,i in zip(terminated,info):
                    w = True if t and 'battle_won' in i and i['battle_won'] else False
                    win_tag.append(w)
            else:
                if not self.args.mat_test and not self.args.env_type == 'mpe':
                    win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
                elif self.args.env_type == 'mpe':
                    win_tag = False
                else:
                    win_tag = True if terminated else False


            o.append(obs)
            s.append(state)

            if self.args.env_type == 'mpe':
                obs = next_obs


            if self.mp_tag:
                u.append(np.reshape(actions, [self.n_agents,self.args.n_episodes, 1]))
            else:
                u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            if self.mp_tag:
                r.append(reward)
                # MULTIPROCESS
                terminate.append(terminated)

                tmp_padded = []
                for i,t,at in zip(range(len(terminated)),terminated,already_terminated ):
                    if at:
                        tmp_padded.append(1.)
                    else:
                        tmp_padded.append(0.)
                        if t:
                            already_terminated[i] = 1
                    episode_reward[i] += reward[i]
                padded.append(tmp_padded)
            else:
                r.append([reward])
                # MULTIPROCESS
                terminate.append([terminated])
                padded.append([0.])
                episode_reward += reward
            # MULTIPROCESS
            step += 1
            # print('step = {}'.format(step))
            # print('reward = {}'.format(reward))
            # print('terminated = {}'.format(terminated))
            # print('episode_reward = {}'.format(episode_reward))
            if not self.args.alg.startswith('maac') and self.args.epsilon_anneal_scale == 'step':
                for i in range(self.args.n_episodes):
                    epsilon = epsilon - self.anneal_epsilon  if epsilon > self.min_epsilon else epsilon
            if not self.args.alg.startswith('maac') and self.args.epsilon_anneal_scale == 'epoch_step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

                # print('episode_step = {},step_epsilon = {}'.format(step,epsilon))
            # if self.args.mat_test:
            #     epsilon = 1

            if step == self.episode_limit:
                if self.mp_tag:
                    terminated = np.ones(self.args.n_episodes)
                else:
                    terminated = 1
            end = None
            if self.mp_tag:
                end = all(terminated)
            else:
                end = terminated
            if end:
                break
        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            if self.args.env_type == 'mpe':
                avail_action = np.ones(self.args.n_actions)
            else:
                avail_action = self.env.get_avail_agent_actions(agent_id)

            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            if self.mp_tag:
                o.append(np.zeros((self.n_agents,self.args.n_episodes, self.obs_shape)))
                u.append(np.zeros([self.n_agents,self.args.n_episodes, 1]))
                s.append(np.zeros([self.args.n_episodes,self.state_shape]))
                r.append(np.zeros(self.args.n_episodes))
                o_next.append(np.zeros((self.n_agents,self.args.n_episodes, self.obs_shape)))
                s_next.append(np.zeros([self.args.n_episodes,self.state_shape]))
                u_onehot.append(np.zeros([self.n_agents,self.args.n_episodes, self.n_actions]))
                avail_u.append(np.zeros((self.n_agents,self.args.n_episodes, self.n_actions)))
                avail_u_next.append(np.zeros((self.n_agents,self.args.n_episodes, self.n_actions)))
                padded.append(np.ones(self.args.n_episodes))
                terminate.append(np.ones(self.args.n_episodes))
            else:
                o.append(np.zeros((self.n_agents, self.obs_shape)))
                u.append(np.zeros([self.n_agents, 1]))
                s.append(np.zeros(self.state_shape))
                r.append([0.])
                o_next.append(np.zeros((self.n_agents, self.obs_shape)))
                s_next.append(np.zeros(self.state_shape))
                u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u.append(np.zeros((self.n_agents, self.n_actions)))
                avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
                padded.append([1.])
                terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        if self.mp_tag:
            for key in episode.keys():
                # print('{}_prev_shape = {}'.format(key,np.shape(episode[key])))
                episode[key] = my_change_shape(episode[key])
                # print('{}_after_shape = {}'.format(key, np.shape(episode[key])))
        else:
            for key in episode.keys():
                episode[key] = np.array([episode[key]])
                # print('key = {}, shape = {}'.format(key,np.shape(episode[key])))
        episode['terminated'] =np.array( episode['terminated'],dtype = np.float32)
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        return episode, episode_reward,win_tag,step * self.args.n_episodes


# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(self.args.n_episodes)
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        while not terminated:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # get the weights of all actions for all agents
            weights = self.agents.get_action_weights(np.array(obs), last_action)

            # choose action for each agent
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(weights[agent_id], avail_action, epsilon, evaluate)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, _ = self.env.step(actions)
            if step == self.episode_limit - 1:
                terminated = 1

            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if terminated:
            #     time.sleep(1)
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        if self.mp_tag:
            for key in episode.keys():
                episode[key] = my_change_shape(episode[key])
        else:
            for key in episode.keys():
                episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
            # print('Epsilon is ', self.epsilon)
        return episode, episode_reward
