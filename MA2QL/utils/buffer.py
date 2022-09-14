import numpy as np
from torch import Tensor
from torch.autograd import Variable
import torch

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self,  num_agents, obs_dims, ac_dims,buff_size,episode_length):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.buff_size = buff_size
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.padded_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((buff_size,episode_length, odim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((buff_size,episode_length, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros((buff_size,episode_length), dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((buff_size,episode_length, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros((buff_size,episode_length), dtype=np.uint8))
            self.padded_buffs.append(np.zeros((buff_size, episode_length), dtype=np.float32))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, episode ): #observations, actions, rewards, next_observations, dones
        observations = episode['o']
        actions = episode['a']
        rewards = episode['r']
        next_observations = episode['next_o']
        dones = episode['done']
        padded = episode['padded']
        # obs = next_obs = (12,25,5,30) (eps_length,thread,agent,odim), agent_actions = (12,25,5,5) (eps_length,agent,thread,adim),
        # rewards = (12,25,5) (eps_length,thread,agent),dones = (12,25,5) (eps_length,thread,agent)

        nepisode = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nepisode > self.buff_size:
            rollover = self.buff_size - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover,axis = 0)
                self.padded_buffs[agent_i] = np.roll(self.padded_buffs[agent_i],
                                                  rollover,axis = 0)
            self.curr_i = 0
            self.filled_i = self.buff_size
        for agent_i in range(self.num_agents):
            # print('obs_buff[agent_i] = {}'.format(np.shape(self.obs_buffs[agent_i]))) (40000,25,30)
            # print('observation = {}'.format(np.shape(observations))) (12,25,5,30)
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nepisode] = observations[:,:, agent_i]
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nepisode] = actions[:,:,agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nepisode] = rewards[:,:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nepisode] = next_observations[:,:, agent_i]
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nepisode] = dones[:,:, agent_i]
            self.padded_buffs[agent_i][self.curr_i:self.curr_i + nepisode] = padded[:,:,agent_i]
        self.curr_i += nepisode
        if self.filled_i < self.buff_size:
            self.filled_i += nepisode
        if self.curr_i == self.buff_size:
            self.curr_i = 0

    def sample(self, N,latest = False, to_gpu=False, norm_rews=True):
        if latest:
            sample_num = min(self.filled_i,N)
            inds = self.filled_i - 1 - np.arange(sample_num)
        else:
            inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=True)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = cast([(self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std()
                        for i in range(self.num_agents)] )
        else:
            ret_rews = cast([self.rew_buffs[i][inds] for i in range(self.num_agents)])
        batch = {}
        batch['o'] = cast([self.obs_buffs[i][inds] for i in range(self.num_agents)])
        # print('batch_o_shape = {}'.format(batch['o'].shape))
        batch['o'] = batch['o'].permute(1,2,0,3 )
        batch['a'] =cast([self.ac_buffs[i][inds] for i in range(self.num_agents)])
        batch['a'] = batch['a'].permute(1, 2, 0, 3)
        # print('batch_a = {}'.format(batch['a'].shape))
        batch['r'] = ret_rews.permute(1, 2, 0)
        batch['next_o'] = cast([self.next_obs_buffs[i][inds] for i in range(self.num_agents)])
        batch['next_o'] = batch['next_o'].permute(1, 2, 0, 3)
        batch['done'] = cast([self.done_buffs[i][inds] for i in range(self.num_agents)])
        batch['done'] = batch['done'].permute(1, 2, 0)
        batch['padded'] = cast([self.padded_buffs[i][inds] for i in range(self.num_agents)])
        batch['padded'] = batch['padded'].permute(1, 2, 0)
        return batch

    def get_average_rewards(self, N):
        if self.filled_i == self.buff_size:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
