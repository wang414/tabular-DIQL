import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit

        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        if self.args.buffer_type == 'episode':
            self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                            'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                            's': np.empty([self.size, self.episode_limit, self.state_shape]),
                            'r': np.empty([self.size, self.episode_limit, 1]),
                            'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                            's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                            'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                            'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                            'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                            'padded': np.empty([self.size, self.episode_limit, 1]),
                            'terminated': np.empty([self.size, self.episode_limit, 1])
                            }
            if self.args.alg == 'maven':
                self.buffers['z'] = np.empty([self.size, self.args.noise_dim])
        elif self.args.buffer_type == 'step':
            self.buffers = {'o': np.empty([self.size, self.n_agents, self.obs_shape]),
                            'u': np.empty([self.size, self.n_agents, 1]),
                            's': np.empty([self.size,  self.state_shape]),
                            'r': np.empty([self.size,  1]),
                            'o_next': np.empty([self.size,  self.n_agents, self.obs_shape]),
                            's_next': np.empty([self.size, self.state_shape]),
                            'avail_u': np.empty([self.size,  self.n_agents, self.n_actions]),
                            'avail_u_next': np.empty([self.size,  self.n_agents, self.n_actions]),
                            'u_onehot': np.empty([self.size,  self.n_agents, self.n_actions]),
                            'padded': np.empty([self.size,  1]),
                            'terminated': np.empty([self.size, 1])
                            }
            if self.args.alg == 'maven':
                self.buffers['z'] = np.empty([self.size, self.args.noise_dim])
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            if self.args.buffer_type == 'episode':
                idxs = self._get_storage_idx(inc=batch_size)
                # store the informations
                self.buffers['o'][idxs] = episode_batch['o']
                self.buffers['u'][idxs] = episode_batch['u']
                self.buffers['s'][idxs] = episode_batch['s']
                self.buffers['r'][idxs] = episode_batch['r']
                self.buffers['o_next'][idxs] = episode_batch['o_next']
                self.buffers['s_next'][idxs] = episode_batch['s_next']
                self.buffers['avail_u'][idxs] = episode_batch['avail_u']
                self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
                self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
                self.buffers['padded'][idxs] = episode_batch['padded']
                self.buffers['terminated'][idxs] = episode_batch['terminated']
                if self.args.alg == 'maven':
                    self.buffers['z'][idxs] = episode_batch['z']
            else:
                for b in range(batch_size):
                    l = episode_batch['o'][b].shape[0]
                    idxs = self._get_storage_idx(inc=l)

                    # store the informations
                    self.buffers['o'][idxs] = episode_batch['o'][b]
                    self.buffers['u'][idxs] = episode_batch['u'][b]
                    self.buffers['s'][idxs] = episode_batch['s'][b]
                    self.buffers['r'][idxs] = episode_batch['r'][b]
                    self.buffers['o_next'][idxs] = episode_batch['o_next'][b]
                    self.buffers['s_next'][idxs] = episode_batch['s_next'][b]
                    self.buffers['avail_u'][idxs] = episode_batch['avail_u'][b]
                    self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next'][b]
                    self.buffers['u_onehot'][idxs] = episode_batch['u_onehot'][b]
                    self.buffers['padded'][idxs] = episode_batch['padded'][b]
                    self.buffers['terminated'][idxs] = episode_batch['terminated'][b]
                    if self.args.alg == 'maven':
                        self.buffers['z'][idxs] = episode_batch['z'][b]

    def sample(self, batch_size,latest=False):
        temp_buffer = {}
        if latest:
            sample_num = min(self.current_size, batch_size)
            idx = self.current_size - 1 - np.arange(sample_num)
        else:
            idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def mix_sample(self, on_batch_size,off_batch_size,division):

        on_sample_num = min(self.current_size, self.args.mix_on_buffer_size)
        used_off_batch_size = min(self.current_size,off_batch_size)
        used_on_batch_size = min(on_sample_num,on_batch_size)
        ep_ids = np.random.choice(self.current_size, used_off_batch_size , replace=False)
        on_ids = np.random.choice(on_sample_num, used_on_batch_size, replace=False)
        on_ids = self.current_idx - 1 - on_ids

        temp_buffer = {}

        idx = ep_ids

        for i in on_ids:
            idx = np.append(idx, [i])
        print('current_size = {} on_sample_num = {} on_ids = {} ep_ids = {} final_ids = {}'.format(self.current_size,
                                                                                                   on_sample_num,
                                                                                                   on_ids, ep_ids, idx))
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
