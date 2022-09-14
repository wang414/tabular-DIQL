"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import torch as th
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv
class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True






# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelEnvSC2:

    def __init__(self, env_fns):
        self.batch_size = len(env_fns)

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(env_fn)))
                            for worker_conn,env_fn in zip(self.worker_conns,env_fns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.closed = 0


    def get_env_info(self):
        return self.env_info
    def get_state(self):
        state = []
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_state", None))
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            state.append(data['state'])
        # print('from_env::state = {}'.format(np.shape(state)))
        return state
    def get_avail_agent_actions(self,agent_id):
        avail_agent_actions = []
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_avail_agent_actions", agent_id))
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            avail_agent_actions.append(data['avail_agent_actions'])
        # print('from_env::avail_agent_actions = {}'.format(np.shape(avail_agent_actions)))
        return avail_agent_actions
    def get_obs(self):
        obs = []
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_obs", None))
        for parent_conn in self.parent_conns:
            data =  parent_conn.recv()
            obs.append(data['obs'])
        obs = np.array(obs).transpose([1,0,2])
        # print('from_env::obs_shape = {}'.format(np.shape(obs)))
        return obs
    def save_replay(self):
        pass
    def step(self,actions,dones_before,info_before):
        rewards = []
        dones = []
        infos = []
        # print('actions = {}'.format(actions))
        actions = np.array(actions).transpose([1,0])
        # print('actions_shape = {}'.format(np.shape(actions)))
        for parent_conn, action,done in zip( self.parent_conns, actions,dones_before):
            # print('action = {}'.format(action))
            # print('action_type = {}'.format(type(action)))
            if not done:
                parent_conn.send(('step', action))
        for parent_conn,done,i in zip(self.parent_conns,dones_before,info_before):
            if not done:
                data = parent_conn.recv()
                rewards.append(data['reward'])
                dones.append(data['terminated'])
                infos.append(data['info'])
            else:
                rewards.append(0.)
                dones.append(True)
                infos.append(i)
        # print('from_env::rewards_shape = {}'.format(np.shape(rewards)))
        # print('from_env::dones_shape = {}'.format(np.shape(dones)))
        return rewards, dones, infos
    def close(self):
        if self.closed:
            return
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def reset(self):
        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))



def run(self, test_mode=False):
    self.reset()

    all_terminated = False
    episode_returns = [0 for _ in range(self.batch_size)]
    episode_lengths = [0 for _ in range(self.batch_size)]
    self.mac.init_hidden(batch_size=self.batch_size)
    terminated = [False for _ in range(self.batch_size)]
    envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
    final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

    while True:

        # Pass the entire batch of experiences up till now to the agents
        # Receive the actions for each agent at this timestep in a batch for each un-terminated env
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
        cpu_actions = actions.to("cpu").numpy()

        # Update the actions taken
        actions_chosen = {
            "actions": actions.unsqueeze(1)
        }
        self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

        # Send actions to each env
        action_idx = 0
        for idx, parent_conn in enumerate(self.parent_conns):
            if idx in envs_not_terminated: # We produced actions for this env
                if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                    parent_conn.send(("step", cpu_actions[action_idx]))
                action_idx += 1 # actions is not a list over every env

        # Update envs_not_terminated
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        all_terminated = all(terminated)
        if all_terminated:
            break

        # Post step data we will insert for the current timestep
        post_transition_data = {
            "reward": [],
            "terminated": []
        }
        # Data for the next step we will insert in order to select an action
        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        # Receive data back for each unterminated env
        for idx, parent_conn in enumerate(self.parent_conns):
            if not terminated[idx]:
                data = parent_conn.recv()
                # Remaining data for this current timestep
                post_transition_data["reward"].append((data["reward"],))

                episode_returns[idx] += data["reward"]
                episode_lengths[idx] += 1
                if not test_mode:
                    self.env_steps_this_run += 1

                env_terminated = False
                if data["terminated"]:
                    final_env_infos.append(data["info"])
                if data["terminated"] and not data["info"].get("episode_limit", False):
                    env_terminated = True
                terminated[idx] = data["terminated"]
                post_transition_data["terminated"].append((env_terminated,))

                # Data for the next timestep needed to select an action
                pre_transition_data["state"].append(data["state"])
                pre_transition_data["avail_actions"].append(data["avail_actions"])
                pre_transition_data["obs"].append(data["obs"])

        # Add post_transiton data into the batch
        self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

        # Move onto the next timestep
        self.t += 1

        # Add the pre-transition data
        self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

    if not test_mode:
        self.t_env += self.env_steps_this_run

    # Get stats back for each env
    for parent_conn in self.parent_conns:
        parent_conn.send(("get_stats",None))

    env_stats = []
    for parent_conn in self.parent_conns:
        env_stat = parent_conn.recv()
        env_stats.append(env_stat)

    cur_stats = self.test_stats if test_mode else self.train_stats
    cur_returns = self.test_returns if test_mode else self.train_returns
    log_prefix = "test_" if test_mode else ""
    infos = [cur_stats] + final_env_infos
    cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
    cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
    cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

    cur_returns.extend(episode_returns)

    n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
    if test_mode and (len(self.test_returns) == n_test_runs):
        self._log(cur_returns, cur_stats, log_prefix)
    elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
        self._log(cur_returns, cur_stats, log_prefix)
        if hasattr(self.mac.action_selector, "epsilon"):
            self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
        self.log_train_stats_t = self.t_env

    return self.batch


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            remote.send({
                # Data for the next timestep needed to pick an action
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == 'get_state':
            remote.send({
                'state':env.get_state()
            })
        elif cmd == 'get_obs':
            remote.send({
                'obs':env.get_obs()
            })
        elif cmd == "reset":
            env.reset()
        elif cmd == 'get_avail_agent_actions':
            remote.send({
                'avail_agent_actions': env.get_avail_agent_actions(data)
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done):
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return