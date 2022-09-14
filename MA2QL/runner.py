import numpy as np
import os
import time
from common.rollout import RolloutWorker, CommRolloutWorker
from tensorboardX import SummaryWriter
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from pathlib import Path
# from tqdm import tqdm
# from smac.env import StarCraft2Env




class Runner:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        # if self.args.mat_test:
        #     print('self.args.mat_test = {}'.format(self.args.mat_test))
        #     for i in range(3):
        #         for j in range(3):
        #             actions = [i,j]
        #             reward, terminated, info = self.env.step(actions)
        #             print('checking environment actions = {}, reward = {}'.format(actions,reward))

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)

        print('self.args.alg = {}, self.args.epsilon_reset = {}'.format(self.args.alg,self.args.epsilon_reset))
        if self.args.alg == 'aga' and self.args.epsilon_reset:

            self.rolloutWorker.epsilon_reset()

        if args.alg in ['iac','i-dmac','i-sac'] or args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1 or args.alg.find('div') > -1 or args.on_policy_multi_batch:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)

        self.mp_tag = self.args.n_episodes > 1
        # 用来保存plt和pkl
        if self.args.special_name is None:
            self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map + '/' + '{}x{}'.format(self.args.rnn_hidden_dim,self.args.qmix_hidden_dim)
        else:
            self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map + '/' + '{}'.format(
                self.args.special_name)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        model_dir = Path(self.save_path)
        if not model_dir.exists():
            run_num = 1
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                             model_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                run_num = 1
            else:
                run_num = max(exst_run_nums) + 1
        self.win_rates = []
        self.episode_rewards = []
        self.init_run_num = run_num
    def plt(self, num,mode = 'epoch'):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()

        if mode == 'epoch':
            plt.subplot(2, 1, 1)
            plt.plot(self.episode_count, self.win_rates)
            plt.xlabel('epoch')
            plt.ylabel('win_rate')

            plt.subplot(2, 1, 2)
            plt.plot(self.episode_count, self.episode_rewards)
            plt.xlabel('epoch')
            plt.ylabel('episode_rewards')
        elif mode == 'step':
            plt.subplot(2, 1, 1)
            plt.plot(self.t_envs, self.win_rates)
            plt.xlabel('t_envs')
            plt.ylabel('win_rate')

            plt.subplot(2, 1, 2)
            plt.plot(self.t_envs, self.episode_rewards)
            plt.xlabel('t_envs')
            plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}_mode_{}.png'.format(num,mode), format='png')


        if self.args.mat_test:
            if self.args.check_mat_policy:
                plt.figure()

                plt.cla()

                plt.plot(self.t_envs, self.pi_judge)
                plt.xlabel('step')
                plt.ylabel('pi_judge')

                plt.savefig(self.save_path + '/pi_judge_{}_mode_{}.png'.format(num, mode), format='png')
            if self.env.evaluate_mat:
                plt.figure()

                plt.cla()

                plt.plot(self.long_step_count,self.traverse)
                plt.xlabel('step')
                plt.ylabel('traverse')

                plt.savefig(self.save_path + '/traverse_{}_mode_{}.png'.format(num, mode), format='png')

                plt.figure()

                plt.cla()

                plt.plot(self.long_step_count, self.relative_traverse)
                plt.xlabel('step')
                plt.ylabel('traverse')

                plt.savefig(self.save_path + '/relative_traverse_{}_mode_{}.png'.format(num, mode), format='png')

                plt.figure()

                plt.cla()

                plt.plot(self.long_step_count, self.static_return)
                plt.xlabel('step')
                plt.ylabel('static_return')

                plt.savefig(self.save_path + '/static_return_{}_mode_{}.png'.format(num, mode), format='png')

    def run(self, num):
        if self.args.mat_test:
            if self.args.check_mat_policy:
                self.pi_judge = []
            if self.env.evaluate_mat:
                self.env.reset_evaluate()
                self.traverse = []
                self.abs_traverse = []
                self.relative_traverse = []
                self.long_step_count = []
                self.static_return = []
        self.win_rates = []
        self.episode_rewards = []
        self.t_envs = []
        self.episode_count = []

        model_dir = Path(self.save_path)
        if not model_dir.exists():
            run_num = 1
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                             model_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                run_num = 1
            else:
                run_num = max(exst_run_nums) + 1
        final_num = run_num
        logdir = self.save_path + '/' + 'run{}/logs'.format(final_num)
        if not os.path.exists(self.save_path):
            os.makedirs(logdir)
        logger = SummaryWriter(str(logdir))
        np.save(self.save_path + '/args_{}'.format(final_num), self.args)
        print('args_type = {}'.format(type(self.args)))
        with open(self.save_path + '/args_txt_{}.txt'.format(final_num),'w') as f:
            f.write(str(self.args))
        #
        # plt.figure()
        # plt.axis([0, self.args.n_epoch, 0, 100])
        train_steps = 0

        # print('Run {} start'.format(num))
        t_env = 0
        start_time = time.time()
        last_t_log = -self.args.step_log_interval - 1
        if self.args.log_mode == 'epoch':
            epoch_len = self.args.n_epoch // self.args.n_episodes + 1
        elif self.args.log_mode == 'step':
            epoch_len = self.args.t_max
        soft_mean_rewards = 0

        n_episodes = self.args.n_episodes
        if self.args.use_minibatch:
            sample_size = self.args.aga_minibatch
        else:
            sample_size = self.args.batch_size

        for epoch in range(epoch_len):
            curr_time = time.time()
            used_time = curr_time - start_time + 1e-12
            if epoch > 0 and (epoch * n_episodes) % sample_size == 0:
                print('Run_current {} Run_total {}, train epoch {} episode_num = {} t_env = {} alg {} rnn {} qmix {}'.format(
                num + 1, final_num, epoch, epoch * self.args.n_episodes, t_env, self.args.alg, self.args.rnn_hidden_dim,
                self.args.qmix_hidden_dim))
                print('start for {} second , speed = {} steps/second'.format(used_time,t_env / used_time))
                print('soft_mean_rewards = {}'.format(soft_mean_rewards))
                print('t_env = {}, t_env - last_t_log = {}, self.args.step_log_interval = {}, tag = {} '.format(t_env , t_env - last_t_log, self.args.step_log_interval ,( t_env  -  last_t_log) / self.args.step_log_interval ))
            if self.args.log_mode == 'epoch' and epoch % self.args.evaluate_cycle  == 0 or \
                self.args.log_mode == 'step' and( t_env  -  last_t_log) / self.args.step_log_interval >= 1.0:
                #MULTIPROCESS
                win_rate, episode_reward = self.evaluate()
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.t_envs.append(t_env)
                self.episode_count.append(epoch * self.args.n_episodes)
                last_t_log = t_env
                # print('win_rate is ', win_rate)
                if self.args.mat_test:
                    if self.args.check_mat_policy:
                        p_j = self.agents.policy.check_policy()
                        self.pi_judge.append(p_j)
                    if self.env.evaluate_mat:
                        traverse, abs_traverse, relative_traverse,static_return = self.env.eval_traverse()
                        self.traverse.append( traverse)
                        self.abs_traverse.append(abs_traverse)
                        self.relative_traverse.append(relative_traverse)
                        self.static_return.append(static_return)
                        self.long_step_count.append(self.env.long_step_count)
                if self.args.auto_draw:
                    self.plt(final_num, mode=self.args.log_mode)

                print('evaluate_episode_reward = {}, t_env = {}'.format(episode_reward, t_env))

                np.save(self.save_path + '/win_rates_{}'.format(final_num), self.win_rates)
                np.save(self.save_path + '/episode_rewards_{}'.format(final_num), self.episode_rewards)
                np.save(self.save_path + '/t_env_{}'.format(final_num), self.t_envs)
                np.save(self.save_path + '/episode_count_{}'.format(final_num), self.episode_count)
                if self.args.mat_test:
                    if self.env.evaluate_mat:
                        np.save(self.save_path + '/traverse_{}'.format(final_num), self.traverse)
                        np.save(self.save_path + '/static_return_{}'.format(final_num), self.static_return)
                        np.save(self.save_path + '/abs_traverse_{}'.format(final_num), self.abs_traverse)
                        np.save(self.save_path + '/relative_traverse_{}'.format(final_num), self.relative_traverse)

            # 收集self.args.n_episodes个episodes
            mean_episode_reward = 0
            # print('sample_for_epoch_{}'.format(epoch))
            episode, episode_reward, _ ,episode_step = self.rolloutWorker.generate_episode()
            # MULTIPROCESS
            if self.mp_tag:
                mean_episode_reward += np.sum(episode_reward)
            else:

                mean_episode_reward += episode_reward
            t_env += episode_step
            # print('train_episode_reward = {}'.format(episode_reward))



            mean_episode_reward /= self.args.n_episodes
            # print('final_mean_episode_reward = {}'.format(mean_episode_reward))
            logger.add_scalar('mean_episode_reward',mean_episode_reward,epoch)
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            # MULTIPROCESS
            soft_mean_rewards = 0.99 * soft_mean_rewards + 0.01 * mean_episode_reward

            episode_batch = episode
            if self.args.alg == 'dapo' or self.args.alg == 'ippo':
                self.buffer.store_episode(episode_batch)
                sample_size = self.args.batch_size
                update_num = self.args.train_steps
                n_episodes = self.args.n_episodes
                if epoch > 0 and (epoch * n_episodes) % sample_size == 0:
                    if self.args.aga_tag and self.args.idv_para and self.args.optim_reset and self.agents.policy.aga_update_tag:
                        print('reset_optim for agent {}'.format(self.agents.policy.update_index))
                        self.agents.policy.reset_optim(self.agents.policy.update_index,t_env)
                    for train_step in range(update_num):
                        print('train_step {}'.format(train_step))


                        mini_batch = self.buffer.sample(min(self.buffer.current_size, sample_size), latest=True)

                        print('sample_size = {}'.format(min(self.buffer.current_size, sample_size)))
                        ret = self.agents.train(mini_batch, train_steps, epsilon=self.rolloutWorker.epsilon, logger=logger)


                        if self.args.alg.endswith('div_soft'):
                            self.agents.policy.soft_update_policy()
                        if ret is not None:
                            if self.args.alg == 'ippo':
                                q_loss,policy_loss = ret
                                logger.add_scalar('q_loss', q_loss, train_steps)
                                logger.add_scalar('policy_loss', policy_loss, train_steps)
                        train_steps += 1
                    if self.args.aga_tag:
                        self.agents.policy.update_update_tag()
                    self.agents.policy.hard_update_policy()
            elif self.args.alg == 'aga':
                self.buffer.store_episode(episode_batch)

                sample_size = self.args.aga_sample_size

                if self.args.new_period and self.agents.policy.aga_update_tag:
                    update_num = self.args.train_steps * self.args.period
                else:
                    update_num = self.args.train_steps
                n_episodes = self.args.n_episodes
                if epoch > 0 and (epoch * n_episodes) % sample_size == 0:

                    if self.args.aga_tag and self.args.idv_para and self.args.optim_reset and self.agents.policy.aga_update_tag:
                        print('reset_optim for agent {}'.format(self.agents.policy.update_index))
                        self.agents.policy.reset_optim(self.agents.policy.update_index,t_env)
                    for train_step in range(update_num):
                        print('train_step {}'.format(train_step))


                        batch_size = self.args.batch_size


                        if self.args.sample_mode == 'episode':
                            real_sample = min(self.buffer.current_size, batch_size )
                        elif self.args.sample_mode == 'step':
                            real_sample = min(self.buffer.current_step_size, batch_size * self.args.episode_limit // update_num)


                        mini_batch = self.buffer.sample(real_sample)
                        print('sample_size = {}, sample_mode = {}'.format(real_sample,self.args.sample_mode))
                        ret = self.agents.train(mini_batch, train_steps, epsilon=self.rolloutWorker.epsilon, logger=logger)

                        if not self.args.target_dec:
                            print('soft_update_critic')
                            self.agents.policy.soft_update_critic()

                        if self.args.alg.endswith('div_soft'):
                            self.agents.policy.soft_update_policy()
                        if ret is not None:
                            if self.args.alg.startswith(
                                    'qmix_div') and self.args.check_alg == False or self.args.alg.startswith('qplex_div'):
                                q_loss_ret, policy_loss, bias_ret = ret
                                q_loss, q_target, q_val, q_bias = q_loss_ret
                                logger.add_scalar('q_loss', q_loss, train_steps)
                                logger.add_scalar('q_target', q_target, train_steps)
                                logger.add_scalar('q_val', q_val, train_steps)
                                logger.add_scalar('q_bias', q_bias, train_steps)
                                logger.add_scalar('policy_loss', policy_loss, train_steps)
                                logger.add_scalar('bias_no_baseline', bias_ret[0], train_steps)
                                logger.add_scalar('bias_with_baseline', bias_ret[1], train_steps)
                        train_steps += 1
                        # print('learning rate = {}'.format(self.agents.policy.optimizer[0].state_dict()['param_groups'][0]['lr']))
                    if self.args.aga_tag:
                        self.agents.policy.update_update_tag()
                    if self.args.value_clip:
                        self.agents.policy.update_old()
                    if self.args.epsilon_reset:
                        print('before reset epsilon = {}'.format(self.rolloutWorker.epsilon))
                        self.rolloutWorker.epsilon_reset()
                        print('after reset epsilon = {}'.format(self.rolloutWorker.epsilon))
                if self.args.alg.endswith('div_hard'):
                    self.agents.policy.hard_update_policy()
            elif self.args.alg.find('coma') > -1 and (self.args.alg.find('div') == -1 or self.args.check_alg) or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                print('on policy')
                if self.args.on_policy_multi_batch:
                    self.buffer.store_episode(episode_batch)
                    n_episodes = self.args.n_episodes
                    sample_size = self.args.batch_size
                    if epoch > 0 and (epoch * n_episodes) % sample_size == 0:
                        mini_batch = self.buffer.sample(min(self.buffer.current_size, sample_size), latest=True)
                        self.agents.train(mini_batch, train_steps, epsilon=self.rolloutWorker.epsilon)
                else:
                    self.agents.train(episode_batch, train_steps, epsilon=self.rolloutWorker.epsilon)
                train_steps += 1
            elif self.args.alg in ['iac','i-dmac','i-sac']:
                self.buffer.store_episode(episode_batch)

                if self.args.alpha_decay:
                    curr_alpha = 1 / self.args.reward_scale
                    new_alpha = max(self.args.alpha_min, (1 - t_env / self.args.alpha_step) * self.args.alpha_max)
                    print('t_env = {} curr_alpha = {} new_alpha = {}'.format(t_env, curr_alpha, new_alpha))
                    self.args.reward_scale = 1 / new_alpha
                sample_size = self.args.batch_size
                update_num = self.args.train_steps
                if self.args.alg == 'iac' and not self.args.idv_off_policy:
                    update_tag = (epoch > 0 and (epoch * n_episodes) % sample_size == 0)
                    latest_tag = True
                else:
                    update_tag = True
                    latest_tag = False
                if update_tag:
                    for train_step in range(update_num):
                        print('train_step {}'.format(train_step))
                        mini_batch = self.buffer.sample(min(self.buffer.current_size, sample_size),latest=latest_tag)
                        print('sample_size = {}'.format(min(self.buffer.current_size, sample_size)))
                        ret = self.agents.train(mini_batch, train_steps, epsilon=self.rolloutWorker.epsilon, logger=logger)
                        # print('in runner ,len_ret = {}, ret = {}'.format(len(ret),ret))
                        policy_loss = ret[0]
                        q_loss = ret[1]
                        if self.args.classic_sac:
                            sac_v_loss = ret[2]
                            logger.add_scalar('sac_v_loss', sac_v_loss, train_steps)
                        if self.args.alg in ['i-dmac','i-sac'] and not self.args.classic_sac:
                            bias_ret = ret[2]
                            logger.add_scalar('bias_no_baseline', bias_ret[0], train_steps)
                            logger.add_scalar('bias_with_baseline', bias_ret[1], train_steps)
                        if self.args.alg == 'i-dmac':
                            self.agents.policy.soft_update_policy(self.args.dmac_tau)
                        train_steps += 1

                        logger.add_scalar('q_loss', q_loss, train_steps)
                        logger.add_scalar('policy_loss', policy_loss, train_steps)

            else:
                print('off_policy')
                self.buffer.store_episode(episode_batch)

                if self.args.alpha_decay:
                    curr_alpha = 1 / self.args.reward_scale
                    new_alpha = max(self.args.alpha_min, (1 - t_env / self.args.alpha_step ) * self.args.alpha_max )
                    print('t_env = {} curr_alpha = {} new_alpha = {}'.format(t_env,curr_alpha,new_alpha))
                    self.args.reward_scale = 1 / new_alpha

                if self.args.check_alg and self.args.alg == 'qmix':
                    update_num = self.args.train_steps_for_qmix
                else:
                    update_num = self.args.train_steps
                sample_size = self.args.batch_size
                update_num = self.args.train_steps
                for train_step in range(update_num):
                    print('train_step {}'.format(train_step))

                    if self.args.mix_sample:
                        mini_batch = self.buffer.mix_sample(self.args.mix_on_batch_size, self.args.mix_off_batch_size,
                                                            self.args.train_steps)
                        off_sample_size = min(self.buffer.current_size, self.args.mix_off_batch_size)
                        sample_size = min(self.buffer.current_size,
                                          self.args.mix_on_batch_size + self.args.mix_off_batch_size)
                        on_sample_size = sample_size - self.args.mix_off_batch_size
                        print('off_sample_size = {},on_sample_size = {}'.format(off_sample_size, on_sample_size))
                    else:
                        mini_batch = self.buffer.sample(min(self.buffer.current_size, sample_size))
                        print('sample_size = {}'.format(min(self.buffer.current_size, sample_size)))
                    ret = self.agents.train(mini_batch, train_steps,epsilon = self.rolloutWorker.epsilon,logger = logger)
                    # print('ret = {}'.format(ret))
                    if self.args.alg.endswith('div_soft') or self.args.alg == 'dop_dmac':
                        self.agents.policy.soft_update_policy(self.args.dmac_tau)
                    if ret is not None:
                        if self.args.logger_check_output or self.args.alg.startswith('qmix_div') and self.args.check_alg == False or self.args.alg.startswith('qplex_div'):
                            q_loss_ret, policy_loss, bias_ret = ret
                            q_loss, q_target, q_val, q_bias = q_loss_ret
                            key_list = ['q_loss','q_target','q_val','q_bias','policy_loss','bias_no_baseline','bias_with_baseline']
                            value_list = [q_loss, q_target, q_val, q_bias,policy_loss,bias_ret[0],bias_ret[1] ]
                            for k,v in zip(key_list,value_list):
                                if v is not None:
                                    print('logger for key {} value = {}'.format(k,v))
                                    logger.add_scalar(k, v, train_steps)
                    train_steps += 1
                if self.args.alg.endswith('div_hard'):
                    self.agents.policy.hard_update_policy()
            if (self.args.alg != 'aga') and self.args.target_update_cycle is not None and train_steps > 0 and train_steps % self.args.target_update_cycle == 0:
                print('update_target!!!!!!!!!!!!!')
                self.agents.policy.hard_update_critic()

            if self.args.log_mode == 'step' and t_env >= self.args.t_max:
                break

        if self.args.auto_draw:
            self.plt(final_num, mode=self.args.log_mode)
        np.save(self.save_path + '/win_rates_{}'.format(final_num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(final_num), self.episode_rewards)
        np.save(self.save_path + '/t_env_{}'.format(final_num), self.t_envs)
        np.save(self.save_path + '/episode_count_{}'.format(final_num), self.episode_count)
        if self.args.mat_test:
            if self.env.evaluate_mat:
                np.save(self.save_path + '/traverse_{}'.format(final_num), self.traverse)
                np.save(self.save_path + '/static_return_{}'.format(final_num), self.static_return)
                np.save(self.save_path + '/abs_traverse_{}'.format(final_num), self.abs_traverse)
                np.save(self.save_path + '/relative_traverse_{}'.format(final_num), self.relative_traverse)

        self.agents.policy.save_model(train_steps)

        logger.close()
    def evaluate(self):
        win_number = 0
        episode_rewards = 0

        epoch_num = max(1 , self.args.evaluate_epoch // self.args.n_episodes)
        for epoch in range(epoch_num):
            print('evaluate_for_epoch_{}'.format(epoch))
            _, episode_reward,win_tag,_ = self.rolloutWorker.generate_episode(evaluate=True)
            print('episode_reward = {}\nwin_tag = {}'.format(episode_reward,win_tag))
            if self.mp_tag:
                episode_rewards += np.sum(episode_reward)
                win_number += np.sum(win_tag)
            else:
                episode_rewards += episode_reward
                if win_tag:
                    win_number += 1
            print('sum_episode_reward = {}'.format(episode_rewards))
        eval_episode_num = epoch_num * self.args.n_episodes
        return win_number / eval_episode_num, episode_rewards / eval_episode_num


