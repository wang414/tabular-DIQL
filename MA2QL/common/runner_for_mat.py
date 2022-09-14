class Runner_for_mat:
    def __init__(self, env, args):
        self.env = env

        # 用来在一个稀疏奖赏的环境上评估算法的好坏，胜利为1，失败为-1，其他普通的一步为0


        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map + '/' + '{}x{}'.format(self.args.rnn_hidden_dim,self.args.qmix_hidden_dim)
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
        self.init_run_num = run_num
    def run(self, num):
        final_num = self.init_run_num + num
        logdir = self.save_path + '/' + 'run{}/logs'.format(final_num)
        if not os.path.exists(self.save_path):
            os.makedirs(logdir)
        logger = SummaryWriter(str(logdir))


        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        win_rates = []
        episode_rewards = []
        v_inf = []
        train_steps = 0
        # print('Run {} start'.format(num))
        for epoch in range(self.args.n_epoch):
            print('Run {}, train epoch {} alg {} rnn {} qmix {}'.format(final_num, epoch,self.args.alg,self.args.rnn_hidden_dim,self.args.qmix_hidden_dim))

            # if epoch % self.args.evaluate_cycle == 0:
            #     win_rate, episode_reward = self.evaluate()
            #     # print('win_rate is ', win_rate)
            #     win_rates.append(win_rate)
            #     episode_rewards.append(episode_reward)
            #
            #     print('evaluate_episode_reward = {}'.format(episode_reward))
            #
            #     plt.cla()
            #     plt.subplot(2, 1, 1)
            #     plt.plot(range(len(win_rates)), win_rates)
            #     plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
            #     plt.ylabel('win_rate')
            #
            #     plt.subplot(2, 1, 2)
            #     plt.plot(range(len(episode_rewards)), episode_rewards)
            #     plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
            #     plt.ylabel('episode_rewards')
            #
            #     plt.savefig(self.save_path + '/plt_{}.png'.format(final_num), format='png')
            #     np.save(self.save_path + '/win_rates_{}'.format(final_num), win_rates)
            #     np.save(self.save_path + '/episode_rewards_{}'.format(final_num), episode_rewards)

            episodes = []
            # 收集self.args.n_episodes个episodes
            mean_episode_reward = 0
            for episode_idx in range(self.args.n_episodes):
                # print('sample_for_epoch_{}'.format(epoch))
                episode, episode_reward,_,episode_step = self.rolloutWorker.generate_episode(episode_idx)
                mean_episode_reward += episode_reward
                t_env
                print('train_episode_reward = {}'.format(episode_reward))
                print('mean_episode_reward = {}'.format(mean_episode_reward))
                episodes.append(episode)
            mean_episode_reward /= self.args.n_episodes
            print('final_mean_episode_reward = {}'.format(mean_episode_reward))
            logger.add_scalar('mean_episode_reward',mean_episode_reward,epoch)
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 and (self.args.alg.find('div') == -1 or self.args.check_alg) or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, epsilon = self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)

                if self.args.check_alg and self.args.alg == 'qmix':
                    update_num = self.args.train_steps_for_qmix
                else:
                    update_num = self.args.train_steps
                for train_step in range(update_num):
                    print('train_step {}'.format(train_step))
                    if self.args.check_alg and self.args.alg == 'qmix':
                        sample_size = self.args.batch_size_for_qmix
                    else:
                        sample_size = self.args.batch_size
                    print('sample_size = {}'.format(sample_size))
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, sample_size))
                    ret = self.agents.train(mini_batch, train_steps,epsilon = self.rolloutWorker.epsilon,logger = logger)

                    if self.args.alg.endswith('div_soft'):
                        self.agents.policy.soft_update_policy()
                    if ret is not None:
                        if self.args.alg.startswith('qmix_div') and self.args.check_alg == False or self.args.alg.startswith('qplex_div'):
                            q_loss_ret, policy_loss, bias_ret = ret
                            q_loss, q_target, q_val, q_bias = q_loss_ret
                            logger.add_scalar('q_loss', q_loss, train_steps)
                            logger.add_scalar('q_target', q_target, train_steps)
                            logger.add_scalar('q_val', q_val, train_steps)
                            logger.add_scalar('q_bias', q_bias, train_steps)
                            logger.add_scalar('policy_loss', policy_loss, train_steps)
                            logger.add_scalar('bias_no_baseline', bias_ret[0], train_steps)
                            logger.add_scalar('bias_with_baseline', bias_ret[1], train_steps)
                        if self.args.mat_test:
                            v_inf.append(ret)
                            if train_steps % 20 == 0:
                                np.save(self.save_path + '/v_inf_{}'.format(final_num), win_rates)
                    train_steps += 1
                if self.args.alg.endswith('div_hard'):
                    self.agents.policy.hard_update_policy()
            if self.args.target_update_cycle is not None and train_steps > 0 and train_steps % self.args.target_update_cycle == 0:
                print('update_target!!!!!!!!!!!!!')
                self.agents.policy.hard_update_critic()

        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(win_rates)), win_rates)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(final_num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(final_num), win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(final_num), episode_rewards)


        logger.close()
    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            # print('evaluate_for_epoch_{}'.format(epoch))
            _, episode_reward,win_tag = self.rolloutWorker.generate_episode(evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def evaluate_sparse(self):
        win_number = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward,win_tag = self.evaluateWorker.generate_episode(evaluate=True)
            result = 'win' if episode_reward > 0 else 'defeat'
            print('Epoch {}: {}'.format(epoch, result))
            if episode_reward > 0:
                win_number += 1
        self.env_evaluate.close()
        return win_number / self.args.evaluate_epoch








