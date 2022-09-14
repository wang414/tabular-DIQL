import numpy as np
import torch
# from policy.ippo import IdvPolicy
from policy.qplex import QPLEX
from policy.gaac import GAAC
from policy.vdn import VDN
from policy.qmix import QMIX
from policy.coma import COMA
from policy.coma_div import COMA_DIV
from policy.aga import AGA
from policy.qmix_div import QMIX_DIV
from policy.maac_div import MAAC_DIV
from policy.maac import MAAC
from policy.reinforce import Reinforce
from policy.central_v import CentralV
from policy.qtran_alt import QtranAlt
from policy.qtran_base import QtranBase
from policy.maven import MAVEN
from torch.distributions import Categorical

# Agent no communication
class Agents:
    def __init__(self, args):
        self.mp_tag = args.n_episodes > 1
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'vdn':
            self.policy = VDN(args)
        elif args.alg == 'qmix':
            self.policy = QMIX(args)
        elif args.alg.startswith('gaac') or args.alg.startswith('maac'):
            self.policy = GAAC(args)
        elif args.alg.startswith('qmix_div') :
            self.policy = QMIX_DIV(args)
        elif args.alg == 'coma':
            self.policy = COMA(args)
        elif args.alg.startswith('coma_div') :
            print('yes get it')
            self.policy = COMA_DIV(args)
        elif args.alg.startswith('qplex'):
            self.policy = QPLEX(args)
        elif args.alg == 'qtran_alt':
            self.policy = QtranAlt(args)
        elif args.alg == 'aga':
            self.policy = AGA(args)
        elif args.alg == 'qtran_base':
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            self.policy = Reinforce(args)
        # elif args.alg in ['ippo','iac','i-dmac','i-sac']:
        #     self.policy = IdvPolicy(args)
        else:
            raise Exception("No such algorithm")
        self.args = args

        print('Init Agents')

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        # print('choose_action::epsilon = {}'.format(epsilon))
        # if self.args.alg.startswith('maac'):
        #     action = self.policy.choose_action(obs, avail_actions, agent_num)
        #     return action
        # print('choose_action_for_{}'.format(agent_num))

        inputs = obs.copy()
        if self.mp_tag:
            avail_actions_ind = []
            for ava in avail_actions:
                ava_id =  np.nonzero(ava)[0]
                ava_id = np.array(ava_id,dtype = np.int32)
                avail_actions_ind.append(ava_id)
        else:
            avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose
        # print('avail_actions_ind_shape = {}'.format(np.shape(avail_actions_ind)))
        # print('avail_actions_ind = {}'.format(avail_actions_ind))
        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.
        if self.mp_tag:
            agent_id = np.expand_dims(agent_id,axis = 0)
            agent_id = np.tile(agent_id,[self.args.n_episodes,1])
        # print('input_shape = {}'.format(np.shape(inputs)))
        # print('last_action_shape = {}'.format(np.shape(last_action)))
        # print('agent_id_shape = {}'.format(np.shape(agent_id)))
        # print('no_rnn = {}'.format(self.args.no_rnn))
        if not self.args.no_rnn:
            if self.args.last_action:
                inputs = np.hstack((inputs, last_action))
            if self.args.reuse_network:
                inputs = np.hstack((inputs, agent_id))

        if self.args.alg in ['ippo','iac','i-dmac','i-sac'] or self.args.alg.startswith('qmix_div') and self.args.check_alg == False or self.args.alg.startswith('qplex_div'):
            if self.args.behavior_check_output:
                print('hidden_state_case 1')
            if self.args.idv_para:
                hidden_state = self.policy.eval_policy_hidden[agent_num]
            else:
                hidden_state = self.policy.eval_policy_hidden[:, agent_num, :]
        else:
            # print('hidden_state_case 2')
            if self.args.idv_para:
                if self.args.target_dec and not self.policy.aga_update_tag and not evaluate:
                    if self.args.behavior_check_output:
                        print('hidden_state_case 2')
                    hidden_state = self.policy.target_hidden[agent_num]
                else:
                    if self.args.behavior_check_output:
                        print('hidden_state_case 3')
                    hidden_state = self.policy.eval_hidden[agent_num]
            else:
                if self.args.target_dec and not self.policy.aga_update_tag and not evaluate:
                    if self.args.behavior_check_output:
                        print('hidden_state_case 4')
                    hidden_state = self.policy.target_hidden[:, agent_num, :]
                else:
                    if self.args.behavior_check_output:
                        print('hidden_state_case 5')
                    hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        if not self.mp_tag:
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
            avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        else:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            avail_actions = torch.tensor(avail_actions, dtype=torch.float32)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        elif self.args.alg  in ['ippo','iac','i-dmac','i-sac'] or  self.args.alg.startswith('qmix_div') and self.args.check_alg == False or self.args.alg.startswith('qplex_div'):
            #
            if self.args.behavior_check_output:
                print('q_value_case 1')
            if self.args.idv_para:
                q_value, self.policy.eval_policy_hidden[agent_num] = self.policy.eval_policy_rnn[agent_num].forward(inputs,
                                                                                                               hidden_state)
            else:
                q_value, self.policy.eval_policy_hidden[:, agent_num, :] = self.policy.eval_policy_rnn.forward(inputs,
                                                                                                               hidden_state)
        else:
            # print('q_value_case 2')
            if self.args.idv_para:
                if self.args.target_dec and not self.policy.aga_update_tag and not evaluate:
                    if self.args.behavior_check_output:
                        print('q_value_case 2')
                    q_value, self.policy.target_hidden[agent_num] = self.policy.target_rnn[agent_num](inputs, hidden_state)
                else:
                    if self.args.behavior_check_output:
                        print('q_value_case 3')
                    q_value, self.policy.eval_hidden[agent_num] = self.policy.eval_rnn[agent_num](inputs, hidden_state)
            else:
                if self.args.target_dec and not self.policy.aga_update_tag and not evaluate:
                    if self.args.behavior_check_output:
                        print('q_value_case 4')
                    q_value, self.policy.target_hidden[:, agent_num, :] = self.policy.target_rnn(inputs, hidden_state)
                else:
                    if self.args.behavior_check_output:
                        print('q_value_case 5')
                    # print('inputs = {} hidden_state = {} eval_rnn {}'.format(inputs.shape,hidden_state.shape,self.policy.eval_rnn))
                    q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # choose action from q value
        if self.args.alg in ['ippo','iac','i-dmac','i-sac']:
            if self.args.behavior_check_output:
                print('action case 1')
            if self.args.alg == 'ippo':
                action = self._choose_action_for_ippo(q_value.cpu(), avail_actions, epsilon, evaluate)
            else:
                action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate,use_epsilon_greedy = 0)

        elif self.args.alg.find('div') > -1 and not self.args.check_alg or self.args.alg.startswith('coma')  or self.args.alg.startswith('gaac') or self.args.alg.startswith('maac') or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            if evaluate and self.args.test_greedy:
                if self.args.behavior_check_output:
                    print('q_action case test_greedy')
                action = torch.argmax(q_value)
            else:
                if self.args.alg== 'maac' or self.args.alg == 'gaac' or self.args.alg.find('div') > -1:
                    use_epsilon_greedy = 0
                else:
                    use_epsilon_greedy = 1
                # print('soft_max_action, use_epsilon_greedy = {}'.format(use_epsilon_greedy))
                if self.args.behavior_check_output:
                    print('action case 2 use epsilon greedy = {}'.format(use_epsilon_greedy))
                action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate,use_epsilon_greedy)
        elif self.args.alg == 'maddpg':
            if self.args.behavior_check_output:
                print('action case 3')
            action = self._choose_action_for_maddpg(q_value.cpu(), avail_actions,evaluate)
        else:
            # print('epsilon_greedy, epsilon = {}'.format(epsilon))
            if self.args.behavior_check_output:
                print('action case 4')
            q_value[avail_actions == 0.0] = - float("inf")
            if self.mp_tag:
                action = []
                for ava_id,q in zip(avail_actions_ind,q_value):
                    # print('ava_id = {}, q = {}'.format(ava_id,q))
                    if evaluate:
                        if self.args.behavior_check_output:
                            print('q_action case 1')
                        a = torch.argmax(q)
                    else:
                        if self.args.alg == 'aga' and self.args.aga_tag == True:
                            if self.args.single_exp == True:
                                if self.policy.aga_update_tag:
                                    if agent_num == self.policy.update_index:
                                        if np.random.uniform() < epsilon:
                                            if self.args.behavior_check_output:
                                                print('q_action case 2 uniform')
                                            a = np.random.choice(ava_id)  # action是一个整数
                                        else:
                                            if self.args.behavior_check_output:
                                                print('q_action case 2 max')
                                            a = torch.argmax(q)
                                    else:
                                        if self.args.behavior_check_output:
                                            print('q_action case 3')
                                        a = torch.argmax(q)
                                else:
                                    if self.args.stop_other_exp:
                                        if self.args.behavior_check_output:
                                            print('q_action case 4')
                                        a = torch.argmax(q)
                                    else:
                                        if np.random.uniform() < epsilon:
                                            if self.args.behavior_check_output:
                                                print('q_action case 5 uniform')
                                            a = np.random.choice(ava_id)  # action是一个整数
                                        else:
                                            if self.args.behavior_check_output:
                                                print('q_action case 5 max')
                                            a = torch.argmax(q)
                            else:
                                if np.random.uniform() < epsilon:
                                    if self.args.behavior_check_output:
                                        print('q_action case 6 uniform')
                                    a = np.random.choice(ava_id)  # action是一个整数
                                else:
                                    if self.args.behavior_check_output:
                                        print('q_action case 6 max ')
                                    a = torch.argmax(q)
                        else:
                            if np.random.uniform() < epsilon:
                                if self.args.behavior_check_output:
                                    print('q_action case 7 uniform')
                                a = np.random.choice(ava_id)  # action是一个整数
                            else:
                                if self.args.behavior_check_output:
                                    print('q_action case 7 max')
                                a = torch.argmax(q)
                    # print('a = {}'.format(a))
                    action.append(a)
                action = torch.tensor(action,dtype = torch.int32)
            else:
                if evaluate:
                    if self.args.behavior_check_output:
                        print('q_action case 8')
                    action = torch.argmax(q_value)
                else:
                    if self.args.alg == 'aga' and self.args.aga_tag == True:
                        if self.args.single_exp == True:
                            if self.policy.aga_update_tag:
                                if agent_num == self.policy.update_index:
                                    # print('case 1')
                                    if np.random.uniform() < epsilon:
                                        if self.args.behavior_check_output:
                                            print('q_action case 9 uniform')
                                        action = np.random.choice(avail_actions_ind)  # action是一个整数
                                    else:
                                        if self.args.behavior_check_output:
                                            print('q_action case 9 max')
                                        action = torch.argmax(q_value)
                                else:
                                    # print('case 2')
                                    if self.args.behavior_check_output:
                                        print('q_action case 10')
                                    action = torch.argmax(q_value)
                            else:
                                if self.args.stop_other_exp:
                                    # print('case 3')
                                    if self.args.behavior_check_output:
                                        print('q_action case 11')
                                    action = torch.argmax(q_value)
                                else:
                                    # print('case 4')

                                    if np.random.uniform() < epsilon:
                                        if self.args.behavior_check_output:
                                            print('q_action case 12 uniform')
                                        action = np.random.choice(avail_actions_ind)  # action是一个整数
                                    else:
                                        if self.args.behavior_check_output:
                                            print('q_action case 12 max')
                                        action = torch.argmax(q_value)
                        else:
                            # print('case 5')
                            if np.random.uniform() < epsilon:
                                if self.args.behavior_check_output:
                                    print('q_action case 13 uniform')
                                action = np.random.choice(avail_actions_ind)  # action是一个整数
                            else:
                                if self.args.behavior_check_output:
                                    print('q_action case 13 max')
                                action = torch.argmax(q_value)
                    else:
                        # print('case 6')
                        if np.random.uniform() < epsilon:
                            if self.args.behavior_check_output:
                                print('q_action case 14 uniform')
                            action = np.random.choice(avail_actions_ind)  # action是一个整数
                        else:
                            if self.args.behavior_check_output:
                                print('q_action case 14 max')
                            action = torch.argmax(q_value)
        # print('action = {}'.format(action))
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False,use_epsilon_greedy = True):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        if use_epsilon_greedy:
            # add noise of epsilon
            prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
            prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

            """
            不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
            会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
            """

            if epsilon == 0 and evaluate:
                action = torch.argmax(prob,dim = -1)
            else:
                action = Categorical(prob).sample().long()
        else:
            # print('evaluate = {}'.format(evaluate))
            prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0
            action = Categorical(prob).sample().long()
        return action

    def _choose_action_for_ippo(self, inputs, avail_actions, epsilon, evaluate=False,use_epsilon_greedy = True):
        """
        :param inputs: # q_value of all actions
        """
        # action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布

        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[
            -1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        #
        # prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0
        action = Categorical(prob).sample().long()
        return action

    def _choose_action_for_maddpg(self, inputs, avail_actions, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        inputs[avail_actions == 0.0] = - 999999999
        # 先将Actor网络的输出通过softmax转换成概率分布

        if not evaluate:
            action = gumbel_softmax(inputs, hard=True)
        else:
            action = onehot_from_logits(inputs)
        action = torch.argmax(action, dim=-1)
        return action
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None,logger = None):  # coma needs epsilon for training
        print('train::epsilon = {}'.format(epsilon))
        # different episode has different length, so we need to get max length of the batch
        if self.args.sample_mode == 'step':
            max_episode_len = batch['o'].shape[1]
        else:
            max_episode_len = self._get_max_episode_len(batch)
        print('max_episode_len = {}'.format(max_episode_len))
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        ret = None
        if self.args.alg.startswith('coma_div') or self.args.alg.startswith('qmix_div') or self.args.alg.startswith('qplex_div'):
            ret = self.policy.learn(batch, max_episode_len, train_step, epsilon,logger = logger)
        elif self.args.alg in ['ippo','iac','i-dmac','i-sac']:
            ret = self.policy.learn(batch, max_episode_len, train_step, epsilon)
        else:
            self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
        if self.args.mat_test and not (self.args.alg in ['ippo','iac','i-dmac','i-sac']):
            if self.args.q_table_output:
                try:
                    ret = self.policy.get_all_q_table(batch,max_episode_len)
                except:
                    pass
        return ret

# Agent for communication
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init CommAgents')

    # 根据weights得到概率，然后再根据epsilon选动作
    def choose_action(self, weights, avail_actions, epsilon, evaluate=False):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # 可以选择的动作的个数
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # 在训练的时候给概率分布添加噪音
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            # 测试时直接选最大的
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = list()
        inputs.append(obs)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat([x for x in inputs], dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
        weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
        weights = weights.reshape(self.args.n_agents, self.args.n_actions)
        return weights.cpu()

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma在训练时也需要epsilon计算动作的执行概率
        # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)










