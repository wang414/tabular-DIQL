import torch
import numpy as np
import os
import argparse
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic
from torch.distributions import Categorical

MSELoss = torch.nn.MSELoss()

def td_lambda_target(batch, max_episode_len, q_targets, next_bias, args):  # 用来通过TD(lambda)计算y
    # batch维度为(episode个数, max_episode_len， n_agents，n_actions)
    # q_targets维度为(episode个数, max_episode_len， n_agents)
    # print('batch_padded = {}'.format(batch["padded"].shape))
    # print('batch_done = {}'.format(batch["done"].shape))
    episode_num = batch['o'].shape[0]
    mask = (1 - batch["padded"].float())  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
    done = (1 -  batch['terminated'].float()) # 用来把episode最后一条经验中的q_target置0
    # 把reward维度从(episode个数, max_episode_len, 1)变成(episode个数, max_episode_len, n_agents)
    r = batch['r']
    # 计算n_step_return
    '''
    1. 每条经验都有若干个n_step_return，所以给一个最大的max_episode_len维度用来装n_step_return
    最后一维,第n个数代表 n+1 step。
    2. 因为batch中各个episode的长度不一样，所以需要用mask将多出的n-step return置为0，
    否则的话会影响后面的lambda return。第t条经验的lambda return是和它后面的所有n-step return有关的，
    如果没有置0，在计算td-error后再置0是来不及的
    3. done用来将超出当前episode长度的q_targets和r置为0
    '''
    n_step_return = torch.zeros((episode_num, max_episode_len, args.n_agents, max_episode_len))
    for transition_idx in range(max_episode_len - 1, -1, -1):
        # 最后计算1 step return
        n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + args.gamma * (q_targets[:, transition_idx] - next_bias[:,transition_idx]) * done[:, transition_idx]) * mask[:, transition_idx]        # 经验transition_idx上的obs有max_episode_len - transition_idx个return, 分别计算每种step return
        # 同时要注意n step return对应的index为n-1
        for n in range(1, max_episode_len - transition_idx):
            # t时刻的n step return =r + gamma * (t + 1 时刻的 n-1 step return)
            # n=1除外, 1 step return =r + gamma * (t + 1 时刻的 Q)
            n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + args.gamma * (n_step_return[:, transition_idx + 1, :, n - 1] - next_bias[:,transition_idx])) * mask[:, transition_idx]
        # 计算lambda return
    '''
    lambda_return 维度为(episode个数, max_episode_len， n_agents)，每条经验中，每个agent都有一个lambda return
    '''
    # print('args.td_lambda = {}'.format(args.td_lambda))
    lambda_return = torch.zeros((episode_num, max_episode_len, args.n_agents))
    for transition_idx in range(max_episode_len):
        returns = torch.zeros((episode_num, args.n_agents))
        for n in range(1, max_episode_len - transition_idx):
            returns += pow(args.td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
        lambda_return[:, transition_idx] = (1 - args.td_lambda) * returns + \
                                           pow(args.td_lambda, max_episode_len - transition_idx - 1) * \
                                           n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
    return lambda_return




class RNN(nn.Module):
    # obs_shape应该是obs_shape+n_actions+n_agents，还要输入当前agent的上一个动作和agent编号，这样就可以只使用一个神经网络
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        # print('obs = {}'.format(np.shape(obs)))
        # print('hidden_state = {}'.format(np.shape(hidden_state)))
        x = F.relu(self.fc1(obs))
        # print('x = {}'.format(np.shape(x)))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # print('h_in = {}'.format(np.shape(h_in)))
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

check_tag = False
eps_tag = False
abs_tag = False
state_info = 2
policy_log_tag = False


# arguments of coma
class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        two_layer_hidden = args.hyper_hidden_dim
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, two_layer_hidden ),
                                          nn.ReLU(),
                                          nn.Linear(two_layer_hidden , args.n_agents * args.qmix_hidden_dim))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, two_layer_hidden ),
                                          nn.ReLU(),
                                          nn.Linear(two_layer_hidden , args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 =nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )
        if args.two_hyper_layers:
            self.state_encoder = nn.Sequential(nn.Linear(args.state_shape, two_layer_hidden),
                                               nn.ReLU(),
                                               nn.Linear(two_layer_hidden, args.qmix_hidden_dim),
                                               nn.ReLU())
            self.q_val_encoder = nn.Sequential(nn.Linear(args.n_agents, two_layer_hidden),
                                               nn.ReLU(),
                                               nn.Linear(two_layer_hidden, args.qmix_hidden_dim),
                                               nn.ReLU()
                                               )
        else:
            self.state_encoder = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                               nn.ReLU())
            self.q_val_encoder = nn.Sequential(nn.Linear(args.n_agents, args.qmix_hidden_dim),
                                               nn.ReLU())

        self.no_state_mlp = nn.Sequential(nn.Linear(args.n_agents, args.qmix_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.qmix_hidden_dim, 1)
                                          )
        self.state_mlp = nn.Sequential(nn.Linear(args.qmix_hidden_dim * 2, two_layer_hidden),
                                       nn.ReLU(),
                                       nn.Linear(two_layer_hidden, 1)
                                       )
    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        episode_num = q_values.size(0)
        if self.args.state_info == 0 or self.args.check_alg == True:
            q_values = q_values.view(-1, 1,
                                     self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
            states = states.reshape(-1, self.args.state_shape)  # (episode_num * max_episode_len, state_shape)
            if abs_tag or self.args.check_alg == True:
                w1 = torch.abs(self.hyper_w1(states))  # (1920, 160)
            else:
                w1 = self.hyper_w1(states)
            b1 = self.hyper_b1(states)  # (1920, 32)

            w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # (1920, 5, 32)
            b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)

            hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (1920, 1, 32)

            if abs_tag or self.args.check_alg == True:
                w2 = torch.abs(self.hyper_w2(states))  # (1920, 32)
            else:
                w2 = self.hyper_w2(states)
            b2 = self.hyper_b2(states)  # (1920, 1)

            w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
            b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

            q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
            q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)
        elif self.args.state_info == 1:
            q_values = q_values.reshape([-1,self.args.n_agents])
            q_total = self.no_state_mlp(q_values)
            q_total = q_total.reshape([episode_num,-1,1])
        else:
            q_values = q_values.reshape([-1, self.args.n_agents])
            states = states.reshape([-1,self.args.state_shape])
            q_encode = self.q_val_encoder(q_values)
            states_encode = self.state_encoder(states)
            final_inputs = torch.cat([q_encode,states_encode],dim = -1)
            q_total = self.state_mlp(final_inputs)
            q_total = q_total.reshape([episode_num, -1, 1])
        return q_total



class QMIX_DIV:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        self.eval_policy_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_policy_rnn = RNN(input_shape, args)

        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)

        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)
        print('rnn::\n{}'.format(self.eval_policy_rnn))
        print('mixer::\n{}'.format(self.eval_qmix_net))
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
            self.eval_policy_rnn.cuda()
            self.target_policy_rnn.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/newest_rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/newest_rnn_net_params.pkl'
                path_qmix = self.model_dir + '/newest_qmix_net_params.pkl'
                path_policy_rnn = self.model_dir + '/newest_rnn_policy_params.pkl'
                self.eval_policy_rnn.load_state_dict(torch.load(path_policy_rnn))
                self.eval_rnn.load_state_dict(torch.load(path_rnn))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix))

                self.target_policy_rnn.load_state_dict(self.eval_policy_rnn.state_dict())

                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        # self.target_policy_rnn.load_state_dict(self.eval_policy_rnn.state_dict())


        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        self.policy_parameters = list(self.eval_policy_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
            self.policy_optimizer = torch.optim.RMSprop(self.policy_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        self.eval_policy_hidden = None
        self.target_policy_hidden = None
        self.trans_cnt = 0
        self.sac_tag = self.args.alg.find('sac') > -1
        print('Init alg QMIX_DIV')

    def _get_actor_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs_next = []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给inputs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                tmp_acs = torch.zeros_like(u_onehot[:, transition_idx])
            else:
                tmp_acs = u_onehot[:, transition_idx - 1]
            tmp_acs_next = u_onehot[:, transition_idx]
            inputs.append(tmp_acs)
            inputs_next.append(tmp_acs_next)
        if self.args.reuse_network:
            # 因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            acs_tag = torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1)
            inputs.append(acs_tag)
            inputs_next.append(acs_tag)
        # 要把inputs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.float().reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.float().reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        # TODO 检查inputs_next是不是相当于inputs向后移动一条
        return inputs, inputs_next

    def _get_action_prob(self, batch, max_episode_len, epsilon, if_target=False):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        # print('avail_action_shape = {}'.format(avail_actions.shape))
        avail_actions_next = batch['avail_u_next']

        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                if if_target:
                    self.target_policy_hidden = self.target_policy_hidden.cuda()
                else:
                    self.eval_policy_hidden = self.eval_policy_hidden.cuda()
            if if_target:
                outputs, self.target_policy_hidden = self.target_policy_rnn(inputs, self.target_policy_hidden)
            else:
                outputs, self.eval_policy_hidden = self.eval_policy_rnn(inputs,self.eval_policy_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            # 把q_eval维度重新变回(8, 5,n_actions)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)
        if if_target:
            last_outputs, self.target_policy_hidden = self.target_policy_rnn(inputs_next, self.target_policy_hidden)
        else:
            last_outputs, self.eval_policy_hidden = self.eval_policy_rnn(inputs_next, self.eval_policy_hidden)
        last_outputs = last_outputs.view(episode_num, self.n_agents, -1)
        last_prob = torch.nn.functional.softmax(last_outputs, dim=-1)
        action_prob.append(last_prob)

        # 得的action_prob是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        action_prob = torch.stack(action_prob, dim=1).cpu()

        init_action_prob = torch.tensor(action_prob).detach()


        avail_actions = torch.cat([avail_actions, avail_actions_next[:, -1].unsqueeze(1)], dim=1)
        action_prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        # 因为上面把不能执行的动作概率置为0，所以概率和不为1了，这里要重新正则化一下。执行过程中Categorical会自己正则化。
        prob_sum = torch.tensor(action_prob).detach()
        # print('prob_sum_shape_1 = {}'.format(prob_sum.shape))
        prob_sum = prob_sum.sum(dim = -1)
        # print('prob_sum_shape_2 = {}'.format(prob_sum.shape))
        prob_sum = prob_sum.reshape([-1])
        # print('prob_sum_shape_3 = {}'.format(prob_sum.shape))
        for i in range(len(prob_sum)):
            if prob_sum[i] == 0:
                prob_sum[i] = 1
        prob_sum = prob_sum.reshape(*action_prob.shape[:-1]).unsqueeze(-1)
        # print('prob_sum_shape_4 = {}'.format(prob_sum.shape))
        action_prob = action_prob / prob_sum
        # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan。
        # 因此需要再一次将该经验对应的概率置为0
        action_prob[avail_actions == 0] = 0.0
        # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan。
        # 因此需要再一次将该经验对应的概率置为0
        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob,init_action_prob
    def soft_update_critic(self,tau = 0.01):
        soft_update(self.target_rnn, self.eval_rnn, tau)
        soft_update(self.target_qmix_net,self.eval_qmix_net,tau)
    def hard_update_critic(self):
        hard_update(self.target_rnn, self.eval_rnn)
        hard_update(self.target_qmix_net, self.eval_qmix_net)
    def soft_update_policy(self,tau= 0.01):
        soft_update(self.target_policy_rnn,self.eval_policy_rnn,tau)
    def hard_update_policy(self,tau= 0.01):
        hard_update(self.target_policy_rnn,self.eval_policy_rnn)
    def learn(self, batch, max_episode_len, train_step, epsilon=None,logger = None,policy_logger_list = None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u_onehot, r,   done,avail_u_next = batch['s'], batch['s_next'], batch['u_onehot'], \
                                                             batch['r'],   \
                                                             batch['terminated'],batch['avail_u_next']
        avail_action = batch['avail_u']

        done = done.mean(dim = 2,keepdim = True)
        r = r.mean(dim = 2,keepdim=True)

        mask_init = (1 - batch["padded"].float())
        zero_mask = torch.zeros([mask_init.shape[0], 1, mask_init.shape[2]])

        mask_init_next = torch.cat([mask_init[:, 1:, :], zero_mask], dim=1)
        mask_next = mask_init_next.repeat(1, 1, self.n_agents)
        mask = mask_init.repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习


        if self.args.cuda:
            s = s.cuda()
            u_onehot = u_onehot.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            done = done.cuda()
            mask = mask.cuda()
            mask_next = mask_next.cuda()
            avail_action = avail_action.cuda()

        u_for_q = torch.argmax(u_onehot, dim=3, keepdim=True)
        q_evals_all, q_targets_all = self.get_q_values(batch, max_episode_len)
        q_evals_for_q = torch.gather(q_evals_all, dim=3, index=u_for_q).squeeze(3)
        q_total_eval_for_q = self.eval_qmix_net(q_evals_for_q, s)  # (32,50,1)

        policy_loss_ret,bias_ret = None,None
        if not self.args.check_alg:
            all_action_prob,init_action_prob = self._get_action_prob(batch, max_episode_len, epsilon)  # 每个agent的所有动作的概率
            all_target_action_prob,_ = self._get_action_prob(batch, max_episode_len, epsilon, if_target=True)

            curr_action_prob = all_action_prob[:, :-1]
            curr_target_action_prob = all_target_action_prob[:, :-1]


            policy_with_eps = None

            sample_prob = curr_action_prob.detach()


            mask_for_action = mask.unsqueeze(3).repeat(1, 1, 1, avail_action.shape[-1])
            sample_prob[mask_for_action == 0] = 1


            u_for_policy = Categorical(sample_prob).sample().unsqueeze(3)

            pi_taken = torch.gather(curr_action_prob, dim=3, index=u_for_policy).squeeze(3)  # 每个agent的选择的动作对应的概率
            target_pi_taken = torch.gather(curr_target_action_prob, dim=3, index=u_for_policy).squeeze(3)
            pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
            target_pi_taken[mask == 0] = 1.0

            next_action_prob = all_action_prob[:, 1:].clone().detach()  # 每个agent的所有动作的概率
            next_target_action_prob = all_target_action_prob[:, 1:].clone().detach()

            # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)

            # print('q_evals_all_shape = {}'.format(q_evals_all.shape))

            # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
            q_evals_for_policy = torch.gather(q_evals_all, dim=3, index=u_for_policy).squeeze(3)
            q_total_eval_for_policy = self.eval_qmix_net(q_evals_for_policy, s) # (32,50,1)


            # print('q_total_eval = {}'.format(q_total_eval.shape))
            q_table = torch.zeros_like(q_evals_all)
            if self.args.cuda:
                q_table = q_table.cuda()
            for a_i in range(self.n_agents):
                for ac in range(self.n_actions):
                    q_input = q_evals_for_policy.clone().detach()
                    q_input[:,:,a_i] = q_evals_all[:,:,a_i,ac]
                    q_table[:,:,a_i,ac] = self.eval_qmix_net(q_input,s).squeeze(2)


            # q_baseline = (q_table * curr_action_prob).sum(dim = 3) # (32,50,5)
            q_baseline = (q_table * curr_action_prob).sum(dim = 3)
            # print('q_baseline = {}'.format(q_baseline.shape))
            bias_ret = []
            if self.sac_tag:
                curr_bias = torch.log(pi_taken)
            else:
                curr_bias = torch.log(pi_taken) - torch.log(target_pi_taken) # (32,50,5)
            # print('curr_bias1 = {}'.format(curr_bias.shape))
            bias_ret.append(curr_bias.mean())

            curr_action_prob_for_log = torch.tensor(curr_action_prob)
            curr_action_prob_for_log[avail_action == 0] = 1.0
            curr_action_prob_for_log[mask_for_action == 0] = 1.0

            curr_target_action_prob_for_log = torch.tensor(curr_target_action_prob)
            curr_target_action_prob_for_log[avail_action == 0] = 1.0
            curr_target_action_prob_for_log[mask_for_action == 0] = 1.0
            if self.sac_tag:
                all_log_diff = torch.log(curr_action_prob_for_log)
            else:
                all_log_diff = torch.log(curr_action_prob_for_log) - torch.log(curr_target_action_prob_for_log)
            # print("all_log_diff = {}".format(all_log_diff))
            bias_baseline = (all_log_diff * curr_action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
            curr_bias -= bias_baseline
            # print('bias_baseline = {}'.format(bias_baseline.shape))
            bias_ret.append(curr_bias.mean())

            curr_bias /= self.args.reward_scale  # (32,50,5)
            adv = (q_total_eval_for_policy - q_baseline - curr_bias).detach()
            # print('curr_bias2 = {}'.format(curr_bias.shape))
            log_pi_taken = torch.log(pi_taken) # (32,50,5)
            # print('log_pi_taken = {}'.format(log_pi_taken.shape))
            policy_loss = - ((adv * log_pi_taken) * mask).sum() / mask.sum() # (32,50,5)
            # print('policy_loss = {}'.format(policy_loss.shape))
            policy_loss = policy_loss.sum()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_parameters, self.args.grad_norm_clip)
            self.policy_optimizer.step()

            print('***************************************************************************')
            print('policy_loss = {}'.format(policy_loss))
            policy_loss_ret = policy_loss

            u_next = u_for_q[:, 1:]
            # print('u_for_q_shape = {},u_next_shape = {}'.format(u_for_q.shape,u_next.shape))
            padded_u_next = torch.zeros(*u_for_q[:, -1].shape, dtype=torch.long).unsqueeze(1)
            if self.args.cuda:
                padded_u_next = padded_u_next.cuda()
            u_next = torch.cat((u_next, padded_u_next), dim=1)

            next_pi_taken = torch.gather(next_action_prob, dim=3, index=u_next).squeeze(3)
            next_target_pi_taken = torch.gather(next_target_action_prob, dim=3, index=u_next).squeeze(3)
            next_pi_taken[mask_next == 0] = 1.0
            next_target_pi_taken[mask_next == 0] = 1.0
            if self.sac_tag:
                next_bias_sum = torch.log(next_pi_taken)
            else:
                next_bias_sum = torch.log(next_pi_taken) - torch.log(next_target_pi_taken)
            next_bias_sum = next_bias_sum.sum(dim = 2,keepdim=True)
            next_bias_sum /= self.args.reward_scale

            q_targets = torch.gather(q_targets_all,dim = 3,index= u_next).squeeze(3)
        else:
            q_targets_all[avail_u_next == 0.0] = - 9999999
            q_targets = q_targets_all.max(dim=3)[0]

        q_total_target = self.target_qmix_net(q_targets, s_next)
        # print('q_total_target = {}'.format(q_total_target.shape))
        # print('done = {}'.format(done.shape))
        targets_rec = None
        q_val_rec = None
        q_bias = None
        if self.args.check_alg == False:
            if self.args.use_td_lambda:
                targets = td_lambda_target(batch, max_episode_len, q_total_target.cpu(), next_bias_sum.cpu(), self.args)
            else:
                targets = r + self.args.gamma * (q_total_target - next_bias_sum) * (1 - done)
            targets_rec = targets.detach().mean()
            q_val_rec = q_total_eval_for_q.detach().mean()
            q_bias = next_bias_sum.detach().mean()
        else:
            targets = r + self.args.gamma * q_total_target * (1 - done)

        if self.args.cuda:
            targets = targets.cuda()

        td_error = (q_total_eval_for_q - targets.detach())
        # print('mask = {}'.format(mask.shape))
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error
        # print('masked_td_error = {}'.format(masked_td_error.shape))
        # loss = masked_td_error.pow(2).mean()
        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()



        print('***************************************************************************')
        print('q_loss = {}'.format(loss))
        print('***************************************************************************')
        q_loss_ret = [loss.item(), targets_rec, q_val_rec, q_bias]
        return q_loss_ret,policy_loss_ret,bias_ret

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据

        inputs = torch.cat([x.float().reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.float().reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        # TODO 检查inputs_next是不是相当于inputs向后移动一条
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        if self.args.cuda:
            q_evals = q_evals.cuda()
            q_targets = q_targets.cuda()
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_policy_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_policy_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
        torch.save(self.eval_policy_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_policy_params.pkl')
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/newest_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/newest_rnn_net_params.pkl')
        torch.save(self.eval_policy_rnn.state_dict(), self.model_dir + '/newest_rnn_policy_params.pkl')







