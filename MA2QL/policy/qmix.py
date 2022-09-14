import torch
import numpy as np
import os
import argparse
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from torch.distributions import Categorical
from network.base_net import RNN
from network.qmix_net import QMixNet
from common.utils import td_lambda_target
class QMIX:
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


        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)

        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)
        print('rnn::\n{}'.format(self.eval_rnn))
        print('mixer::\n{}'.format(self.eval_qmix_net))
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/newest_rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/newest_rnn_net_params.pkl'
                path_qmix = self.model_dir + '/newest_qmix_net_params.pkl'
                self.eval_rnn.load_state_dict(torch.load(path_rnn))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix))


                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        # self.target_policy_rnn.load_state_dict(self.eval_policy_rnn.state_dict())


        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        self.eval_policy_hidden = None
        self.target_policy_hidden = None
        self.trans_cnt = 0
        print('Init alg QMIX_DIV')


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
        curr_q_evals = q_evals_all[:, :-1]
        next_q_evals = q_evals_all[:, 1:]
        next_q_targets = q_targets_all[:, 1:]

        q_evals_for_q = torch.gather(curr_q_evals, dim=3, index=u_for_q).squeeze(3)
        q_total_eval_for_q = self.eval_qmix_net(q_evals_for_q, s)  # (32,50,1)

        policy_loss_ret, bias_ret = None, None

        if self.args.double_q:
            next_q_evals = next_q_evals.clone().detach()
            next_q_evals[avail_u_next == 0.0] = - 9999999
            next_acs = torch.argmax(next_q_evals, dim=3, keepdim=True)
            q_targets = torch.gather(next_q_targets, dim=3, index=next_acs).squeeze(3)
        else:
            next_q_targets[avail_u_next == 0.0] = - 9999999
            q_targets = next_q_targets.max(dim=3)[0]

        q_total_target = self.target_qmix_net(q_targets, s_next)
        # print('q_total_target = {}'.format(q_total_target.shape))
        # print('done = {}'.format(done.shape))
        targets_rec = None
        q_val_rec = None
        q_bias = None

        if self.args.use_td_lambda:
            targets = td_lambda_target(batch, max_episode_len, q_total_target.cpu(),self.args)
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
            q_target, self.target_hidden = self.target_rnn(inputs, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_eval, self.eval_hidden = self.eval_rnn(inputs_next,self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
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

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/newest_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/newest_rnn_net_params.pkl')



class QMIX_prev:
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
        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr,eps=args.optim_eps)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QMIX')
    def hard_update_critic(self):
        hard_update(self.target_rnn, self.eval_rnn)
        hard_update(self.target_qmix_net, self.eval_qmix_net)
    def double_q_learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals_all, q_targets_all = self.get_q_values(batch, max_episode_len)
        curr_q_evals = q_evals_all[:,:-1]
        next_q_targets = q_targets_all[:,1:]
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(curr_q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q
        if self.args.double_q:
            print('double_q')
            next_q_evals = q_evals_all[:,1:].clone().detach()
            next_q_evals[avail_u_next == 0.0] = - 9999999
            cur_max_actions = torch.argmax(next_q_evals,dim = 3,keepdim=True)
            q_targets = torch.gather(next_q_targets,dim = 3,index = cur_max_actions).squeeze(3)
        else:
            next_q_targets[avail_u_next == 0.0] = - 9999999
            q_targets = next_q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
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

        q_targets_all[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets_all.max(dim=3)[0]

        q_total_target = self.target_qmix_net(q_targets, s_next)
        # print('q_total_target = {}'.format(q_total_target.shape))
        # print('done = {}'.format(done.shape))
        targets_rec = None
        q_val_rec = None
        q_bias = None

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
    def prev_learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()




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
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def double_q_get_q_values(self, batch, max_episode_len):
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
            q_target, self.target_hidden = self.target_rnn(inputs, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_eval, self.eval_hidden = self.eval_rnn(inputs_next,self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
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
        return q_evals, q_targets

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

    def prev_get_q_values(self, batch, max_episode_len):
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
        return q_evals, q_targets
    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')