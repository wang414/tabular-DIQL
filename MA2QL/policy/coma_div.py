import torch
import os
import numpy as np
import math
from network.base_net import RNN
from network.commnet import CommNet
from network.g2anet import G2ANet
from network.coma_critic import ComaCritic
from torch.distributions import Categorical
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from common.utils import td_lambda_target as td_lambda_target_no_bias

def td_lambda_target(batch, max_episode_len, q_targets, next_bias, args):  # 用来通过TD(lambda)计算y
    # batch维度为(episode个数, max_episode_len， n_agents，n_actions)
    # q_targets维度为(episode个数, max_episode_len， n_agents)
    # print('batch_padded = {}'.format(batch["padded"].shape))
    # print('batch_done = {}'.format(batch['terminated'].shape))
    episode_num = batch['o'].shape[0]
    mask = (1 - batch["padded"].float())  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
    done = (1 - batch['terminated'].float()) # 用来把episode最后一条经验中的q_target置0
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
    lambda_return = torch.zeros((episode_num, max_episode_len, args.n_agents))
    for transition_idx in range(max_episode_len):
        returns = torch.zeros((episode_num, args.n_agents))
        for n in range(1, max_episode_len - transition_idx):
            returns += pow(args.td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
        lambda_return[:, transition_idx] = (1 - args.td_lambda) * returns + \
                                           pow(args.td_lambda, max_episode_len - transition_idx - 1) * \
                                           n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
    return lambda_return


class COMA_DIV:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape  # actor网络输入的维度，和vdn、qmix的rnn输入维度一样，使用同一个网络结构
        critic_input_shape = self._get_critic_input_shape()  # critic网络输入的维度
        # 根据参数决定RNN的输入维度


        if not self.args.no_rnn:
            if args.last_action:
                actor_input_shape += self.n_actions
            if args.reuse_network:
                actor_input_shape += self.n_agents
        # 神经网络
        # 每个agent选动作的网络,输出当前agent所有动作对应的概率，用该概率选动作的时候还需要用softmax再运算一次。
        if self.args.alg.startswith('coma_div'):
            print('Init alg {}'.format(self.args.alg))

            if self.args.idv_comparing and self.args.idv_para:
                self.eval_rnn = [RNN(actor_input_shape, args) for _ in range(self.n_agents)]
            else:
                self.eval_rnn = RNN(actor_input_shape, args)
        elif self.args.alg == 'coma+commnet':
            print('Init alg coma+commnet')
            self.eval_rnn = CommNet(actor_input_shape, args)
        elif self.args.alg == 'coma+g2anet':
            print('Init alg coma+g2anet')
            self.eval_rnn = G2ANet(actor_input_shape, args)
        else:
            raise Exception("No such algorithm")
        if not self.args.check_alg:
            if self.args.idv_comparing and self.args.idv_para:
                self.target_rnn = [RNN(actor_input_shape, args) for _ in range(self.n_agents)]
            else:
                self.target_rnn = RNN(actor_input_shape, args)
        # 得到当前agent的所有可执行动作对应的联合Q值，得到之后需要用该Q值和actor网络输出的概率计算advantage

        if self.args.idv_comparing and self.args.idv_para:
            self.eval_critic = [ComaCritic(critic_input_shape, self.args) for _ in range(self.n_agents)]
            self.target_critic = [ComaCritic(critic_input_shape, self.args) for _ in range(self.n_agents)]
        else:
            self.eval_critic = ComaCritic(critic_input_shape, self.args)
            self.target_critic = ComaCritic(critic_input_shape, self.args)

        if self.args.cuda:
            if self.args.idv_comparing and self.args.idv_para:
                for a in range(self.args.n_agents):
                    self.eval_rnn[a].cuda()
                    self.eval_critic[a].cuda()
                    self.target_critic[a].cuda()
                    if not self.args.check_alg:
                        self.target_rnn[a].cuda()
            else:
                self.eval_rnn.cuda()
                self.eval_critic.cuda()
                self.target_critic.cuda()
                if not self.args.check_alg:
                    self.target_rnn.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        print(self.model_dir)
        print(os.path.exists(self.model_dir))
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/newest_rnn_params.pkl'):
                path_rnn = self.model_dir + '/newest_rnn_params.pkl'
                path_coma = self.model_dir + '/newest_critic_params.pkl'
                # path_rnn = self.model_dir + '/100_rnn_params.pkl'
                # path_coma = self.model_dir + '/100_critic_params.pkl'
                self.eval_rnn.load_state_dict(torch.load(path_rnn))
                self.eval_critic.load_state_dict(torch.load(path_coma))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_coma))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        if self.args.idv_comparing and self.args.idv_para:
            for a in range(self.args.n_agents):
                self.target_critic[a].load_state_dict(self.eval_critic[a].state_dict())
        else:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())

        if self.args.idv_comparing and self.args.idv_para:
            self.policy_parameters = [list(x.parameters()) for x in self.eval_rnn]
            self.critic_parameters = [list(x.parameters()) for x in self.eval_critic]
        else:
            self.policy_parameters = list(self.eval_rnn.parameters())
            self.critic_parameters = list(self.eval_critic.parameters())

        if args.optimizer == "RMS":

            if self.args.idv_comparing and self.args.idv_para:
                self.policy_optimizer = [torch.optim.RMSprop(x, lr=args.lr_actor) for x in self.policy_parameters]
                self.critic_optimizer = [torch.optim.RMSprop(x, lr=args.lr_critic) for x in self.critic_parameters]
            else:
                self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.lr_critic)
                self.policy_optimizer = torch.optim.RMSprop(self.policy_parameters, lr=args.lr_actor)
        self.args = args

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden
        self.eval_hidden = None
        self.target_hidden = None
        self.sac_tag = self.args.alg.find('sac') > -1

        # self.eval_V = self.eval_critic
        # self.target_V = self.target_critic
        #
        # self.old_policy_hidden = self.target_hidden
        # self.eval_policy_hidden = self.eval_hidden
        #
        # self.old_policy_rnn = self.target_rnn
        # self.eval_policy_rnn = self.eval_rnn

    def _get_critic_input_shape(self):
        # state
        if self.args.idv_comparing:
            input_shape = self.obs_shape
        else:
            input_shape = self.state_shape  # 48
            # obs
            input_shape += self.obs_shape  # 30
            # agent_id
            input_shape += self.n_agents  # 3
            # 所有agent的当前动作和上一个动作
            input_shape += self.n_actions * self.n_agents * 2  # 54

        return input_shape
    def soft_update_policy(self,tau= 0.001):
        if self.args.idv_comparing and self.args.idv_para:
            for a in range(self.args.n_agents):
                soft_update(self.target_rnn[a], self.eval_rnn[a], tau)
        else:
            soft_update(self.target_rnn,self.eval_rnn,tau)
    def hard_update_policy(self):
        if self.args.idv_comparing and self.args.idv_para:
            for a in range(self.args.n_agents):
                hard_update(self.target_rnn[a], self.eval_rnn[a])
        else:
            hard_update(self.target_rnn,self.eval_rnn)
    def hard_update_critic(self):
        if self.args.idv_comparing and self.args.idv_para:
            for a in range(self.args.n_agents):
                hard_update(self.target_critic[a], self.eval_critic[a])
        else:
            hard_update(self.target_critic,self.eval_critic)
    def _get_action_prob(self, batch, max_episode_len, epsilon, if_target=False):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        # print('avail_action_shape = {}'.format(avail_actions.shape))
        avail_actions_next = batch['avail_u_next']

        action_prob = []
        for transition_idx in range(max_episode_len):
            # if not if_target and transition_idx == 0:
                # print('trans_idx = {}\n\nhidden_before = {}'.format(transition_idx,self.eval_hidden))
            inputs,inputs_next = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                if if_target:
                    self.target_hidden = self.target_hidden.cuda()
                else:
                    self.eval_hidden = self.eval_hidden.cuda()
            if if_target:
                outputs, self.target_hidden = self.target_rnn(inputs, self.target_hidden)
            else:
                outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            # 把q_eval维度重新变回(8, 5,n_actions)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)

            action_prob.append(prob)
        if if_target:
            last_outputs, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
        else:
            last_outputs,self.eval_hidden = self.eval_rnn(inputs_next,self.eval_hidden)
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
    def COMA_DIV_learn(self, batch, max_episode_len, train_step, epsilon=None,logger = None,policy_logger_list = None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
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
        q_total_eval_for_q = q_evals_for_q
        u_next = u_for_q[:, 1:]
        # print('u_for_q_shape = {},u_next_shape = {}'.format(u_for_q.shape,u_next.shape))
        padded_u_next = torch.zeros(*u_for_q[:, -1].shape, dtype=torch.long).unsqueeze(1)
        if self.args.cuda:
            padded_u_next = padded_u_next.cuda()
        u_next = torch.cat((u_next, padded_u_next), dim=1)
        q_targets = torch.gather(q_targets_all,dim = 3,index= u_next).squeeze(3)


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
            q_total_eval_for_policy =  q_evals_for_policy# (32,50,1)


            # print('q_total_eval = {}'.format(q_total_eval.shape))
            q_table = q_evals_all

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

        # print('q_total_target = {}'.format(q_total_target.shape))
        # print('done = {}'.format(done.shape))
        targets_rec = None
        q_val_rec = None
        q_bias = None
        if self.args.check_alg:
            targets = td_lambda_target_no_bias(batch, max_episode_len, q_targets.cpu(), self.args)
        else:
            if self.args.test_no_use_td_lambda:
                targets = r + self.args.gamma * (q_targets - next_bias_sum) * (1 - done)
            else:
                targets = td_lambda_target(batch, max_episode_len, q_targets.cpu(), next_bias_sum.cpu(),self.args)

        if self.args.cuda:
            targets = targets.cuda()

        td_error = (q_total_eval_for_q - targets.detach())
        # print('mask = {}'.format(mask.shape))
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error
        # print('masked_td_error = {}'.format(masked_td_error.shape))
        # loss = masked_td_error.pow(2).mean()
        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()



        print('***************************************************************************')
        print('q_loss = {}'.format(loss))
        print('***************************************************************************')
        q_loss_ret = [loss.item(), targets_rec, q_val_rec, q_bias]
        return q_loss_ret,policy_loss_ret,bias_ret

    def prev_get_inputs(self, batch, transition_idx):
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
    def prev_learn(self, batch, max_episode_len, train_step, epsilon):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        bias_mean_no_baseline = None
        bias_mean_baseline = None

        for key in batch.keys():
            batch[key] = torch.tensor(batch[key],dtype=torch.float32)
            # print('batch[{}] = {}'.format(key,batch[key].shape))
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        u_onehot, r,  done = batch['u_onehot'], batch['r'],   batch['terminated']
        avail_action = batch['avail_u']
        u = torch.argmax(u_onehot,dim = 3, keepdim= True)
        # print('u = {},u_onehot = {}'.format(u.shape,u_onehot.shape) )
        # print('u = {}'.format(u.shape))
        mask_init = (1 - batch["padded"].float())
        zero_mask = torch.zeros([mask_init.shape[0],1,mask_init.shape[2]])
        mask_init_next = torch.cat([mask_init[:,1:,:],zero_mask ],dim = 1)
        mask_next = mask_init_next.repeat(1,1,self.n_agents)
        mask = mask_init.repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        # print('mask = {}'.format(mask.shape))
        if self.args.cuda:
            u = u.cuda()
            mask = mask.cuda()
            u_onehot = u_onehot.cuda()
            r = r.cuda()
            done = done.cuda()
            mask = mask.cuda()
            mask_next = mask_next.cuda()
            avail_action = avail_action.cuda()
        # 根据经验计算每个agent的Ｑ值,从而跟新Critic网络。然后计算各个动作执行的概率，从而计算advantage去更新Actor。

        u_next = u[:, 1:]
        padded_u_next = torch.zeros(*u[:, -1].shape, dtype=torch.long).unsqueeze(1)
        if not self.args.check_alg:
            padded_u_next = padded_u_next.cuda()
        u_next = torch.cat((u_next, padded_u_next), dim=1)

        all_action_prob = self._get_action_prob(batch, max_episode_len, epsilon)  # 每个agent的所有动作的概率
        curr_action_prob = all_action_prob[:,:-1]
        next_action_prob =  all_action_prob[:,1:]# 每个agent的所有动作的概率
        next_pi_taken = torch.gather(next_action_prob, dim=3, index=u_next).squeeze(3)
        next_pi_taken[mask_next == 0] = 1.0

        if not self.args.check_alg:
            all_target_action_prob = self._get_action_prob(batch, max_episode_len, epsilon, if_target=True)
            curr_target_action_prob = all_target_action_prob[:, :-1]
            next_target_action_prob = all_target_action_prob[:, 1:]
            next_target_pi_taken = torch.gather(next_target_action_prob, dim=3, index=u_next).squeeze(3)
            next_target_pi_taken[mask_next == 0] = 1.0

            next_bias = torch.log(next_pi_taken) - torch.log(next_target_pi_taken)
            next_bias = next_bias.sum(dim=2, keepdim=True).repeat(1, 1, self.n_agents)
            next_bias /= self.args.reward_scale

        if not self.args.check_alg:
            q_values,q_loss = self._train_critic(batch, max_episode_len, train_step,next_bias.clone().detach() )  # 训练critic网络，并且得到每个agent的所有动作的Ｑ值
        else:
            q_values, q_loss = self._train_critic(batch, max_episode_len, train_step)
            # print('q_values = {},q_loss = {}'.format(q_values,q_loss))
        if math.isnan(q_loss):
            1/0
        # print('curr_action_prob_shape = {}'.format(curr_action_prob.shape))

        mask_for_action = mask.unsqueeze(3).repeat(1,1,1,avail_action.shape[-1])
        prev_shape = curr_action_prob.shape
        sample_prob = curr_action_prob.detach()
        sample_prob[mask_for_action == 0] = 1
        # print('sample_prob = {}'.format(sample_prob))
        action_sample = Categorical(sample_prob).sample().unsqueeze(3)
        # print('action_sample_shape_1 = {}'.format(action_sample.shape))
        # print('curr_action_prob_shape_after = {}'.format(curr_action_prob.shape))
        # print('action_sample_shape_2 = {}'.format(action_sample.shape))

        q_taken = torch.gather(q_values, dim=3, index=action_sample).squeeze(3)  # 每个agent的选择的动作对应的Ｑ值
        pi_taken = torch.gather(curr_action_prob, dim=3, index=action_sample).squeeze(3)  # 每个agent的选择的动作对应的概率
        pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
        log_pi_taken = torch.log(pi_taken)
        curr_action_prob_for_log = torch.tensor(curr_action_prob)
        curr_action_prob_for_log[avail_action == 0] = 1.0
        curr_action_prob_for_log[mask_for_action == 0] = 1.0


        if not self.args.check_alg:
            target_pi_taken = torch.gather(curr_target_action_prob, dim=3, index=action_sample).squeeze(3)
            target_pi_taken[mask == 0] = 1.0
            log_target_pi_taken = torch.log(target_pi_taken)
            bias = log_pi_taken - log_target_pi_taken
            bias_mean_no_baseline = bias.mean()

            curr_target_action_prob_for_log = torch.tensor(curr_target_action_prob)
            curr_target_action_prob_for_log[avail_action == 0] = 1.0
            curr_target_action_prob_for_log[mask_for_action == 0] = 1.0

            all_log_diff = torch.log(curr_action_prob_for_log) - torch.log(curr_target_action_prob_for_log)
            bias_baseline = (all_log_diff * curr_action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
            bias = bias - bias_baseline
            bias_mean_baseline = bias.mean()
            bias /= self.args.reward_scale
        # print("q_taken = {}".format(q_taken))
        # print('bias2 = {}'.format(bias))
        # 计算advantage

        baseline = (q_values * curr_action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
        if self.args.check_alg:
            advantage = (q_taken - baseline).detach()
        else:
            advantage = (q_taken - baseline - bias).detach()
        # print('baseline = {},advantage = {}'.format(baseline, advantage))
        loss = - ((advantage * log_pi_taken) * mask).sum() / mask.sum()
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        policy_loss = loss.item()

        if math.isnan(policy_loss):
            1 / 0
        return q_loss,policy_loss,[bias_mean_no_baseline,bias_mean_baseline]


    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        # 取出所有episode上该transition_idx的经验
        obs, obs_next,s, s_next  = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx], \
                                batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        u_onehot = batch['u_onehot'][:, transition_idx]
        if transition_idx != max_episode_len - 1:
            u_onehot_next = batch['u_onehot'][:, transition_idx + 1]
        else:
            u_onehot_next = torch.zeros(*u_onehot.shape)
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]
        # 因为coma的critic用到的是所有agent的动作，所以要把u_onehot最后一个维度上当前agent的动作变成所有agent的动作
        # print('u_onehot = {}'.format(u_onehot.shape))
        # print('episode_num = {}'.format(episode_num))
        u_onehot = u_onehot.reshape((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        u_onehot_next = u_onehot_next.reshape((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        # print('u_onehot_new = {}'.format(u_onehot.shape))
        if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
            u_onehot_last = torch.zeros_like(u_onehot)
        else:
            u_onehot_last = batch['u_onehot'][:, transition_idx - 1]
            u_onehot_last = u_onehot_last.reshape((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        # print('u_onehot = {}'.format(u_onehot.shape))
        # print('u_onehot_next = {}'.format(u_onehot_next.shape))
        # print('u_onehot_last = {}'.format(u_onehot_last.shape))
        inputs, inputs_next = [], []
        # 添加状态
        if self.args.idv_comparing:
            inputs.append(obs)
            inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号




            inputs = torch.cat([x.float().reshape(episode_num, self.args.n_agents, -1) for x in inputs], dim=2)
            inputs_next = torch.cat([x.float().reshape(episode_num, self.args.n_agents, -1) for x in inputs_next],
                                    dim=2)
        # 添加obs
        else:
            inputs.append(s)
            inputs_next.append(s_next)
            inputs.append(obs)
            inputs_next.append(obs_next)
            # 添加所有agent的上一个动作
            inputs.append(u_onehot_last)
            inputs_next.append(u_onehot)

            # 添加当前动作
            '''
            因为coma对于当前动作，输入的是其他agent的当前动作，不输入当前agent的动作，为了方便起见，每次虽然输入当前agent的
            当前动作，但是将其置为0相量，也就相当于没有输入。
            '''
            action_mask = (1 - torch.eye(self.n_agents))  # th.eye（）生成一个二维对角矩阵
            # print('action_mask_1 = {}'.format(action_mask.shape))
            # 得到一个矩阵action_mask，用来将(episode_num, n_agents, n_agents * n_actions)的actions中每个agent自己的动作变成0向量
            action_mask = action_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
            # print('action_mask_2_shape = {}'.format(action_mask.shape))
            # print('action_mask_2 = {}'.format(action_mask))

            # print('u_onehot * action_mask_shape = {}'.format((u_onehot * action_mask).shape))
            # print('u_onehot * action_mask = {}'.format(u_onehot * action_mask))
            inputs.append(u_onehot * action_mask.unsqueeze(0))
            inputs_next.append(u_onehot_next * action_mask.unsqueeze(0))

            # 添加agent编号对应的one-hot向量
            '''
            因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在后面添加对应的向量
            即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            '''
            mat_show = torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1)
            # print('torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1)_shape = {}'.format(mat_show.shape))
            # print('torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1) = {}'.format(mat_show))
            inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

            # 要把inputs中的5项输入拼起来，并且要把其维度从(episode_num, n_agents, inputs)三维转换成(episode_num * n_agents, inputs)二维
            inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
            inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)
            # print('inputs = {}'.format(inputs.shape))
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            # 神经网络输入的是(episode_num * n_agents, inputs)二维数据，得到的是(episode_num * n_agents， n_actions)二维数据
            q_eval = self.eval_critic(inputs)
            q_target = self.target_critic(inputs_next)
            # print('get_q_values_for_transition {}\ninputs = {}\ninputs_next = {}\nq_eval = {}\nq_targets = {}'.format( \
            #     transition_idx,inputs,inputs_next,q_eval,q_targets))
            # 把q值的维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_evals和q_targets是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        if self.args.cuda:
            q_evals = q_evals.cuda()
            q_targets = q_targets.cuda()
        return q_evals, q_targets

    def _get_actor_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs,obs_next, u_onehot = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx],batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs_next = []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给inputs添加上一个动作、agent编号
        if not self.args.no_rnn:
            if self.args.last_action:
                if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                    inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
                else:
                    inputs.append(u_onehot[:, transition_idx - 1])
                inputs_next.append(u_onehot[:,transition_idx])
            if self.args.reuse_network:
                # 因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在dim_1上添加对应的向量
                # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
                # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
                inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
                inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把inputs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        # TODO 检查inputs_next是不是相当于inputs向后移动一条
        return inputs,inputs_next

    def prev_get_action_prob(self, batch, max_episode_len, epsilon,if_target = False):

        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        # print('avail_action_shape = {}'.format(avail_actions.shape))
        avail_actions_next = batch['avail_u_next']
        action_prob = []
        for transition_idx in range(max_episode_len):
            # if not if_target and transition_idx == 0:
                # print('trans_idx = {}\n\nhidden_before = {}'.format(transition_idx,self.eval_hidden))
            inputs,inputs_next = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                if if_target:
                    self.target_hidden = self.target_hidden.cuda()
                else:
                    self.eval_hidden = self.eval_hidden.cuda()
            if if_target:
                outputs, self.target_hidden = self.target_rnn(inputs, self.target_hidden)
            else:
                outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            # 把q_eval维度重新变回(8, 5,n_actions)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)

            action_prob.append(prob)
        if if_target:
            last_outputs, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
        else:
            last_outputs,self.eval_hidden = self.eval_rnn(inputs_next,self.eval_hidden)
        last_outputs = last_outputs.view(episode_num, self.n_agents, -1)
        last_prob = torch.nn.functional.softmax(last_outputs, dim=-1)
        action_prob.append(last_prob)

        avail_actions = torch.cat([avail_actions,avail_actions_next[:,-1].unsqueeze(1)],dim = 1 )
        # print('avail_action_shape_after = {}'.format(avail_actions.shape))
        # 得的action_prob是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        action_prob = torch.stack(action_prob, dim=1).cpu()
        if self.args.check_alg:
            action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1,
                                                                                avail_actions.shape[-1])  # 可以选择的动作的个数
            action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
        # if not if_target:
            # print('action_prob_before_mask = {}'.format(action_prob))
        action_prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0
        # 因为上面把不能执行的动作概率置为0，所以概率和不为1了，这里要重新正则化一下。执行过程中Categorical会自己正则化。
        prob_sum = torch.tensor(action_prob).detach()
        prob_sum = prob_sum.sum(dim = -1)
        prob_sum = prob_sum.reshape([-1])
        for i in range(len(prob_sum)):
            if prob_sum[i] == 0:
                prob_sum[i] = 1
        prob_sum = prob_sum.reshape(*action_prob.shape[:-1]).unsqueeze(-1)
        action_prob = action_prob / prob_sum
        # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan。
        # 因此需要再一次将该经验对应的概率置为0
        action_prob[avail_actions == 0] = 0.0
        # if not if_target:
            # print('action_prob_after_final = {}'.format(action_prob))
        check = action_prob < 0
        check = check.sum()
        # print('check = {}'.format(check))
        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden
        if self.args.idv_comparing and self.args.idv_para:
            self.eval_hidden = []
            self.target_hidden = []
            for i in range(self.n_agents):
                self.eval_hidden.append(torch.zeros((episode_num, self.args.rnn_hidden_dim)))
                self.target_hidden.append(torch.zeros((episode_num, self.args.rnn_hidden_dim)))

        else:
            self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
            self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))


    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/newest_critic_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir +'/newest_rnn_params.pkl')

    def learn(self, batch, max_episode_len, train_step, epsilon=None, logger=None,
              policy_logger_list=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数

        loss, policy_loss = 0, 0
        bias_ret = [0, 0]
        for a in range(self.args.n_agents):

            ret = self.dmac_learn_idv(batch, max_episode_len, train_step, agent_id=a, epsilon=epsilon,
                                          logger=logger, policy_logger_list=policy_logger_list)

            idv_policy_loss = ret[0]
            idv_loss = ret[1]
            loss += idv_loss
            policy_loss += idv_policy_loss


            idv_bias_ret = ret[2]
            bias_ret[0] += idv_bias_ret[0]
            bias_ret[1] += idv_bias_ret[1]
        loss /= self.args.n_agents
        policy_loss /= self.args.n_agents
        ret = [policy_loss, loss]


        bias_ret[0] /= self.args.n_agents
        bias_ret[1] /= self.args.n_agents
        ret.append(bias_ret)
        q_loss_ret = [loss, None, None, None]
        return q_loss_ret,policy_loss,bias_ret


    def dmac_learn_idv(self, batch, max_episode_len, train_step, agent_id,epsilon=None,logger = None,policy_logger_list = None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
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

        mask_next = mask_next[:, :, agent_id].unsqueeze(2)
        avail_action = avail_action[:, :, agent_id].unsqueeze(2)
        mask = mask[:, :, agent_id].unsqueeze(2)
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
        u_for_q = u_for_q[:, :, agent_id, :].unsqueeze(2)

        u_next = u_for_q[:, 1:]
        # print('u_for_q_shape = {},u_next_shape = {}'.format(u_for_q.shape,u_next.shape))
        padded_u_next = torch.zeros(*u_for_q[:, -1].shape, dtype=torch.long).unsqueeze(1)
        if self.args.cuda:
            padded_u_next = padded_u_next.cuda()
        u_next = torch.cat((u_next, padded_u_next), dim=1)

        u_for_q = u_for_q[:, :, agent_id, :].unsqueeze(2)
        u_next = u_next[:, :, agent_id, :].unsqueeze(2)

        q_evals_all, q_targets_all = self.dmac_get_q_values(batch, max_episode_len, agent_id=agent_id)

        q_evals_for_q = torch.gather(q_evals_all, dim=3, index=u_for_q).squeeze(3)
        q_total_eval_for_q = q_evals_for_q
        q_targets = torch.gather(q_targets_all,dim = 3,index= u_next).squeeze(3)


        policy_loss_ret,bias_ret = None,None

        all_action_prob,init_action_prob = self.dmac_get_action_prob(batch, max_episode_len, epsilon, agent_id = agent_id)  # 每个agent的所有动作的概率
        all_target_action_prob,_ = self.dmac_get_action_prob(batch, max_episode_len, epsilon, if_target=True, agent_id = agent_id)

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
        q_total_eval_for_policy =  q_evals_for_policy# (32,50,1)


        # print('q_total_eval = {}'.format(q_total_eval.shape))
        q_table = q_evals_all

        # q_baseline = (q_table * curr_action_prob).sum(dim = 3) # (32,50,5)
        q_baseline = (q_table * curr_action_prob).sum(dim = 3)
        # print('q_baseline = {}'.format(q_baseline.shape))
        bias_ret = []
        if self.args.alg == 'i-sac':
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
        if self.args.alg == 'i-sac':
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

        if self.args.idv_para:
            self.policy_optimizer[agent_id].zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_parameters[agent_id], self.args.grad_norm_clip)
            self.policy_optimizer[agent_id].step()
        else:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_parameters, self.args.grad_norm_clip)
            self.policy_optimizer.step()

        print('***************************************************************************')
        print('policy_loss = {}'.format(policy_loss))
        policy_loss_ret = policy_loss



        next_pi_taken = torch.gather(next_action_prob, dim=3, index=u_next).squeeze(3)
        next_target_pi_taken = torch.gather(next_target_action_prob, dim=3, index=u_next).squeeze(3)
        next_pi_taken[mask_next == 0] = 1.0
        next_target_pi_taken[mask_next == 0] = 1.0
        if self.args.alg == 'i-sac':
            next_bias_sum = torch.log(next_pi_taken)
        else:
            next_bias_sum = torch.log(next_pi_taken) - torch.log(next_target_pi_taken)
        next_bias_sum = next_bias_sum.sum(dim = 2,keepdim=True)
        next_bias_sum /= self.args.reward_scale

        # print('q_total_target = {}'.format(q_total_target.shape))
        # print('done = {}'.format(done.shape))
        targets_rec = None
        q_val_rec = None
        q_bias = None

        # if self.args.test_no_use_td_lambda:
        #     targets = r + self.args.gamma * (q_targets - next_bias_sum) * (1 - done)
        # else:
        #     targets = td_lambda_target(batch, max_episode_len, q_targets.cpu(), next_bias_sum.cpu(),self.args)
        targets = r + self.args.gamma * (q_targets - next_bias_sum) * (1 - done)

        if self.args.cuda:
            targets = targets.cuda()

        td_error = (q_total_eval_for_q - targets.detach())
        # print('mask = {}'.format(mask.shape))
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error
        # print('masked_td_error = {}'.format(masked_td_error.shape))
        # loss = masked_td_error.pow(2).mean()
        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.idv_para:
            self.critic_optimizer[agent_id].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_parameters[agent_id], self.args.grad_norm_clip)
            self.critic_optimizer[agent_id].step()
        else:
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
            self.critic_optimizer.step()


        print('***************************************************************************')
        print('q_loss = {}'.format(loss))
        print('***************************************************************************')

        ret = [policy_loss,loss]
        # print('self.alg == {}'.format(self.alg))

            # print('here?')
        ret.append(bias_ret)
        # print('ret = {}'.format(len(ret)))
        return ret

    def dmac_get_q_values(self, batch, max_episode_len,agent_id = None):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            # 神经网络输入的是(episode_num * n_agents, inputs)二维数据，得到的是(episode_num * n_agents， n_actions)二维数据
            if self.args.idv_para:
                tmp_inputs = inputs[:,agent_id,:]
                tmp_inputs_next = inputs_next[:,agent_id,:]
                eval_critic = self.eval_critic[agent_id]
                target_critic = self.target_critic[agent_id]
            else:
                tmp_inputs = inputs
                tmp_inputs_next = inputs_next
                eval_critic = self.eval_critic
                target_critic = self.target_critic
            # print('tmp_inputs = {}, eval_V = {}'.format(tmp_inputs.shape,self.eval_critic))
            q_eval = eval_critic(tmp_inputs)
            q_target = target_critic(tmp_inputs_next)
            # print('get_q_values_for_transition {}\ninputs = {}\ninputs_next = {}\nq_eval = {}\nq_targets = {}'.format( \
            #     transition_idx,inputs,inputs_next,q_eval,q_targets))
            # 把q值的维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, 1, -1)
            q_target = q_target.view(episode_num, 1, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_evals和q_targets是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        if self.args.cuda:
            q_evals = q_evals.cuda()
            q_targets = q_targets.cuda()
        return q_evals, q_targets
    def dmac_get_action_prob(self, batch, max_episode_len, epsilon, if_target=False,agent_id = None):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        # print('avail_action_shape = {}'.format(avail_actions.shape))
        avail_actions_next = batch['avail_u_next']
        if self.args.idv_para:
            avail_actions = avail_actions[:, :, agent_id, :].unsqueeze(2)
            avail_actions_next = avail_actions_next[:, :, agent_id, :].unsqueeze(2)
        action_prob = []
        for transition_idx in range(max_episode_len):
            # if not if_target and transition_idx == 0:
                # print('trans_idx = {}\n\nhidden_before = {}'.format(transition_idx,self.eval_hidden))
            inputs,inputs_next = self._get_policy_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                if self.args.idv_para:
                    if if_target:
                        self.target_hidden[agent_id] = self.target_hidden[agent_id].cuda()
                    else:
                        self.eval_hidden[agent_id] = self.eval_hidden[agent_id].cuda()
                else:
                    if if_target:
                        self.target_hidden = self.target_hidden.cuda()
                    else:
                        self.eval_hidden = self.eval_hidden.cuda()
            if self.args.idv_para:
                tmp_eval_policy_hidden = self.eval_hidden[agent_id]
                tmp_target_policy_hidden = self.target_hidden[agent_id]
                tmp_inputs = inputs[:,agent_id]
                tmp_inputs_next = inputs_next[:,agent_id]
            else:
                tmp_eval_policy_hidden = self.eval_hidden
                tmp_target_policy_hidden = self.target_hidden
                tmp_inputs = inputs
                tmp_inputs_next = inputs_next
            if self.args.idv_para:
                if if_target:
                    outputs, self.target_hidden[agent_id] = self.target_rnn[agent_id](tmp_inputs, tmp_target_policy_hidden)
                else:
                    outputs, self.eval_hidden[agent_id] = self.eval_rnn[agent_id](tmp_inputs, tmp_eval_policy_hidden)
            else:
                if if_target:
                    outputs, self.target_hidden = self.target_rnn(tmp_inputs, tmp_target_policy_hidden)
                else:
                    outputs, self.eval_hidden = self.eval_rnn(tmp_inputs, tmp_eval_policy_hidden)
            # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            # 把q_eval维度重新变回(8, 5,n_actions)
            outputs = outputs.view(episode_num, 1, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)

            action_prob.append(prob)
        if self.args.idv_para:
            if if_target:
                last_outputs, self.target_hidden[agent_id] = self.target_rnn[agent_id](tmp_inputs_next, tmp_target_policy_hidden)
            else:
                last_outputs,self.eval_hidden[agent_id] = self.eval_rnn[agent_id](tmp_inputs_next,tmp_eval_policy_hidden)
        else:
            if if_target:
                last_outputs, self.target_hidden = self.target_rnn(tmp_inputs_next, tmp_target_policy_hidden)
            else:
                last_outputs,self.eval_hidden = self.eval_rnn(tmp_inputs_next,tmp_eval_policy_hidden)
        last_outputs = last_outputs.view(episode_num, 1, -1)
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


    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号


        if self.args.idv_comparing:
            # print('obs = {}'.format(obs.shape))
            if self.args.idv_para:
                inputs = torch.cat([x.float().reshape(episode_num, self.args.n_agents, -1) for x in inputs], dim=2)
                inputs_next = torch.cat([x.float().reshape(episode_num, self.args.n_agents, -1) for x in inputs_next],
                                        dim=2)
            else:
                inputs = torch.cat([x.float().reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
                inputs_next = torch.cat([x.float().reshape(episode_num * self.args.n_agents, -1) for x in inputs_next],
                                        dim=1)
        else:
            if self.args.idv_para:
                inputs = torch.cat([x.float().reshape(episode_num, self.args.n_agents, -1) for x in inputs], dim=2)
                inputs_next = torch.cat([x.float().reshape(episode_num, self.args.n_agents, -1) for x in inputs_next],
                                        dim=2)
            else:
                if self.args.reuse_network:
                    # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
                    # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
                    # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
                    inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
                    inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
                # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
                # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据

                inputs = torch.cat([x.float().reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
                inputs_next = torch.cat([x.float().reshape(episode_num * self.args.n_agents, -1) for x in inputs_next],
                                        dim=1)
        # TODO 检查inputs_next是不是相当于inputs向后移动一条
        return inputs, inputs_next

    def _get_policy_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号


        if self.args.idv_comparing:
            if self.args.idv_para:
                inputs = torch.cat([x.float().reshape(episode_num, self.args.n_agents, -1) for x in inputs], dim=2)
                inputs_next = torch.cat([x.float().reshape(episode_num, self.args.n_agents, -1) for x in inputs_next],
                                        dim=2)
            else:
                inputs = torch.cat([x.float().reshape(episode_num* self.args.n_agents, -1) for x in inputs], dim=1)
                inputs_next = torch.cat([x.float().reshape(episode_num* self.args.n_agents, -1) for x in inputs_next],
                                        dim=1)
        else:
            if self.args.idv_para:
                inputs = torch.cat([x.float().reshape(episode_num, self.args.n_agents, -1) for x in inputs], dim=2)
                inputs_next = torch.cat([x.float().reshape(episode_num, self.args.n_agents, -1) for x in inputs_next],
                                        dim=2)
            else:
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
                inputs_next = torch.cat([x.float().reshape(episode_num * self.args.n_agents, -1) for x in inputs_next],
                                        dim=1)
        # TODO 检查inputs_next是不是相当于inputs向后移动一条
        return inputs, inputs_next



