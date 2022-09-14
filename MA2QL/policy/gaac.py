import torch
import os
import numpy as np
import math
from network.base_net import RNN
from network.commnet import CommNet
from network.g2anet import G2ANet
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



class GAAC:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape  # actor网络输入的维度，和vdn、qmix的rnn输入维度一样，使用同一个网络结构
        # 根据参数决定RNN的输入维度
        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents
        self.args = args

        # 神经网络
        # 每个agent选动作的网络,输出当前agent所有动作对应的概率，用该概率选动作的时候还需要用softmax再运算一次。

        self.eval_rnn = RNN(actor_input_shape, args)
        self.target_rnn = RNN(actor_input_shape, args)
        self.eval_critic = G2ANet(actor_input_shape, args)
        self.target_critic = G2ANet(actor_input_shape, args)
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_params.pkl'):
                path_rnn = self.model_dir + '/rnn_params.pkl'
                self.eval_rnn.load_state_dict(torch.load(path_rnn))
                print('Successfully load the model: {}'.format(path_rnn))
            else:
                raise Exception("No model!")

        self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.rnn_parameters = list(self.eval_rnn.parameters())
        self.critic_parameters = list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.lr_critic)
            self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=args.lr_actor)


        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden
        self.div_reg = self.args.alg.find('div') > -1
        self.eval_hidden = None
        self.target_hidden = None
        self.critic_eval_hidden = None
        self.critic_target_hidden = None
    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.critic_eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.critic_target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
    def soft_update_policy(self,tau= 0.001):
        soft_update(self.target_rnn,self.eval_rnn,tau)
    def hard_update_policy(self):
        hard_update(self.target_rnn,self.eval_rnn)
    def hard_update_critic(self):
        hard_update(self.target_critic,self.eval_critic)
    def _get_action_prob(self, batch, max_episode_len, epsilon,if_target = False):

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

        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob

    def _get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        u_onehot, r, done = batch['u_onehot'], batch['r'], batch['terminated']
        u = torch.argmax(u_onehot,dim = 3, keepdim= True)
        u_next = u[:, 1:]
        padded_u_next = torch.zeros(*u[:, -1].shape, dtype=torch.long).unsqueeze(1)
        eye = torch.eye(self.n_actions)
        if self.args.cuda:
            padded_u_next = padded_u_next.cuda()
            eye = eye.cuda()
            u_next = u_next.cuda()
            u = u.cuda()
        u_next = torch.cat((u_next, padded_u_next), dim=1)
        u_input_all = eye[u].squeeze(3)
        u_next_input_all = eye[u_next].squeeze(3)

        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_actor_inputs(batch, transition_idx)
            u_input = u_input_all[:,transition_idx].reshape(self.n_agents * episode_num,-1)
            u_next_input = u_next_input_all[:,transition_idx].reshape(self.n_agents * episode_num,-1)
            if self.args.cuda:
                u_input = u_input.cuda()
                u_next_input = u_next_input.cuda()
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.critic_target_hidden =  self.critic_target_hidden.cuda()
                self.critic_eval_hidden = self.critic_eval_hidden.cuda()
            q_target, self.critic_target_hidden = self.target_critic(inputs_next, self.critic_target_hidden,u_next_input)
            q_eval, self.critic_eval_hidden = self.eval_critic(inputs,self.critic_eval_hidden,u_input)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)


            # 把q值的维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_evals和q_targets是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def _train_critic(self, batch, max_episode_len, train_step,next_bias = None):
        onehot_u, r, done = batch['u_onehot'], batch['r'], batch['terminated']
        # print('onehot_u = {}'.format(onehot_u.shape))
        u = torch.argmax(onehot_u, dim=-1, keepdim=True)
        u_next = u[:, 1:]
        padded_u_next = torch.zeros(*u[:, -1].shape, dtype=torch.long).unsqueeze(1)
        u_next = torch.cat((u_next, padded_u_next), dim=1)
        mask = (1 - batch["padded"].float())  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            r = r.cuda()
            done = done.cuda()
            u = u.cuda()
            u_next = u_next.cuda()
            mask = mask.cuda()

        q_evals, q_next_target = self._get_q_values(batch, max_episode_len)
        q_values = q_evals.clone()  # 在函数的最后返回，用来计算advantage从而更新actor
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了

        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_next_target = torch.gather(q_next_target, dim=3, index=u_next).squeeze(3)

        if self.args.cuda:
            q_next_target = q_next_target.cuda()
            next_bias = next_bias.cuda()

        if self.args.use_td_lambda:
            targets = td_lambda_target(batch, max_episode_len, q_next_target.cpu(), next_bias.cpu(), self.args)
        else:
            targets = r + self.args.gamma * (q_next_target - next_bias) * (1 - done)

        if self.args.cuda:
            targets = targets.cuda()
        td_error = targets.detach() - q_evals
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # print('Loss is ', loss)
        self.critic_optimizer.zero_grad()
        loss.backward()
        # print('q_loss = {}'.format(loss.item()))
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        return q_values, loss.item()


    def learn(self, batch, max_episode_len, train_step, epsilon):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
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
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
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
        if self.args.cuda:
            padded_u_next = padded_u_next.cuda()
        u_next = torch.cat((u_next, padded_u_next), dim=1)

        all_action_prob = self._get_action_prob(batch, max_episode_len, epsilon)  # 每个agent的所有动作的概率
        curr_action_prob = all_action_prob[:,:-1]
        next_action_prob =  all_action_prob[:,1:]# 每个agent的所有动作的概率
        next_pi_taken = torch.gather(next_action_prob, dim=3, index=u_next).squeeze(3)
        next_pi_taken[mask_next == 0] = 1.0


        all_target_action_prob = self._get_action_prob(batch, max_episode_len, epsilon, if_target=True)
        curr_target_action_prob = all_target_action_prob[:, :-1]
        next_target_action_prob = all_target_action_prob[:, 1:]
        next_target_pi_taken = torch.gather(next_target_action_prob, dim=3, index=u_next).squeeze(3)
        next_target_pi_taken[mask_next == 0] = 1.0
        if self.div_reg:
            next_bias = torch.log(next_pi_taken) - torch.log(next_target_pi_taken)
            next_bias = next_bias.sum(dim=2, keepdim=True).repeat(1, 1, self.n_agents)
            next_bias /= self.args.reward_scale
        else:
            next_bias = torch.log(next_target_pi_taken)
            next_bias /= self.args.reward_scale

        q_values, q_loss = self._train_critic(batch, max_episode_len, train_step, next_bias.clone().detach())

        mask_for_action = mask.unsqueeze(3).repeat(1,1,1,avail_action.shape[-1])

        sample_prob = curr_action_prob.detach()
        sample_prob[mask_for_action == 0] = 1

        action_sample = Categorical(sample_prob).sample().unsqueeze(3)

        q_taken = torch.gather(q_values, dim=3, index=action_sample).squeeze(3)  # 每个agent的选择的动作对应的Ｑ值
        pi_taken = torch.gather(curr_action_prob, dim=3, index=action_sample).squeeze(3)  # 每个agent的选择的动作对应的概率
        pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
        log_pi_taken = torch.log(pi_taken)
        curr_action_prob_for_log = torch.tensor(curr_action_prob)
        curr_action_prob_for_log[avail_action == 0] = 1.0
        curr_action_prob_for_log[mask_for_action == 0] = 1.0


        if self.div_reg:
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
        if self.div_reg:
            advantage = (q_taken - baseline - bias).detach()
        else:
            advantage = (q_taken - baseline - log_pi_taken / self.args.reward_scale).detach()
            
        # print('baseline = {},advantage = {}'.format(baseline, advantage))
        loss = - ((advantage * log_pi_taken) * mask).sum() / mask.sum()
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        policy_loss = loss.item()

        return q_loss,policy_loss,[bias_mean_no_baseline,bias_mean_baseline]


    def _get_actor_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs,obs_next, u_onehot = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx],batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs_next = []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给inputs添加上一个动作、agent编号

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




    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')