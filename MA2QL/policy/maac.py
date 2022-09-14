import torch
import torch.nn.functional as F
from network.base_net import RNN
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic
from torch.distributions import Categorical
import os
MSELoss = torch.nn.MSELoss()
def cast(x):
    return torch.tensor(x)
def change_shape(x):
    dim = len(x.shape)
    # print('dim = {}'.format(dim))
    if dim == 3:
        return x.permute(2,0,1).reshape([x.shape[2],-1]).double()
    elif dim == 4:
        return x.permute(2, 0, 1,3).reshape([x.shape[2], -1,x.shape[3]]).double()
def normalize(x):
    x_sum = x.sum(dim=-1)
    for i in range(len(x_sum)):
        if x_sum[i] == 0:
            x_sum[i] = 1
    x_sum = x_sum.unsqueeze(1)
    return x / x_sum

def get_norm_pi(pi,avail):
    pi = torch.tensor(pi).detach()
    pi[avail == 0] = 0
    pi = normalize(pi)
    pi[avail == 0] = 0
    return pi
def get_log_pi(pi,avail,m):
    if len(m.shape) == 1 :
        m = m.unsqueeze(1).repeat(1,pi.shape[1])
    # print('pi_shape = {}'.format(pi.shape))
    # print('avail_shape = {}'.format(avail.shape))
    # print('m_shape = {}'.format(m.shape))
    pi = torch.tensor(pi).detach()
    pi[avail == 0] = 1
    pi[m == 0] = 1
    pi = torch.log(pi)
    return pi
class MAAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self,args, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.args = args
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

        self.eval_rnn = RNN(actor_input_shape, args)
        self.target_rnn = RNN(actor_input_shape, args)
        self.rnn_parameters = list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=args.pi_lr)

        self.nagents = len(sa_size)



        self.critic = AttentionCritic(sa_size,self.state_shape, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size,self.state_shape, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,weight_decay=1e-3)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        self.eval_hidden = None
        self.target_hidden = None
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

    def _get_action_prob(self, batch, max_episode_len, epsilon = 0,if_target = False):

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
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
    def learn(self, sample,max_episode_len, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        episode_num = sample['o'].shape[0]
        self.init_hidden(episode_num)
        for key in sample.keys():
            sample[key] = torch.tensor(sample[key],dtype=torch.float32)
        s,next_s,avail_action,avail_action_next,obs, acs, rews, next_obs, dones = \
        sample['s'],sample['s_next'],sample['avail_u'],sample['avail_u_next'],sample['o'],sample['u_onehot'],sample['r'],sample['o_next'],sample['terminated']
        mask_init = (1 - sample["padded"])

        # obs, acs, rews, next_obs, dones,mask = cast([obs, acs, rews, next_obs, dones,mask])
        mask_init = mask_init.float()
        zero_mask = torch.zeros([mask_init.shape[0],1,mask_init.shape[2]])
        # print('zero_mask_shape = {}'.format(zero_mask.shape))
        # print('mask_init_part_shape = {}'.format(mask_init[:,1:,:].shape))
        mask_init_next = torch.cat([mask_init[:,1:,:], zero_mask],dim = 1)

        mask = mask_init.repeat(1,1,self.nagents)
        mask_next = mask_init_next.repeat(1,1,self.nagents)

        rews = rews.repeat(1,1,self.nagents)
        dones = dones.repeat(1,1,self.nagents)
        # print('mask_init_next_shape = {}'.format(mask_init_next.shape))
        # print('critic_mask_shape = {}'.format(mask_init.shape))
        # print('action_shape_before = {}'.format(acs.shape))
        # print('action_shape_after = {}'.format(acs.shape))

        # print('obs_shape = {}'.format(obs.shape))
        # obs = (episode_num,episode_length,agent, ob_dim)
        # Q loss
        all_action_prob = self._get_action_prob(sample, max_episode_len)  # 每个agent的所有动作的概率
        curr_action_prob = all_action_prob[:, :-1]
        next_action_prob = all_action_prob[:, 1:]  # 每个agent的所有动作的概率
        all_target_action_prob = self._get_action_prob(sample, max_episode_len,if_target=True)
        curr_target_action_prob = all_target_action_prob[:, :-1]
        next_target_action_prob = all_target_action_prob[:, 1:]

        rews,dones,obs,next_obs,acs,mask,mask_next,avail_action,avail_action_next,next_target_action_prob,curr_action_prob,next_action_prob,curr_target_action_prob =\
        [ change_shape(x) for x in [rews,dones,obs,next_obs,acs,mask,mask_next,avail_action,avail_action_next,next_target_action_prob,curr_action_prob,next_action_prob,curr_target_action_prob] ]
        # Q loss
        next_acs = []
        next_log_pis = []
        for pi, ob,ava_next,m_next in zip(next_target_action_prob, next_obs,avail_action_next,mask_next):

            curr_next_probs = pi

            sample_prob = curr_next_probs
            m_acs = m_next.unsqueeze(1).repeat(1, sample_prob.shape[1])
            sample_prob[m_acs == 0] = 1

            curr_next_ac_int = Categorical(sample_prob).sample()

            ac_dim = curr_next_probs.shape[-1]
            eye = torch.eye(ac_dim)
            curr_next_ac = eye[curr_next_ac_int].double()
            #NEED MASK?
            curr_next_log_probs = get_log_pi(curr_next_probs,ava_next,m_next)
            curr_next_log_pi = (curr_next_ac * curr_next_log_probs).sum(dim=1).view(-1, 1)

            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)

        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in,next_s)
        critic_rets = self.critic(critic_in,s, regularize=True,
                                  logger=logger, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs),m in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets,mask):
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale

            td_error = target_q - pq
            # print('td_error_shape = {}'.format(td_error.shape))
            # print('m_shape = {}'.format(m.shape))
            m = m.unsqueeze(1)
            td_error_masked = td_error * m
            # print('td_error_masked_shape = {}'.format(td_error_masked.shape))
            q_loss += (td_error_masked ** 2).sum() / m.sum()
            # q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

        samp_acs = []
        all_probs = []
        all_log_pis = []

        for a_i, pi, ob, ava, m in zip(range(self.nagents), curr_action_prob, obs, avail_action, mask):


            probs = pi

            _probs = torch.tensor(probs).detach()
            _probs = get_norm_pi(_probs, ava)

            m_acs = m.unsqueeze(1).repeat(1, _probs.shape[-1])
            sample_prob = _probs
            sample_prob[m_acs == 0] = 1
            curr_ac_int = Categorical(sample_prob).sample()

            ac_dim = _probs.shape[-1]
            eye = torch.eye(ac_dim)
            curr_ac = eye[curr_ac_int].double()

            probs = probs.cpu()
            probs = probs * ava
            probs_sum = probs.sum(dim=-1)
            for i in range(len(probs_sum)):
                if probs_sum[i] == 0:
                    probs_sum[i] = 1
            probs_sum = probs_sum.unsqueeze(1)
            probs = probs / probs_sum
            probs[ava == 0] = 0
            curr_ac_int = curr_ac_int.unsqueeze(1)
            pi_taken = torch.gather(probs, dim=1, index=curr_ac_int)
            m = m.unsqueeze(1)
            pi_taken[m == 0] = 1.0
            log_pi = torch.log(pi_taken)

            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in,s, return_all_q=True)

        pol_loss = 0
        for a_i, probs, log_pi, (q, all_q), ava, m in zip(range(self.nagents), all_probs,
                                                          all_log_pis,
                                                          critic_rets, avail_action, mask):

            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v

            m = m.unsqueeze(1)
            if soft:
                adv = (log_pi / self.reward_scale - pol_target).detach()
                pol_loss += ((log_pi * adv) * m).sum() / m.sum()
            else:
                adv = (-pol_target).detach()
                pol_loss += ((log_pi * adv) * m).sum() / m.sum()
            # for reg in pol_regs:
            #     pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
        self.rnn_optimizer.zero_grad()

        disable_gradients(self.critic)
        pol_loss.backward()
        enable_gradients(self.critic)

        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        policy_loss = pol_loss.item()
        if logger is not None:
            logger.add_scalar('policy_loss',
                              policy_loss, self.niter)


    def update_target_critic(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
    def hard_update_policy(self):
        for a_i,a in enumerate(self.agents):
             hard_update(a.target_policy, a.policy)
    def soft_update_policy(self):
        for a_i,a in enumerate(self.agents):
             soft_update(a.target_policy, a.policy)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save_model(self, train_step):
        """
        Save trained parameters of all agents into one file
        """
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': {'policy':self.eval_rnn.state_dict(),
                                      'target_policy':self.target_rnn.state_dict(),
                                       'policy_optimizer':self.rnn_optimizer.state_dict()},
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict,  self.model_dir + '/' + num + '_maac_model.pkl')

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space,
                              env.observation_space):
            agent_init_params.append({'num_in_pol': obsp.shape[0],
                                      'num_out_pol': acsp.n})
            sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_args(cls, args):
        agent_init_params = []
        sa_size = args.sa_size
        for o_size, a_size in sa_size:
            agent_init_params.append({'num_in_pol': o_size,
                                      'num_out_pol': a_size})

        init_dict = {'gamma': args.gamma, 'tau': args.tau,
                     'pi_lr': args.pi_lr, 'q_lr': args.q_lr,
                     'reward_scale': args.reward_scale,
                     'pol_hidden_dim': args.pol_hidden_dim,
                     'critic_hidden_dim': args.critic_hidden_dim,
                     'attend_heads': args.attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': args.sa_size,'args':args}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']


        agent_params = save_dict[agent_params]
        instance.eval_rnn.load_state_dict(agent_params['policy'])
        instance.target_rnn.load_state_dict(agent_params['target_policy'])
        instance.rnn_optimizer.load_state_dict(agent_params['policy_optimizer'])
        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance