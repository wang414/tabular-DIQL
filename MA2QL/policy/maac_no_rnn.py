import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic
from torch.distributions import Categorical

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
    def __init__(self, agent_init_params, sa_size,
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
        self.nagents = len(sa_size)

        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **params)
                         for params in agent_init_params]
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
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

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]
    def choose_action(self,obs,avail_action,agent_num):
        a = self.agents[agent_num]
        avail_action = torch.tensor(avail_action,dtype=torch.float32)
        obs = torch.tensor(obs).double()
        # obs = obs.repeat([2,1])
        obs_dim = len(obs.shape)
        if obs_dim == 1:
            obs = obs.unsqueeze(0)
        avail_action_dim = len(avail_action.shape)
        if avail_action_dim == 1 :
            avail_action = avail_action.unsqueeze(0)
        # print('obs_shape = {}'.format(obs.shape))
        logit = a.policy(obs)
        prob = F.softmax(logit,dim = -1)
        prob = prob.detach()
        # print('prob_all = {}'.format(prob))
        # prob = prob[0]
        # print('prob_0_before = {}'.format(prob))
        # prob[avail_action == 0] = 0.0
        prob[avail_action == 0] = 0
        # print('avail_action_shape = {}'.format(avail_action.shape))
        # print('prob_0_after = {}'.format(prob))
        action = Categorical(prob).sample().long()
        # print('action_shape = {}'.format(action.shape))
        return action
    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        for key in sample:
            sample[key] = cast(sample[key])
        avail_action,avail_action_next,obs, acs, rews, next_obs, dones =\
        sample['avail_u'],sample['avail_u_next'],sample['o'],sample['u_onehot'],sample['r'],sample['o_next'],sample['terminated']
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
        rews,dones,obs,next_obs,acs,mask,mask_next,avail_action,avail_action_next =\
        [ change_shape(x) for x in [rews,dones,obs,next_obs,acs,mask,mask_next,avail_action,avail_action_next] ]
        # Q loss
        next_acs = []
        next_log_pis = []
        for pi, ob,ava_next,m_next in zip(self.target_policies, next_obs,avail_action_next,mask_next):
            curr_next_logit = pi(ob)
            curr_next_probs = F.softmax(curr_next_logit)
            curr_next_probs = get_norm_pi(curr_next_probs,ava_next)

            sample_prob = curr_next_probs
            m_acs = m_next.unsqueeze(1).repeat(1, sample_prob.shape[1])
            sample_prob[m_acs == 0] = 1

            curr_next_ac_int = Categorical(sample_prob).sample()

            ac_dim = curr_next_probs.shape[-1]
            eye = torch.eye(ac_dim)
            curr_next_ac = eye[curr_next_ac_int].double()

            curr_next_log_probs = get_log_pi(curr_next_probs,ava_next,m_next)
            curr_next_log_pi = (curr_next_ac * curr_next_log_probs).sum(dim=1).view(-1, 1)

            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)

        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True,
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

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        for key in sample:
            sample[key] = cast(sample[key])
        avail_action,avail_action_next,obs, acs, rews, next_obs, dones =\
        sample['avail_u'],sample['avail_u_next'],sample['o'],sample['u_onehot'],sample['r'],sample['o_next'],sample['terminated']
        mask_init = (1 - sample["padded"])
        # obs, acs, rews, next_obs, dones,mask = cast([obs, acs, rews, next_obs, dones,mask])
        mask_init = mask_init.float()
        zero_mask = torch.zeros([mask_init.shape[0],1,mask_init.shape[2]])
        # print('zero_mask_shape = {}'.format(zero_mask.shape))
        # print('mask_init_part_shape = {}'.format(mask_init[:,1:,:].shape))
        mask_init_next = torch.cat([mask_init[:,1:,:], zero_mask],dim = 1)

        mask = mask_init.repeat(1,1,self.nagents)
        mask_next = mask_init_next.repeat(1,1,self.nagents)
        rews = rews.repeat(1, 1, self.nagents)
        dones = dones.repeat(1, 1, self.nagents)

        rews,dones,obs, next_obs, acs, mask, mask_next, avail_action, avail_action_next = \
            [change_shape(x) for x in [rews,dones,obs, next_obs, acs, mask, mask_next, avail_action, avail_action_next]]
        # print('policy_mask_shape = {}'.format(mask.shape))


        samp_acs = []
        all_probs = []
        all_log_pis = []

        for a_i, pi, ob,ava,m in zip(range(self.nagents), self.policies, obs,avail_action,mask):
            logit = pi(ob)

            probs = F.softmax(logit, dim=-1)

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
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, (q, all_q),ava,m in zip(range(self.nagents), all_probs,
                                                            all_log_pis,
                                                            critic_rets,avail_action,mask):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v

            m = m.unsqueeze(1)
            if soft:
                adv = (log_pi / self.reward_scale - pol_target).detach()
                pol_loss = ((log_pi * adv) * m).sum() / m.sum()
            else:
                adv = (-pol_target).detach()
                pol_loss = ((log_pi * adv) * m).sum() / m.sum()
            # for reg in pol_regs:
            #     pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm(
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                  pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                                  grad_norm, self.niter)


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

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

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
                     'sa_size': args.sa_size}
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
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance