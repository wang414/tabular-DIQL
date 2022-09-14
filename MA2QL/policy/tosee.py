def _get_action_prob(self, batch, max_episode_len, epsilon, if_target=False):
    episode_num = batch['o'].shape[0]
    avail_actions = batch['avail_u']
    # print('avail_action_shape = {}'.format(avail_actions.shape))
    avail_actions_next = batch['avail_u_next']
    action_prob = []
    for transition_idx in range(max_episode_len):
        # if not if_target and transition_idx == 0:
        # print('trans_idx = {}\n\nhidden_before = {}'.format(transition_idx,self.eval_hidden))
        inputs, inputs_next = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
        if self.args.cuda:
            inputs = inputs.cuda()
            if if_target:
                self.target_hidden = self.target_hidden.cuda()
            else:
                self.eval_hidden = self.eval_hidden.cuda()
        if if_target:
            outputs, self.target_hidden = self.target_rnn(inputs, self.target_hidden)
        else:
            outputs, self.eval_hidden = self.eval_rnn(inputs,
                                                      self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
        # 把q_eval维度重新变回(8, 5,n_actions)
        outputs = outputs.view(episode_num, self.n_agents, -1)
        prob = torch.nn.functional.softmax(outputs, dim=-1)

        action_prob.append(prob)
    if if_target:
        last_outputs, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
    else:
        last_outputs, self.eval_hidden = self.eval_rnn(inputs_next, self.eval_hidden)
    last_outputs = last_outputs.view(episode_num, self.n_agents, -1)
    last_prob = torch.nn.functional.softmax(last_outputs, dim=-1)
    action_prob.append(last_prob)

    avail_actions = torch.cat([avail_actions, avail_actions_next[:, -1].unsqueeze(1)], dim=1)
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
    prob_sum = prob_sum.sum(dim=-1)
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