import torch
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward

import einops

def combine_first_two(tensor):
    if tensor is None:
        return None
    return tensor.view((tensor.size(0) * tensor.size(1),) + tensor.size()[2:])

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, contrastive_flag, ids=None):
        opt = self.opt
        
        out = {}
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif contrastive_flag:
            # fc_feats: batch_size x (n_distractors+1) x d
            # att_feats: batch_size x (n_distractors+1) x n_obj x d
            # labels: batch_size x (n_distractors+1) x num_captions x T
            # att_masks: batch_size x (n_distractors+1) x n_obj
            alpha = opt.pragmatic_incremental_alpha
            batch_size, per_image_dim, _ = fc_feats.size()
            batch_size_, per_image_dim_, num_captions, T = labels.size()
            assert batch_size == batch_size_ and per_image_dim == per_image_dim_

            if opt.pragmatic_distractor_candidate_type in ['closest', 'random']:
                num_distractors = opt.pragmatic_distractors
            elif opt.pragmatic_distractor_candidate_type == 'batch':
                num_distractors = batch_size - 1
            else:
                raise ValueError(
                    f"invalid --pragmatic_distractor_candidate_type {opt.pragmatic_distractor_candidate_type}"
                )
            assert num_distractors+1 == per_image_dim

            fc_feats_target, fc_feats_distr = fc_feats.split((1, num_distractors), dim=1)
            att_feats_target, att_feats_distr = att_feats.split((1, num_distractors), dim=1)
            att_masks_target, att_masks_distr = att_masks.split((1, num_distractors), dim=1)

            labels_from_target = labels[:,0]
            masks_from_target = masks[:,0]

            labels_replace_distractor_target = labels.clone()
            masks_replace_distractor_target = masks.clone()

            labels_replace_distractor_target[:,1:] = labels_from_target.unsqueeze(1).expand_as(labels_replace_distractor_target[:,1:])
            masks_replace_distractor_target[:,1:] = masks_from_target.unsqueeze(1).expand_as(masks_replace_distractor_target[:,1:])

            outs = self.model(combine_first_two(fc_feats), combine_first_two(att_feats),
                              combine_first_two(labels_replace_distractor_target[..., :-1]), combine_first_two(att_masks))
            # batch_size x (n_distractors+1) x num_captions x T-1 x V
            outs = outs.view(batch_size, per_image_dim, num_captions, T-1, -1)
            V = outs.size(-1)

            num_choices = 2

            outs_target, outs_distractors = outs.split((1, num_distractors), dim=1)
            outs_target = outs_target.expand_as(outs_distractors)

            # batch_size x (n_distractors) x num_captions x num_choices x T-1 x V
            outs_comparative = torch.stack((outs_target, outs_distractors), 3)
            assert outs_comparative.size(-2) == T-1

            labels_from_target_expanded = labels_from_target.unsqueeze(1).unsqueeze(3).unsqueeze(4)\
                .expand(batch_size, num_distractors, num_captions, num_choices, 1, T)

            masks_from_target_expanded = masks_from_target.unsqueeze(1).unsqueeze(3) \
                .expand(batch_size, num_distractors, num_captions, num_choices, T)

            def select_label(tensor, t):
                return tensor.gather(-1, labels_from_target_expanded[...,t]).squeeze(-1)

            # batch_size x (n_distractors) x num_captions x 2
            log_priors = torch.full(outs_comparative.size()[:-2], 1./outs_comparative.size(3)).log().to(outs_comparative)

            log_s1_sums = torch.zeros_like(log_priors)
            # batch_size x (n_distractors) x num_captions x num_choices
            word_counts = torch.zeros_like(log_s1_sums)

            for t in range(T-1):
                # batch_size x (n_distractors) x num_captions x num_choices x V
                log_s0 = outs_comparative[...,t,:]
                log_priors_expanded = log_priors.unsqueeze(-1).expand_as(log_s0)
                log_l0 = (log_s0 + log_priors_expanded).log_softmax(3)
                log_s1 = (log_s0 + (log_l0 * alpha)).log_softmax(4)
                log_s1_chosen = select_label(log_s1, t+1)
                this_mask = masks_from_target_expanded[...,t+1]
                log_s1_sums += (log_s1_chosen * this_mask)
                word_counts += this_mask
                if opt.pragmatic_incremental_l1_uses == 's0':
                    s_to_use = log_s0
                elif opt.pragmatic_incremental_l1_uses == 's1':
                    s_to_use = log_s1
                else:
                    raise ValueError("invalid --pragmatic_incremental_l1_uses {}".format(opt.pragmatic_incremental_l1_uses))
                log_l1 = (select_label(log_l0, t+1) + select_label(s_to_use, t+1)).log_softmax(3)
                log_priors = log_l1

            # log p(c | i, i') for caption c, target image i and distractor i'
            # batch_size x n_distractors x num_captions
            target_log_seq_s1 = log_s1_sums[...,0]

            # TODO: a model that incorporates object features too
            # log p(i' | i) for target image i and distractor i'
            # batch_size x n_distractors
            distractor_log_probs = self.model.distractor_log_probs(fc_feats_target, fc_feats_distr)

            # log p(c, i' | i))
            # batch_size x n_distractors x num_captions
            joint_log_s1 = target_log_seq_s1 + distractor_log_probs.unsqueeze(-1).expand_as(target_log_seq_s1)

            # batch_size x num_captions
            if opt.contrastive_em == 'hard':
                obj_log_s1 = joint_log_s1.max(1).values
            elif opt.contrastive_em == 'soft':
                obj_log_s1 = joint_log_s1.logsumexp(1)

            # TODO: this is a bit weird: scaling to be similar to the loss used in non-contrastive training, but
            #  will hopefully prevent having to mess with the LRs too much
            loss = -(obj_log_s1.sum()) / word_counts[:,0,:,0].sum()
        elif not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
