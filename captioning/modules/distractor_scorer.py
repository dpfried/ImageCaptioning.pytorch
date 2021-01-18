import torch
from torch import nn

import opt_einsum

BIG_NEG = -1e9

class DistractorScorer(torch.nn.Module):
    def __init__(self, opt):
        super(DistractorScorer, self).__init__()
        self.opt = opt
        hidden_size = opt.pragmatic_distractor_scoring_hidden_size
        self.use_object_features = vars(opt).get('pragmatic_distractor_scoring_use_objects', 0)
        self.scorer = nn.Sequential(nn.Linear(opt.fc_feat_size*2, hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(opt.drop_prob_lm),
                                    nn.Linear(hidden_size, 1))
        if self.use_object_features:
            self.object_inner = nn.Bilinear(opt.att_feat_size, opt.att_feat_size, out_features=1, bias=False)
            self.object_inner.weight.data.copy_(torch.eye(opt.att_feat_size))
            self.object_scorer = nn.Sequential(nn.Linear(opt.att_feat_size*2, hidden_size),
                                               nn.ReLU(),
                                               nn.Dropout(opt.drop_prob_lm),
                                               nn.Linear(hidden_size, 1))

    def forward(self, fc_feats_target, fc_feats_distr, att_feats_target, att_feats_distr, att_masks_target, att_masks_distr):
        # fc_feats_target: batch_size x 1 x d
        # fc_feats_distr: batch_size x n_distractors x d
        # att_feats_target: batch_size x 1 x n_obj x _
        # att_feats_distr: batch_size x n_distractors x n_obj x _
        # att_masks_target: batch_size x 1 x n_obj
        # att_masks_distr: batch_size x n_distractors x n_obj
        cat_feats = torch.cat((fc_feats_target.expand_as(fc_feats_distr), fc_feats_distr), -1)
        # batch_size x n_distractors
        scores = self.scorer(cat_feats).squeeze(-1)
        if self.use_object_features:
            att_feats_target_expand = att_feats_target.expand_as(att_feats_distr)
            att_masks_target_expand = att_masks_target.expand_as(att_masks_distr)
            # batch_size x n_distractors x n_obj[target] x n_obj[distr]
            inners = opt_einsum.contract("bixd,biye,de->bixy",
                                att_feats_target_expand,
                                att_feats_distr,
                                self.object_inner.weight.squeeze(0),
                                )
            mask_outer = opt_einsum.contract("bix,biy->bixy", att_masks_target_expand, att_masks_distr)
            inners = torch.masked_fill(inners, (1-mask_outer).bool(), BIG_NEG)
            # batch_size x n_distractors x n_obj
            target_weights = inners.max(3).values.softmax(-1)
            # batch_size x n_distractors x n_obj
            distr_weights = inners.max(2).values.softmax(-1)

            # batch_size x n_distractors x d
            target_feats = opt_einsum.contract("bixd,bix->bid", att_feats_target_expand, target_weights)
            distr_feats = opt_einsum.contract("bixd,bix->bid", att_feats_distr, distr_weights)

            cat_att_feats = torch.cat((target_feats, distr_feats), -1)
            att_feat_scores = self.object_scorer(cat_att_feats).squeeze(-1)
            scores = scores + att_feat_scores
        log_probs = scores.log_softmax(-1)
        return log_probs