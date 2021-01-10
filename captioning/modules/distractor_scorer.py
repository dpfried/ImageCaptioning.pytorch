import torch
from torch import nn

class DistractorScorer(torch.nn.Module):
    def __init__(self, opt):
        super(DistractorScorer, self).__init__()
        self.opt = opt
        hidden_size = opt.pragmatic_distractor_scoring_hidden_size
        self.scorer = nn.Sequential(nn.Linear(opt.fc_feat_size*2, hidden_size),
                                      nn.ReLU(),
                                      nn.Dropout(opt.drop_prob_lm),
                                      nn.Linear(hidden_size, 1))

    def forward(self, fc_feats_target, fc_feats_distr):
        # fc_feats_target: batch_size x 1 x d
        # fc_feats_distr: batch_size x n_distractors x d
        cat_feats = torch.cat((fc_feats_target.expand_as(fc_feats_distr), fc_feats_distr), -1)
        # batch_size x n_distractors x 1
        scores = self.scorer(cat_feats)
        log_probs = scores.squeeze(-1).log_softmax(-1)
        return log_probs