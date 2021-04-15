import torch
from torch import nn
import itertools
import random

#contexts = ["A", "X", "B", "Y", "AX", "BX", "BY", "AY", "Z", "AZ", "BZ"]
# pairs = {}
# all_tokens = set()
# for context in contexts:
#     tokens = []
#     tokens.append(context.lower())
#     for c in context.lower():
#         tokens.append(c)
#     pairs[context] = tokens
#     all_tokens.update(tokens)
# tokens = list(sorted(all_tokens))

# TRAINING_DATA = [
#     ("AX", "ax"),
#     ("AY", "ay"),
#     ("BX", "bx"),
#     ("BY", "by"),
# ]

# TRAINING_DATA = [
#     ("Z", "z"),
#     ("AX", "a"),
#     ("AY", "a"),
#     ("BX", "bx"),
#     ("BY", "by"),
# ]

# TEST_DATA = [
#     ("AZ", "a"),
#     ("BZ", "b"),
# ]

# contexts = ["AB", "B", "A", "C"]
# tokens = ["a", "b", "c"]

contexts = ["AB", "B", "A", "C"]
#contexts = ["X", "B", "A", "C"]
#contexts = ["AB", "A", "C"]
tokens = ["a", "b", "c"]

#TRAINING_DATA = [
#    #("AB", "b"),
#    ("A", "a"),
#    ("B", "b"),
#    ("C", "c"),
#]

TRAINING_DATA = [
    ("AB", "b"),
    #("X", "b"),
    ("A", "a"),
    ("B", "b"),
    ("C", "c"),
]

pairs = {
    #'AB': list("ab"),
    'AB': list("b"),
    # 'X': list("ab"),
    # 'X': list("a"),
    #'X': list("b"),
    'A': list("a"),
    'B': list("b"),
    'C': list("c"),
}

TEST_DATA = TRAINING_DATA

N_CONTEXT_DISTRACTORS = [1]

context_indices = {k: i for i, k in enumerate(contexts)}
token_indices = {k: i for i, k in enumerate(tokens)}

TOKEN_DIM = 0
CONTEXT_DIM = 1

EPS=1e-3
BIG_NEG=-1e9


def renormalize(mat,dim):
    #return mat / mat.sum(dim=dim, keepdim=True)
    return mat.log_softmax(dim=dim)

def sn(s0, n=1):
    dist = s0.clone()
    for i in range(n):
        dist = renormalize(dist, CONTEXT_DIM)
        dist = renormalize(dist, TOKEN_DIM)
    return dist

def print_row_dist(dist):
    print('\t'.join('{}:{:.4f}'.format(tok, torch.tensor(val).exp()) for val, tok in dist))

def search_distractors(target_context, target_token, s0, distractor_scores=None):
    max_lp = BIG_NEG
    all_lp = []
    max_distractors = None
    max_dist = None

    target_context_index = context_indices[target_context]
    target_token_index = token_indices[target_token]

    if distractor_scores is not None:
        target_distractor_scores = distractor_scores[target_context_index]

    #for n_context_distractors in [1, 2, 3, 4]:
    for n_context_distractors in N_CONTEXT_DISTRACTORS:
        for n_token_distractors in [len(tokens) - 1]:
            for context_distractors in itertools.combinations(list(ctx for ctx in contexts if ctx != target_context), n_context_distractors):
                context_distractor_indices = list(context_indices[ctx] for ctx in context_distractors)
                if distractor_scores is not None:
                    this_distractor_score = target_distractor_scores[context_distractor_indices].sum()
                else:
                    this_distractor_score = 0
                for token_distractors in itertools.combinations(list(tok for tok in tokens if tok != target_token), n_token_distractors):
                    token_distractor_indices = list(token_indices[tok] for tok in token_distractors)

                    sub_s0 = s0[[target_token_index] + token_distractor_indices][:,[target_context_index] + context_distractor_indices]

                    s1 = sn(sub_s0, n=1)
                    lp = s1[0,0] + this_distractor_score
                    all_lp.append(lp)
                    this_distractors = [target_token] + list(token_distractors), [target_context] + list(context_distractors)
                    this_dist = s1[:,0]
                    if lp > max_lp:
                        max_distractors = this_distractors
                        max_lp = lp
                        max_dists = this_dist
    if distractor_scores is None:
        sum_lp = torch.stack(all_lp, -1).logsumexp(-1) - torch.tensor(len(all_lp)).float().log()
    else:
        sum_lp = torch.stack(all_lp, -1).logsumexp(-1)
    return max_lp, sum_lp, max_distractors, max_dists

class ScoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.single_contexts = [x for x in contexts if len(x) == 1]
        self.single_tokens = [x for x in tokens if len(x) == 1]
        self.single_context_indices = {
            x: ix for ix, x in enumerate(self.single_contexts)
        }
        self.single_token_indices = {
            x: ix for ix, x in enumerate(self.single_tokens)
        }
        self.context_embeddings = nn.EmbeddingBag(len(self.single_contexts), 20)
        self.token_embeddings = nn.EmbeddingBag(len(self.single_tokens), 20)

        self.distractor_scoring = nn.Bilinear(20, 20, 1, bias=False)

    def make_indices_and_offsets(self, to_embed, lookup):
        offsets = []
        indices = []
        off = 0
        for x in to_embed:
            offsets.append(off)
            indices.extend([lookup[emb] for emb in x])
            off += len(x)
        #print(offsets, indices)
        return torch.LongTensor(indices), torch.LongTensor(offsets)
    
    def embed_contexts(self, contexts):
        return self.context_embeddings(*self.make_indices_and_offsets(
            contexts, self.single_context_indices
        ))

    def embed_tokens(self, tokens):
        return self.token_embeddings(*self.make_indices_and_offsets(
            tokens, self.single_token_indices
        ))

    def make_s0(self):
        context_emb = self.embed_contexts(contexts)
        token_emb = self.embed_tokens(tokens)
        inner = torch.einsum("ax,bx->ab",token_emb,context_emb)
        return inner.log_softmax(TOKEN_DIM)

    def make_distractor_scores(self):
        context_emb = self.embed_contexts(contexts)
        inner = torch.einsum("ax,by,dxy->dab",context_emb, context_emb, self.distractor_scoring.weight)
        return inner.squeeze(0).log_softmax(-1)

compat = torch.full((len(tokens), len(contexts)), EPS)
for context, tks in pairs.items():
    for tk in tks:
        compat[token_indices[tk],context_indices[context]] = 1 / len(tks)


# target_context = "BX"
# target_token = "ay"
# max_prob, max_distractors = search_distractors(target_context, target_token, compat)


def print_speaker_dist(dist, contexts=contexts):
    for context in contexts:
        print(context)
        print_row_dist(list(sorted(zip(dist[:,context_indices[context]].tolist(), tokens), reverse=True))[:4])
        print()

#print_speaker_dist(scoring_model.make_s0())

SEED = 1

score_distractors = True

#for training_method in ['s0', 's1', 's1_latent_hard', 's1_latent_soft']:
#for training_method in ['s1_latent_hard']:
for training_method in ['s1_latent_soft']:
#for training_method in ['s1_latent_soft', 's1_latent_hard']:
#for training_method in []:
    print("training: {}".format(training_method))
    torch.manual_seed(SEED)
    scoring_model = ScoringModel()

    opt = torch.optim.Adam(scoring_model.parameters())

    scoring_model.train()

    data = TRAINING_DATA[:]

    for epoch in range(1000):
        opt.zero_grad()
        s0 = scoring_model.make_s0()
        if score_distractors:
            distractor_scores = scoring_model.make_distractor_scores()
        else:
            distractor_scores = None

        loss = 0
        for target_context, target_token in data:
            if training_method in ['s0', 's1']:
                if training_method == 's0':
                    log_speaker_lps = s0
                elif training_method == 's1':
                    log_speaker_lps = sn(s0, n=1)
                target_context_index = context_indices[target_context]
                target_token_index = token_indices[target_token]
                loss -= log_speaker_lps[target_token_index,target_context_index]
            elif training_method in ['s1_latent_hard', 's1_latent_soft']:
                max_lp, sum_lp, _, _ = search_distractors(target_context, target_token, s0, distractor_scores)
                if training_method == 's1_latent_hard':
                    loss -= max_lp
                else:
                    loss -= sum_lp

        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            print("{}: {:.4f} [{:.4f}]".format(epoch, loss, (-loss).exp()))

    del target_token

    s0 = scoring_model.make_s0()
    if score_distractors:
        distractor_scores = scoring_model.make_distractor_scores()
    else:
        distractor_scores = None

    test_contexts = [td[0] for td in TEST_DATA]
    #test_contexts = contexts

    print("learned literal")
    print_speaker_dist(s0, test_contexts)

    print("learned pragmatic")
    print_speaker_dist(sn(s0,n=1), test_contexts)


    print("argmax latent")
    for target_context, target_token in TEST_DATA:
    #for target_context in test_contexts:
        print(target_context)
        max_prob, sum_prob, (token_distractors, context_distractors), max_dist = search_distractors(target_context, target_token, s0, distractor_scores)
        print(context_distractors)
        print_row_dist(list(sorted(zip(max_dist.tolist(), list(token_distractors)), reverse=True))[:4])
        #is_max = [dist.argmax() == 0 for dist in max_dist]
        #print(max_prob, max_distractors, is_max)
        print()
