from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from PIL import Image

from torch.nn.utils.rnn import pad_sequence

import itertools
import tqdm
import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
from . import misc as utils

# load coco-caption if available
from ..data.dataloader import DataLoader
from ..models import AttModel


SPICE_THREADS=4

try:
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except Exception as e:
    print(e)
    print('Warning: coco-caption not available')

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

PAD_ID = 0

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def getCOCO(dataset):
    if 'coco' in dataset:
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'data/f30k_captions4eval.json'
    return COCO(annFile)

def prediction_filename(eval_kwargs: dict):
    id = eval_kwargs['id']
    split = eval_kwargs['split']
    pred_fn = os.path.join('eval_results/', '.saved_pred_'+ id + '_' + split + '.pth')
    return pred_fn

def save_predictions(eval_kwargs: dict, predictions, n_predictions, verbose_predictions):
    split = eval_kwargs.get('split', 'val')
    verbose_pred_fname = os.path.join('eval_results/', f"pred_verbose_{eval_kwargs['id']}_{split}.pth")
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    pred_fn = prediction_filename(eval_kwargs)
    print(f'saving to {pred_fn}')
    torch.save((predictions, n_predictions), pred_fn)

    if eval_kwargs.get('save_verbose_predictions', 0):
        print(f"saving verbose predictions to {verbose_pred_fname}")
        torch.save(verbose_predictions, verbose_pred_fname)

def language_eval(dataset, preds, preds_n, eval_kwargs, split):
    model_id = eval_kwargs['id']
    eval_oracle = eval_kwargs.get('eval_oracle', 0)
    
    # create output dictionary
    out = {}

    if len(preds_n) > 0:
        # vocab size and novel sentences
        if 'coco' in dataset:
            dataset_file = 'data/dataset_coco.json'
        elif 'flickr30k' in dataset or 'f30k' in dataset:
            dataset_file = 'data/dataset_flickr30k.json'
        training_sentences = set([' '.join(__['tokens']) for _ in json.load(open(dataset_file))['images'] if not _['split'] in ['val', 'test'] for __ in _['sentences']])
        generated_sentences = set([_['caption'] for _ in preds_n])
        novels = generated_sentences - training_sentences
        out['novel_sentences'] = float(len(novels)) / len(preds_n)
        tmp = [_.split() for _ in generated_sentences]
        words = []
        for _ in tmp:
            words += _
        out['vocab_size'] = len(set(words))

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')

    coco = getCOCO(dataset)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]
    mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
    mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes, spice_threads=SPICE_THREADS, 
                           scorers_to_run=['bleu', 'meteor', 'rouge', 'cider', 'wmd'])
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy

    imgToEval = cocoEval.imgToEval
    # for k in list(imgToEval.values())[0]['SPICE'].keys():
    #     if k != 'All':
    #         out['SPICE_'+k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
    #         out['SPICE_'+k] = (out['SPICE_'+k][out['SPICE_'+k]==out['SPICE_'+k]]).mean()
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    # if len(preds_n) > 0:
    #     from . import eval_multi
    #     cache_path_n = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '_n.json')
    #     allspice = eval_multi.eval_allspice(dataset, preds_n, model_id, split)
    #     out.update(allspice['overall'])
    #     div_stats = eval_multi.eval_div_stats(dataset, preds_n, model_id, split)
    #     out.update(div_stats['overall'])
    #     if eval_oracle:
    #         oracle = eval_multi.eval_oracle(dataset, preds_n, model_id, split)
    #         out.update(oracle['overall'])
    #     else:
    #         oracle = None
    #     self_cider = eval_multi.eval_self_cider(dataset, preds_n, model_id, split)
    #     out.update(self_cider['overall'])
    #     with open(cache_path_n, 'w') as outfile:
    #         json.dump({'allspice': allspice, 'div_stats': div_stats, 'oracle': oracle, 'self_cider': self_cider}, outfile)
        
    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def generate_pragmatic(model: AttModel, loader: DataLoader, fc_feats, att_feats, att_masks, data, eval_kwargs):
    # generate candidate utterances
    keep_all_scores = eval_kwargs.get('pragmatic_serialize_all_scores', 0)
    input_data = fc_feats, att_feats, att_masks, data
    n_imgs = fc_feats.size(0)
    n_predictions, seqs, log_probs = generate_caption_candidates(
        model, input_data, eval_kwargs, loader=loader,
    )
    # seqs: n_images x n_captions x T
    # log_probs: n_images x n_captions
    n_imgs_, n_captions = log_probs.size()
    assert n_imgs == n_imgs_, (n_imgs, n_imgs_)
    s0_weight = eval_kwargs['pragmatic_s0_weight']
    if not (0.0 <= s0_weight <= 1.0):
        raise ValueError(f"s0_weight {s0_weight} not in [0, 1]")
    all_s0_scores_s = []
    all_s1_scores_s = []
    all_s0s1_scores_s = []
    target_s0_scores_s = []
    target_s1_scores_s = []
    target_s0s1_scores_s = []
    best_seqs = []
    best_scores = []
    device = fc_feats.device
    image_context_paths_s = []
    image_context_ids_s = []
    candidates_s = []
    k_neighbors = eval_kwargs['pragmatic_distractors']
    nearest_neighbor_index = loader.indices[eval_kwargs['pragmatic_distractor_split']]
    all_neighbors = nearest_neighbor_index.get_neighbor_batch(
        loader, fc_feats.cpu().numpy(), k_neighbors=k_neighbors,
        include_self=True, self_indices=[data['infos'][img_ix]['ix'] for img_ix in range(n_imgs)],
        neighbor_type=eval_kwargs['pragmatic_distractor_candidate_type']
    )
    for img_ix in range(n_imgs):
        neighbor_infos = all_neighbors['infos'][img_ix*(k_neighbors+1):(img_ix+1)*(k_neighbors+1)]
        assert len(neighbor_infos) == k_neighbors+1
        # n_captions x (1 + pragmatic_distractors)
        # [:,0]: scores for the captions for the target image
        s0_scores = model.cross_product_scores(
            all_neighbors['fc_feats'][img_ix].to(device),
            all_neighbors['att_feats'][img_ix].to(device),
            all_neighbors['att_masks'][img_ix].to(device) if all_neighbors['att_masks'] is not None else None,
            seqs[img_ix]
        )
        candidates_s.append(utils.decode_sequence(model.vocab, seqs[img_ix]))
        # n_captions
        l1_scores = s0_scores.log_softmax(dim=1)
        s1_scores = l1_scores.log_softmax(dim=0)
        s0s1_scores = s0_scores * s0_weight + s1_scores * (1.0 - s0_weight)

        target_s0s1_scores = s0s1_scores[:,0]

        image_context_paths_s.append([d['file_path'] for d in neighbor_infos])
        image_context_ids_s.append([d['id'] for d in neighbor_infos])

        all_s0_scores_s.append(s0_scores.detach().cpu().numpy())
        all_s1_scores_s.append(s1_scores.detach().cpu().numpy())
        all_s0s1_scores_s.append(s0s1_scores.detach().cpu().numpy())
        target_s0_scores_s.append(s0_scores[:,0].detach().cpu().numpy())
        target_s1_scores_s.append(s1_scores[:,0].detach().cpu().numpy())
        target_s0s1_scores_s.append(target_s0s1_scores.detach().cpu().numpy())

        best_score, best_ix = target_s0s1_scores.max(-1)
        best_scores.append(best_score)
        best_seqs.append(seqs[img_ix][best_ix])
    seq = pad_sequence(best_seqs, batch_first=True, padding_value=PAD_ID)
    scores = torch.stack(best_scores, -1)
    entropy = torch.zeros_like(scores)
    perplexity = torch.zeros_like(scores)
    extras = {
        'target_s0_scores': target_s0_scores_s,
        'target_s1_scores': target_s1_scores_s,
        'target_s0s1_scores': target_s0s1_scores_s,
        'chosen_target_s0s1_scores': scores.detach().cpu().numpy(),
        'candidates': candidates_s,
        'context_paths': image_context_paths_s,
        'context_ids': image_context_ids_s,
    }
    if keep_all_scores:
        extras.update({
            'all_s0_scores': all_s0_scores_s,
            'all_s1_scores': all_s1_scores_s,
            'all_s0s1_scores': all_s0s1_scores_s,
        })
    return seq, entropy, perplexity, extras

def search_distractors(s0_cap_by_img_score_mat, num_distractors_to_choose, s0_weight):
    if not (0.0 <= s0_weight <= 1.0):
        raise ValueError(f"s0_weight {s0_weight} must be in [0, 1]")
    n_cap, n_img = s0_cap_by_img_score_mat.size()
    best_distractors = None
    best_score = None
    best_cap = None
    for distractors in itertools.combinations(range(1, n_img), num_distractors_to_choose):
        img_indices = list(itertools.chain((0,), distractors))
        sub_mat = s0_cap_by_img_score_mat[:,img_indices]
        l1 = sub_mat.log_softmax(1)
        s1 = l1.log_softmax(0)
        target_s0_scores = s0_cap_by_img_score_mat[:,0]
        target_s1_scores = s1[:,0]
        target_s0s1_scores = s0_weight * target_s0_scores + (1 - s0_weight) * target_s1_scores
        this_best_score, this_best_cap = target_s0s1_scores.max(-1)
        if best_score is None or this_best_score > best_score:
            best_distractors = distractors
            best_score = this_best_score
            best_cap = this_best_cap
    return best_cap, best_distractors, best_score

def pragmatic_choose_from_candidates(instance_candidates: dict, eval_kwargs):
    device = eval_kwargs.get('device', 'cuda')
    # instance_candidates: candidate captions and scores for a single image
    prediction = {}
    for key in ['image_id', 'candidates', 'perplexity', 'entropy', 'context_paths', 'context_ids']:
        prediction[key] = instance_candidates[key]
    candidate_captions = instance_candidates['candidates']
    nonempty_indices, nonempty_captions = zip(*[(ix, cap) for ix, cap in enumerate(candidate_captions) if cap])
    nonempty_indices = torch.tensor(nonempty_indices).long()
    if eval_kwargs['pragmatic_inference']:
        assert not eval_kwargs['mbr_inference']
        num_distractors = eval_kwargs['pragmatic_distractors']
        s0_weight = eval_kwargs['pragmatic_s0_weight']
        # n_captions x n_images

        # target image has index 0; indices 1-end are for distractor images
        s0_scores = torch.tensor(instance_candidates['all_s0_scores'])
        if num_distractors >= s0_scores.size(1):
            raise ValueError(f"not enough distractors in serialized candidates. {num_distractors} required; {s0_scores.size(1) - 1} available")
        else:
            s0_scores = s0_scores[:,:num_distractors+1]
        s0_scores = s0_scores[nonempty_indices]
        s0_scores = s0_scores.to(device)
        distractor_type = eval_kwargs.get('pragmatic_distractor_type', 'closest')
        if distractor_type == 'closest':
            # use all distractors
            num_to_choose = num_distractors
        elif distractor_type == 'choose_within_closest':
            num_to_choose = eval_kwargs.get('pragmatic_distractors_to_choose', 1)
        else:
            raise NotImplementedError(f"invalid --pragmatic_distractor_type {distractor_type}")
        best_cap, best_distractors, best_scores = search_distractors(s0_scores, num_to_choose, s0_weight)

        caption = nonempty_captions[best_cap]
    else:
        raise NotImplementedError()
    prediction['caption'] = caption
    # TODO: copy scores to this
    verbose_prediction = prediction
    return prediction, verbose_prediction


def clip_choose_from_candidates(instance_candidates: dict, eval_kwargs, clip_model, clip_transform):
    from clip.clip import tokenize
    device = eval_kwargs.get('device', 'cuda')
    s0_weight = eval_kwargs['clip_s0_weight']
    save_verbose_predictions = eval_kwargs.get('save_verbose_predictions', 0)
    if not (0.0 <= s0_weight <= 1.0):
        raise ValueError(f"--clip_s0_weight {s0_weight} must be in [0, 1]")

    prediction = {}
    for key in ['image_id', 'candidates', 'perplexity', 'entropy', 'context_paths', 'context_ids']:
        prediction[key] = instance_candidates[key]

    target_path = instance_candidates['context_paths'][0]
    candidate_captions = instance_candidates['candidates']
    nonempty_indices, nonempty_captions = zip(*[(ix, cap) for ix, cap in enumerate(candidate_captions) if cap])
    nonempty_captions = list(nonempty_captions)
    nonempty_indices = torch.tensor(nonempty_indices).long()

    s0_scores_full = torch.tensor(instance_candidates['target_s0_scores'])
    s0_scores = s0_scores_full[nonempty_indices]

    image = clip_transform(Image.open(target_path)).unsqueeze(0).to(device)
    text = tokenize(nonempty_captions).to(device)

    with torch.no_grad():
        # TODO: why do these differ?
        logits_per_image, logits_per_text = clip_model(image, text)
        clip_log_probs = logits_per_image.log_softmax(dim=-1)

    clip_log_probs = clip_log_probs.to(s0_scores.device)
    joint_scores = s0_weight * s0_scores + (1 - s0_weight) * clip_log_probs
    best_score, best_cap = joint_scores.max(-1)
    caption = nonempty_captions[best_cap]
    prediction['caption'] = caption

    if save_verbose_predictions:
        clip_log_probs_full = torch.zeros_like(s0_scores_full)
        clip_log_probs_full[nonempty_indices] = clip_log_probs.float()

        joint_scores_full = torch.zeros_like(s0_scores_full)
        joint_scores_full[nonempty_indices] = joint_scores

        verbose_prediction = prediction.copy()

        verbose_prediction['s0_scores'] = instance_candidates['target_s0_scores']
        verbose_prediction['clip_scores'] = clip_log_probs_full.detach().cpu().numpy()
        verbose_prediction['joint_scores'] = joint_scores_full.detach().cpu().numpy()
        verbose_prediction['distinct_candidates'] = len(set(candidate_captions))
    else:
        verbose_prediction = None
    return prediction, verbose_prediction

def mbr_choose_from_candidates(instance_candidates: dict, eval_kwargs, sent_rep_model, sent_rep_tokenizer):
    device = eval_kwargs.get('device', 'cuda')
    mbr_type = eval_kwargs.get('mbr_type', 'bert_cosine_sim')
    if mbr_type != 'bert_cosine_sim':
        raise NotImplementedError(f"--mbr_type: {mbr_type}")
    s0_weight = eval_kwargs['mbr_s0_weight']
    if not (0.0 <= s0_weight <= 1.0):
        raise ValueError(f"--mbr_s0_weight {s0_weight} must be in [0, 1]")

    prediction = {}
    for key in ['image_id', 'candidates', 'perplexity', 'entropy', 'context_paths', 'context_ids']:
        prediction[key] = instance_candidates[key]
    candidate_captions = instance_candidates['candidates']
    nonempty_indices, nonempty_captions = zip(*[(ix, cap) for ix, cap in enumerate(candidate_captions) if cap])
    nonempty_captions = list(nonempty_captions)
    nonempty_indices = torch.tensor(nonempty_indices).long()

    s0_scores = torch.tensor(instance_candidates['target_s0_scores'])
    s0_scores = s0_scores[nonempty_indices]
    inputs = sent_rep_tokenizer(nonempty_captions, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = sent_rep_model(**inputs)
    h = outputs['last_hidden_state'][:,0]
    h_rescaled = h / h.norm(2, dim=-1, keepdim=True)
    cosine_sim = torch.einsum("xh,yh->xy", (h_rescaled, h_rescaled))
    mbr_scores = cosine_sim.mean(-1).log_softmax(-1)
    mbr_scores = mbr_scores.to(s0_scores.device)
    joint_scores = s0_weight * s0_scores + (1 - s0_weight) * mbr_scores
    best_score, best_cap = joint_scores.max(-1)
    caption = nonempty_captions[best_cap]
    prediction['caption'] = caption
    # TODO: add scores to this
    verbose_prediction = prediction
    return prediction, verbose_prediction


def eval_split_from_serialized(path, eval_kwargs={}):
    device = eval_kwargs.get('device', 'cuda')
    pragmatic_inference = eval_kwargs.get('pragmatic_inference', 0)
    mbr_inference = eval_kwargs.get('mbr_inference', 0)
    clip_inference = eval_kwargs.get('clip_inference', 0)

    if sum([mbr_inference, pragmatic_inference, clip_inference]) > 1:
        raise ValueError("can't do multiple of --pragmatic_inference, --mbr_inference, --clip_inference: {}".format(
            [mbr_inference, pragmatic_inference, clip_inference]
        ))

    if mbr_inference:
        from transformers import AutoTokenizer, AutoModel
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        bert_model = AutoModel.from_pretrained("bert-large-uncased")
        bert_model = bert_model.to(device)

    if clip_inference:
        from clip import clip
        clip_model, clip_transform = clip.load("ViT-B/32", device=device)

    candidates = torch.load(path)
    predictions = []
    verbose_predictions = []

    num_images = eval_kwargs.get('num_images', None)
    if num_images is not None:
        if num_images > len(candidates):
            raise ValueError(f"--num_images={num_images} but only {len(candidates)} images found in serialized file {path}")
        candidates = candidates[:num_images]

    for instance_candidates in tqdm.tqdm(candidates, ncols=80):
        if mbr_inference:
            prediction, verbose_prediction = mbr_choose_from_candidates(instance_candidates, eval_kwargs, bert_model, bert_tokenizer)
        elif clip_inference:
            prediction, verbose_prediction = clip_choose_from_candidates(
                instance_candidates, eval_kwargs, clip_model, clip_transform,
            )
        else:
            prediction, verbose_prediction = pragmatic_choose_from_candidates(instance_candidates, eval_kwargs)
        predictions.append(prediction)
        verbose_predictions.append(verbose_prediction)

    return predictions, verbose_predictions

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_captions = eval_kwargs.get('verbose_captions', 0)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration
    device = eval_kwargs.get('device', 'cuda')

    pragmatic_inference = eval_kwargs.get('pragmatic_inference', 0)
    contrastive = eval_kwargs['sample_method'] == 'contrastive_beam_search'

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = [] # when sample_n > 1
    verbose_predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        if labels is not None and verbose_loss:
            # forward the model to get loss
            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        with torch.no_grad():
            tmp_eval_kwargs = eval_kwargs.copy()
            if pragmatic_inference:
                seq, entropy, perplexity, extras = generate_pragmatic(model, loader, fc_feats, att_feats, att_masks, data, tmp_eval_kwargs)
                seq = seq.data
            else:
                tmp_eval_kwargs.update({'sample_n': 1})
                # forward the model to also get generated samples for each image
                seq, seq_logprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample',
                                          loader=loader, data=data)
                seq = seq.data
                extras = {}
                entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq>0).to(seq_logprobs).sum(1)+1)
                perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq>0).to(seq_logprobs).sum(1)+1)
        
        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(fc_feats.shape[0]):
                print('\n'.join([utils.decode_sequence(model.vocab, _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(model.vocab, seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            verbose_entry = entry.copy()
            if contrastive:
                # DataParallel wrapper doesn't make attrs accessible
                if isinstance(model, torch.nn.DataParallel):
                    underlying_model = model.module
                else:
                    underlying_model = model
                neighbor_infos = underlying_model.neighbor_infos[k]
                verbose_entry['context_paths'] = [d['file_path'] for d in neighbor_infos]
                verbose_entry['context_ids'] = [d['id'] for d in neighbor_infos]
            if extras:
                for key, value in extras.items():
                    assert len(value) == len(sents)
                    verbose_entry[key] = value[k]
            predictions.append(entry)
            verbose_predictions.append(verbose_entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose_captions:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        if sample_n > 1:
            eval_split_n(model, n_predictions, [fc_feats, att_feats, att_masks, data], eval_kwargs, loader=loader)
        
        # ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation performance... %d/%d (%f)' %(n, ix1, loss))

        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
        n_predictions = sorted(n_predictions, key=lambda x: x['perplexity'])

    save_predictions(eval_kwargs, predictions, n_predictions, verbose_predictions)
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, n_predictions, eval_kwargs, split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


def eval_split_n(model, n_predictions, input_data, eval_kwargs={}, loader=None):
    new_predictions, seqs, log_probs = generate_caption_candidates(model, input_data, eval_kwargs, loader=loader)
    n_predictions.extend(new_predictions)

# Only run when sample_n > 0
def generate_caption_candidates(model, input_data, eval_kwargs={}, loader=None):
    n_predictions = []
    verbose_captions = eval_kwargs.get('verbose_captions', 0)
    # beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')

    fc_feats, att_feats, att_masks, data = input_data

    n_imgs = fc_feats.size(0)

    tmp_eval_kwargs = eval_kwargs.copy()
    if sample_n_method in ['bs', 'contrastive_bs', 'gumbel_bs']:
        # case 1 sample_n == beam size
        contrastive = sample_n_method == 'contrastive_bs'
        if contrastive:
            tmp_eval_kwargs['sample_method'] = 'contrastive_beam_search'
        if sample_n_method == 'gumbel_bs':
            tmp_eval_kwargs['sample_method'] = 'gumbel_beam_search'
        tmp_eval_kwargs.update({'sample_n': 1, 'beam_size': sample_n, 'group_size': 1}) # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample', data=data, loader=loader)
        seqs = []
        log_probs = []
        for k in range(fc_feats.shape[0]):
            beams = [model.done_beams[k][_]['seq'] for _ in range(sample_n)]
            stacked_beams = pad_sequence(beams, batch_first=True, padding_value=0)
            seqs.extend(beams)
            _log_prob = torch.stack([model.done_beams[k][i]['unaug_log_prob'] for i in range(sample_n)]).flatten()
            log_probs.append(_log_prob)
            _sents = utils.decode_sequence(model.vocab, stacked_beams)
            for sent_ix, sent in enumerate(_sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'log_prob': _log_prob[sent_ix].item()}
                if contrastive:
                    neighbor_infos = model.neighbor_infos[sent_ix]
                    entry['context_paths'] = [d['file_path'] for d in neighbor_infos]
                    entry['context_ids'] = [d['id'] for d in neighbor_infos]
                n_predictions.append(entry)
        seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
        log_probs = torch.cat(log_probs, 0)
    # case 2 sample / gumbel / topk sampling/ nucleus sampling
    elif sample_n_method == 'sample' or \
            sample_n_method == 'gumbel' or \
            sample_n_method.startswith('top'):
        tmp_eval_kwargs.update({'sample_n': sample_n, 'sample_method': sample_n_method, 'beam_size': 1}) # randomness from sample
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample',
                                          loader=loader, data=data)
        _sents = utils.decode_sequence(model.vocab, _seq)
        _log_prob = _sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(1)
        _perplexity = - _log_prob / ((_seq>0).to(_sampleLogprobs).sum(1)+1)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'seq': _seq[k], 'caption': sent, 'perplexity': _perplexity[k].item(), 'log_prob': _log_prob[k].item()}
            n_predictions.append(entry)
        seqs = _seq
        log_probs = _log_prob
    elif sample_n_method == 'dbs':
        # Use diverse beam search
        raise NotImplementedError("set seqs to be the returned candidates (a batch_size*sample_n x T array) and log_probs to bbe log probabilities (batch_size*sample_n)")
        tmp_eval_kwargs.update({'beam_size': sample_n * beam_size, 'group_size': sample_n}) # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample', loader=loader, data=data)
        for k in range(loader.batch_size):
            _sents = utils.decode_sequence(model.vocab, torch.stack([model.done_beams[k][_]['seq'] for _ in range(0, sample_n*beam_size, beam_size)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    elif sample_n_method in ['dgreedy', 'dsample', 'dtopk', 'dtopp']:
        raise NotImplementedError("set log_probs to bbe log probabilities (batch_size*sample_n)")
        tmp_eval_kwargs.update({'sample_method': sample_n_method[1:], 'group_size': sample_n, 'beam_size':1}) # randomness from softmax
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample',
                                          loader=loader, data=data)
        _sents = utils.decode_sequence(model.vocab, _seq)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
            n_predictions.append(entry)
        seqs = _seq
    else:
        raise ValueError(f"invalid sample_n_method {sample_n_method}")
    if verbose_captions:
        for entry in sorted(n_predictions[-fc_feats.shape[0] * sample_n:], key=lambda x: x['image_id']):
            print('image %s: %s' %(entry['image_id'], entry['caption']))
    seqs = einops.rearrange(seqs, "(n_imgs n_caps) T -> n_imgs n_caps T", n_imgs=n_imgs, n_caps=sample_n)
    log_probs = einops.rearrange(log_probs, "(n_imgs n_caps) -> n_imgs n_caps", n_imgs=n_imgs, n_caps=sample_n)
    return n_predictions, seqs, log_probs