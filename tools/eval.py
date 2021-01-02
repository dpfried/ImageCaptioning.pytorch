from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
import sys
from six.moves import cPickle

import pprint

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
from captioning.data.dataloaderraw import *
import captioning.utils.eval_utils as eval_utils
import argparse
import captioning.utils.misc as utils
import captioning.modules.losses as losses
import torch

if __name__ == "__main__":
    # Input arguments and options
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--model', type=str, default='',
                    help='path to model to evaluate')
    parser.add_argument('--cnn_model', type=str,  default='resnet101',
                    help='resnet101, resnet152')
    parser.add_argument('--infos_path', type=str, default='',
                    help='path to infos to evaluate')
    parser.add_argument('--only_lang_eval', type=int, default=0,
                    help='lang eval on saved results')
    parser.add_argument('--force', type=int, default=0,
                    help='force to evaluate no matter if there are results available')
    parser.add_argument('--device', type=str, default='cuda',
                    help='cpu or cuda')
    parser.add_argument('--from_serialized_candidates')
    parser.add_argument('--save_verbose_predictions', type=int, default=0, help='write predictions to eval_results/pred_verbose_{id}_{split}.pth')
    opts.add_loader_options(parser)
    opts.add_eval_options(parser)
    opts.add_diversity_opts(parser)
    opts.add_pragmatics_opts(parser)
    opts.add_mbr_opts(parser)
    opt = parser.parse_args()

    print(' '.join(sys.argv))
    utils.dump_git_status()
    pprint.pprint(vars(opt))

    # Load infos
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)

    # override and collect parameters
    replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
    ignore = ['start_from']

    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if not k in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

    vocab = infos['vocab'] # ix -> word mapping

    pred_fn = os.path.join('eval_results/', '.saved_pred_'+ opt.id + '_' + opt.split + '.pth')
    result_fn = os.path.join('eval_results/', opt.id + '_' + opt.split + '.json')

    def print_lang_stats(lang_stats):
        stat_keys = ['Bleu_1', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
        if lang_stats:
            pprint.pprint(lang_stats)
            print(','.join(stat_keys))
            print(','.join('{:.4f}'.format(lang_stats[key]) if key in lang_stats else '----'
                           for key in stat_keys ))

    def vis_predictions(split_predictions):
        if opt.dump_json == 1:
            # dump the json
            json.dump(split_predictions, open('vis/vis.json', 'w'))

    if opt.from_serialized_candidates or opt.only_lang_eval == 1 or (not opt.force and os.path.isfile(pred_fn)):
        # if results existed, then skip, unless force is on
        if opt.from_serialized_candidates:
            predictions = eval_utils.eval_split_from_serialized(
                opt.from_serialized_candidates, vars(opt)
            )
            n_predictions = []
            print(f'saving to {pred_fn}')
            torch.save((predictions, n_predictions), pred_fn)
        else:
            if not opt.force:
                try:
                    if os.path.isfile(result_fn):
                        print(result_fn)
                        json.load(open(result_fn, 'r'))
                        print('already evaluated')
                        os._exit(0)
                except:
                    pass

            predictions, n_predictions = torch.load(pred_fn)
        lang_stats = eval_utils.language_eval(opt.input_json, predictions, n_predictions, vars(opt), opt.split)
        print_lang_stats(lang_stats)
        vis_predictions(predictions)
        os._exit(0)

    # At this point only_lang_eval if 0
    if not opt.force:
        # Check out if
        try:
            # if no pred exists, then continue
            tmp = torch.load(pred_fn)
            # if language_eval == 1, and no pred exists, then continue
            if opt.language_eval == 1:
                json.load(open(result_fn, 'r'))
            print('Result is already there')
            os._exit(0)
        except:
            pass

    # Setup the model
    opt.vocab = vocab
    model = models.setup(opt)
    del opt.vocab
    model.load_state_dict(torch.load(opt.model, map_location='cpu'))
    model.to(opt.device)
    model.eval()
    crit = losses.LanguageModelCriterion()

    # Create the Data Loader instance
    if len(opt.image_folder) == 0:
        if opt.pragmatic_inference:
            loader = DataLoader(opt,
                                build_nearest_neighbor_indices_for_splits=['train'],
                                index_serialization_root_path=opt.index_serialization_root_path)
        else:
            loader = DataLoader(opt)
    else:
        assert not opt.pragmatic_inference
        loader = DataLoaderRaw({'folder_path': opt.image_folder,
                                'coco_json': opt.coco_json,
                                'batch_size': opt.batch_size,
                                'cnn_model': opt.cnn_model})
    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    loader.dataset.ix_to_word = infos['vocab']


    # Set sample options
    opt.dataset = opt.input_json
    loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
            vars(opt))

    print('loss: ', loss)
    print_lang_stats(lang_stats)
    vis_predictions(split_predictions)

