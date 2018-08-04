#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from pprint import pprint

sys.path.append('../')

from opts.fig_5.fully_supervised_human36_14k import opt as sup_pretrain_h36
from opts.fig_5.rel_supervised_human36_14k import opt as rel_pretrain_h36
from opts.fig_5.fully_supervised_lsp_14k import opt as sup_test_lsp
from opts.fig_5.rel_supervised_lsp_14k import opt as rel_test_lsp
from opts.fig_5.rel_supervised_human36_lsp_14k import opt as rel_finetune_lsp
from opts.fig_5.rel_supervised_train_lsp_14_k import opt as rel_train_test_lsp

opt_files_h36 = [sup_pretrain_h36, rel_pretrain_h36]
opt_files_lsp = [sup_test_lsp, rel_test_lsp,
                 rel_finetune_lsp, rel_train_test_lsp]

from main_human36 import main_human
from main_lsp import main_lsp

results  = []

# all the experiments are saved here
checkpoint_dir = '../checkpoint/fig_5'

# run the pretraining on human36 dataset
for i, exp_opt in enumerate(opt_files_h36):

    print("\n==================Options=================")
    pprint(vars(exp_opt), indent=4)

    if not os.path.isdir(exp_opt.ckpt): os.makedirs(exp_opt.ckpt)
    if not os.path.isdir(exp_opt.ckpt_ims): os.makedirs(exp_opt.ckpt_ims)
    print("==========================================\n")

    err_test_best, err_test_last, err_test_actions_last = main_human(exp_opt)
    print("Testing Errors:")
    print(" - Best:            [{}]".format(round(err_test_best, 2)))
    print(" - Last:            [{}]".format(round(err_test_last, 2)))
    print(" - Last Avg Action: [{}]".format(round(np.mean(err_test_actions_last), 2)))

    result = {}
    result['exp'] = exp_opt.exp
    result['err_test_best'] = err_test_best
    result['err_test_last'] = err_test_last
    result['err_test_actions_last'] = err_test_actions_last
    result['err_test_actions_last_mean'] = np.mean(err_test_actions_last)

    results.append(result)

# run the finetuning and test on lsp dataset
for i, exp_opt in enumerate(opt_files_lsp):

    save_log = True
    return_poses = True

    print("\n==================Options=================")
    pprint(vars(exp_opt), indent=4)

    if not os.path.isdir(exp_opt.ckpt): os.makedirs(exp_opt.ckpt)
    if not os.path.isdir(exp_opt.ckpt_ims): os.makedirs(exp_opt.ckpt_ims)
    print("==========================================\n")

    err_test_best, err_test_last, all_poses = main_lsp(opt, save_log, return_poses)
    print("Testing Errors:")
    print(" - Best: [{}]".format(err_test_best))
    print(" - Last: [{}]".format(err_test_last))

    np.save(exp_opt.ckpt + "/all_poses.npy", all_poses)

    result = {}
    result['exp'] = exp_opt.exp
    result['err_test_best'] = err_test_best
    result['err_test_last'] = err_test_last

    results.append(result)

with open('%s/fig_5_results.json'%(checkpoint_dir),'w') as fp:
    json.dump(results, fp)
