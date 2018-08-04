#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from pprint import pprint

sys.path.append('../')

from opts.fig_1d.fully_supervised import opt as opt_full_sup
from opts.fig_1d.rel_supervised import opt as opt_rel_sup

from main_human36 import main_human as main

sup_train_data_perc = [p/100. for p in [.0625, .125, .25, 5, 25, 100]]
sup_train_last_err  = []

rel_train_data_perc = [p/100. for p in [25, 50, 100, 100, 100]]
rel_num_pairs       = [1, 1, 1, 17, 'all']
rel_train_last_err  = []

num_epochs = 25
save_ims   = True
# all the experiments are saved here
checkpoint_dir = '../checkpoint/fig_1d'

# run the experiments for 3d supervised method
for i, p in enumerate(sup_train_data_perc):
    exp_opt = opt_full_sup

    exp_opt.epochs         = num_epochs
    exp_opt.amt_train_data = p
    exp_opt.save_ims       = save_ims

    exp_opt.ckpt = checkpoint_dir
    exp_opt.exp  = 'fully_supervised_%d'%i
    exp_opt.ckpt = os.path.join(exp_opt.ckpt, exp_opt.exp)
    exp_opt.ckpt_ims = exp_opt.ckpt + '/ims'

    print("\n==================Options=================")
    pprint(vars(exp_opt), indent=4)

    if not os.path.isdir(exp_opt.ckpt): os.makedirs(exp_opt.ckpt)
    if not os.path.isdir(exp_opt.ckpt_ims): os.makedirs(exp_opt.ckpt_ims)
    print("==========================================\n")

    err_test_best, err_test_last, err_test_actions_last = main(exp_opt)
    print("Testing Errors:")
    print(" - Best:            [{}]".format(round(err_test_best, 2)))
    print(" - Last:            [{}]".format(round(err_test_last, 2)))
    print(" - Last Avg Action: [{}]".format(round(np.mean(err_test_actions_last), 2)))

    sup_train_last_err.append(err_test_last)

# run the experiments for rel supervised method
for i, (p,n) in enumerate(zip(rel_train_data_perc,rel_num_pairs)):
    exp_opt = opt_rel_sup

    exp_opt.epochs         = num_epochs
    exp_opt.amt_train_data = p
    exp_opt.num_pairs      = n
    exp_opt.save_ims       = save_ims

    exp_opt.ckpt = checkpoint_dir
    exp_opt.exp  = 'rel_supervised_%d'%i
    exp_opt.ckpt = os.path.join(exp_opt.ckpt, exp_opt.exp)
    exp_opt.ckpt_ims = exp_opt.ckpt + '/ims'

    print("\n==================Options=================")
    pprint(vars(exp_opt), indent=4)

    if not os.path.isdir(exp_opt.ckpt): os.makedirs(exp_opt.ckpt)
    if not os.path.isdir(exp_opt.ckpt_ims): os.makedirs(exp_opt.ckpt_ims)
    print("==========================================\n")

    err_test_best, err_test_last, err_test_actions_last = main(exp_opt)
    print("Testing Errors:")
    print(" - Best:            [{}]".format(round(err_test_best, 2)))
    print(" - Last:            [{}]".format(round(err_test_last, 2)))
    print(" - Last Avg Action: [{}]".format(round(np.mean(err_test_actions_last), 2)))

    rel_train_last_err.append(err_test_last)

res_dict = {'fully_supervised':sup_train_last_err,
            'rel_supervised':rel_train_last_err,
            'sup_train_data_perc':sup_train_data_perc,
            'rel_train_data_perc':rel_train_data_perc,
            'rel_num_pairs':rel_num_pairs}

with open('%s/fig_1d_results.json'%(checkpoint_dir),'w') as fp:
    json.dump(res_dict, fp)
