#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from pprint import pprint

sys.path.append('../')

from opts.fig_4c.fully_supervised import opt as opt_full_sup
from opts.fig_4c.rel_supervised import opt as opt_rel_sup

from main_human36 import main_human as main

test_amount_noise = [0, 5, 10, 15, 20, 'detections']

sup_train_last_err  = []
rel_train_last_err  = []

num_epochs = 25
save_ims = True
# all the experiments are saved here
checkpoint_dir = '../checkpoint/fig_4c_test_noise'

# run the experiments for 3d supervised method
for i, n in enumerate(test_amount_noise):
    exp_opt = opt_full_sup

    exp_opt.epochs   = num_epochs
    exp_opt.save_ims = save_ims

    if n != 'detections':
        exp_opt.test_noise_type_2d   = 'normal'
        exp_opt.test_noise_amount_2d = n
    else:
        exp_opt.dataset_type         = 'shg_16k'
        exp_opt.test_noise_type_2d   = n
        exp_opt.test_noise_amount_2d = 0

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

for i, n in enumerate(test_amount_noise):
    exp_opt = opt_rel_sup

    exp_opt.epochs   = num_epochs
    exp_opt.save_ims = save_ims

    if n != 'detections':
        exp_opt.test_noise_type_2d = 'normal'
        exp_opt.test_noise_amount_2d   = n
    else:
        exp_opt.dataset_type       = 'shg_16k'
        exp_opt.test_noise_type_2d = n
        exp_opt.test_noise_amount_2d   = 0

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
            'test_amount_noise':test_amount_noise}

with open('%s/fig_4c_test_noise_results.json'%(checkpoint_dir),'w') as fp:
    json.dump(res_dict, fp)
