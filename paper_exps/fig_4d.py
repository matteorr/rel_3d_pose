#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from pprint import pprint

sys.path.append('../')

from opts.fig_4d.rel_supervised import opt as opt_rel_sup

from main_human36 import main_human as main

train_amount_noise = [0., .25, 'mturk', .5, .75, 1.]

results  = []

num_epochs = 25
save_ims = True
# all the experiments are saved here
checkpoint_dir = '../checkpoint/fig_4d'

# run the experiments for 3d supervised method
for i, n in enumerate(train_amount_noise):
    exp_opt = opt_rel_sup

    exp_opt.epochs   = num_epochs
    exp_opt.save_ims = save_ims

    exp_opt.rel_labels_noise_prob = n

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

    result = {}
    result['exp'] = exp_opt.exp
    result['amt_noise'] = n
    result['err_test_best'] = err_test_best
    result['err_test_last'] = err_test_last
    result['err_test_actions_last'] = err_test_actions_last
    result['err_test_actions_last_mean'] = np.mean(err_test_actions_last)

    results.append(result)

with open('%s/fig_4d_results.json'%(checkpoint_dir),'w') as fp:
    json.dump(results, fp)
