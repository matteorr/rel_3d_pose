#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from pprint import pprint

sys.path.append('../')

from opts.fig_4ab.rel_supervised import opt as opt_row_ours
from opts.fig_4ab.fully_supervised import opt as opt_row_sup

from main_human36 import main_human as main

opt_list = [opt_row_ours, opt_row_sup]

results = []

num_epochs = 25
save_ims   = True
# all the experiments are saved here - for this expeirment we can also load the results from table 1
checkpoint_dir = '../checkpoint/fig_4ab'

only_run_test = False
return_proc_all = True
save_log = True

for i, exp_opt in enumerate(opt_list):

    exp_opt.epochs   = num_epochs
    exp_opt.save_ims = save_ims

    exp_opt.ckpt = checkpoint_dir
    exp_opt.ckpt = os.path.join(exp_opt.ckpt, exp_opt.exp)
    exp_opt.ckpt_ims = exp_opt.ckpt + '/ims'

    if only_run_test:
        exp_opt.load = exp_opt.ckpt + '/test_ckpt_last.pth.tar'
        exp_opt.resume = True
        exp_opt.is_train = False
        exp_opt.epochs = num_epochs+1
        save_log = False
        save_ims = False

    print("\n==================Options=================")
    pprint(vars(exp_opt), indent=4)
    if not os.path.isdir(exp_opt.ckpt): os.makedirs(exp_opt.ckpt)
    if not os.path.isdir(exp_opt.ckpt_ims): os.makedirs(exp_opt.ckpt_ims)
    print("==========================================\n")

    err_test_best, err_test_last, err_test_actions_last, all_dist_proc, all_poses_proc, data_dict_test \
        = main(exp_opt, save_log, return_proc_all)
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

    # save the errors
    np.save(checkpoint_dir + '/' + exp_opt.exp + '/all_dist_proc', all_dist_proc)
    # save the poses
    np.save(checkpoint_dir + '/' + exp_opt.exp + '/all_poses_proc', all_poses_proc)
    # save the test data dict info
    np.savez(checkpoint_dir + '/' + exp_opt.exp + '/subjects_info', data_dict_test)


with open('%s/fig_4ab_results.json'%(checkpoint_dir),'w') as fp:
    json.dump(results, fp)
