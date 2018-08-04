#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from pprint import pprint

sys.path.append('../')

from main_human36 import main_human as main

# ordering of table 1 in the camera ready paper
# 1) 3d supervised - opt_row_sup
from opts.table_1.row_sup import opt as opt_row_sup
# 2) Ours - opt_row_ours
from opts.table_1.row_ours import opt as opt_row_ours
# 3) Known focal length (previously called weak projective)
from opts.table_1.row_3 import opt as opt_row_3
# 4) Skeleton from [41]
from opts.table_1.row_4 import opt as opt_row_4
# 5) No skeleton loss
from opts.table_1.row_5 import opt as opt_row_5
# 6) All pairs
from opts.table_1.row_6 import opt as opt_row_6
# 7) Distance tolerance
from opts.table_1.row_7 import opt as opt_row_7

opt_list = [opt_row_sup, opt_row_ours, opt_row_3, opt_row_4, opt_row_5, opt_row_6, opt_row_7]

results = []

num_epochs = 25
save_ims   = True
save_log = True
only_run_test = False

# all the experiments are saved here
checkpoint_dir = '../checkpoint/table_1'

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

    print("\n==================Options=================")
    pprint(vars(exp_opt), indent=4)
    if not os.path.isdir(exp_opt.ckpt): os.makedirs(exp_opt.ckpt)
    if not os.path.isdir(exp_opt.ckpt_ims): os.makedirs(exp_opt.ckpt_ims)
    print("==========================================\n")

    err_test_best, err_test_last, err_test_actions_last = main(exp_opt, save_log)
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

with open('%s/table_1_results.json'%(checkpoint_dir),'w') as fp:
    json.dump(results, fp)
