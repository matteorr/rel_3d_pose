#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from pprint import pprint

sys.path.append('../')

from opts.table_2.sup_17_gt_gt import opt as opt_row_1
from opts.table_2.our_17_gt_gt import opt as opt_row_2
from opts.table_2.our_17_gt_gt_n import opt as opt_row_3
from opts.table_2.sup_16_sh_sh import opt as opt_row_4
from opts.table_2.our_16_sh_sh import opt as opt_row_5
from opts.table_2.our_16_sh_sh_n import opt as opt_row_6

from main_human36 import main_human as main


opt_list = [opt_row_1, opt_row_2, opt_row_3, opt_row_4, opt_row_5, opt_row_6]

results = []

only_run_test = False  # set to True to only evaluate on trained models
num_epochs = 25
save_ims   = True
save_log   = True

# all the experiments are saved here
checkpoint_dir = '../checkpoint/table_2'

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
    result['err_test_last'] = err_test_last
    result['err_test_actions_last'] = err_test_actions_last
    result['err_test_actions_last_mean'] = np.mean(err_test_actions_last)

    results.append(result)


with open('%s/table_2_results.json'%(checkpoint_dir),'w') as fp:
    json.dump(results, fp)

print("\n")
for rr in results:
    print("\n" + rr['exp'])
    print([round(aa,1) for aa in rr['err_test_actions_last']])
    print(round(rr['err_test_actions_last_mean'],1))
