"""
Main script for creating plots for Human3.6 MTurk user study.

data should be stored in ../mturk_data/FILENAME.pkl

You will need the crowdsourcing code from here to run this:
https://github.com/gvanhorn38/crowdsourcing
"""

import pickle
import numpy as np
import sys, os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sns.set(style="whitegrid")

sys.path.append('./user_study')
import combine_worker_preds as cmb
import load_gt_for_mturk as mturk_gt
import user_study_utils as utils


def compute_gt(combined_labels, coords, gt_set):
    gt = {}
    for im_id in combined_labels['images']:
        im = combined_labels['images'][im_id]
        depths = coords[int(im['raw_id'])][:, 2]
        d0 = depths[im['kp0']]
        d1 = depths[im['kp1']]
        label = int(d1 < d0)
        dis = np.abs(d1 - d0)
        gt[im['id']] = {'label': label, 'dis': dis, 'kp0':im['kp0'], 'kp1':im['kp1']}
    print gt_set, np.array([gg['label'] for gg in gt.itervalues()]).mean(), '% pos GT out of', len(gt)
    return gt


def plot_image(im_id, gs_id):
    print ii, per_im_overall_correct[im_id], un_images[im_id], im_paths[int(un_images[im_id])]
    c_ind = int(un_images[im_id])
    im = plt.imread(im_paths[c_ind])
    ax = plt.subplot(gs_id, aspect='equal')
    ax.imshow(im, aspect='auto')
    for jj, gg in enumerate(im_ids_kps[im_id]):
        print gg, keypoints[gt[gg]['kp0']], keypoints[gt[gg]['kp1']]
        #plt.plot(coords_2d[c_ind][gt[gg]['kp0'],0], coords_2d[c_ind][gt[gg]['kp1'],1], 'b.')
        x0 = coords_2d[c_ind][gt[gg]['kp0'],0]# + np.random.randint(-5,5)
        x1 = coords_2d[c_ind][gt[gg]['kp1'],0]# + np.random.randint(-5,5)
        y0 = coords_2d[c_ind][gt[gg]['kp0'],1]# + np.random.randint(-5,5)
        y1 = coords_2d[c_ind][gt[gg]['kp1'],1]# + np.random.randint(-5,5)
        #col = sns.color_palette()[jj]
        col = ['g', 'b', 'c', 'm', 'y', 'k'][jj]
        #plt.plot(x0, y0, 'o', c=col, markeredgewidth=1.0, markeredgecolor='k')
        #plt.plot(x1, y1, 'o', c=col, markeredgewidth=1.0, markeredgecolor='k')
        plt.plot(x0, y0, 'o', c='r', ms=5)
        plt.plot(x1, y1, 'o', c='r', ms=5)
        plt.plot([x0, x1], [y0, y1], c=col)
        plt.title(str(int(per_im_overall_correct[im_id]*100)) + '%')
    ax.axis('off')


h36_root = '../../../h5_annotations'
turk_results_file = '../mturk_data/HUMAN36_FINALIZED_HITS_2018-07-05_20-01-41.pkl'
human_36m_file_path = '../mturk_data/human36m_train_mturk_subset.json'
# op_dir = 'plots/'
#
# if not os.path.isdir(op_dir):
#     os.makedirs(op_dir)

im_dir = '../pics/'
hf_cameras_file  = h36_root + '/cameras.h5'

images_to_exclude = [279782]  # exclude tutorial image
title_font = 18
label_font = 15

plt.close('all')

# load turk data
with open(turk_results_file) as dp:
    data_turk = pickle.load(dp)['_all_assignments']
mturk_results = utils.parse_mturk(data_turk, images_to_exclude)
ims_of_interest = list(set([mm['im_id'] for mm in mturk_results]))

# run crowd sourcing to combine
combined_labels = cmb.resolve_labels(mturk_results)

# get gt labels
cam_coords_orig, cam_coords, coords_2d, im_paths, keypoints = mturk_gt.load_h36_coords(h36_root,
                                                      human_36m_file_path, hf_cameras_file,
                                                      ims_of_interest, im_dir)

# compute GT stats
gt = compute_gt(combined_labels, cam_coords, 'Corrected')
gt_orig = compute_gt(combined_labels, cam_coords_orig, 'Original')


# generate stats
images = np.array([im['raw_id'] for im in combined_labels['images'].itervalues()])
un_workers = np.array([ww['id'] for ww in combined_labels['workers'].itervalues()])
un_images = np.unique(images)
images_kps = np.array([im['id'] for im in combined_labels['images'].itervalues()])
risks_kps = np.array([im['risk'] for im in combined_labels['images'].itervalues()])
merge_correct_kps = np.array([im['label']['label'] == gt[im['id']]['label'] for im in combined_labels['images'].itervalues()])
distance_kps = np.array([gt[im['id']]['dis'] for im in combined_labels['images'].itervalues()])
gt_overall = np.array([gt[rr['im_id_joint']]['label'] for rr in mturk_results])
correct_overall = np.array([rr['ann'] == (2*gt[rr['im_id_joint']]['label']-1) for rr in mturk_results])
images_overall = np.array([rr['im_id'] for rr in mturk_results])
workers_overall = np.array([rr['w_id'] for rr in mturk_results])

# stats for original labels
distance_kps_orig = np.array([gt_orig[im['id']]['dis'] for im in combined_labels['images'].itervalues()])
merge_correct_kps_orig = np.array([im['label']['label'] == gt_orig[im['id']]['label'] for im in combined_labels['images'].itervalues()])
correct_overall_orig = np.array([rr['ann'] == (2*gt_orig[rr['im_id_joint']]['label']-1) for rr in mturk_results])


# majority vote
mt_ims = np.array([rr['im_id_joint'] for rr in mturk_results])
mt_anns = np.array([(rr['ann']+1)/2 for rr in mturk_results])
mv_correct_kps = np.zeros(len(images_kps))
mv_percent_correct = np.zeros(len(images_kps))
for ii, im in enumerate(images_kps):
    inds = np.where(mt_ims == im)[0]
    pred = np.mean(mt_anns[inds])
    mv_correct_kps[ii] = (pred>0.5) == gt[im]['label']
    mv_percent_correct[ii] = (mt_anns[inds] == gt[im]['label']).mean()

print 'overall num annotations      ', len(mturk_results)
print 'numer of images              ', un_images.shape[0]
print 'num of pairs                 ', len(combined_labels['images'])
print 'acc (overall)                ', round(correct_overall.mean(),4)
print 'acc_orig (overall)           ', round(correct_overall_orig.mean(),4)
print 'acc merged  (per pair)       ', round(merge_correct_kps.mean(),4)
print 'acc merged_orig (per pair)   ', round(merge_correct_kps_orig.mean(),4)
print 'acc mv      (per pair)       ', round(mv_correct_kps.mean(),4)


# per image stats
per_im_risk = np.zeros(un_images.shape[0])
per_im_merged_correct = np.zeros(un_images.shape[0])
per_im_overall_correct = np.zeros(un_images.shape[0])
im_ids_kps = []
for ii, im in enumerate(un_images):
    inds = np.where(images == im)[0]
    im_ids_kps.append(images_kps[inds])

    per_im_risk[ii] = risks_kps[inds].mean()
    per_im_merged_correct[ii] = merge_correct_kps[inds].mean()
    inds_ov = np.where(images_overall == int(im))[0]
    per_im_overall_correct[ii] = correct_overall[inds_ov].mean()

# show hardest and easiest images
num_exs = 5
easiest_ims = np.argsort(per_im_overall_correct)[-num_exs:][::-1]
hardest_ims = np.argsort(per_im_overall_correct)[:num_exs]

plt.figure(0, (12,4))
gs = gridspec.GridSpec(1, num_exs)
print '\neasiest images'
for kk, im_id in enumerate(easiest_ims):
    plot_image(im_id, gs[kk])
# plt.savefig(op_dir + 'h36_easy.pdf')

plt.figure(1, (12,4))
gs = gridspec.GridSpec(1, num_exs)
print '\nhardest images'
for kk, im_id in enumerate(hardest_ims):
    plot_image(im_id, gs[kk])
# plt.savefig(op_dir + 'h36_hard.pdf')
#plt.suptitle('Easiest and Hardest Images', fontsize=title_font)


# per worker stats
per_worker_acc = np.zeros(un_workers.shape[0])
per_worker_acc_orig = np.zeros(un_workers.shape[0])
per_worker_tn = np.zeros(un_workers.shape[0])
per_worker_tp = np.zeros(un_workers.shape[0])
for ii, ww in enumerate(un_workers):
    inds = np.where(workers_overall == ww)[0]
    per_worker_acc[ii] = correct_overall[inds].mean()
    per_worker_tn[ii] = ((correct_overall[inds]==1) & (gt_overall[inds] == 0)).sum() / float((gt_overall[inds] == 0).sum())
    per_worker_tp[ii] = ((correct_overall[inds]==1) & (gt_overall[inds] == 1)).sum() / float((gt_overall[inds] == 1).sum())

    inds = np.where(workers_overall == ww)[0]
    per_worker_acc_orig[ii] = correct_overall_orig[inds].mean()


#
# image risk
cumul_merge_correct_kps = np.zeros(risks_kps.shape[0])
for ii, s_id in enumerate(np.argsort(risks_kps)):
    inds = np.where(risks_kps<=risks_kps[s_id])[0]
    cumul_merge_correct_kps[ii] = merge_correct_kps[inds].mean()

plt.figure(2, figsize=(5, 5))
plt.plot(np.sort(risks_kps))
plt.plot(cumul_merge_correct_kps, color=sns.color_palette()[2])
plt.title('Risk of Comparisons', fontsize=title_font)
plt.xlabel('sorted keypoint pairs', fontsize=label_font)
plt.ylabel('% correct pairwise comparisons', fontsize=label_font, color=sns.color_palette()[2])
ax = plt.gca().twinx()
ax.set_ylabel('risk', fontsize=label_font, color=sns.color_palette()[0])
ax.set_yticks([])
plt.xlim([0, len(risks_kps)])
plt.grid('on')
plt.tight_layout()
plt.show()


#
# image accuracy
plt.figure(3, figsize=(5, 5))
plt.plot(np.sort(per_im_merged_correct*100))
#plt.plot(np.sort(per_im_overall_correct*100))
plt.plot([0, len(per_im_overall_correct)], [50, 50], ':', c=sns.xkcd_rgb["faded green"])
plt.title('Image Difficulty', fontsize=title_font)
plt.xlabel('images sorted hardest to easiest', fontsize=label_font)
plt.ylabel('% correct pairwise comparisons', fontsize=label_font)
plt.xlim([0, len(per_im_overall_correct)])
plt.grid('on')
plt.tight_layout()
plt.show()
# plt.savefig(op_dir + 'image_difficulty_corr.pdf')

#
# distance dist of GT
plt.figure(4, figsize=(5, 5))
hist, bins = np.histogram(distance_kps/10.0, bins=50)
hist = 100*hist.astype(np.float) / hist.sum()
bin_width = bins[1]-bins[0]
plt.bar(bins[:-1]+bin_width/2.0, hist, width=bin_width)
plt.title('Relative Distance Between Pairs', fontsize=title_font)
plt.ylabel('% of pairs', fontsize=label_font)
plt.xlabel('relative distance between keypoints (cm)', fontsize=label_font)
plt.xlim([0,distance_kps.max()/10.0])
plt.grid('on')
plt.tight_layout()
plt.show()


#
# risk against distance
plt.figure(5, figsize=(5, 5))
plt.plot(distance_kps/10, risks_kps, '.')
plt.title('Risk vs Distance', fontsize=title_font)
plt.ylabel('risk', fontsize=label_font)
plt.xlabel('distance between keypoints (cm)', fontsize=label_font)
plt.grid('on')
plt.tight_layout()
plt.show()


#
# worker skill - from crowdsourcing model
# 0 TPR, 1 TNR - diagnoal entries of confusion m
worker_skill_0 = [combined_labels['workers'][ww]['skill'][0] for ww in combined_labels['workers']]
worker_skill_1 = [combined_labels['workers'][ww]['skill'][1] for ww in combined_labels['workers']]
plt.figure(6, figsize=(5, 5))
plt.plot(worker_skill_0, worker_skill_1, '.')
plt.title('Worker Skill', fontsize=title_font)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('tpr', fontsize=label_font)
plt.ylabel('tnr', fontsize=label_font)
plt.grid('on')
plt.tight_layout()
plt.show()


#
# worker bias
plt.figure(7, figsize=(5, 5))
plt.plot([0, 1.0], [0, 1.0], ':', c=sns.xkcd_rgb["faded green"])
plt.plot(per_worker_tn, per_worker_tp, '.')
plt.title('Worker Bias', fontsize=title_font)
delta = 0.005
plt.xlim([-delta, 1+delta])
plt.ylim([-delta, 1+delta])
plt.xlabel('true negatives', fontsize=label_font)
plt.ylabel('true positives', fontsize=label_font)
plt.grid('on')
plt.tight_layout()
plt.show()
# plt.savefig(op_dir + 'annotator_bias_corr.pdf')


#
# plot accuracy vs distance
merg_intervals, merg_binned_acc, merg_binned_inds = utils.bin_vals(distance_kps, merge_correct_kps, bin_size=10)
merg_intervals_orig, merg_binned_acc_orig, merg_binned_inds_orig = utils.bin_vals(distance_kps_orig, merge_correct_kps_orig, bin_size=10)

#mv_intervals, mv_binned_acc, mv_binned_inds = utils.bin_vals(distance_kps, mv_correct_kps, bin_size=10)
#print merge_correct_kps[merg_binned_inds[65]].shape[0]  # check the amount in that bin

plt.figure(8, figsize=(5, 5))
plt.plot(merg_intervals/10.0, merg_binned_acc*100, '.', label='corrected cam', ms=10)
plt.xlabel('relative distance between keypoints (cm)', fontsize=label_font)
plt.ylabel('% correct pairwise comparisons', fontsize=label_font)
plt.title('Accuracy vs Distance', fontsize=title_font)
plt.grid('on')
plt.tight_layout()
plt.show()
# plt.savefig(op_dir + 'acc_v_dis_corr.pdf')

plt.figure(9, figsize=(5, 5))
plt.plot(merg_intervals/10.0, merg_binned_acc*100, '.', label='corrected cam', ms=10)
plt.plot(merg_intervals_orig/10.0, merg_binned_acc_orig*100, '.', label='original cam', ms=10)
plt.xlabel('relative distance between keypoints (cm)', fontsize=label_font)
plt.ylabel('% correct pairwise comparisons', fontsize=label_font)
plt.title('Accuracy vs Distance', fontsize=title_font)
plt.legend(frameon=True)
plt.grid('on')
plt.tight_layout()
plt.show()
# plt.savefig(op_dir + 'acc_v_dis_both.pdf')

#
# per worker acc
plt.figure(10, figsize=(5, 5))
bin_width = 10
hist, bins = np.histogram(per_worker_acc*100, range=(0,100))
plt.bar(bins[:-1]+bin_width/2.0, 100*hist.astype(np.float) / hist.sum(), width=bin_width, label='corrected cam')
plt.xlim([0,100])
plt.title('Annotator Accuracy', fontsize=title_font)
plt.xlabel('% correct pairwise comparisons', fontsize=label_font)
plt.ylabel('% of annotators', fontsize=label_font)
plt.tight_layout()
plt.show()
# plt.savefig(op_dir + 'annotator_acc_corr.pdf')

#
# per worker acc
plt.figure(11, figsize=(10, 5))

plt.subplot(121)
bin_width = 10
hist, bins = np.histogram(per_worker_acc*100, range=(0,100))
plt.bar(bins[:-1]+bin_width/2.0, 100*hist.astype(np.float) / hist.sum(), width=bin_width, label='corrected cam', color=sns.color_palette()[0])
plt.xlim([0,100])
plt.ylim([0,40])
plt.xlabel('% correct pairwise comparisons', fontsize=label_font)
plt.ylabel('% of annotators', fontsize=label_font)
plt.legend(frameon=True, loc=2)


plt.subplot(122)
hist, bins = np.histogram(per_worker_acc_orig*100, range=(0,100))
plt.bar(bins[:-1]+bin_width/2.0, 100*hist.astype(np.float) / hist.sum(), width=bin_width, label='original cam', color=sns.color_palette()[1])
plt.xlim([0,100])
plt.ylim([0,40])
plt.xlabel('% correct pairwise comparisons', fontsize=label_font)
plt.ylabel('% of annotators', fontsize=label_font)
plt.legend(frameon=True, loc=2)

plt.suptitle('Annotator Accuracy', fontsize=title_font)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig(op_dir + 'annotator_acc_both.pdf')

plt.show()
