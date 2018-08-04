"""
Plots random annotations from LSP with merged labels.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import sys
import os
from PIL import Image
import matplotlib.gridspec as gridspec

sys.path.append('./user_study')
import user_study_utils as utils

lsp_root = '../../../lsp_dataset/'
merged_mturk_anns = '../mturk_data/relative_depth_lsp_v1.json'
turk_results_file_experts = '../mturk_data/LSP_MRO_HITS_2018-05-04_19-25-04.pkl'
# op_dir = 'plots/'
# if not os.path.isdir(op_dir):
#     os.makedirs(op_dir)

plt.close('all')
np.random.seed(1000000)
im_height = 300.0

# load merged mturk annotations
with open(merged_mturk_anns) as da:
    lsp = json.load(da)
joint_names = lsp['joint_names']

# shorten the joint names
joint_names = [jj.replace('right', 'r') for jj in joint_names]
joint_names = [jj.replace('left', 'l') for jj in joint_names]
joint_names = [jj.replace('shoulder', 'shldr') for jj in joint_names]
joint_names = [jj.replace('elbow', 'elbw') for jj in joint_names]
joint_names = [jj.replace('wrist', 'wrst') for jj in joint_names]
joint_names = [jj.replace('ankle', 'ankl') for jj in joint_names]


# load mturk annotations - expert
with open(turk_results_file_experts) as dp:
    data_turk_expert = pickle.load(dp)['_all_assignments']
mturk_results_expert = utils.parse_mturk(data_turk_expert, [], False)

# check expert consensus
unique_expert_ims = np.unique([aa['im_id_joint'] for aa in mturk_results_expert]).tolist()
unique_experts = np.unique([aa['w_id'] for aa in mturk_results_expert]).tolist()
ex_votes = np.zeros((len(unique_expert_ims), len(unique_experts)))
for mm in mturk_results_expert:
    im_ind = unique_expert_ims.index(mm['im_id_joint'])
    w_ind = unique_experts.index(mm['w_id'])
    ex_votes[im_ind, w_ind] = mm['ann']
agreement = (ex_votes.sum(1) == (-ex_votes.shape[1])).sum() + (ex_votes.sum(1) == (ex_votes.shape[1])).sum()
print round(100*(agreement / float(ex_votes.shape[0])),2), 'percent agreement among all three'
ex_votes_two = ex_votes[:, [1,2]]
agreement_two = (ex_votes_two.sum(1) == (-ex_votes_two.shape[1])).sum() + (ex_votes_two.sum(1) == (ex_votes_two.shape[1])).sum()
print round(100*(agreement_two / float(ex_votes_two.shape[0])),2), 'percent agreement among two\n'


# show expert annotation
rand_pair = np.random.choice(unique_expert_ims)
rand_im_id = int(rand_pair[:rand_pair.find('_')])
votes = [mm for mm in mturk_results_expert if mm['im_id_joint'] == rand_pair]
im_path = lsp_root + 'images/' + 'im' + str(rand_im_id+1).zfill(4) + '.jpg'
keypoints = np.array(lsp['images'][rand_im_id]['keypoints']).reshape(14,2)
print rand_pair
kp0 = votes[0]['kp0']
kp1 = votes[0]['kp1']
print joint_names[kp0], joint_names[kp1]
for ww in unique_experts:
    for vv in votes:
        if vv['w_id'] == ww:
            if vv['ann'] == -1:
                print vv['w_id'], '\t', vv['ann'], '\t', joint_names[kp0], 'is closer than', joint_names[kp1]
            else:
                print vv['w_id'], '\t', vv['ann'], '\t', joint_names[kp1], 'is closer than', joint_names[kp0]
print '\n'

if False:
    plt.figure(0)
    im = plt.imread(im_path)
    plt.imshow(im)
    plt.plot(keypoints[kp0,0], keypoints[kp0,1], 'b.' ,ms=15)
    plt.plot(keypoints[kp1,0], keypoints[kp1,1], 'b.' ,ms=15)
    plt.plot(keypoints[:,0], keypoints[:,1], 'r.' ,ms=2)
    plt.show()

# compare expert results to mturk - messy
inds = np.where(ex_votes_two[:,0] == ex_votes_two[:,1])[0]
ex_votes_label = (ex_votes_two[inds, 0]+1)/2
expert_agreed_ims = np.array(unique_expert_ims)[inds]
crowd_pred = np.zeros(inds.shape[0])
for ii in lsp['images']:
    for aa in ii['anns']:
        if aa['original_id'] in expert_agreed_ims:
            ind = np.where(expert_agreed_ims == aa['original_id'])[0]
            crowd_pred[ind] = aa['label']
print round(100*(crowd_pred == ex_votes_label).mean(),2), 'agreement crowd and experts on', expert_agreed_ims.shape[0], 'pairs\n'


# choose single random image
if False:
    rand_ind = np.random.randint(len(lsp['images']))
    print rand_ind

    kp0 = [aa['kp0'] for aa in lsp['images'][rand_ind]['anns']]
    kp1 = [aa['kp1'] for aa in lsp['images'][rand_ind]['anns']]
    risk = [aa['risk'] for aa in lsp['images'][rand_ind]['anns']]
    prob = [aa['prob'] for aa in lsp['images'][rand_ind]['anns']]
    label = [aa['label'] for aa in lsp['images'][rand_ind]['anns']]

    keypoints = [np.array(lsp['images'][rand_ind]['keypoints']).reshape(14,2)]*len(kp0)
    occluded = [np.array(lsp['images'][rand_ind]['occluded'])]*len(kp0)
    im_files = [lsp_root + 'images/' + 'im' + str(rand_ind+1).zfill(4) + '.jpg']*len(kp0)


# show multiple different people
if True:
    num_ims = 10
    rand_inds = np.random.randint(0, len(lsp['images']), num_ims)
    keypoints = [np.array(lsp['images'][rand_ind]['keypoints']).reshape(14,2) for rand_ind in rand_inds]
    occluded = [np.array(lsp['images'][rand_ind]['occluded']) for rand_ind in rand_inds]
    im_files = [lsp_root + 'images/' + 'im' + str(rand_ind+1).zfill(4) + '.jpg' for rand_ind in rand_inds]

    kp0 = []
    kp1 = []
    risk = []
    prob = []
    label = []
    for rand_ind in rand_inds:
        rand_kp = np.random.randint(len(lsp['images'][rand_ind]['anns']))
        kp0.append(lsp['images'][rand_ind]['anns'][rand_kp]['kp0'])
        kp1.append(lsp['images'][rand_ind]['anns'][rand_kp]['kp1'])
        risk.append(lsp['images'][rand_ind]['anns'][rand_kp]['risk'])
        prob.append(lsp['images'][rand_ind]['anns'][rand_kp]['prob'])
        label.append(lsp['images'][rand_ind]['anns'][rand_kp]['label'])


# sort from easiest to most difficulty
inds = np.argsort(risk)
keypoints = np.array(keypoints)[inds]
occluded = np.array(occluded)[inds]
im_files = np.array(im_files)[inds]

kp0 = np.array(kp0)[inds]
kp1 = np.array(kp1)[inds]
risk = np.array(risk)[inds]
prob = np.array(prob)[inds]
label = np.array(label)[inds]


plt.figure(1, (16,2))
ims = [Image.open(im_files[kk]) for kk in range(len(kp0))]
imsr = []
ratios = []
for im in ims:
    ss = im.size
    ratio = im_height / ss[1]
    im = im.resize((int(ss[0]*ratio), int(ss[1]*ratio)))
    imsr.append(im)
    ratios.append(ratio)

widths = np.array([im.size[0] for im in imsr])
widths = widths / float(widths.min())

gs = gridspec.GridSpec(1, len(imsr), width_ratios=widths.tolist())
for kk in range(len(imsr)):
    ax = plt.subplot(gs[kk], aspect='equal')
    ax.imshow(imsr[kk], aspect='auto')
    #ax.text(10, im_height-10, format(risk[kk], '.3f'), fontsize=12, bbox=dict(facecolor='green', alpha=0.5, pad=3))
    ax.axis('off')

    if label[kk] == 1:
        cols = ['#ef4343', '#8def43']
        #plt.title(joint_names[kp1[kk]] + ' <\n' + joint_names[kp0[kk]] + '\n' + str(round(risk[kk],3)), fontsize=10)
        plt.title(joint_names[kp1[kk]] + '<' + joint_names[kp0[kk]], fontsize=10)
        print joint_names[kp1[kk]] + '<' + joint_names[kp0[kk]] + '\t' + str(round(risk[kk],4))
    else:
        cols = ['#8def43', '#ef4343']
        #plt.title(joint_names[kp0[kk]] + ' <\n' + joint_names[kp1[kk]] + '\n' + str(round(risk[kk],3)), fontsize=10)
        plt.title(joint_names[kp0[kk]] + '<' + joint_names[kp1[kk]], fontsize=10)
        print joint_names[kp0[kk]] + '<' + joint_names[kp1[kk]] + '\t' + str(round(risk[kk],4))

    ratio = ratios[kk]
    # xx = [ratio*keypoints[kk][kp0[kk], 0], ratio*keypoints[kk][kp1[kk], 0]]
    # yy = [ratio*keypoints[kk][kp0[kk], 1], ratio*keypoints[kk][kp1[kk], 1]]
    # plt.plot(xx, yy, c='b')
    plt.plot(ratio*keypoints[kk][kp0[kk], 0], ratio*keypoints[kk][kp0[kk], 1], 'o', c=cols[0], markeredgewidth=1.0, markeredgecolor='k')
    plt.plot(ratio*keypoints[kk][kp1[kk], 0], ratio*keypoints[kk][kp1[kk], 1], 'o', c=cols[1], markeredgewidth=1.0, markeredgecolor='k')
plt.show()

# plt.savefig(op_dir + 'lsp_turk.pdf')
