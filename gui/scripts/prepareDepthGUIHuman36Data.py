'''
Written for the cocoa_depth Amazon Mechanical Turk GUI.

Intended to extract the needed keypoint annotations from the Human3.6
dataset.
'''

#######################################################################
# GENERAL PURPOSE MODULES

import random
import json
import sys
import numpy as np

sys.path.append('../')
import config

env = sys.argv[1] if len(sys.argv) >= 2 else 'dev'

if env == 'dev':
    config = config.DevelopmentConfig
elif env == 'prod':
    config = config.ProductionConfig
else:
    raise ValueError('Invalid environment name')

#######################################################################
# CONSTANTS

NUMBER_SUBJECTS_IN_HIT = 5
NUMBER_HIT_ASSIGNMENTS = 5
NUMBER_COMPS_IN_SUBJ = 5

VERBOSE = False

DATASET_MIN_KEYPOINTS = 17

NUM_DATASET_SUBJS = 1000

with open(config.HUMAN_ANNOTATION_FILE) as f:
    _human_dataset = json.load(f)

########################################################################
# MONGO CLIENT DATABASE SETUP

from pymongo import MongoClient
_mongo_client = MongoClient()

_mongo_db = _mongo_client.cocoa_depth_human36m

# TODO: Write a IF EXISTS DROP function
_mongo_db.depth_amt_gui_data.drop()
_mongo_db.keypoint_labels.drop()
_mongo_db.depth_hit_id2human_subj_id.drop()
_mongo_db.human_subj_id2depth_hit_id.drop()

_mongo_coll_1 = _mongo_db.depth_amt_gui_data
_mongo_coll_2 = _mongo_db.keypoint_labels
_mongo_coll_3 = _mongo_db.depth_hit_id2human_subj_id
_mongo_coll_4 = _mongo_db.human_subj_id2depth_hit_id

########################################################################
# STORE KEYPOINTS AND KEYPOINT LABELS TO BE USED

_human_labels = _human_dataset['pose'][0]['keypoints']
_human_keypoint_labels = _human_labels[:]

# create a single document in the collection and insert it
_mongo_keypoint_entry = {"_keypoint_labels": _human_keypoint_labels}
res = _mongo_coll_2.insert(_mongo_keypoint_entry)

########################################################################
# SELECT IMAGES TO ANNOTATE

_human_anns = _human_dataset['annotations']
_human_images = _human_dataset['images']

_index_shuffle = range(len(_human_anns))
random.shuffle(_index_shuffle)
_index_shuffle = _index_shuffle[:NUM_DATASET_SUBJS]
_human_anns = [_human_anns[i] for i in _index_shuffle]
_human_images = [_human_images[i] for i in _index_shuffle]

DATASET_IMG_ID_LIST = [h['id'] for h in _human_anns]

_dataset_subj_id_list = []
count = 0
for _human_ann, _human_img in zip(_human_anns, _human_images):
    count += 1
    _dataset_subj_id_list.append(_human_ann['id'])
    if VERBOSE:
        print "[%d]/[%d] -> subject id [%d]" % (count, NUM_DATASET_SUBJS,
                                                _human_ann['id'])
    else:
        if count == 1 or count == NUM_DATASET_SUBJS:
            print "[%d]/[%d] -> subject id [%d]" % (count, NUM_DATASET_SUBJS,
                                                    _human_ann['id'])

    # Add the visible flag so it matches the data structure of mscoco
    _new_keypoints = []
    for i, pt in enumerate(_human_ann['kpts_2d']):
        _new_keypoints.append(pt)
        if i % 2 == 1:
            _new_keypoints.append(2)
    _human_ann['kpts_2d'] = _new_keypoints

    # Extract width and height of image to create a fake bounding box.
    _img_width = _human_img['width']
    _img_height = _human_img['height']

    # Pairs
    num_pts = 17
    num_comps = NUMBER_COMPS_IN_SUBJ
    pairs = [(i, j) for i in range(num_pts) for j in range(num_pts) if i < j]
    pair_idxs = np.random.choice(len(pairs), num_comps, replace=False)
    comps = np.take(pairs, pair_idxs, axis=0).tolist()

    for _ci, c in enumerate(comps):
        comps[_ci] = c if bool(random.getrandbits(1)) else [c[1], c[0]]

    # create a document in the collection for every annotation
    _mongo_ann_entry = \
        {"_human_img_id": _human_ann['i_id'],
         "_human_subj_id": _human_ann['id'],
         "_image_keypoints": _human_ann['kpts_2d'],
         "_keypoints_bbox": [0, 0, _img_width, _img_height],
         "_human_img_src": config.HUMAN_IMAGES_SERVER_FOLDER + _human_img['filename'],
           "_comps": comps,
        }
    # insert the document in the collection
    res = _mongo_coll_1.insert(_mongo_ann_entry)

# randomly add an element to the end of array until full number is divisible by 10
DATASET_SUBJ_ID_LIST = _dataset_subj_id_list
while (len(DATASET_SUBJ_ID_LIST) % NUMBER_SUBJECTS_IN_HIT != 0):
    DATASET_SUBJ_ID_LIST.append(random.choice(DATASET_SUBJ_ID_LIST))

l = len(DATASET_SUBJ_ID_LIST)
print "_____________________________________________________________"
print "Organizing HITs"
print " - Augmented number of subjects:       [%d]" % l
print " - Number of subjects per HIT:         [%d]" % NUMBER_SUBJECTS_IN_HIT
print " - Number of annotators per subject:   [%d]" % NUMBER_HIT_ASSIGNMENTS
print " - Total number of HITs needed:        [%d]" % (
    NUMBER_HIT_ASSIGNMENTS * (l / NUMBER_SUBJECTS_IN_HIT))
print "_____________________________________________________________"

amt_hit_id = 0
for ii in range(0, NUMBER_HIT_ASSIGNMENTS):
    random.shuffle(DATASET_SUBJ_ID_LIST)

    for jj in range(0, l, NUMBER_SUBJECTS_IN_HIT):
        amt_hit_id = amt_hit_id + 1
        _amt_hit_people_list = DATASET_SUBJ_ID_LIST[jj:jj +
                                                    NUMBER_SUBJECTS_IN_HIT]

        if VERBOSE:
            print "HITId: [%d] -> human subjects: [%s]" % (
                amt_hit_id, str(_amt_hit_people_list))
        else:
            if amt_hit_id == 1 or amt_hit_id == (NUMBER_HIT_ASSIGNMENTS *
                                                 (l / NUMBER_SUBJECTS_IN_HIT)):
                print "HITId: [%d] -> human subjects: [%s]" % (
                    amt_hit_id, str(_amt_hit_people_list))

        # create a document in the collection for every group of human ids
        _mongo_hit = \
            {"_amt_hit_id": amt_hit_id,
             "_human_subjs_ids": _amt_hit_people_list}
        # insert the document in the collection
        res = _mongo_coll_3.insert(_mongo_hit)

        for pp in _amt_hit_people_list:
            _mongo_rev_hit = \
            {"_human_subj_id": pp,
             "_amt_hit_id": amt_hit_id}
            res = _mongo_coll_4.insert(_mongo_rev_hit)
