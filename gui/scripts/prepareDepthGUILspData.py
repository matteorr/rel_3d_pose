'''
Written for the cocoa_depth Amazon Mechanical Turk GUI.

Intended to extract the needed keypoint annotations from the LSP
dataset.
'''

#######################################################################
# GENERAL PURPOSE MODULES

import pickle
import random
import json
import sys
import numpy as np
from PIL import Image

#######################################################################
# CONSTANTS

SEED = 421
random.seed(SEED)

NUMBER_SUBJECTS_IN_HIT = 5
NUMBER_HIT_ASSIGNMENTS = 1#5
NUMBER_COMPS_IN_SUBJ = 5

VERBOSE = False

NUM_DATASET_SUBJS = 2000

#######################################################################
# LSP ANNOTATIONS SETUP

LSP_ANNOTATION_FILE = '/home/ubuntu/datasets/lsp/CO_LSP_train2016.json'
LSP_IMAGES_FOLDER = 'http://vision.caltech.edu/~mronchi/data/LSP/images/'
LSP_IMAGES_SERVER_FOLDER = '/static/images/lsp/'

LSP_RAW_RESULT_PATH = "/home/ubuntu/amt_guis/cocoa_depth/hits/lsp/LSP_FINALIZED_HITS_2018-04-27_17-33-24.pkl"
lsp_result_data = pickle.load(open(LSP_RAW_RESULT_PATH, 'r'))
lsp_result_data = lsp_result_data['_good_assignments'] + lsp_result_data['_flagged_assignments']

subj_id_to_comps = {}
for d in lsp_result_data:
    for t in d['trials']:
        subj_id_to_comps[t['lsp_subj_id']] = [[int(c) for c in comp.split(',')] for comp in t['depth']['keypoint_comparisons_res'].keys()]

with open( LSP_ANNOTATION_FILE ) as f:
    _lsp_dataset = json.load(f)

########################################################################
# MONGO CLIENT DATABASE SETUP

from pymongo import MongoClient
_mongo_client = MongoClient()

_mongo_db = _mongo_client.cocoa_depth_lsp

# TODO: Write a IF EXISTS DROP function
_mongo_db.depth_amt_gui_data.drop()
_mongo_db.keypoint_labels.drop()
_mongo_db.depth_hit_id2lsp_subj_id.drop()
_mongo_db.lsp_subj_id2depth_hit_id.drop()

_mongo_coll_1 = _mongo_db.depth_amt_gui_data
_mongo_coll_2 = _mongo_db.keypoint_labels
_mongo_coll_3 = _mongo_db.depth_hit_id2lsp_subj_id
_mongo_coll_4 = _mongo_db.lsp_subj_id2depth_hit_id


########################################################################
# STORE KEYPOINTS AND KEYPOINT LABELS TO BE USED

_lsp_labels = _lsp_dataset['keypoints']
_lsp_keypoint_labels = _lsp_labels[:]

# create a single document in the collection and insert it
_mongo_keypoint_entry = { "_keypoint_labels": _lsp_keypoint_labels }
res = _mongo_coll_2.insert( _mongo_keypoint_entry )


########################################################################
# SELECT IMAGES TO ANNOTATE

_lsp_anns = _lsp_dataset['annotations']
#_lsp_images = _lsp_dataset['images']

_index_shuffle = range(len(_lsp_anns))
random.shuffle(_index_shuffle)
_index_shuffle = _index_shuffle[:NUM_DATASET_SUBJS]
_lsp_anns = [_lsp_anns[i] for i in _index_shuffle]
#_lsp_images = [_lsp_images[i] for i in _index_shuffle]

DATASET_IMG_ID_LIST = [h['id'] for h in _lsp_anns]

_dataset_subj_id_list = []
count = 0
for _lsp_ann in _lsp_anns:
    count += 1
    _dataset_subj_id_list.append(_lsp_ann['id']) 
    if VERBOSE:
        print "[%d]/[%d] -> subject id [%d]" %( count, NUM_DATASET_SUBJS, _lsp_ann['id'] )
    else:
        if count == 1 or count == NUM_DATASET_SUBJS:
            print "[%d]/[%d] -> subject id [%d]" %( count, NUM_DATASET_SUBJS, _lsp_ann['id'] )
   
    # Add the visible flag so it matches the data structure of mscoco
    _new_keypoints = [] 
    for i, pt in enumerate(_lsp_ann['2d_keypoints']):
        _new_keypoints.append(pt) 
        if i % 2 == 1:
            _new_keypoints.append(2) 
    _lsp_ann['2d_keypoints'] = _new_keypoints

    # Extract width and height of image to create a fake bounding box.
    #_img_width = _lsp_img['width']
    #_img_height = _lsp_img['height']

    # Pairs
    num_pts = 14 
    num_comps = NUMBER_COMPS_IN_SUBJ 
    pairs = [(i, j) for i in range(num_pts) for j in range(num_pts) if i < j]
    pair_idxs = np.random.choice(len(pairs), num_comps, replace=False) 
    comps = np.take(pairs, pair_idxs, axis=0).tolist()

    # create a document in the collection for every annotation 
    _mongo_ann_entry = \
        {"_lsp_img_id": _lsp_ann['id'],
         "_lsp_subj_id": _lsp_ann['id'],
         "_image_keypoints": _lsp_ann['2d_keypoints'],
         # "_keypoints_bbox": [0, 0, _img_width, _img_height],
         "_lsp_img_src": LSP_IMAGES_SERVER_FOLDER + _lsp_ann['image'],
	 "_comps": comps, 
        }
    # insert the document in the collection
    res = _mongo_coll_1.insert( _mongo_ann_entry )


# randomly add an element to the end of array until full number is divisible by 10
DATASET_SUBJ_ID_LIST = _dataset_subj_id_list
while ( len( DATASET_SUBJ_ID_LIST ) % NUMBER_SUBJECTS_IN_HIT != 0 ):
    DATASET_SUBJ_ID_LIST.append( random.choice( DATASET_SUBJ_ID_LIST ) )

l = len( DATASET_SUBJ_ID_LIST )
print "_____________________________________________________________"
print "Organizing HITs"
print " - Augmented number of subjects:       [%d]" % l
print " - Number of subjects per HIT:         [%d]" % NUMBER_SUBJECTS_IN_HIT
print " - Number of annotators per subject:   [%d]" % NUMBER_HIT_ASSIGNMENTS
print " - Total number of HITs needed:        [%d]" %(NUMBER_HIT_ASSIGNMENTS * (l / NUMBER_SUBJECTS_IN_HIT))
print "_____________________________________________________________"

amt_hit_id = 0
for ii in range( 0, NUMBER_HIT_ASSIGNMENTS ):
	l = len(DATASET_SUBJ_ID_LIST)
    else:
        random.shuffle( DATASET_SUBJ_ID_LIST )
    
    for jj in range( 0, l, NUMBER_SUBJECTS_IN_HIT ):
        amt_hit_id = amt_hit_id + 1
        _amt_hit_people_list = DATASET_SUBJ_ID_LIST[jj:jj + NUMBER_SUBJECTS_IN_HIT]
        
        if VERBOSE:
            print "HITId: [%d] -> lsp subjects: [%s]" %( amt_hit_id, str( _amt_hit_people_list ))
        else:
            if amt_hit_id == 1 or amt_hit_id == (NUMBER_HIT_ASSIGNMENTS * (l / NUMBER_SUBJECTS_IN_HIT)): 
                print "HITId: [%d] -> lsp subjects: [%s]" %( amt_hit_id, str( _amt_hit_people_list ))
        
        # create a document in the collection for every group of lsp ids
        _mongo_hit = \
            {"_amt_hit_id": amt_hit_id, 
             "_lsp_subjs_ids": _amt_hit_people_list}
        # insert the document in the collection
        res = _mongo_coll_3.insert(_mongo_hit)
        
        for pp in _amt_hit_people_list:
            _mongo_rev_hit = \
            {"_lsp_subj_id": pp,
             "_amt_hit_id": amt_hit_id}
            res = _mongo_coll_4.insert( _mongo_rev_hit )




