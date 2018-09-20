#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

################################################################################
################################################################################

HUMAN_36M_KEYPOINTS = \
  ['mid_hip',
   'right_hip', 'right_knee', 'right_ankle', 'right_foot_base', 'right_foot_tip',
   'left_hip', 'left_knee', 'left_ankle', 'left_foot_base', 'left_foot_tip',
   'mid_hip_2', 'mid_spine', 'neck', 'chin', 'head', 'neck_2',
   'left_shoulder', 'left_elbow', 'left_wrist', 'left_wrist_2','left_palm',
   'left_thumb','left_thumb_2', 'neck_3', 'right_shoulder', 'right_elbow',
   'right_wrist', 'right_wrist_2','right_palm','right_thumb','right_thumb_3']


################################################################################
# 14 keypoints - same as LSP
################################################################################

HUMAN_36M_14K_3D = \
  ['right_ankle','right_knee', 'right_hip',
   'left_hip', 'left_knee', 'left_ankle',
   'right_wrist', 'right_elbow', 'right_shoulder',
   'left_shoulder', 'left_elbow', 'left_wrist',
   'neck', 'head']

HUMAN_36M_14K_2D = HUMAN_36M_14K_3D

HUMAN_36M_14K_ROOT = 'neck'

HUMAN_36M_14K_SKELETON_3D = [['right_knee', 'right_hip'],
            ['right_ankle', 'right_knee'],
            ['left_hip', 'left_knee'],
            ['left_knee', 'left_ankle'],
            ['right_hip','left_hip'],
            ['left_shoulder', 'left_elbow'],
            ['left_elbow', 'left_wrist'],
            ['right_elbow', 'right_shoulder'],
            ['right_wrist', 'right_elbow'],
            ['left_shoulder','neck'],
            ['right_shoulder','neck'],
            ['left_hip','left_shoulder'],
            ['right_hip','right_shoulder'],
            ['neck','head']]

# NOTE: this is actually taken from human3.6, but could be a generic skeleton
AVG_PERSON_14K_SKELETON_3D_LENS = \
            [449.20465088, 445.5809021 , 449.20339966, 445.5802002 ,
             268.09924316, 279.75796509, 249.522995  , 279.76083374,
             249.52223206, 148.83453369, 148.83372498, 438.89956665,
             423.5843811,  174.09963989]

HUMAN_36M_14K_SKELETON_2D = HUMAN_36M_14K_SKELETON_3D

HUMAN_36M_14K_SKELETON_NAMES = [
            'right_thigh',
            'right_foreleg',
            'left_thigh',
            'left_foreleg',
            'pelvic_girdle',
            'left_upper_arm',
            'left_forearm',
            'right_upper_arm',
            'right_forearm',
            'left_shoulder_girdle',
            'right_shoulder_girdle',
            'left_trunk',
            'right_trunk',
            'neck']

# CONTAINES THE LIMBS THAT SHOULD BE IGNORED FOR PROPORTION PURPOSES
HUMAN_36M_14K_SKELETON_MASK = ['left_trunk', 'right_trunk']

HUMAN_36M_14K_UNIT_LENGTH_LIMB = ['right_ankle','right_knee']

HUMAN_36M_14K_RIGHT_LIMBS = [['right_knee','right_hip'],
            ['right_ankle', 'right_knee'],
            ['right_shoulder','neck'],
            ['right_elbow', 'right_shoulder'],
            ['right_wrist', 'right_elbow'],
            ['right_hip','right_shoulder']]

HUMAN_36M_14K_LEFT_LIMBS = [['left_hip', 'left_knee'],
            ['left_knee', 'left_ankle'],
            ['left_shoulder','neck'],
            ['left_shoulder', 'left_elbow'],
            ['left_elbow', 'left_wrist'],
            ['left_hip','left_shoulder']]


################################################################################
# 16 keypoints - Stacked HourGlass - same as 17 but no chin
################################################################################

HUMAN_36M_16K_3D = \
  ['right_ankle','right_knee', 'right_hip',
   'left_hip', 'left_knee', 'left_ankle',
   'mid_hip', 'mid_spine', 'neck', 'head',
   'right_wrist', 'right_elbow', 'right_shoulder',
   'left_shoulder', 'left_elbow', 'left_wrist']

HUMAN_36M_16K_2D = HUMAN_36M_16K_3D

HUMAN_36M_16K_ROOT = 'mid_hip'

HUMAN_36M_16K_SKELETON_3D = [['mid_hip', 'right_hip'],
            ['right_hip', 'right_knee'],
            ['right_knee', 'right_ankle'],
            ['mid_hip', 'left_hip'],
            ['left_hip', 'left_knee'],
            ['left_knee', 'left_ankle'],
            ['mid_hip', 'mid_spine'],
            ['mid_spine', 'neck'],
            ['neck', 'head'],
            ['neck', 'left_shoulder'],
            ['left_shoulder', 'left_elbow'],
            ['left_elbow', 'left_wrist'],
            ['neck', 'right_shoulder'],
            ['right_shoulder', 'right_elbow'],
            ['right_elbow', 'right_wrist']]

AVG_PERSON_16K_SKELETON_3D_LENS = None

HUMAN_36M_16K_SKELETON_2D = HUMAN_36M_16K_SKELETON_3D

HUMAN_36M_16K_SKELETON_NAMES = ['right_pelvic_girdle',
            'right_thigh',
            'right_foreleg',
            'left_pelvic_girdle',
            'left_thigh',
            'left_foreleg',
            'low_spine',
            'high_spine',
            'head',
            'left_shoulder_girdle',
            'left_upper_arm',
            'left_forearm',
            'right_shoulder_girdle',
            'right_upper_arm',
            'right_forearm']

HUMAN_36M_16K_SKELETON_MASK = []

HUMAN_36M_16K_UNIT_LENGTH_LIMB  = ['right_knee', 'right_ankle']

HUMAN_36M_16K_RIGHT_LIMBS = [['mid_hip', 'right_hip'],
            ['right_hip', 'right_knee'],
            ['right_knee', 'right_ankle'],
            ['neck', 'right_shoulder'],
            ['right_shoulder', 'right_elbow'],
            ['right_elbow', 'right_wrist']]

HUMAN_36M_16K_LEFT_LIMBS = [['mid_hip', 'left_hip'],
            ['left_hip', 'left_knee'],
            ['left_knee', 'left_ankle'],
            ['neck', 'left_shoulder'],
            ['left_shoulder', 'left_elbow'],
            ['left_elbow', 'left_wrist']]


################################################################################
# 17 keypoints - default
################################################################################

HUMAN_36M_17K_3D = \
  ['mid_hip',
   'right_hip','right_knee', 'right_ankle',
   'left_hip', 'left_knee', 'left_ankle',
   'mid_spine', 'neck', 'chin', 'head',
   'left_shoulder', 'left_elbow', 'left_wrist',
   'right_shoulder', 'right_elbow', 'right_wrist']

HUMAN_36M_17K_2D = HUMAN_36M_17K_3D

HUMAN_36M_17K_ROOT = 'mid_hip'

HUMAN_36M_17K_SKELETON_3D = [['mid_hip', 'right_hip'],
            ['right_hip', 'right_knee'],
            ['right_knee', 'right_ankle'],
            ['mid_hip', 'left_hip'],
            ['left_hip', 'left_knee'],
            ['left_knee', 'left_ankle'],
            ['mid_hip', 'mid_spine'],
            ['mid_spine', 'neck'],
            ['neck', 'chin'],
            ['chin', 'head'],
            ['neck', 'left_shoulder'],
            ['left_shoulder', 'left_elbow'],
            ['left_elbow', 'left_wrist'],
            ['neck', 'right_shoulder'],
            ['right_shoulder', 'right_elbow'],
            ['right_elbow', 'right_wrist']]

AVG_PERSON_17K_SKELETON_3D_LENS = \
    [7,19,20,7,19,20,11.5,12.5,3.5,3.5,9,15,14,9,15,14]

HUMAN_36M_17K_SKELETON_2D = HUMAN_36M_17K_SKELETON_3D

HUMAN_36M_17K_SKELETON_NAMES = ['right_pelvic_girdle',
            'right_thigh',
            'right_foreleg',
            'left_pelvic_girdle',
            'left_thigh',
            'left_foreleg',
            'low_spine',
            'high_spine',
            'neck',
            'head',
            'left_shoulder_girdle',
            'left_upper_arm',
            'left_forearm',
            'right_shoulder_girdle',
            'right_upper_arm',
            'right_forearm']

HUMAN_36M_17K_SKELETON_MASK = []

HUMAN_36M_17K_UNIT_LENGTH_LIMB  = ['right_knee', 'right_ankle']

HUMAN_36M_17K_RIGHT_LIMBS = [['mid_hip', 'right_hip'],
            ['right_hip', 'right_knee'],
            ['right_knee', 'right_ankle'],
            ['neck', 'right_shoulder'],
            ['right_shoulder', 'right_elbow'],
            ['right_elbow', 'right_wrist']]

HUMAN_36M_17K_LEFT_LIMBS = [['mid_hip', 'left_hip'],
            ['left_hip', 'left_knee'],
            ['left_knee', 'left_ankle'],
            ['neck', 'left_shoulder'],
            ['left_shoulder', 'left_elbow'],
            ['left_elbow', 'left_wrist']]

################################################################################
# 19 keypoints
################################################################################

HUMAN_36M_19K_3D = \
  ['head', 'chin', 'neck',
   'right_shoulder', 'right_elbow', 'right_wrist',
   'left_shoulder',  'left_elbow',  'left_wrist',
   'mid_spine', 'mid_hip',
   'right_hip', 'right_knee', 'right_ankle', 'right_foot_tip',
   'left_hip',  'left_knee',  'left_ankle',  'left_foot_tip']

HUMAN_36M_19K_2D = HUMAN_36M_19K_3D

HUMAN_36M_19K_ROOT = 'mid_hip'

HUMAN_36M_19K_SKELETON_3D = \
    [['mid_hip', 'right_hip'],
     ['right_hip', 'right_knee'],
     ['right_knee', 'right_ankle'],
     ['right_ankle', 'right_foot_tip'],
     ['mid_hip', 'left_hip'],
     ['left_hip', 'left_knee'],
     ['left_knee', 'left_ankle'],
     ['left_ankle', 'left_foot_tip'],
     ['mid_hip', 'mid_spine'],
     ['mid_spine', 'neck'],
     ['neck', 'chin'],
     ['chin', 'head'],
     ['neck', 'left_shoulder'],
     ['left_shoulder', 'left_elbow'],
     ['left_elbow', 'left_wrist'],
     ['neck', 'right_shoulder'],
     ['right_shoulder', 'right_elbow'],
     ['right_elbow', 'right_wrist']]

AVG_PERSON_19K_SKELETON_3D_LENS = None

HUMAN_36M_19K_SKELETON_2D = HUMAN_36M_19K_SKELETON_3D

HUMAN_36M_19K_SKELETON_NAMES = \
    ['right_pelvic_girdle',
     'right_thigh',
     'right_foreleg',
     'right_foot',
     'left_pelvic_girdle',
     'left_thigh',
     'left_foreleg',
     'left_foot'
     'low_spine',
     'high_spine',
     'neck',
     'head',
     'left_shoulder_girdle',
     'left_upper_arm',
     'left_forearm',
     'right_shoulder_girdle',
     'right_upper_arm',
     'right_forearm']

HUMAN_36M_19K_SKELETON_MASK = []

HUMAN_36M_19K_UNIT_LENGTH_LIMB  = ['right_knee', 'right_ankle']

HUMAN_36M_19K_RIGHT_LIMBS = \
    [['mid_hip', 'right_hip'],
     ['right_hip', 'right_knee'],
     ['right_knee', 'right_ankle'],
     ['right_ankle', 'right_foot_tip'],
     ['neck', 'right_shoulder'],
     ['right_shoulder', 'right_elbow'],
     ['right_elbow', 'right_wrist']]

HUMAN_36M_19K_LEFT_LIMBS = \
    [['mid_hip', 'left_hip'],
     ['left_hip', 'left_knee'],
     ['left_knee', 'left_ankle'],
     ['left_ankle', 'left_foot_tip'],
     ['neck', 'left_shoulder'],
     ['left_shoulder', 'left_elbow'],
     ['left_elbow', 'left_wrist']]

################################################################################
# OPENPOSE keypoints
################################################################################

OPENPOSE_17K_3D = \
  ['chin', 'neck',
   'right_shoulder', 'right_elbow', 'right_wrist',
   'left_shoulder',  'left_elbow',  'left_wrist',
   'mid_hip',
   'right_hip', 'right_knee', 'right_ankle', 'right_foot_tip',
   'left_hip',  'left_knee',  'left_ankle',  'left_foot_tip']

OPENPOSE_17K_2D = OPENPOSE_17K_3D

OPENPOSE_17K_ROOT = 'mid_hip'

OPENPOSE_17K_SKELETON_3D = \
    [['mid_hip', 'right_hip'],
     ['right_hip', 'right_knee'],
     ['right_knee', 'right_ankle'],
     ['right_ankle', 'right_foot_tip'],
     ['mid_hip', 'left_hip'],
     ['left_hip', 'left_knee'],
     ['left_knee', 'left_ankle'],
     ['left_ankle', 'left_foot_tip'],
     ['mid_hip', 'neck'],
     ['neck', 'chin'],
     ['neck', 'left_shoulder'],
     ['left_shoulder', 'left_elbow'],
     ['left_elbow', 'left_wrist'],
     ['neck', 'right_shoulder'],
     ['right_shoulder', 'right_elbow'],
     ['right_elbow', 'right_wrist']]

AVG_PERSON_OPENPOSE_17K_SKELETON_3D_LENS = None

OPENPOSE_17K_SKELETON_2D = OPENPOSE_17K_SKELETON_3D

OPENPOSE_17K_SKELETON_NAMES = \
    ['right_pelvic_girdle',
     'right_thigh',
     'right_foreleg',
     'right_foot',
     'left_pelvic_girdle',
     'left_thigh',
     'left_foreleg',
     'left_foot'
     'spine',
     'neck',
     'head',
     'left_shoulder_girdle',
     'left_upper_arm',
     'left_forearm',
     'right_shoulder_girdle',
     'right_upper_arm',
     'right_forearm']

OPENPOSE_17K_SKELETON_MASK = []

OPENPOSE_17K_UNIT_LENGTH_LIMB  = ['right_knee', 'right_ankle']

OPENPOSE_17K_RIGHT_LIMBS = \
    [['mid_hip', 'right_hip'],
     ['right_hip', 'right_knee'],
     ['right_knee', 'right_ankle'],
     ['right_ankle', 'right_foot_tip'],
     ['neck', 'right_shoulder'],
     ['right_shoulder', 'right_elbow'],
     ['right_elbow', 'right_wrist']]

OPENPOSE_17K_LEFT_LIMBS = \
    [['mid_hip', 'left_hip'],
     ['left_hip', 'left_knee'],
     ['left_knee', 'left_ankle'],
     ['left_ankle', 'left_foot_tip'],
     ['neck', 'left_shoulder'],
     ['left_shoulder', 'left_elbow'],
     ['left_elbow', 'left_wrist']]

################################################################################
################################################################################

class DatasetMisc(object):
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type

        if self.dataset_type == 'openpose_17k':
            self.init_openpose_17k()

        elif self.dataset_type == 'human36_19k':
            self.init_human_36_19k()

        elif self.dataset_type == 'human36_17k':
            self.init_human_36_17k()

        elif self.dataset_type == 'human36_14k':
            self.init_human_36_14k()

        elif self.dataset_type == 'lsp_14k':
            self.init_human_36_14k()

        elif self.dataset_type == 'shg_16k':
            self.init_shg_16k()

        else:
            raise ValueError("Unknown dataset type %s"%self.dataset_type)

        self.ROOT_IDX_2D  = self.KEYPOINTS_2D.index(self.ROOT_KPT)
        self.ROOT_IDX_3D  = self.KEYPOINTS_3D.index(self.ROOT_KPT)

        self.SKELETON_2D_IDX = [[self.KEYPOINTS_2D.index(k[0]),self.KEYPOINTS_2D.index(k[1])] for k in self.SKELETON_2D]
        self.SKELETON_3D_IDX = [[self.KEYPOINTS_3D.index(k[0]),self.KEYPOINTS_3D.index(k[1])] for k in self.SKELETON_3D]
        self.RIGHT_LIMBS_IDX = [[self.KEYPOINTS_3D.index(k[0]),self.KEYPOINTS_3D.index(k[1])] for k in self.RIGHT_LIMBS]
        self.LEFT_LIMBS_IDX  = [[self.KEYPOINTS_3D.index(k[0]),self.KEYPOINTS_3D.index(k[1])] for k in self.LEFT_LIMBS]
        self.UNIT_LENGTH_LIMB_IDX = self.SKELETON_3D.index(self.UNIT_LENGTH_LIMB)

        self.NUM_KEYPOINTS_2D = len(self.KEYPOINTS_2D)
        self.NUM_KEYPOINTS_3D = len(self.KEYPOINTS_3D)

        self.SKELETON_2D_COLOR = [1 if 'right' in ss[1] else 0 for ss in self.SKELETON_2D]
        self.SKELETON_3D_COLOR = [1 if 'right' in ss[1] else 0 for ss in self.SKELETON_3D]

    def init_human_36_14k(self):
        self.KEYPOINTS_2D = HUMAN_36M_14K_2D
        self.KEYPOINTS_3D = HUMAN_36M_14K_3D
        self.ROOT_KPT     = HUMAN_36M_14K_ROOT
        self.SKELETON_2D  = HUMAN_36M_14K_SKELETON_2D
        self.SKELETON_3D  = HUMAN_36M_14K_SKELETON_3D
        self.UNIT_LENGTH_LIMB = HUMAN_36M_14K_UNIT_LENGTH_LIMB
        self.RIGHT_LIMBS      = HUMAN_36M_14K_RIGHT_LIMBS
        self.LEFT_LIMBS       = HUMAN_36M_14K_LEFT_LIMBS
        self.SKELETON_3D_LENS_AVG_PERSON = AVG_PERSON_14K_SKELETON_3D_LENS
        self.SKELETON_3D_NAMES = HUMAN_36M_14K_SKELETON_NAMES
        self.SKELETON_3D_MASK  = [self.SKELETON_3D_NAMES.index(i) for i in HUMAN_36M_14K_SKELETON_MASK]

    def init_human_36_17k(self):
        self.KEYPOINTS_2D = HUMAN_36M_17K_2D
        self.KEYPOINTS_3D = HUMAN_36M_17K_3D
        self.ROOT_KPT     = HUMAN_36M_17K_ROOT
        self.SKELETON_2D  = HUMAN_36M_17K_SKELETON_2D
        self.SKELETON_3D  = HUMAN_36M_17K_SKELETON_3D
        self.UNIT_LENGTH_LIMB = HUMAN_36M_17K_UNIT_LENGTH_LIMB
        self.RIGHT_LIMBS      = HUMAN_36M_17K_RIGHT_LIMBS
        self.LEFT_LIMBS       = HUMAN_36M_17K_LEFT_LIMBS
        self.SKELETON_3D_LENS_AVG_PERSON = AVG_PERSON_17K_SKELETON_3D_LENS
        self.SKELETON_3D_NAMES = HUMAN_36M_17K_SKELETON_NAMES
        self.SKELETON_3D_MASK  = [self.SKELETON_3D_NAMES.index(i) for i in HUMAN_36M_17K_SKELETON_MASK]

    def init_human_36_19k(self):
        self.KEYPOINTS_2D = HUMAN_36M_19K_2D
        self.KEYPOINTS_3D = HUMAN_36M_19K_3D
        self.ROOT_KPT     = HUMAN_36M_19K_ROOT
        self.SKELETON_2D  = HUMAN_36M_19K_SKELETON_2D
        self.SKELETON_3D  = HUMAN_36M_19K_SKELETON_3D
        self.UNIT_LENGTH_LIMB = HUMAN_36M_19K_UNIT_LENGTH_LIMB
        self.RIGHT_LIMBS      = HUMAN_36M_19K_RIGHT_LIMBS
        self.LEFT_LIMBS       = HUMAN_36M_19K_LEFT_LIMBS
        self.SKELETON_3D_LENS_AVG_PERSON = AVG_PERSON_19K_SKELETON_3D_LENS
        self.SKELETON_3D_NAMES = HUMAN_36M_19K_SKELETON_NAMES
        self.SKELETON_3D_MASK  = [self.SKELETON_3D_NAMES.index(i) for i in HUMAN_36M_19K_SKELETON_MASK]

    def init_openpose_17k(self):
        self.KEYPOINTS_2D = OPENPOSE_17K_2D
        self.KEYPOINTS_3D = OPENPOSE_17K_3D
        self.ROOT_KPT     = OPENPOSE_17K_ROOT
        self.SKELETON_2D  = OPENPOSE_17K_SKELETON_2D
        self.SKELETON_3D  = OPENPOSE_17K_SKELETON_3D
        self.UNIT_LENGTH_LIMB = OPENPOSE_17K_UNIT_LENGTH_LIMB
        self.RIGHT_LIMBS      = OPENPOSE_17K_RIGHT_LIMBS
        self.LEFT_LIMBS       = OPENPOSE_17K_LEFT_LIMBS
        self.SKELETON_3D_LENS_AVG_PERSON = AVG_PERSON_OPENPOSE_17K_SKELETON_3D_LENS
        self.SKELETON_3D_NAMES = OPENPOSE_17K_SKELETON_NAMES
        self.SKELETON_3D_MASK  = [self.SKELETON_3D_NAMES.index(i) for i in OPENPOSE_17K_SKELETON_MASK]

    def init_shg_16k(self):
        self.KEYPOINTS_2D = HUMAN_36M_16K_2D
        self.KEYPOINTS_3D = HUMAN_36M_16K_3D
        self.ROOT_KPT     = HUMAN_36M_16K_ROOT
        self.SKELETON_2D  = HUMAN_36M_16K_SKELETON_2D
        self.SKELETON_3D  = HUMAN_36M_16K_SKELETON_3D
        self.UNIT_LENGTH_LIMB = HUMAN_36M_16K_UNIT_LENGTH_LIMB
        self.RIGHT_LIMBS      = HUMAN_36M_16K_RIGHT_LIMBS
        self.LEFT_LIMBS       = HUMAN_36M_16K_LEFT_LIMBS
        self.SKELETON_3D_LENS_AVG_PERSON = AVG_PERSON_16K_SKELETON_3D_LENS
        self.SKELETON_3D_NAMES = HUMAN_36M_16K_SKELETON_NAMES
        self.SKELETON_3D_MASK  = [self.SKELETON_3D_NAMES.index(i) for i in HUMAN_36M_16K_SKELETON_MASK]

    def get_skeleton_pairs(self):
        if self.KEYPOINTS_2D != self.KEYPOINTS_3D or \
           self.SKELETON_2D  != self.SKELETON_3D:
            raise NotImplementedError("Currently our method works only if 2d/3d keypoints match.")

        skeleton_pairs_2d = [[self.KEYPOINTS_2D.index(kpt) for kpt in limb] for limb in self.SKELETON_2D]
        skeleton_pairs_3d = [[self.KEYPOINTS_3D.index(kpt) for kpt in limb] for limb in self.SKELETON_3D]

        return skeleton_pairs_2d, skeleton_pairs_3d

    def get_keypoints(self):
        if self.KEYPOINTS_2D != self.KEYPOINTS_3D or \
           self.SKELETON_2D  != self.SKELETON_3D:
            raise NotImplementedError("Currently our method works only if 2d/3d keypoints match.")

        keypoints_2d_indxs = [HUMAN_36M_KEYPOINTS.index(k) for k in self.KEYPOINTS_2D]
        keypoints_3d_indxs = [HUMAN_36M_KEYPOINTS.index(k) for k in self.KEYPOINTS_3D]

        return keypoints_2d_indxs, keypoints_3d_indxs

    def get_skeleton_root_idx(self):
        if self.KEYPOINTS_2D != self.KEYPOINTS_3D or \
           self.SKELETON_2D  != self.SKELETON_3D:
            raise NotImplementedError("Currently our method works only if 2d/3d keypoints match.")

        assert self.ROOT_KPT in self.KEYPOINTS_2D
        assert self.ROOT_KPT in self.KEYPOINTS_3D

        skeleton_root_indx_2d = self.KEYPOINTS_2D.index(self.ROOT_KPT)
        skeleton_root_indx_3d = self.KEYPOINTS_3D.index(self.ROOT_KPT)

        return skeleton_root_indx_2d, skeleton_root_indx_3d
