
pose_config = {

    'KEYPOINT_NAMES':
        ['head', 'chin', 'neck',
         'right_shoulder', 'right_elbow', 'right_wrist',
         'left_shoulder',  'left_elbow',  'left_wrist',
         'mid_spine', 'mid_hip',
         'right_hip', 'right_knee', 'right_ankle', 'right_foot_tip',
         'left_hip',  'left_knee',  'left_ankle',  'left_foot_tip'],

    'KEYPOINT_COLORS':
        ['#4286f4','#4286f4', '#4286f4',
         '#27a30b','#27a30b', '#27a30b',
         '#9e1809','#9e1809', '#9e1809',
         '#4286f4', '#4286f4',
         '#27a30b', '#27a30b', '#27a30b', '#27a30b',
         '#9e1809','#9e1809','#9e1809','#9e1809'],

    'SKELETON_NAMES':
        [['mid_hip', 'right_hip'],
         ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'], ['right_ankle', 'right_foot_tip'],
         ['mid_hip', 'left_hip'],
         ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],  ['left_ankle', 'left_foot_tip'],
         ['mid_hip', 'mid_spine'], ['mid_spine', 'neck'],
         ['neck', 'chin'], ['chin', 'head'],
         ['neck', 'left_shoulder'],
         ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
         ['neck', 'right_shoulder'],
         ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist']],

    'SKELETON_COLORS':
        ['#ffe20c',
         '#a57e60', '#a57e60', '#a57e60',
         '#ffe20c',
         '#a57e60', '#a57e60', '#a57e60',
         '#ffe20c', '#ffe20c',
         '#a00899', '#a00899',
         '#ffe20c',
         '#0c99ff', '#0c99ff',
         '#ffe20c',
         '#0c99ff', '#0c99ff']

}

pose_config['SKELETON_IDXS'] = \
        [[pose_config['KEYPOINT_NAMES'].index(k[0]),
          pose_config['KEYPOINT_NAMES'].index(k[1])] for k in pose_config['SKELETON_NAMES']]
