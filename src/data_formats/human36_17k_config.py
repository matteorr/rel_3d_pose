
pose_config = {

    'KEYPOINT_NAMES':
        ['mid_hip',
         'right_hip','right_knee', 'right_ankle',
         'left_hip', 'left_knee', 'left_ankle',
         'mid_spine',
         'neck', 'nose', 'head',
         'left_shoulder', 'left_elbow', 'left_wrist',
         'right_shoulder', 'right_elbow', 'right_wrist'],

    'KEYPOINT_COLORS':
        ['#4286f4',
         '#27a30b', '#27a30b', '#27a30b',
         '#9e1809', '#9e1809', '#9e1809',
         '#4286f4',
         '#4286f4', '#4286f4', '#4286f4',
         '#9e1809', '#9e1809', '#9e1809',
         '#27a30b', '#27a30b', '#27a30b'],

    'SKELETON_NAMES':
        [['neck','nose'], ['nose','head'],
         ['neck','right_shoulder'], ['neck','left_shoulder'],
         ['right_shoulder','right_elbow'], ['right_elbow','right_wrist'],
         ['left_shoulder','left_elbow'], ['left_elbow','left_wrist'],
         ['neck','mid_spine'], ['mid_spine','mid_hip'],
         ['right_hip','mid_hip'], ['left_hip','mid_hip'],
         ['right_hip','right_knee'], ['right_knee','right_ankle'],
         ['left_hip','left_knee'], ['left_knee','left_ankle']],

    'SKELETON_COLORS':
        ['#a00899', '#a00899',
         '#ffe20c', '#ffe20c',
         '#0c99ff', '#0c99ff',
         '#0c99ff', '#0c99ff',
         '#ffe20c', '#ffe20c',
         '#ffe20c', '#ffe20c',
         '#a57e60', '#a57e60',
         '#a57e60', '#a57e60']

}

pose_config_br = {

    'KEYPOINT_NAMES':
        ['mid_hip',
         'right_hip','right_knee', 'right_ankle',
         'left_hip', 'left_knee', 'left_ankle',
         'mid_spine',
         'neck', 'nose', 'head',
         'left_shoulder', 'left_elbow', 'left_wrist',
         'right_shoulder', 'right_elbow', 'right_wrist'],

    'KEYPOINT_COLORS':
        ['#37DB34',
         '#37DB34', '#37DB34', '#37DB34',
         '#37DB34', '#37DB34', '#37DB34',
         '#37DB34',
         '#37DB34', '#37DB34', '#37DB34',
         '#37DB34', '#37DB34', '#37DB34',
         '#37DB34', '#37DB34', '#37DB34'],

    'SKELETON_NAMES':
        [['neck','nose'], ['nose','head'],
         ['neck','right_shoulder'], ['neck','left_shoulder'],
         ['right_shoulder','right_elbow'], ['right_elbow','right_wrist'],
         ['left_shoulder','left_elbow'], ['left_elbow','left_wrist'],
         ['neck','mid_spine'], ['mid_spine','mid_hip'],
         ['right_hip','mid_hip'], ['left_hip','mid_hip'],
         ['right_hip','right_knee'], ['right_knee','right_ankle'],
         ['left_hip','left_knee'], ['left_knee','left_ankle']],

    'SKELETON_COLORS':
        ['#EC4C3C', '#EC4C3C',
         '#3798DB', '#EC4C3C',
         '#3798DB', '#3798DB',
         '#EC4C3C', '#EC4C3C',
         '#EC4C3C', '#EC4C3C',
         '#3798DB', '#EC4C3C',
         '#3798DB', '#3798DB',
         '#EC4C3C', '#EC4C3C']

}

pose_config['SKELETON_IDXS'] = \
        [[pose_config['KEYPOINT_NAMES'].index(k[0]),
          pose_config['KEYPOINT_NAMES'].index(k[1])] for k in pose_config['SKELETON_NAMES']]

pose_config_br['SKELETON_IDXS'] = \
        [[pose_config_br['KEYPOINT_NAMES'].index(k[0]),
          pose_config_br['KEYPOINT_NAMES'].index(k[1])] for k in pose_config_br['SKELETON_NAMES']]
