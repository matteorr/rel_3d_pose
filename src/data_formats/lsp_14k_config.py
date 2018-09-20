pose_config = {

    'KEYPOINT_NAMES':
        ['right_ankle','right_knee','right_hip',
         'left_hip', 'left_knee', 'left_ankle',
         'right_wrist', 'right_elbow', 'right_shoulder',
         'left_shoulder', 'left_elbow', 'left_wrist',
         'neck', 'nose'],

    'KEYPOINT_COLORS':
        ['#27a30b', '#27a30b', '#27a30b',
         '#9e1809', '#9e1809', '#9e1809',
         '#27a30b', '#27a30b', '#27a30b',
         '#9e1809', '#9e1809', '#9e1809',
         '#4286f4', '#4286f4'],

    'SKELETON_NAMES':
        [['nose','neck'], ['neck','right_shoulder'], ['neck','left_shoulder'],
         ['right_shoulder','right_elbow'], ['right_elbow','right_wrist'],
         ['left_shoulder','left_elbow'], ['left_elbow','left_wrist'],
         ['right_shoulder','right_hip'], ['left_shoulder','left_hip'],
         ['right_hip','left_hip'],
         ['right_hip','right_knee'], ['right_knee','right_ankle'],
         ['left_hip','left_knee'], ['left_knee','left_ankle']],

    'SKELETON_COLORS':
        ['#a00899', '#ffe20c', '#ffe20c',
         '#0c99ff', '#0c99ff',
         '#0c99ff', '#0c99ff',
         '#ffe20c', '#ffe20c',
         '#ffe20c',
         '#a57e60', '#a57e60',
         '#a57e60', '#a57e60']

}

pose_config['SKELETON_IDXS'] = \
        [[pose_config['KEYPOINT_NAMES'].index(k[0]),
          pose_config['KEYPOINT_NAMES'].index(k[1])] for k in pose_config['SKELETON_NAMES']]
