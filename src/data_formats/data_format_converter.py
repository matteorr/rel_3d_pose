import numpy as np

class DataFormatConverter():

    def __init__(self, pose_config_source, pose_config_target):
        self.pose_config_source = pose_config_source
        self.pose_config_target = pose_config_target

        self.num_source_kpts = len(self.pose_config_source['KEYPOINT_NAMES'])
        self.num_target_kpts = len(self.pose_config_target['KEYPOINT_NAMES'])

        # assert that at all keypoints in target are contained in source
        self.index_map = []
        err_str = "Bad data formats. All target keypoints must be in source!"
        for k in self.pose_config_target['KEYPOINT_NAMES']:
            assert k in self.pose_config_source['KEYPOINT_NAMES'], err_str
            self.index_map.append(self.pose_config_source['KEYPOINT_NAMES'].index(k))

    def source_to_target(self, source_pose_array, dim=2):
        if len(source_pose_array.shape) == 1:
            len_arrays = len(source_pose_array)
            num_arrays = 1
            source_pose_array = source_pose_array[np.newaxis,...]
        else:
            num_arrays, len_arrays = source_pose_array.shape

        err_str = "Bad input. Array doesn't match source data format length."
        assert len_arrays == self.num_source_kpts * dim, err_str

        # transform an array in the source format to the target format
        if dim == 1:
            return source_pose_array[:,self.index_map]

        else:
            t_x = source_pose_array[:,0::dim][:,self.index_map]
            t_y = source_pose_array[:,1::dim][:,self.index_map]

            if dim == 3:
                t_z = source_pose_array[:,2::dim][:,self.index_map]
                target_pose_array = np.vstack((t_x,t_y,t_z)).reshape((-1,self.num_target_kpts * dim),order='F')

            elif dim == 2:
                target_pose_array = np.vstack((t_x,t_y)).reshape((-1,self.num_target_kpts * dim),order='F')

            else:
                raise ValueError("Dimension has to be one of: [1,2,3].")

        return target_pose_array


    def target_to_source(self, target_pose_array, dim=2):
        if len(target_pose_array.shape) == 1:
            len_arrays = len(target_pose_array)
            num_arrays = 1
            target_pose_array = target_pose_array[np.newaxis,...]
        else:
            num_arrays, len_arrays = target_pose_array.shape

        err_str = "Bad input. Array doesn't match target data format length."
        assert len_arrays == self.num_target_kpts * dim, err_str

        inv = np.nan # -1
        source_pose_array = inv * np.ones((num_arrays, dim * self.num_source_kpts))

        if dim == 1:
            source_pose_array[:, self.index_map] = target_pose_array

        elif dim in [2,3]:
            # transform an array in the target format to the source format
            s_x = inv * np.ones((num_arrays, self.num_source_kpts))
            s_x[:, self.index_map] = target_pose_array[:, 0::dim]
            source_pose_array[:, 0::dim] = s_x

            s_y = inv * np.ones((num_arrays, self.num_source_kpts))
            s_y[:, self.index_map] = target_pose_array[:, 1::dim]
            source_pose_array[:, 1::dim] = s_y

            if dim == 3:
                s_z = inv * np.ones((num_arrays, self.num_source_kpts))
                s_z[:, self.index_map] = target_pose_array[:, 2::dim]
                source_pose_array[:, 2::dim] = s_z

        else:
            raise ValueError("Dimension has to be one of: [1,2,3].")

        return source_pose_array
