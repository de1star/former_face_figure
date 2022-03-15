import h5py
import os
import numpy as np
import pickle
from tqdm import tqdm


def main():
    data_types = ['tf_exp', 'tf_pose', 'tf_shape', 'tf_rot', 'tf_trans']
    with h5py.File('../data/mhi_mimicry.hdf5') as f:
        for sessions_idx, sessions_info in tqdm(f['sessions'].items()):
            cur_video_len = sessions_info['participants/P2/tf_pose'].shape[0]
            p1_params_dict = {}
            p2_params_dict = {}
            for data_type in data_types:
                p1_params_dict[data_type] = sessions_info['participants/P1/' + data_type]
                p2_params_dict[data_type] = sessions_info['participants/P2/' + data_type]
            p1_tf_shape = p1_params_dict['tf_shape'][:, :100]
            p1_tf_exp = p1_params_dict['tf_exp'][:, :50]
            p1_tf_pose = p1_params_dict['tf_pose']
            p1_tf_rot = np.concatenate((p1_params_dict['tf_rot'], np.zeros([cur_video_len, 3])), axis=1)
            p1_vector = np.concatenate((p1_tf_shape, p1_tf_exp, p1_tf_rot, p1_tf_pose[:, :3], p1_tf_pose[:, 3:9]),
                                       axis=1)

            p2_tf_shape = p2_params_dict['tf_shape'][:, :100]
            p2_tf_exp = p2_params_dict['tf_exp'][:, :50]
            p2_tf_pose = p2_params_dict['tf_pose']
            p2_tf_rot = np.concatenate((p2_params_dict['tf_rot'], np.zeros([cur_video_len, 3])), axis=1)
            p2_vector = np.concatenate((p2_tf_shape, p2_tf_exp, p2_tf_rot, p2_tf_pose[:, :3], p2_tf_pose[:, 3:9]),
                                       axis=1)
            data_pair = {'input': p1_vector, 'output': p2_vector}
            if not os.path.exists('../data/train'):
                os.makedirs('../data/train')
            if not os.path.exists('../data/valid'):
                os.makedirs('../data/valid')
            if not os.path.exists('../data/test'):
                os.makedirs('../data/test')
            if int(sessions_idx) > 4:
                with open(f'../data/train/{sessions_idx}.pkl', 'wb') as data_f:
                    pickle.dump(data_pair, data_f)
            elif int(sessions_idx) > 2:
                with open(f'../data/valid/{sessions_idx}.pkl', 'wb') as data_f:
                    pickle.dump(data_pair, data_f)
            else:
                with open(f'../data/test/{sessions_idx}.pkl', 'wb') as data_f:
                    pickle.dump(data_pair, data_f)


if __name__ == '__main__':
    main()