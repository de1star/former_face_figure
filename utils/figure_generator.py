import h5py
import torch
import numpy as np
import trimesh
import pyrender

from tqdm import tqdm
import sys
sys.path.append("..")
from FLAME import FLAME
from config import get_config
from others import utils


def test():
    data_path = '../data/mhi_mimicry.hdf5'
    data_types = ['tf_exp', 'tf_pose', 'tf_shape', 'tf_rot', 'tf_trans']
    config = get_config()
    flamelayer = FLAME(config)
    with h5py.File(data_path, 'r') as f:
        frame = 500
        for sessions_idx, sessions_info in tqdm(f['sessions'].items()):
            if sessions_idx != '1':
                continue
            cur_video_len = sessions_info['participants/P2/tf_pose'].shape[0]
            p1_params_dict = {}
            p2_params_dict = {}
            for data_type in data_types:
                p1_params_dict[data_type] = sessions_info['participants/P1/' + data_type]
                p2_params_dict[data_type] = sessions_info['participants/P2/' + data_type]
            print(cur_video_len)
            print(p1_params_dict)
            print(p2_params_dict)
            break
        tf_shape = p1_params_dict['tf_shape'][frame][:100]
        tf_exp = p1_params_dict['tf_exp'][frame][:50]
        tf_pose = p1_params_dict['tf_pose'][frame]
        tf_rot = np.concatenate((p1_params_dict['tf_rot'][frame], np.zeros(3)))#, p1_params_dict['tf_rot'][0]))
        vertice, landmark = flamelayer(shape_params=torch.from_numpy(tf_shape).float().unsqueeze(0),
                                       expression_params=torch.from_numpy(tf_exp).float().unsqueeze(0),
                                       pose_params=torch.from_numpy(tf_rot).float().unsqueeze(0),
                                       neck_pose=torch.from_numpy(tf_pose[0:3]).float().unsqueeze(0),
                                       eye_pose=torch.from_numpy(tf_pose[3:9]).float().unsqueeze(0))
        print(vertice.size(), landmark.size())
        faces = flamelayer.faces
        vertices = vertice[0].detach().cpu().numpy().squeeze()
        joints = landmark[0].detach().cpu().numpy().squeeze()
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

        tri_mesh = trimesh.Trimesh(vertices, faces,
                                   vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0)
        camera_pose = utils.create_pose(0, 0, 0,
                                        0, -0.04, 0.35)
        scene.add(mesh)
        scene.add(camera, pose=camera_pose)
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        #scene.add(joints_pcl)
        pyrender.Viewer(scene, use_raymond_lighting=True,
                        viewer_flags={#'rotate':True,
                                      'show_world_axis': True,
                                      'show_mesh_axes': True},)


if __name__ == '__main__':
    test()
