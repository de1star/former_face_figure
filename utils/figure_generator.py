import h5py
import torch
import numpy as np
import trimesh
import pyrender
import pickle
import os
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
import sys
sys.path.append("..")
from FLAME import FLAME
from config import get_config
from others import utils


class Figure_Generator():
    def __init__(self):
        config = get_config()
        self.flamelayer = FLAME(config)
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0)
        self.camera_pose = utils.create_pose(0, 0, 0,
                                             0, -0.04, 0.35)

    def frames_to_video(self, frame_path, prefix):
        save_dir = f'../demos/videos/{prefix}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imgs = [img for img in os.listdir(frame_path)]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 60
        video_width, video_height = 1920, 1080
        videoWriter = cv2.VideoWriter(save_dir, fourcc, fps, (video_width, video_height))


    def generate(self, flame_vectors, prefix):
        save_dir = f'../demos/frames/{prefix}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in tqdm(range(flame_vectors.shape[0])):
            tf_shape = flame_vectors[i, :100]
            tf_exp = flame_vectors[i, 100:150]
            tf_rot = flame_vectors[i, 150:156]
            tf_pose = flame_vectors[i, 156:]
            vertice, landmark = self.flamelayer(shape_params=torch.from_numpy(tf_shape).float().unsqueeze(0),
                                                expression_params=torch.from_numpy(tf_exp).float().unsqueeze(0),
                                                pose_params=torch.from_numpy(tf_rot).float().unsqueeze(0),
                                                neck_pose=torch.from_numpy(tf_pose[0:3]).float().unsqueeze(0),
                                                eye_pose=torch.from_numpy(tf_pose[3:9]).float().unsqueeze(0))
            faces = self.flamelayer.faces
            vertices = vertice[0].detach().cpu().numpy().squeeze()
            joints = landmark[0].detach().cpu().numpy().squeeze()
            vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

            tri_mesh = trimesh.Trimesh(vertices, faces,
                                       vertex_colors=vertex_colors)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            # scene.clear()
            scene = pyrender.Scene()
            light = pyrender.SpotLight(color=np.ones(3), intensity=0.7,
                                       innerConeAngle=np.pi / 16.0,
                                       outerConeAngle=np.pi / 6.0)
            scene.add(mesh)
            scene.add(self.camera, pose=self.camera_pose)
            scene.add(light, pose=self.camera_pose)
            r = pyrender.OffscreenRenderer(200, 200)
            color, _ = r.render(scene)
            idx = str(i).rjust(6, '0')
            plt.imsave(f'{save_dir}/{idx}.jpg', color)
            r.delete()


def test1():
    generator = Figure_Generator()
    data_path = '../data/mhi_mimicry.hdf5'
    data_types = ['tf_exp', 'tf_pose', 'tf_shape', 'tf_rot', 'tf_trans']
    with h5py.File(data_path, 'r') as f:
        for sessions_idx, sessions_info in tqdm(f['sessions'].items()):
            if sessions_idx != '1':
                continue
            p1_params_dict = {}
            for data_type in data_types:
                p1_params_dict[data_type] = sessions_info['participants/P1/' + data_type]
            break
        tf_shape = p1_params_dict['tf_shape'][:, :100]
        tf_exp = p1_params_dict['tf_exp'][:, :50]
        tf_pose = p1_params_dict['tf_pose']
        tf_rot = np.concatenate((p1_params_dict['tf_rot'], np.zeros([tf_shape.shape[0], 3])), axis=1)  # , p1_params_dict['tf_rot'][0]))
        vectors = np.concatenate((tf_shape, tf_exp, tf_rot, tf_pose[:, :3], tf_pose[:, 3:9]), axis=1)
        generator.generate(vectors)

def test2():
    generator = Figure_Generator()
    with open('../data/valid/3.pkl', 'rb') as f:
        data = pickle.load(f)
    generator.generate(data['input'], 3)


if __name__ == '__main__':
    test2()
