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

    def generate(self, flame_vectors, save_dir, shape=None, exp=None, rot=None, pose=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in tqdm(range(flame_vectors.shape[0])):
            if shape is not None:
                tf_shape = shape
            else:
                tf_shape = flame_vectors[i, :100]
            if exp is not None:
                tf_exp = exp
            else:
                tf_exp = flame_vectors[i, 100:150]
            if rot is not None:
                tf_rot = rot
            else:
                tf_rot = flame_vectors[i, 150:153]
                tf_rot = np.concatenate((tf_rot, np.zeros(3)))
            if pose is not None:
                tf_pose = pose
            else:
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


def test2():
    generator = Figure_Generator()
    path = 'D:\projects\\former_face_figure\demos\\tensors\\1'
    save_path = 'D:\projects\\former_face_figure\demos\\frames\\1'
    file_list = os.listdir(path)
    for file in file_list[2:]:
        data_path = f"{path}\\{file}"
        save_path = f"{save_path}\\{file}"
        with open(data_path, 'rb') as f:
            data = torch.load(f)[0]
            data = np.array(data)
        generator.generate(data, save_path)

def test3():
    img_path = os.listdir("D:\projects\\former_face_figure\demos\\frames\\1")
    frame = cv2.imread("D:\projects\\former_face_figure\demos\\frames\\1\input.pt\\000000.jpg")
    heigth, width, layers = frame.shape
    path1 = "D:\projects\\former_face_figure\demos\\frames\\1\input.pt"
    path2 = "D:\projects\\former_face_figure\demos\\frames\\1\my_output.pt"
    path3 = "D:\projects\\former_face_figure\demos\\frames\\1\\true_output.pt"
    width *= 3
    video_path = "D:\projects\\former_face_figure\demos\\video"
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    video_name = "1.avi"
    video = cv2.VideoWriter(f"{video_path}\\{video_name}", 0, 25, (width, heigth))
    for i in range(5000):
        idx = str(i).rjust(6, '0')
        img_name = f"{idx}.jpg"
        img1 = cv2.imread(f"{path1}\\{img_name}")
        img2 = cv2.imread(f"{path2}\\{img_name}")
        img3 = cv2.imread(f"{path3}\\{img_name}")
        cat_img = np.concatenate([img1, img2, img3], axis=1)
        video.write(cat_img)
    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    test3()
