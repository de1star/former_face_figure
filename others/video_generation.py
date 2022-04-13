import numpy as np
from FLAME_PyTorch.FLAME import FLAME
import pyrender
import trimesh
import matplotlib.pyplot as plt
import torch
import cv2
import os
from tqdm import tqdm
from others import utils
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# from OpenGL import EGL
# print(os.environ['PYOPENGL_PLATFORM'])


device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
def denormalize( data, config, dataloader):

    start_dim = 0
    result = {}
    for feature in dataloader.dataset.select_features:
        end_dim = getattr(config, f"{feature}_params")
        std = torch.from_numpy(dataloader.dataset.data[feature]["std"][:end_dim].astype(np.float32)).to(device)
        mean = torch.from_numpy(dataloader.dataset.data[feature]["mean"][:end_dim].astype(np.float32)).to(device)
        result[feature] = data[:,start_dim:start_dim +end_dim ] *std +mean
        start_dim+=end_dim
    return result


def flame_generation(result, config, pair_idx=None):
    expression_seq = result["expression"]
    jaw_pose_seq = result["jaw_pose"]
    rotation_seq = result["rotation"]
    flamelayer = FLAME(config)
    flamelayer.to(device)
    # shape_params = mean shapes
    shape_params = torch.zeros(config.FLAME_batch_size, 100).to(device)
    # shape_params = torch.randn(100).unsqueeze(0).repeat(FLAME_batch_size, 1).to(device)
    # global rotation
    # global_rotation = torch.zeros((FLAME_batch_size, 3)).to(device)
    frame_idx = 0
    for start_time in tqdm(range(0, jaw_pose_seq.shape[0] - config.FLAME_batch_size + 1, config.FLAME_batch_size), leave=False):
        # Creating a batch of global poses and expressions
        # neck_pose
        neck_pose = torch.zeros(config.FLAME_batch_size, 3).to(device)
        # jaw_pose
        jaw_pose = jaw_pose_seq[start_time:start_time + config.FLAME_batch_size]
        # expression_params
        expression_params = expression_seq[start_time:start_time + config.FLAME_batch_size]
        # expression_params = torch.zeros((FLAME_batch_size, 50)).to(device)#expression_seq[start_time:start_time + FLAME_batch_size]
        # global rotation
        global_rotation = rotation_seq[start_time:start_time + config.FLAME_batch_size]
        # global_rotation = pose_params[:, :3]
        # shape_params
        # shape_params = shape_seq[start_time:start_time + FLAME_batch_size]
        # Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework
        vertice, landmark = flamelayer(shape_params=shape_params,
                                       expression_params=expression_params,
                                       global_rot=global_rotation,
                                       neck_pose=neck_pose,
                                       jaw_pose=jaw_pose)  # For RingNet project
        # Visualize Landmarks
        # This visualises the static landmarks and the pose dependent dynamic landmarks used for RingNet project
        faces = flamelayer.faces
        for time_idx in range(config.FLAME_batch_size):
            vertices = vertice[time_idx].detach().cpu().numpy().squeeze()
            vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
            tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            scene = pyrender.Scene()
            light = pyrender.SpotLight(color=np.ones(3), intensity=0.7,
                                       innerConeAngle=np.pi / 16.0,
                                       outerConeAngle=np.pi / 6.0)
            # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
            camera_pose = utils.create_pose(0 ,0 ,0,
                                            0 ,-0.04 ,0.35)
            scene.add(mesh)
            scene.add(camera, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            r = pyrender.OffscreenRenderer(200, 200)
            color, _ = r.render(scene)
            if pair_idx is None:
                image_name = f"{frame_idx}.jpg"
            else:
                image_name = f"{frame_idx}.{pair_idx}.jpg"
            plt.imsave(os.path.join(config.temp_folder ,image_name), color)
            r.delete()
            frame_idx += 1



def save_video(video_name, config, fps, output_path, pair=False):
    images = [img for img in os.listdir(config.temp_folder) if not img.endswith(".1.jpg")]
    images.sort(key=lambda name: int(name.split(".")[0]))
    frame = cv2.imread(os.path.join(config.temp_folder, images[0]))
    height, width, layers = frame.shape
    if pair:
        images2 = [img for img in os.listdir(config.temp_folder) if img.endswith(".1.jpg")]
        images2.sort(key=lambda name: int(name.split(".")[0]))
        width *= 2

    # TODO: link fps
    if output_path is None:
        output_name = f'{config.resume_iters}'
        output_name += f'_randZ' if config.random_z else f'_calZ'
        output_folder = os.path.join(config.output_folder, config.result_dir, output_name)
    else:
        output_folder = output_path
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoWriter(os.path.join(output_folder,video_name), 0, fps, (width, height))
    for image, image2 in zip(images, images2):
        cat_image = np.concatenate([cv2.imread(os.path.join(config.temp_folder, image)),
                                    cv2.imread(os.path.join(config.temp_folder, image2))],axis=1)
        video.write(cat_image)
    cv2.destroyAllWindows()
    video.release()

def generate_video(flame_seq, name, config, data_loader, output_path = None):
    flame_seq = denormalize(flame_seq.squeeze(dim=1).permute((1, 0)), config, data_loader)
    flame_generation(flame_seq, config)
    save_video(name, config,data_loader.dataset.data["info"]["fps"], output_path)

def generate_video_pair(flame_seq1,flame_seq2, name, config, data_loader, output_path = None):
    flame_seq1 = denormalize(flame_seq1.squeeze(dim=1).permute((1, 0)), config, data_loader)
    flame_seq2 = denormalize(flame_seq2.squeeze(dim=1).permute((1, 0)), config, data_loader)
    flame_generation(flame_seq1, config)
    flame_generation(flame_seq2, config, 2)
    save_video(name, config,data_loader.dataset.data["info"]["fps"], output_path, pair=True)