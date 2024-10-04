import open3d as o3d
import os
import sys
sys.path.append("./get_pcd/mast3r")
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R

from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View1, focal2fov, fov2focal
import numpy as np
import json
from utils.sh_utils import SH2RGB

from utils.dust3r_pcd_alignment import Pcd_Global_Alignment

from utils.graphics_utils import fov2focal
from dust3r.cloud_opt import fast_pnp, estimate_focal, global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images, rgb
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs

from sfm_free_utils.optimization_sRT import *
from sfm_free_utils.utils import *
import torch.nn.functional as F

import numpy as np
from glob import glob
import torch
import math
import cv2
import re
import time
import torchvision.transforms as transforms

MODEL_PATH = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
Mast3R_MODEL_PATH = "./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

def tensor_to_pil_image(tensor):
    """
    将 PyTorch Tensor 转换为 PIL.Image 对象
    :param tensor: 输入的 Tensor 对象 (C, H, W)
    :return: 转换后的 PIL.Image 对象
    """
    tensor = tensor.squeeze(0).detach().cpu()  
    tensor = tensor.permute(1, 2, 0).numpy() 
    tensor = (tensor * 255).astype(np.uint8)  
    return Image.fromarray(tensor)

def find_valid_matches(output, device="cuda"):

    desc1 = output['pred1']['desc'].squeeze(0).detach()
    desc2 = output['pred2']['desc'].squeeze(0).detach()
    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = output['view1']['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = output['view2']['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    return matches_im0, matches_im1

def parser_res_mast3r(output):

    view1_pts3d = output['pred1']['pts3d']
    view1_conf = output['pred1']['conf']
    view1_img = output['view1']['img']

    view2_pts3d = output['pred2']['pts3d_in_other_view']
    view2_conf = output['pred2']['conf']
    view2_img = output['view2']['img']
    # save view1 and view2 as image
    view1_img_pil = tensor_to_pil_image(view1_img)
    view1_img_pil.save("./results/output_view1.png")
    view2_img_pil = tensor_to_pil_image(view2_img)
    view2_img_pil.save("./results/output_view2.png")
    matches_im0, matches_im1 = find_valid_matches(output)
    
    return view1_pts3d, view1_conf, view1_img, view2_pts3d, view2_conf, view2_img, matches_im0, matches_im1
    
def readSFMFreeSceneInfo(dataset_path, min_conf_thr=3, img_fov=False, focal_known=907.4083251953125, scale=20, device='cuda', fps=1, fast=False):
    # instance model
    model = AsymmetricCroCo3DStereo.from_pretrained(MODEL_PATH).to(device)
    while True:
        img_fps = sorted(glob(f'{dataset_path}/*'), key=natural_sort_key)
        img_fps = img_fps[::fps]
        if len(img_fps) > 1:
            start_time = time.time()
            print("start time: ", start_time)
            break 
 

    split_img = []
    for i in range(len(img_fps) - 1):
        
        split_img.append([img_fps[i], img_fps[i+1]])
    
    # record inference result 
    infer_result = []
    
    for img_pair in split_img:
        images, _ = load_images(img_pair, size=512)

        pairs = make_pairs(images, scene_graph='complete', symmetrize=True)
        output = inference(pairs=pairs, model=model, device=device, batch_size=1)
        infer_result.append(output)
        
    del model
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_mask = scene.get_masks()

    pts3d = [pts3d[i].detach().cpu().numpy() for i in range(len(pts3d))]
    poses = [poses[i].detach().cpu().numpy() for i in range(len(poses))]
    return pts3d, imgs, poses

            
def readSFMFreeSceneInfo_v1(dataset_path, min_conf_thr=3, img_fov=False, focal_known=907.4083251953125, device='cuda', fps=1, fast=False):
    # instance model
    model = AsymmetricMASt3R.from_pretrained(Mast3R_MODEL_PATH).to(device)
    while True:
        img_fps = sorted(glob(f'{dataset_path}/*'), key=natural_sort_key)
        img_fps = img_fps[::fps]
        if len(img_fps) > 1:
            start_time = time.time()
            print("start time: ", start_time)
            break   

    split_img = []
    for i in range(len(img_fps) - 1):
        split_img.append([img_fps[i], img_fps[i+1]])
    
    # record inference result 
    infer_result = []
    
    for img_pair in split_img:
        images, _ = load_images(img_pair, size=512, square_ok=True)
        pairs = make_pairs(images, scene_graph='complete', symmetrize=False)

        output = inference(pairs=pairs, model=model, device=device, batch_size=1)
        infer_result.append(output)
        
    del model
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_mask = scene.get_masks()

    pts3d = [pts3d[i].detach().cpu().numpy() for i in range(len(pts3d))]
    poses = [poses[i].detach().cpu().numpy() for i in range(len(poses))]
    return pts3d, imgs, poses
   

def project_pcd_to_depth(pcds, focals, poses):
    '''
    project:
    camera_pcd = w2c @ pcds
    img_pcd = K @ camera_pcd
    '''
    def project_pcd_to_depth_single(pcd, focal, pose):
        h, w = pcd.shape[:2]
        world_pcd = pcd.reshape(-1, 3)
        suffix = np.ones(world_pcd.shape[0])[:, None]
        world_pcd_homogeneous = np.hstack([world_pcd, suffix])

        camrea_pcd_homogeneous = (np.linalg.inv(pose) @ world_pcd_homogeneous.T).T

        K = np.array([
            [focal, 0, w/2],
            [0, focal, h/2],
            [0, 0, 1]
        ])

        image_pcd = (K @ camrea_pcd_homogeneous[:, :3].T).T
        
        z = image_pcd[:, 2:3].reshape((h, w))

        return z
    
    depths = []
    for pcd, focal, pose in zip(pcds, focals, poses): # seq
        depth = []
        for i, j in zip(pcd, pose): # pairs
            depth.append(project_pcd_to_depth_single(i, focal, j)[None])
        depths.append(np.concatenate(depth))

    return depths

def progressPair(img_pair, org_h, org_w, model, focal_known, img_fov=False, device='cuda',):
    images, _ = load_images(img_pair, size=512, square_ok=True)
    h, w = images[0]['true_shape'][0]

    pairs = make_pairs(images, scene_graph='oneref', symmetrize=False)
    output = inference(pairs=pairs, model=model, device=device, batch_size=1)
    view1_pts3d, view1_conf, view1_img, view2_pts3d, view2_conf, view2_img, matches_im0, matches_im1 = parser_res_mast3r(output)
    pts3d = torch.cat([view1_pts3d, view2_pts3d]) 
    conf = torch.cat([view1_conf, view2_conf])
    imgs = np.concatenate([rgb(view1_img), rgb(view2_img)])
    pose = []
    focal = []
    
    for v, m, match in zip(pts3d, conf, (matches_im0, matches_im1)):
        msk = torch.zeros_like(m).type(torch.bool)
        x_coords = match[:, 0]
        y_coords = match[:, 1]
        msk[y_coords, x_coords] = True

        if img_fov:
            im_focals = fov2focal(math.radians(img_fov), 512)
        elif focal_known:
            im_focals = float(focal_known) * (w / org_w)
        else:
            im_focals = estimate_focal(v)   
        try:
            f, P = fast_pnp(v, im_focals, msk=msk, device=device, niter_PnP=10)
            pose.append(P[None].cpu())
            focal.append(torch.tensor([[f]]))
        except:
            try:
                pose.append(pose[-1])
            except:
                pose.append(torch.eye(4)[None])

            focal.append(torch.tensor([[im_focals]]))
            
    pose = torch.cat(pose)
    focal = torch.cat(focal)
    focal = focal.mean() * (org_w / w)
    top_thr = torch.topk(conf.reshape(-1), int(conf.reshape(-1).shape[0] * 0.8)).values.min()
    c2ws = pose.numpy()
    focals = focal.numpy()
    pts3ds = pts3d.numpy()
    dense_mask = (conf > top_thr).numpy()
    colors = imgs

    depth = project_pcd_to_depth([pts3ds], [focals * (w / org_w)], [c2ws])

    return c2ws, focals, pts3ds, dense_mask, colors, depth

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def get_transmarix(s, R, T):
    scale = s[0]
    r = R[0]
    t = T[0]
    M = get_transform(s=scale, R=r, T=t)
    return M

def get_global_pose(pose, global_pose_srt, global_pose_r):
    new_pose_r = global_pose_r @ pose[:3, :3]
    new_pose_t = global_pose_srt @ pose[: ,-1]
    new_pose= np.eye(4)
    new_pose[:3, :3] = new_pose_r
    new_pose[:, -1] = new_pose_t
    return new_pose

def apply_transformation(points, transformation_matrix):
    assert points.shape[1] == 3, "Points must have 3 coordinates"
    assert transformation_matrix.shape == (4, 4), "Transformation matrix must be 4x4"
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]

def find_rotation_matrix(A, B):
    H = A @ B.T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def get_mast3r_to_world(global_pose, gt_pose):
    c1 = global_pose[0]
    c2 = global_pose[1]
    g1 = gt_pose[0]
    g2 = gt_pose[1]
    c1_t = c1[:3, -1]
    c2_t = c2[:3, -1]
    g1_t = g1[:3, -1]
    g2_t = g2[:3, -1]   
    scale = np.linalg.norm(g2_t - g1_t) / np.linalg.norm(c2_t - c1_t)
    c1_r = c1[:3, :3]
    g1_r = g1[:3, :3]
    R = find_rotation_matrix(c1_r, g1_r)

    c1_t = R @(c1_t*scale)
    c2_t = R @(c2_t*scale)
    t1 = g1_t - c1_t
    t2 = g2_t - c2_t
    if np.linalg.norm(t1)-np.linalg.norm(t2) > 0.01:
        print("Warning: t1 and t2 not equal")
    t = (t1 + t2) / 2
    M = np.eye(4)
    M[:3, :3] = scale*R
    M[:3, -1] = t
    return M, scale, R, t

def _compute_translation_3D(A, B, R):
    """Compute the translation between A and B given the rotation matrix R"""
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    t = -R @ centroid_A + centroid_B

    return t

def get_mast3r_to_world_optimer(global_pose, gt_pose):
    global_homeJ = global_pose[0]
    gt_homeJ = gt_pose[0]
    rotation = gt_homeJ[:3, :3] @ np.linalg.inv(global_homeJ)[:3, :3]
    scales = []
    for i in range(1, len(global_pose)):
        global_t = global_pose[i][:3, -1]
        gt_t = gt_pose[i][:3, -1]

        global_dist = np.linalg.norm(global_homeJ[:3, -1] - global_t)
        gt_dist = np.linalg.norm(gt_homeJ[:3, -1] - gt_t)
        scales.append(global_dist / gt_dist)

    scale = np.mean(scales)
    scale = 1 / scale
    global_pts = []
    real_pts = []
    for i in range(1, len(global_pose)):
        global_t = global_pose[i][:3, -1]
        gt_t = gt_pose[i][:3, -1]
        global_pts.append(global_t)
        real_pts.append(gt_t)

    global_pts = np.array(global_pts) * scale
    real_pts = np.array(real_pts)
    translation = _compute_translation_3D(global_pts.T, real_pts.T, rotation).reshape(
        (3,)
    )
    M = np.eye(4)
    M[:3, :3] = scale*rotation
    M[:3, -1] = translation
    return M, scale, rotation, translation


def visualize_multiple_batches(pts_batch, color_batch, transformation_matrices):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="点云可视化", width=2400, height=1600, left=500, top=500)

    for i, (pts, color, transformation_matrix) in enumerate(zip(pts_batch, color_batch, transformation_matrices)):
        b, w, h, _ = pts.shape
        pts = np.array(pts)
        color = np.array(color)
        pts = pts.reshape(b*w*h, 3).astype(np.float64)
        color = color.reshape(b*w*h, 3).astype(np.float64)
        transformed_points = apply_transformation(pts, transformation_matrix)
        
        
        transformed_points += 6
        transformed_pcd = o3d.geometry.PointCloud()
        
        if np.any(np.isnan(transformed_points)) or np.any(np.isinf(transformed_points)):
            raise ValueError("Transformed points contain NaN or Inf values.")

        transformed_points = np.array(transformed_points, dtype=np.float64)
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
        transformed_pcd.colors = o3d.utility.Vector3dVector(color)
        
        # transformed_pcd = transformed_pcd.voxel_down_sample(voxel_size=0.05)
        vis.add_geometry(transformed_pcd)
        vis.update_geometry(transformed_pcd)
        vis.poll_events()
        vis.update_renderer()
        # time.sleep(10)
        
    vis.run()
    vis.destroy_window()

# def create_camera_cone(pose, size=0.2, color=[0, 0, 1]):
#     """
#     创建一个锥形体来表示相机姿态，并将其放置到指定的姿态（pose）下。
    
#     :param pose: 相机的姿态矩阵（4x4）
#     :param size: 锥形体的大小
#     :param color: 锥形体的颜色
#     :return: 已变换的锥形体
#     """
#     cone = o3d.geometry.TriangleMesh.create_cone(radius=size*0.1, height=size)
#     cone.compute_vertex_normals()
#     cone.paint_uniform_color(color)
    
#     # 将锥形体的底面平移到世界坐标系原点
#     transformation = [[1, 0, 0, 0],
#                       [0, 1, 0, 0],
#                       [0, 0, 1, -size*0.5],
#                       [0, 0, 0, 1]]
    
#     cone.transform(transformation)
#     cone.transform(pose)
#     return cone

def create_camera_axes(pose, size=0.2, origin_color=[1, 0, 0]):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    origin = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.05)
    origin.paint_uniform_color(origin_color)
    origin.transform(pose)
    
    axes.transform(pose)
    
    return axes, origin

def visualize_all(pts_batch, color_batch, camera_poses, gt_poses):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="点云与相机姿态可视化", width=2400, height=1600, left=500, top=500)
    
    for pose in camera_poses:
        camera_axes, camera_origin = create_camera_axes(pose, size=0.1, origin_color=[0, 0, 1])
        vis.add_geometry(camera_axes)
        vis.add_geometry(camera_origin)

    for gt_pose in gt_poses:
        gt_axes, gt_origin = create_camera_axes(gt_pose, size=0.1, origin_color=[1, 0, 0])
        vis.add_geometry(gt_axes)
        vis.add_geometry(gt_origin)
    
    for i, (pts, color) in enumerate(zip(pts_batch, color_batch)):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(color)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    vis.run()
    vis.destroy_window()

def write_pose_to_json(global_pose):
    camera_params = {
        "fl_x": 907.4083251953125,
        "fl_y": 907.3446044921875,
        "cx": 638.7867431640625,
        "cy": 357.056640625,
        "w": 1280,
        "h": 720,
        "camera_model": "OPENCV",
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }

    frames = []
    for i, transform_matrix in enumerate(global_pose):
        frames.append({
            "file_path": f"./images/frame_{i+1:05}.png",  
            "transform_matrix": transform_matrix.tolist()  
        })

    output = {
        **camera_params,
        "frames": frames
    }

    with open('./results/transforms.json', 'w') as f:
        json.dump(output, f, indent=4)

def readSFMFreeSceneInfo_v2(dataset_path, image_length, min_conf_thr=1, img_fov=False, focal_known=907.4083251953125, device='cuda', 
                            fps=1, fast=False):
    model = AsymmetricMASt3R.from_pretrained(Mast3R_MODEL_PATH).to(device)
    pair_len = image_length
    read_images = []
    c2ws_list = []
    focals_list = []
    pts3ds_list = []
    dense_mask_list = []
    colors_list = []
    split_img = []
    global_pose_sRT = []
    global_pose_R = []
    global_pose = []
    trans_matrix = []
    r_matrix = []
    while True:
        if len(read_images) >= pair_len:
            break
        all_files = sorted(glob(f'{dataset_path}/*'), key=natural_sort_key)
        new_files = [f for f in all_files if f not in read_images]
        if len(new_files) >= 1:
            read_images.extend(new_files[:1])
            if len(read_images) == 2:
                time_start = time.time()
                print("pcd time_start:", time_start)
                org_h, org_w = cv2.imread(read_images[0]).shape[:2] 
                img_pair = [read_images[len(read_images)-2], read_images[len(read_images)-1]]
                c2ws, focals, pts3ds, dense_mask, colors, depth = progressPair(img_pair, org_h, org_w, model, focal_known)
                c2ws_list.append(c2ws)
                focals_list.append(focals)
                pts3ds_list.append(pts3ds)
                dense_mask_list.append(dense_mask)
                colors_list.append(colors)
                split_img.append(img_pair)
                global_pose_sRT.append(np.eye(4))
                global_pose_sRT.append(np.eye(4))
                global_pose_R.append(np.eye(3))
                global_pose_R.append(np.eye(3))
                global_pose.append(c2ws[0])
                global_pose.append(c2ws[1])
            elif len(read_images) >= 3:
                img_pair = [read_images[len(read_images)-2], read_images[len(read_images)-1]]
                c2ws, focals, pts3ds, dense_mask, colors, depth = progressPair(img_pair, org_h, org_w, model, focal_known)
                c2ws_list.append(c2ws)
                focals_list.append(focals)
                pts3ds_list.append(pts3ds)
                dense_mask_list.append(dense_mask)
                colors_list.append(colors)
                split_img.append(img_pair)
                idx_now = len(read_images)-3
                scene = dict(
                    pose=[c2ws_list[idx_now], c2ws_list[idx_now+1]],
                    pcds=[pts3ds_list[idx_now], pts3ds_list[idx_now+1]],
                    confs=[dense_mask_list[idx_now], dense_mask_list[idx_now+1]],
                    keyframe=[split_img[idx_now], split_img[idx_now+1]],
                    )
                align_net = Pcd_Global_Alignment(scene=scene, camera_align=True)
                if not fast:
                    align_net.compute_global_alignment(lr=0.01, niter=300, schedule='cosine')    
                    s, R, T = align_net.get_result()
                else:
                    s, R, T = align_net.get_result()
                
                del align_net
                M = get_transmarix(s, R, T)
                trans_matrix.append(M)
                r_matrix.append(R)
                global_pose_sRT_new = np.eye(4)
                global_pose_R_new = np.eye(3)
                for i in range(idx_now, -1, -1):
                    global_pose_sRT_new = trans_matrix[i] @ global_pose_sRT_new
                    global_pose_R_new = r_matrix[i] @ global_pose_R_new
                global_pose_sRT.append(global_pose_sRT_new)
                global_pose_R.append(global_pose_R_new)
                global_pose_new = get_global_pose(pose=c2ws[1], global_pose_srt=global_pose_sRT_new, global_pose_r=global_pose_R_new)
                global_pose.append(global_pose_new)
        else :
            time.sleep(1)
    write_pose_to_json(global_pose=global_pose)
    return pts3ds_list, colors_list, global_pose_sRT, global_pose
    


sceneLoadTypeCallback = {
    "SFMFree_dust3r": readSFMFreeSceneInfo,
    "SFMFree_mast3r": readSFMFreeSceneInfo_v1,
    "SFMFree_realtime": readSFMFreeSceneInfo_v2
}