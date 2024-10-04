import os
import sys
import numpy as np
import open3d as o3d
import cv2
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
sys.path.append("./get_pcd")
from dataset_readers_sfm_free import sceneLoadTypeCallback, visualize_all, apply_transformation, natural_sort_key
from dataset_readers_sfm_free import get_mast3r_to_world, get_mast3r_to_world_optimer
from utils.camera_utils import visualize_camera_poses, global_pose_to_world, compute_relative_pose

def rotation_matrix_to_euler_angles(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    rpy_angles = r.as_euler('xyz', degrees=True)
    print("Roll: {:.2f}, Pitch: {:.2f}, Yaw: {:.2f}".format(rpy_angles[0], rpy_angles[1], rpy_angles[2]))

def get_obj_pcd_array(mask_pts, mask_colors):
    pts = np.concatenate(mask_pts, axis=0)
    colors = np.concatenate(mask_colors, axis=0)
    return pts, colors

def get_world_mask_pts(mask_pts, camera_to_world):
    pts = apply_transformation(mask_pts, camera_to_world)
    return pts

def get_obj_pcd(pts3ds_list, colors_list, shape_list, mask_list):
    if len(shape_list) == 3:
        b = shape_list[0]
        w = shape_list[1]
        h = shape_list[2]
        mask_pts = []
        mask_colors = []
        for i in range(len(pts3ds_list)):
            pts3ds = pts3ds_list[i]
            colors = colors_list[i]
            pts3ds = pts3ds.reshape(b, w, h, 3)
            colors = colors.reshape(b, w, h, 3)
            if i == 0:
                mask1 = mask_list[0]
                mask2 = mask_list[1]
                pts3ds1 = pts3ds[0][mask1>0]
                colors1 = colors[0][mask1>0]
                pts3ds2 = pts3ds[1][mask2>0]
                colors2 = colors[1][mask2>0]
                mask_pts.append(pts3ds1)
                mask_pts.append(pts3ds2)
                mask_colors.append(colors1)
                mask_colors.append(colors2)
            else:
                mask = mask_list[i+1]
                pts3ds = pts3ds[1][mask>0]
                colors = colors[1][mask>0]
                mask_pts.append(pts3ds)
                mask_colors.append(colors)
    else:
        w = shape_list[0]
        h = shape_list[1]
        mask_pts = []
        mask_colors = []
        for i in range(len(pts3ds_list)):
            pts3ds = pts3ds_list[i]
            colors = colors_list[i]
            pts3ds = pts3ds.reshape(w, h, 3)
            colors = colors.reshape(w, h, 3)
            mask = mask_list[i]
            pts3ds = pts3ds[mask>0]
            colors = colors[mask>0]
            mask_pts.append(pts3ds)
            mask_colors.append(colors)
    
    pts, color = get_obj_pcd_array(mask_pts, mask_colors)   
    return pts, color

def visiualize_pcd(points, colors, gg):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([pcd])
    if gg :
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([*grippers, pcd])
    else:
        o3d.visualization.draw_geometries([pcd])

def matrix2rpy(matrix):
    rotation_matrix = matrix[:3, :3]

    r = R.from_matrix(rotation_matrix)

    # (roll, pitch, yaw)
    rpy_degrees = r.as_euler('xyz', degrees=True)
    print("Roll (X):", rpy_degrees[0])
    print("Pitch (Y):", rpy_degrees[1])
    print("Yaw (Z):", rpy_degrees[2])

def run_get_pcd(source_path, img_len, min_conf_thr=1, img_fov=False, focal_known="907.4083251953125", pcd_visualize=True, 
                realtime=True, mast3r=False, scene_scale=1.0, camera_visualize=True):
    if mast3r:
        if realtime == True:
            pts3ds_list, colors_list, global_pose_sRT, global_pose = sceneLoadTypeCallback["SFMFree_realtime"](source_path, img_len, min_conf_thr, img_fov, focal_known)
        else:
            pts3ds_list, colors_list, global_pose = sceneLoadTypeCallback["SFMFree_mast3r"](source_path, min_conf_thr, img_fov, focal_known)
    else:
        pts3ds_list, colors_list, global_pose = sceneLoadTypeCallback["SFMFree_dust3r"](source_path, min_conf_thr, img_fov, focal_known, scene_scale)
    
    base_dir = os.path.dirname(source_path)
    base_dir = os.path.dirname(base_dir)
    pose_path = os.path.join(base_dir, "poses")
    npy_files = sorted([f for f in os.listdir(pose_path) if f.endswith('.npy')], key=natural_sort_key)

    gt_pose = [np.load(os.path.join(pose_path, f)) for f in npy_files]
    
    if camera_visualize:
        mast3r_to_world, s, R, t = get_mast3r_to_world(global_pose, gt_pose)
        pose_to_world = global_pose_to_world(global_pose, s, R, t)
        visualize_camera_poses(pose_to_world, gt_pose)
    
    else:
        mast3r_to_world, _, _, _ = get_mast3r_to_world(global_pose, gt_pose)
    pts3ds_list_world = []
    colors_list_world = []
    if mast3r:
        if realtime == True:
            for idx, (pts3ds, color, trans) in enumerate(zip(pts3ds_list, colors_list, global_pose_sRT[1:])):
                b, w, h, _ = pts3ds.shape
                pts3ds = np.array(pts3ds)
                color = np.array(color)
                pts3ds = pts3ds.reshape(b*w*h, 3).astype(np.float64)
                color = color.reshape(b*w*h, 3).astype(np.float64)
                pts3ds = apply_transformation(pts3ds, trans)
                pts3ds = apply_transformation(pts3ds, mast3r_to_world)
                pts3ds = np.array(pts3ds, dtype=np.float64)
                pts3ds_list_world.append(pts3ds)
                colors_list_world.append(color)
            time_end = time.time()
            print("Time: ", time_end)
        else:
            for pts3ds, color in zip(pts3ds_list, colors_list):
                w, h, _ = pts3ds.shape
                pts3ds = np.array(pts3ds)
                color = np.array(color)
                pts3ds = pts3ds.reshape(w*h, 3).astype(np.float64)
                color = color.reshape(w*h, 3).astype(np.float64)
                pts3ds = apply_transformation(pts3ds, mast3r_to_world)
                pts3ds_list_world.append(pts3ds)
                colors_list_world.append(color)
            time_end = time.time()
            print("Time: ", time_end)
    else:
        for pts3ds, color in zip(pts3ds_list, colors_list):
            w, h, _ = pts3ds.shape
            pts3ds = np.array(pts3ds)
            color = np.array(color)
            pts3ds = pts3ds.reshape(w*h, 3).astype(np.float64)
            color = color.reshape(w*h, 3).astype(np.float64)
            pts3ds = apply_transformation(pts3ds, mast3r_to_world)
            pts3ds_list_world.append(pts3ds)
            colors_list_world.append(color)
        time_end = time.time()
        print("Time: ", time_end)        

    if pcd_visualize:
        visualize_all(pts3ds_list_world, colors_list_world, pose_to_world, gt_pose) 

    if mast3r & realtime:
        shape_list = [b, w, h]
    else:
        shape_list = [w, h]
    return pts3ds_list_world, colors_list_world, shape_list, gt_pose[0]

def pairwise_registration_plane(source, target, threshold, trans_init=np.identity(4)):
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return reg.transformation

def pairwise_registration(source, target, threshold, trans_init=np.identity(4)):

    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg.transformation

def get_true_pcd(source_path, img_len, focal_known=907.4083251953125, pcd_visualize=True, pcd_save=False):
    base_dir = os.path.dirname(source_path)
    base_dir = os.path.dirname(base_dir)
    pose_path = os.path.join(base_dir, "poses")
    npy_files = sorted([f for f in os.listdir(pose_path) if f.endswith('.npy')], key=natural_sort_key)
    poses_list = []

    for npy_file in npy_files:
        file_path = os.path.join(pose_path, npy_file)
        pose_data = np.load(file_path)
        poses_list.append(pose_data)

    for i in range(img_len):
        depth_dir = os.path.join(base_dir, "depths")
        depth_path = os.path.join(depth_dir, f"depth_{i:04}.npy")
        depth_image = np.load(depth_path)  

        rgb_path = os.path.join(source_path, f"frame_{i:04}.png")
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # BGR to RGB

        mask_path = os.path.join(base_dir, "masks")
        mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith('.png')], key=natural_sort_key)
        
        mask_files_path = os.path.join(mask_path, mask_files[i])
        mask_image = cv2.imread(mask_files_path, cv2.IMREAD_GRAYSCALE) 

        mask = mask_image > 0  

        depth_image_masked = np.where(mask, depth_image, 0)

        depth_o3d = o3d.geometry.Image(depth_image_masked)
        rgb_o3d = o3d.geometry.Image(rgb_image)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=rgb_image.shape[1],
            height=rgb_image.shape[0],
            fx=focal_known,  
            fy=focal_known,  
            cx=rgb_image.shape[1] / 2, 
            cy=rgb_image.shape[0] / 2   
        )

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d, 
            depth=depth_o3d, 
            convert_rgb_to_intensity=False 
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, 
            intrinsics
        )

        relative_pose = compute_relative_pose(poses_list[0], poses_list[i])
        R = relative_pose[:3, :3]
        t = relative_pose[:3, -1]

        transformation = np.eye(4)
        transformation[:3, :3] = R  
        transformation[:3, 3] = t   

        pcd.transform(transformation)

        # icp
        if i > 0:
            transformation_icp_point = pairwise_registration(pcd, target_pcd, threshold=0.002)
            pcd.transform(transformation_icp_point)
            # transformation_icp_plane = pairwise_registration_plane(pcd, target_pcd, threshold=0.001)
            # pcd.transform(transformation_icp_plane)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
            pcd = pcd.select_by_index(ind)
        if i == 0:
            global_pcd = pcd
            target_pcd = pcd
        else:
            global_pcd += pcd 
            # global_pcd = global_pcd.voxel_down_sample(voxel_size=0.005) 

    if pcd_visualize:
        o3d.visualization.draw_geometries([global_pcd])

    if pcd_save:
        o3d.io.write_point_cloud("masked_pointcloud.ply", global_pcd)

    return global_pcd


if __name__ == "__main__":
    run_get_pcd(source_path="./data/images", img_len=17)