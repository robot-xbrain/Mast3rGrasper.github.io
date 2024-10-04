import open3d as o3d
import numpy as np
import torch
from scipy.spatial import KDTree

def compute_distances(points_A, points_B):
    tree = KDTree(points_B)
    distances, _ = tree.query(points_A)
    return distances

def compute_accuracy(reconstructed_points, ground_truth_points):
    distances = compute_distances(reconstructed_points, ground_truth_points)
    return np.mean(distances)

def compute_completeness(ground_truth_points, reconstructed_points):
    distances = compute_distances(ground_truth_points, reconstructed_points)
    return np.mean(distances)

# 函数计算整体误差 (Overall error)
def compute_overall_error(accuracy, completeness):
    return (accuracy + completeness) / 2

def chamfer_distance(pcd1, pcd2):
    p1 = np.asarray(pcd1.points)
    p2 = np.asarray(pcd2.points)

    dists_p1_to_p2 = np.min(np.linalg.norm(p1[:, np.newaxis, :] - p2[np.newaxis, :, :], axis=-1), axis=1)
    dists_p2_to_p1 = np.min(np.linalg.norm(p2[:, np.newaxis, :] - p1[np.newaxis, :, :], axis=-1), axis=1)

    chamfer_dist = np.mean(dists_p1_to_p2) + np.mean(dists_p2_to_p1)
    return chamfer_dist

#RMSE
def compute_rmse(source_pcd, target_pcd, threshold=0.02):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.inlier_rmse


# Overlap ratio 重叠度
def compute_overlap_ratio(source_pcd, target_pcd, threshold=0.02):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    overlap_ratio = len(reg_p2p.correspondence_set) / min(len(source_pcd.points), len(target_pcd.points))
    return overlap_ratio

def numpy_to_pointcloud(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd

def calculate_iou(pcd1, pcd2, threshold=0.001):
    pcd2 = numpy_to_pointcloud(pcd2)
    chamfer_dist = chamfer_distance(pcd1, pcd2)
    print(f"Chamfer 距离: {chamfer_dist}")
    overlap = compute_overlap_ratio(pcd1, pcd2)
    print(f"点云重叠度: {overlap}")
    rmse = compute_rmse(pcd1, pcd2)
    print(f"配准后的RMSE: {rmse}")

def calculate_metrics(reconstructed_points, ground_truth_points):
    ground_truth_points = np.asarray(ground_truth_points.points)
    accuracy = compute_accuracy(reconstructed_points, ground_truth_points)
    completeness = compute_completeness(ground_truth_points, reconstructed_points)
    overall_error = compute_overall_error(accuracy, completeness)

    print(f"Accuracy: {accuracy} mm")
    print(f"Completeness: {completeness} mm")
    print(f"Overall Error: {overall_error} mm")
