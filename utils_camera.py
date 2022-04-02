import os

import cv2
from matplotlib.pyplot import axis
import numpy as np


def get_realsense_perspective_matrix(is_vertical=True):
    if is_vertical:
        return get_realsense_perspective_matrix_vertical()
    else:
        return get_realsense_perspective_matrix_horizontal()


def get_realsense_perspective_matrix_horizontal():
    src = np.float32([[74, 109], [37, 654], [841, 128], [833, 706]]) / 1.5
    dst = np.float32([[201, 206], [177, 541], [691, 217], [677, 578]]) / 1.5
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)

    return perspective_matrix


def get_realsense_perspective_matrix_vertical():
    # src = np.float32([[79, 41], [75, 796], [585, 801], [621, 72]]) / 1.5
    # dst = np.float32([[182, 221], [179, 700], [503, 699], [525, 246]]) / 1.5
    src = np.float32([[31, 390], [38, 640], [421, 390], [424, 631]]) / 1.5
    dst = np.float32([[155, 424], [159, 579], [397, 427], [397, 576]]) / 1.5
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)

    return perspective_matrix


def get_realsense_affine_matrix(is_vertical=True):
    if is_vertical:
        return get_realsense_affine_matrix_vertical()
    else:
        return get_realsense_affine_matrix_horizontal()


def get_realsense_affine_matrix_horizontal():
    src = np.float32([[74, 109], [37, 654], [841, 128]]) / 1.5
    dst = np.float32([[201, 206], [177, 541], [691, 217]]) / 1.5
    affine_matrix = cv2.getAffineTransform(src, dst)

    return affine_matrix


def get_realsense_affine_matrix_vertical():
    # src = np.float32([[79, 41], [75, 796], [585, 801]]) / 1.5
    # dst = np.float32([[182, 221], [179, 700], [503, 699]]) / 1.5
    src = np.float32([[31, 390], [38, 640], [421, 390]]) / 1.5
    dst = np.float32([[155, 424], [159, 579], [397, 427]]) / 1.5
    affine_matrix = cv2.getAffineTransform(src, dst)

    return affine_matrix


def gather_depth_by_idx(depth_map: np.ndarray, idx_2d: np.ndarray):
    assert len(idx_2d.shape) == 3, f"{idx_2d.shape} should be 3, now {len(idx_2d.shape)}"
    num_person, num_keypoints, num_axis = idx_2d.shape

    person_depth = np.zeros((num_person, num_keypoints))
    for i in range(num_person):
        for j in range(num_keypoints):
            person_depth[i][j] = depth_map[int(idx_2d[i][j][1])][int(idx_2d[i][j][0])]

    return person_depth


def perspective_transform(perspective_matrix, persons_poses):
    """
    https://stackoverflow.com/questions/53861636/how-can-i-implement-opencvs-perspectivetransform-in-python
    """
    assert len(persons_poses.shape) == 3, f"len({persons_poses.shape}) should be 3, now {len(persons_poses.shape)}"
    num_person, num_keypoints, num_axis = persons_poses.shape

    target_poses = np.zeros_like(persons_poses)
    for i in range(num_person):
        transpose_keypoints = np.concatenate((persons_poses[i], np.ones((num_keypoints, 1))), axis=1)
        target_homo_keypoints = transpose_keypoints.dot(perspective_matrix)
        target_keypoints = target_homo_keypoints[:, :2] / target_homo_keypoints[:, 2].reshape((-1, 1))

        target_poses[i] = target_keypoints

    return target_poses


def merge_keypoints(keypoints, dataset_type: str):
    if type(keypoints) is not np.ndarray:
        keypoints = np.array(keypoints)
    num_person, num_keypoints, num_axis = keypoints.shape

    dataset_type = dataset_type.lower()
    assert dataset_type in ("coco", "muco", "mpii"), f"dataset_type should be one of coco, muco or mpii, now {dataset_type}"

    if dataset_type == "coco":
        keypoints_thorax = np.mean(keypoints[:, 5:7], axis=1)
        keypoints_head = np.mean(keypoints[:, 3:5], axis=1)
        # keypoints_head_top = keypoints_thorax + (keypoints_head - keypoints_thorax) * 2
        keypoints_head_top = keypoints_head
        rearrange_idx = np.array([5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16])
        keypoints_limb = keypoints[:, rearrange_idx]
    elif dataset_type == "muco":
        keypoints_head = keypoints[:, 16]
        keypoints_thorax = keypoints[:, 1]
        keypoints_head_top = keypoints[:, 16]
        rearrange_idx = np.array([5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10])
        keypoints_limb = keypoints[:, rearrange_idx]
    elif dataset_type == "mpii":
        keypoints_head = np.mean([keypoints[:, 8:10]], axis=2)
        keypoints_thorax = keypoints[:, 7]
        keypoints_head_top = keypoints[:, 9]
        rearrange_idx = np.array([15, 14, 13, 12, 11, 10, 5, 4, 3, 2, 1, 0])
        keypoints_limb = keypoints[:, rearrange_idx]

    keypoints = np.hstack([keypoints_head_top[:, np.newaxis, :], keypoints_thorax[:, np.newaxis, :], keypoints_limb])
    return keypoints
