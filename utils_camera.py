import os

import cv2
from matplotlib.pyplot import axis
import numpy as np


def get_realsense_perspective_matrix():
    src = np.float32([[74, 109], [37, 654], [841, 128], [833, 706]]) / 1.5
    dst = np.float32([[201, 206], [177, 541], [691, 217], [677, 578]]) / 1.5
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)

    return perspective_matrix


def gather_depth_by_idx(depth_map: np.ndarray, idx_2d: np.ndarray):
    assert len(idx_2d.shape) == 3, f"{idx_2d.shape} should be 3, now {len(idx_2d.shape)}"
    num_person, num_keypoints, num_axis = idx_2d.shape

    person_depth = np.zeros((num_person, num_keypoints))
    for i in range(num_person):
        for j in range(num_keypoints):
            person_depth[i][j] = depth_map[idx_2d[i][j][1]][idx_2d[i][j][0]]

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
        # calculate mean value for ears
        keypoints_head = np.mean(keypoints[:, :, 3:5], axis=2)
        # keep coordinate data of limbs
        keypoints_limb = keypoints[:, :, 5:]
    elif dataset_type == "muco":
        keypoints_head = keypoints[:, :, 16]
        rearrange_idx = np.array([5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10])
        keypoints_limb = keypoints_limb[:, :, rearrange_idx]
    elif dataset_type == "mpii":
        keypoints_head = np.mean([keypoints[:, :, 8:10]], axis=2)
        rearrange_idx = np.array([13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0])
        keypoints_limb = keypoints[:, :, rearrange_idx]

    keypoints = np.concatenate([keypoints_head, keypoints_limb], axis=2)
    return keypoints