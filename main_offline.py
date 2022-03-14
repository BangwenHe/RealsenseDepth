"""
先拍视频，然后离线推理。

数据：视频+npy的深度图
推理：nanodet+mhp，higherhrnet，？+深度图
"""

import copy
import time

import cv2
import numpy as np
import mmcv
from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from sklearn import pipeline
from tqdm import tqdm

from detector import build_default_detector
# from yamppe3d import get_root_pose_net
from mobile_human_pose import MobileHumanPose
from utils_pose_estimation import draw_skeleton
from utils_camera import get_realsense_perspective_matrix, perspective_transform, gather_depth_by_idx


if __name__ == "__main__":
    video_filepath = "data/offline_test.mp4"
    npy_filepath = "data/offline_test.npy"

    perspective_matrix = get_realsense_perspective_matrix()

    video_cap = cv2.VideoCapture(video_filepath)
    depth_images = np.load(npy_filepath)

    detector = build_default_detector("nanodet-m.yml", "nanodet_m.ckpt", "cuda:0")
    pose_mhp = MobileHumanPose("mobile_human_pose_working_well_256x256.onnx")
    pose_hrhrnet = init_pose_model("/home/tclab/bangwhe/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py",
                              "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth",
                              device="cuda:0")
    pose_hrnet = init_pose_model("/home/tclab/bangwhe/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py")

    bbox_threshold = 0.5
    run_model = True

    video_resolution = (640*2, 480)
    video_fps = 30
    video_cc = cv2.VideoWriter_fourcc(*"mp4v")
    video_filepath = "offline_test.mp4"
    video_writer = cv2.VideoWriter(video_filepath, video_cc, video_fps, video_resolution)

    pbar = tqdm(range(len(depth_images)))
    for i in pbar:
        depth_image = depth_images[i]
        ret, color_image = video_cap.read()

        start = time.time()

        meta, bboxes = detector.inference(color_image)
        vis_image = detector.visualize(bboxes[0], meta, detector.cfg.class_names, bbox_threshold)

        if len(bboxes[0][0]) > 0:
            filtered_bbox = np.array(bboxes[0][0])
            filtered_bbox = filtered_bbox[filtered_bbox[:, -1] > bbox_threshold]
            # root_depth_list, output_pose_2d, output_pose_3d = posenet.inference(color_image, filtered_bbox, bbox_threshold)
            # vis_image = posenet.visualize(vis_image, output_pose_2d)

            output_pose_2d, output_pose_3d, output_scores = pose_mhp(color_image, filtered_bbox)
            vis_image_mhp = pose_mhp.visualize(vis_image, output_pose_2d, output_scores)

            pose_results, returned_outputs = inference_bottom_up_pose_model(
                pose_hrhrnet, color_image)
            vis_image_gt = vis_pose_result(pose_hrhrnet, vis_image, pose_results)

            gt_2d = np.array([res["keypoints"] for res in pose_results])
            if len(gt_2d) > 0:
                # gt_2d_depth_idx = cv2.perspectiveTransform(gt_2d[:, :, :2], perspective_matrix).astype(int)
                gt_2d_depth_idx = perspective_transform(perspective_matrix, gt_2d[:, :, :2]).astype(int)
                gt_2d_depth = gather_depth_by_idx(depth_image, gt_2d_depth_idx)
            
            video_writer.write(np.hstack((vis_image_mhp, vis_image_gt)))

        end = time.time()
        num_person = len(filtered_bbox) if len(bboxes[0][0]) > 0 else 0
        # print(f" pipeline time: {end - start:.3f}s | persons: {num_person}")

        pipeline_fps = 1 / (end - start)
        pbar.set_description_str(f"{num_person} person{'s' if num_person > 1 else ''}")
    
    video_writer.release()