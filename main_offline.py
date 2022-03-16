import copy
import os
from pathlib import Path
import time

import cv2
from cv2 import merge
import numpy as np
import mmcv
from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from sklearn import pipeline
from sklearn.metrics import median_absolute_error
from torch import gather, gt
from tqdm import tqdm
from scipy.ndimage import median_filter

from detector import build_default_detector
# from yamppe3d import get_root_pose_net
from mobile_human_pose import MobileHumanPose
from utils_pose_estimation import draw_skeleton, pixel2cam, vis_3d_multiple_skeleton
from utils_camera import get_realsense_perspective_matrix, merge_keypoints, gather_depth_by_idx
from hourglass_pose import HourglassModel, skeleton


if __name__ == "__main__":
    coco_skeleton = np.array([[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
            [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]])-1

    # video_filepath = "data/offline_vertical2.mp4"
    # npy_filepath = "data/offline_vertical2.npy"
    # video_filepath = "data/offline_sit.mp4"
    # npy_filepath = "data/offline_sit.npy"
    video_filepath = "data/offline_2p.mp4"
    npy_filepath = "data/offline_2p.npy"
    # video_filepath = "data/offline_far.mp4"
    # npy_filepath = "data/offline_far.npy"
    # video_filepath = "data/offline_1.mp4"
    # npy_filepath = "data/offline_1.npy"
    output_dir = "output_2p"
    os.makedirs(output_dir, exist_ok=True)

    perspective_matrix = get_realsense_perspective_matrix(is_vertical=True)

    video_cap = cv2.VideoCapture(video_filepath)
    depth_images = np.load(npy_filepath)

    detector = build_default_detector("nanodet-m.yml", "nanodet_m.ckpt", "cuda:0")
    pose_mhp = MobileHumanPose("mobile_human_pose_working_well_256x256.onnx")
    pose_hrhrnet = init_pose_model("/home/tclab/bangwhe/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py",
                              "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth",
                              device="cuda:0")
    pose_hg = HourglassModel("model.tflite")

    bbox_threshold = 0.5
    run_model = True

    # video_resolution = (640*3, 480)
    video_resolution = (480*3, 640)
    video_fps = 30
    video_cc = cv2.VideoWriter_fourcc(*"mp4v")
    video_filepath = "offline_test.mp4"
    video_writer = cv2.VideoWriter(video_filepath, video_cc, video_fps, video_resolution)

    pbar = tqdm(range(len(depth_images)))
    for i in pbar:
        depth_image = depth_images[i]
        depth_image = median_filter(depth_image, 7)
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

            hg_pose_2d = pose_hg(color_image, filtered_bbox)
            vis_image_hg = HourglassModel.visualize(vis_image, hg_pose_2d)

            gt_2d = np.array([res["keypoints"] for res in pose_results])
            if len(gt_2d) > 0:
                gt_depth_idx = cv2.perspectiveTransform(gt_2d[:, :, :2], perspective_matrix).astype(int)
                # gt_depth_idx = perspective_transform(perspective_matrix, gt_2d[:, :, :2]).astype(int)
                gt_depth = gather_depth_by_idx(depth_image, gt_depth_idx)
                gt_3d = np.concatenate((gt_2d[:, :, :2], gt_depth[..., np.newaxis]), axis=2)
                
                # no empty hole
                if np.where(gt_depth < 50)[0].shape[0] == 0:
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_TURBO)

                    gt_cam_3d = np.zeros_like(gt_3d)
                    for j in range(gt_cam_3d.shape[0]):
                        gt_cam_3d[j] = pixel2cam(gt_3d[j], [604.607, 604.648], [321.509, 239.94])
                    gt_3d_image = vis_3d_multiple_skeleton(gt_cam_3d, np.ones_like(gt_cam_3d), coco_skeleton)
                    vis_image_gt_3d = cv2.warpPerspective(vis_image_gt, perspective_matrix, (480, 640))
                    vis_image_gt_3d = cv2.addWeighted(vis_image_gt_3d, 0.7, depth_colormap, 0.3, 0.0)
                    vis_image_gt_3d = cv2.resize(vis_image_gt_3d, (1600, 1200))
                    gt_3d_image = np.hstack((gt_3d_image, vis_image_gt_3d))
                    cv2.imwrite(f"{output_dir}/gt_3d_image_{i:03d}.jpg", gt_3d_image)

                    mhp_3d = output_pose_3d.copy()
                    mhp_3d[:, :, 1] = -mhp_3d[:, :, 1]
                    mhp_3d_image = vis_3d_multiple_skeleton(mhp_3d, np.ones_like(output_pose_3d))
                    vis_image_mhp_3d = cv2.warpPerspective(vis_image_mhp, perspective_matrix, (480, 640))
                    vis_image_mhp_3d = cv2.addWeighted(vis_image_mhp_3d, 0.7, depth_colormap, 0.3, 0.0)
                    vis_image_mhp_3d = cv2.resize(vis_image_mhp_3d, (1600, 1200))
                    mhp_3d_image = np.hstack((mhp_3d_image, vis_image_mhp_3d))
                    cv2.imwrite(f"{output_dir}/mhp_3d_image_{i:03d}.jpg", mhp_3d_image)

                    # TODO: get depth from our SGBM algorithm
                    hg_depth_idx = cv2.perspectiveTransform(hg_pose_2d, perspective_matrix).astype(int)
                    hg_depth = gather_depth_by_idx(depth_image, hg_depth_idx)
                    hg_pose_3d = np.concatenate((hg_pose_2d, hg_depth[..., np.newaxis]), axis=2)
                    hg_3d_image = vis_3d_multiple_skeleton(hg_pose_3d, np.ones_like(hg_pose_3d), skeleton)
                    vis_image_hg_3d = cv2.warpPerspective(vis_image_hg, perspective_matrix, (480, 640))
                    vis_image_hg_3d = cv2.addWeighted(vis_image_hg_3d, 0.7, depth_colormap, 0.3, 0.0)
                    vis_image_hg_3d = cv2.resize(vis_image_hg_3d, (1600, 1200))
                    hg_3d_image = np.hstack((hg_3d_image, vis_image_hg_3d))
                    cv2.imwrite(f"{output_dir}/hg_3d_image_{i:03d}.jpg", hg_3d_image)

            pose_mhp_2d = merge_keypoints(output_pose_2d, "muco")
            vis_image_mhp = HourglassModel.visualize(vis_image, pose_mhp_2d)
            cv2.putText(vis_image_mhp, "MobileHumanPose", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            pose_gt_2d = merge_keypoints(gt_2d, "coco")
            vis_image_gt = HourglassModel.visualize(vis_image, pose_gt_2d)
            cv2.putText(vis_image_gt, "HigherHRNet", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            pose_hg_2d = hg_pose_2d
            vis_image_hg = vis_image_hg
            cv2.putText(vis_image_hg, "Ours Hourglass", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

            video_writer.write(np.hstack((vis_image_mhp, vis_image_gt, vis_image_hg)))

        end = time.time()
        num_person = len(filtered_bbox) if len(bboxes[0][0]) > 0 else 0
        # print(f" pipeline time: {end - start:.3f}s | persons: {num_person}")

        pipeline_fps = 1 / (end - start)
        pbar.set_description_str(f"{num_person} person{'s' if num_person > 1 else ''}")
    
    video_writer.release()