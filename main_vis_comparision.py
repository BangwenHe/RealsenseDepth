import copy
from email.mime import image
import glob
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

    images_folder = "data/0317/outdoor"
    rectified_images_paths = sorted(glob.glob(os.path.join(images_folder, "*rectified*.png")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))
    origin_images_paths = sorted(glob.glob(os.path.join(images_folder, "*original*.png")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))
    depths_npy_folder = "data/0317/outdoor"
    depths_npy_paths = sorted(glob.glob(os.path.join(images_folder, "depth*.npy")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))

    output_dir = "output_0317_outdoor"
    os.makedirs(output_dir, exist_ok=True)

    # src = np.float32([[202,266], [204,410], [432,407], [427,260]])
    # dst = np.float32([[169,305], [165,423], [353,438], [358,311]])
    # perspective_matrix = cv2.getPerspectiveTransform(src, dst)

    detector = build_default_detector("nanodet-m.yml", "nanodet_m.ckpt", "cuda:0")
    pose_mhp = MobileHumanPose("mobile_human_pose_working_well_256x256.onnx")
    # pose_hrhrnet = init_pose_model("/home/tclab/bangwhe/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py",
    #                           "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth",
    #                           device="cuda:0")
    pose_hg = HourglassModel("model.tflite")

    bbox_threshold = 0.5
    run_model = True
    alpha = 0.7

    # video_resolution = (640*3, 480)
    video_resolution = (480*2, 640)
    video_fps = 30
    video_cc = cv2.VideoWriter_fourcc(*"mp4v")
    video_filepath = "offline_test.mp4"
    video_writer = cv2.VideoWriter(video_filepath, video_cc, video_fps, video_resolution)

    pbar = tqdm(range(len(rectified_images_paths)))
    for i in pbar:
        #7~15 24~34 42~69 74~121
        if not (7 <= i <= 15 or 24 <= i <= 34 or 42 <= i <= 69 or 74 <= i <= 121):
            continue

        depth_image = np.load(depths_npy_paths[i])
        depth_image = median_filter(depth_image, 3)
        # depth_image = cv2.bilateralFilter(depth_image, 15, 75, 75)
        depth_image[depth_image >= 5] = 5
        rectified_image = cv2.imread(rectified_images_paths[i])
        origin_image = cv2.imread(origin_images_paths[i])

        start = time.time()

        meta, bboxes = detector.inference(origin_image)
        vis_image = detector.visualize(bboxes[0], meta, detector.cfg.class_names, bbox_threshold)

        if len(bboxes[0][0]) > 0:
            filtered_bbox = np.array(bboxes[0][0])
            filtered_bbox = filtered_bbox[filtered_bbox[:, -1] > bbox_threshold]

            output_pose_2d, output_pose_3d, output_scores = pose_mhp(origin_image, filtered_bbox)
            vis_image_mhp = pose_mhp.visualize(origin_image, output_pose_2d, output_scores)

            hg_pose_2d = pose_hg(rectified_image, filtered_bbox)
            vis_image_hg = HourglassModel.visualize(rectified_image, hg_pose_2d)
            
            # no empty hole
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=100), cv2.COLORMAP_TURBO)
            print(depths_npy_paths[i])
            cv2.imwrite("depth_colormap.jpg", depth_colormap)

            # TODO: get depth from our SGBM algorithm
            # hg_depth_idx = cv2.perspectiveTransform(hg_pose_2d, perspective_matrix).astype(int)
            hg_depth_idx = hg_pose_2d.copy().astype(int)
            hg_depth = gather_depth_by_idx(depth_image, hg_depth_idx) * 1000

            # avoid inf depth
            if np.where(hg_depth == 5000)[0].shape[0] == 0:
                hg_pose_3d = np.concatenate((hg_pose_2d, hg_depth[..., np.newaxis]), axis=2)
                view_output_dir = f"{output_dir}/{i}/hg"
                os.makedirs(view_output_dir, exist_ok=True)
                hg_3d_image = vis_3d_multiple_skeleton(hg_pose_3d, np.ones_like(hg_pose_3d), skeleton, model_type="hg", output_dir=view_output_dir)
                # vis_image_hg_3d = cv2.warpPerspective(vis_image_hg, perspective_matrix, (480, 640))
                vis_image_hg_3d = vis_image_hg.copy()
                vis_image_hg_3d = cv2.addWeighted(vis_image_hg_3d, alpha, depth_colormap, 1 - alpha, 0.0)
                vis_image_hg_3d = cv2.resize(vis_image_hg_3d, (1600, 1200))
                hg_3d_image = np.hstack((hg_3d_image, vis_image_hg_3d))
                cv2.imwrite(f"{output_dir}/hg_3d_image_{i:03d}.jpg", hg_3d_image)

                mhp_3d = output_pose_3d.copy()
                mhp_3d[:, :, 1] = -mhp_3d[:, :, 1]
                mhp_3d = merge_keypoints(mhp_3d, "muco")
                view_output_dir = f"{output_dir}/{i}/mhp"
                os.makedirs(view_output_dir, exist_ok=True)
                mhp_3d_image = vis_3d_multiple_skeleton(mhp_3d, np.ones_like(output_pose_3d), skeleton, model_type="mhp", output_dir=view_output_dir)
                pose_mhp_2d = merge_keypoints(output_pose_2d, "muco")
                vis_image_mhp = HourglassModel.visualize(vis_image, pose_mhp_2d)
                # vis_image_mhp_3d = cv2.warpPerspective(vis_image_mhp, perspective_matrix, (480, 640))
                vis_image_mhp_3d = vis_image_mhp.copy()
                # vis_image_mhp_3d = cv2.addWeighted(vis_image_mhp_3d, alpha, depth_colormap, 1 - alpha, 0.0)
                vis_image_mhp_3d = cv2.resize(vis_image_mhp_3d, (1600, 1200))
                mhp_3d_image = np.hstack((mhp_3d_image, vis_image_mhp_3d))
                cv2.imwrite(f"{output_dir}/mhp_3d_image_{i:03d}.jpg", mhp_3d_image)

        pose_mhp_2d = merge_keypoints(output_pose_2d, "muco")
        vis_image_mhp = HourglassModel.visualize(vis_image, pose_mhp_2d)
        cv2.putText(vis_image_mhp, "MobileHumanPose", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        pose_hg_2d = hg_pose_2d
        vis_image_hg = vis_image_hg
        cv2.putText(vis_image_hg, "Ours Hourglass", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

        video_writer.write(np.hstack((vis_image_mhp, vis_image_hg)))

        end = time.time()
        num_person = len(filtered_bbox) if len(bboxes[0][0]) > 0 else 0
        # print(f" pipeline time: {end - start:.3f}s | persons: {num_person}")

        pipeline_fps = 1 / (end - start)
        pbar.set_description_str(f"{num_person} person{'s' if num_person > 1 else ''}")
    
    video_writer.release()