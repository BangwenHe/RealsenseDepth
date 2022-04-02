"""
Groundtruth: Mate40 RGB + D435i depth
Baseline: Mate40 RGB
Ours: Mate40 RGB(undistortion rgb image) + sgbm depth image
"""

import copy
import glob
import os
from pathlib import Path
import time

import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import median_filter
from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)

from detector import build_default_detector
# from yamppe3d import get_root_pose_net
from mobile_human_pose import MobileHumanPose
from mobile_human_pose import MobileHumanPose
from utils_pose_estimation import draw_skeleton, pixel2cam, vis_3d_multiple_skeleton
from utils_camera import get_realsense_perspective_matrix, merge_keypoints, gather_depth_by_idx
from hourglass_pose import HourglassModel, skeleton
from ppdet_deploy.keypoint_infer import PredictConfig_KeyPoint, KeyPoint_Detector
from ppdet_deploy.det_keypoint_unite_infer import predict_with_given_det
from detector import build_default_detector
from sgbm_v2 import get_depth, rotate

# ignore pytorch align corners UserWarning
import warnings
warnings.filterwarnings("ignore")


def build_keypoint_detector_input(bboxes):
    num_person = len(bboxes)
    res = {"boxes": np.zeros((num_person, 6), dtype=bboxes.dtype), "boxes_num": np.array([num_person])}

    res["boxes"][:, 1] = bboxes[:, -1]
    res["boxes"][:, 2:] = bboxes[:, :-1]

    return res


def remap_2d_keypoints(keypoints, map_x, map_y):
    num_person, num_keypoints, num_axis = keypoints.shape
    remapped_keypoints = np.zeros_like(keypoints)

    for i in range(num_person):
        for j in range(num_keypoints):
            remapped_keypoints[i][j][0] = map_x[int(keypoints[i][j][1]), int(keypoints[i][j][0])]
            remapped_keypoints[i][j][1] = map_y[int(keypoints[i][j][1]), int(keypoints[i][j][0])]
    
    return remapped_keypoints


def transform_thorax_relative_coordinate(pose_3d):
    pose_3d_relative = pose_3d.copy()
    # pose_3d_relative -= pose_3d_relative[:, 1, :]
    pose_3d_relative[:, :, :2] -= pose_3d_relative[:, 1:2, :2]
    return pose_3d_relative


def compute_mpjpe(gt, result):
    # only use 13 points to compute
    dis = (result - gt) ** 2
    dis = np.sqrt(dis.sum(axis=2))
    dis = np.mean(dis, axis=1)
    mpjpe = np.mean(dis)
    return mpjpe


def compute_depth_error(gt, result):
    error = (gt[:, :, 2] - result[:, :, 2]) ** 2
    error = np.sqrt(error.sum(axis=1))
    return np.mean(error)


if __name__ == "__main__":
    coco_skeleton = np.array([[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
            [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]])-1

    # MARK: 需要保证图片是相对应的

    expriment_name = "exp2"
    phone_images_folder = f"accuracy_test_data/{expriment_name}/phone_camera"
    depth_camera_images_folder = f"accuracy_test_data/{expriment_name}/depth_camera"
    depth_camera_images_paths = sorted(glob.glob(os.path.join(depth_camera_images_folder, "capture*.jpg")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))
    phone_left_images_paths = sorted(glob.glob(os.path.join(phone_images_folder, "left*.png")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))
    phone_right_images_paths = sorted(glob.glob(os.path.join(phone_images_folder, "right*.png")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))

    gt_npy_folder = f"accuracy_test_data/{expriment_name}/depth_camera"
    gt_depths_npy_paths = sorted(glob.glob(os.path.join(gt_npy_folder, "d435i*.npy")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))

    assert len(depth_camera_images_paths) == len(phone_left_images_paths) == len(phone_right_images_paths) == len(gt_depths_npy_paths)

    output_dir = f"output_accuracy_{expriment_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"output directory: {output_dir}")

    detector = build_default_detector("nanodet-m.yml", "nanodet_m.ckpt", "cuda:0")
    # pose_mhp = MobileHumanPose("mobile_human_pose_working_well_256x256.onnx")
    pose_mhp = MobileHumanPose("mobile_human_pose_working_well_256x256.onnx", rootnet_model_path="3dmppe_rootnet_snapshot_18.pth.tar")
    model_dir = "/home/tclab/bangwhe/PaddleDetection/output_inference/tinypose_256x192"
    pred_config = PredictConfig_KeyPoint(model_dir = "/home/tclab/bangwhe/PaddleDetection/output_inference/tinypose_256x192")
    pose_tp = KeyPoint_Detector(pred_config, model_dir, device="gpu", run_mode="paddle", use_dark=True)
    pose_hrhrnet = init_pose_model("/home/tclab/bangwhe/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py",
                              "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth",
                              device="cuda:0")

    phone_cam_focal = pose_mhp.focal
    phone_cam_princpt = pose_mhp.princpt
    depth_cam_focal = [604.607, 604.648]
    depth_cam_princpt = [321.509, 239.94]
    perspective_matrix = get_realsense_perspective_matrix(is_vertical=True)

    bbox_threshold = 0.5
    run_model = True
    alpha = 0.7
    all_angle = False

    image_shape = [480,640]
    resize_scale = [1.128916, 1.1163591]
    calibration_filepath = "huawei_mate40pro_fixed_3.0m.yml"

    gt_results = []
    tp_results = []
    mhp_results = []

    pbar = tqdm(range(len(depth_camera_images_paths)))
    for i in pbar:
        gt_depth_image = np.load(gt_depths_npy_paths[i])
        vis_z = np.zeros(gt_depth_image.shape,dtype='uint8')
        cv2.normalize(np.abs(gt_depth_image),vis_z,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        vis_z = np.abs(255-vis_z)
        gt_depth_colormap = cv2.applyColorMap(vis_z,cv2.COLORMAP_TURBO)

        depth_camera_bgr_image = cv2.imread(depth_camera_images_paths[i])
        phone_left_bgr_image_origin = rotate(phone_left_images_paths[i])
        phone_left_bgr_image = rotate(phone_left_images_paths[i], scale=resize_scale, shape=image_shape)
        phone_right_bgr_image = rotate(phone_right_images_paths[i])
        detector_image = phone_left_bgr_image_origin.copy()
        # detector_image = phone_left_bgr_image_origin.copy()

        meta, bboxes = detector.inference(detector_image)
        vis_image = detector.visualize(bboxes[0], meta, detector.cfg.class_names, bbox_threshold)
        filtered_bbox = np.array(bboxes[0][0])
        filtered_bbox = filtered_bbox[filtered_bbox[:, -1] > bbox_threshold]
        # strict to only one person
        filtered_bbox = filtered_bbox[0:1]

        _, phone_left_rectified_image, tp_depth_colormap, tp_depth_image, left_map_x, left_map_y = get_depth(calibration_filepath, phone_left_bgr_image, phone_right_bgr_image)
        tp_depth_image = median_filter(tp_depth_image, 7)

        if len(filtered_bbox) > 0:
            # ground truth model is bottom up, so don't need detector
            pose_results, returned_outputs = inference_bottom_up_pose_model(
                pose_hrhrnet, depth_camera_bgr_image)
            # strict to only one person
            pose_results = [max(pose_results, key=lambda pose_result: pose_result["area"])]
            vis_image_gt = vis_pose_result(pose_hrhrnet, depth_camera_bgr_image, pose_results)
            cv2.imwrite(f"debug/gt_debug_{i}.jpg", vis_image_gt)

            output_pose_2d, output_pose_3d, output_scores = pose_mhp(phone_left_bgr_image_origin, filtered_bbox)
            vis_image_mhp = pose_mhp.visualize(phone_left_bgr_image_origin, output_pose_2d, output_scores)

            # do 2d pose estimation in rectified image and transform it back into origin image
            tp_pose_result_rectified = predict_with_given_det(phone_left_rectified_image, build_keypoint_detector_input(filtered_bbox), pose_tp, 1, bbox_threshold, 0, False)
            tp_pose_2d = remap_2d_keypoints(np.array(tp_pose_result_rectified["keypoint"][0])[:, :, :2], left_map_x, left_map_y)
            tp_pose_2d = merge_keypoints(tp_pose_2d, "coco")
            vis_image_tp = HourglassModel.visualize(vis_image, tp_pose_2d)

            # no empty hole
            cv2.imwrite("tp_depth_colormap.jpg", tp_depth_colormap)
            cv2.imwrite("gt_depth_colormap.jpg", gt_depth_colormap)

            tp_depth_idx = np.array(tp_pose_result_rectified["keypoint"][0])[:, :, :2]
            tp_depth_idx = merge_keypoints(tp_depth_idx, "coco")
            tp_depth = gather_depth_by_idx(tp_depth_image, tp_depth_idx) * 1000

            gt_2d = np.array([res["keypoints"] for res in pose_results])
            # avoid inf depth
            if np.where(tp_depth == 5000)[0].shape[0] == 0 and np.where(tp_depth == 0)[0].shape[0] == 0:
                gt_depth_idx = cv2.perspectiveTransform(gt_2d[:, :, :2], perspective_matrix).astype(int)
                gt_depth = gather_depth_by_idx(gt_depth_image, gt_depth_idx)
                gt_3d = np.concatenate((gt_2d[:, :, :2], gt_depth[..., np.newaxis]), axis=2)
                gt_cam_3d = np.zeros_like(gt_3d)
                for j in range(gt_cam_3d.shape[0]):
                    gt_cam_3d[j] = pixel2cam(gt_3d[j], depth_cam_focal, depth_cam_princpt)

                gt_cam_3d = merge_keypoints(gt_cam_3d, "coco")
                view_output_dir = f"{output_dir}/{i}/gt"
                os.makedirs(view_output_dir, exist_ok=True)
                gt_3d_image = vis_3d_multiple_skeleton(gt_cam_3d, np.ones_like(gt_cam_3d), skeleton, model_type="tp", 
                                                        output_dir=view_output_dir, all_angle=all_angle)
                pose_gt_2d = merge_keypoints(gt_2d, "coco")
                vis_image_gt = HourglassModel.visualize(depth_camera_bgr_image, pose_gt_2d)
                vis_image_gt_2d = vis_image_gt.copy()
                vis_image_gt_2d = cv2.resize(vis_image_gt_2d, (1600, 1200))
                gt_depth_colormap_pose_2d = HourglassModel.visualize(gt_depth_colormap, cv2.perspectiveTransform(pose_gt_2d[:, :, :2], perspective_matrix))
                gt_3d_image = np.hstack((gt_3d_image, vis_image_gt_2d, cv2.resize(gt_depth_colormap_pose_2d, (1600, 1200))))
                cv2.imwrite(f"{output_dir}/gt_3d_image_{i:03d}.jpg", gt_3d_image)

                tp_pose_3d = np.concatenate((tp_pose_2d, tp_depth[..., np.newaxis]), axis=2).copy()
                for j in range(tp_pose_3d.shape[0]):
                    tp_pose_3d[j] = pixel2cam(tp_pose_3d[j], phone_cam_focal, phone_cam_princpt)

                view_output_dir = f"{output_dir}/{i}/tp"
                os.makedirs(view_output_dir, exist_ok=True)
                tp_3d_image = vis_3d_multiple_skeleton(tp_pose_3d, np.ones_like(tp_pose_3d), skeleton, model_type="tp", 
                                                        output_dir=view_output_dir, all_angle=all_angle)
                vis_image_tp_3d = vis_image_tp.copy()
                # vis_image_tp_3d = cv2.addWeighted(vis_image_tp_3d, alpha, depth_colormap, 1 - alpha, 0.0)
                vis_image_tp_3d = cv2.resize(vis_image_tp_3d, (1600, 1200))
                tp_depth_colormap_2d = HourglassModel.visualize(tp_depth_colormap, tp_depth_idx)
                tp_3d_image = np.hstack((tp_3d_image, vis_image_tp_3d, cv2.resize(tp_depth_colormap_2d, (1600, 1200))))
                cv2.imwrite(f"{output_dir}/tp_3d_image_{i:03d}.jpg", tp_3d_image)

                mhp_3d = output_pose_3d.copy()
                mhp_3d = merge_keypoints(mhp_3d, "muco")
                view_output_dir = f"{output_dir}/{i}/mhp"
                os.makedirs(view_output_dir, exist_ok=True)
                mhp_3d_image = vis_3d_multiple_skeleton(mhp_3d, np.ones_like(output_pose_3d), skeleton, model_type="mhp", 
                                                        output_dir=view_output_dir, all_angle=all_angle)
                pose_mhp_2d = merge_keypoints(output_pose_2d, "muco")
                vis_image_mhp = HourglassModel.visualize(vis_image, pose_mhp_2d)
                vis_image_mhp_3d = vis_image_mhp.copy()
                vis_image_mhp_3d = cv2.resize(vis_image_mhp_3d, (1600, 1200))
                mhp_3d_image = np.hstack((mhp_3d_image, vis_image_mhp_3d))
                cv2.imwrite(f"{output_dir}/mhp_3d_image_{i:03d}.jpg", mhp_3d_image)

                cv2.putText(vis_image_mhp, "MobileHumanPose", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                cv2.putText(vis_image_gt, "HigherHRNet", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                cv2.putText(vis_image_tp, "Ours Tinypose", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                all_model_result = np.hstack([vis_image_gt, vis_image_tp, vis_image_mhp])
                cv2.imwrite(f"{output_dir}/all_{i:03d}.jpg", all_model_result)

                gt_results.append(gt_cam_3d)
                tp_results.append(tp_pose_3d)
                mhp_results.append(mhp_3d)

        num_person = len(filtered_bbox) if len(bboxes[0][0]) > 0 else 0

        pbar.set_description_str(f"{num_person} person{'s' if num_person > 1 else ''}")

    mpjpe_mhp_arr = []
    mpjpe_tp_arr = []
    depth_error_mhp_arr = []
    depth_error_tp_arr = []

    for idx, (gt, tp, mhp) in enumerate(zip(gt_results, tp_results, mhp_results)):
        gt = transform_thorax_relative_coordinate(gt)
        tp = transform_thorax_relative_coordinate(tp)
        mhp = transform_thorax_relative_coordinate(mhp)

        mpjpe_tp = compute_mpjpe(gt[:, 1:], tp[:, 1:])
        mpjpe_mhp = compute_mpjpe(gt[:, 1:], mhp[:, 1:])
        depth_error_tp = compute_depth_error(gt[:, 1:], tp[:, 1:])
        depth_error_mhp = compute_depth_error(gt[:, 1:], mhp[:, 1:])

        mpjpe_mhp_arr.append(mpjpe_mhp)
        mpjpe_tp_arr.append(mpjpe_tp)
        depth_error_mhp_arr.append(depth_error_mhp)
        depth_error_tp_arr.append(depth_error_tp)

        print(f"mpjpe of ours: {mpjpe_tp:.2f}, mpjpe of mhp: {mpjpe_mhp:.2f}, "
        f"depth error of ours: {depth_error_tp:.2f}, depth error of mhp: {depth_error_mhp:.2f}")

    print(len(mpjpe_mhp_arr), len(phone_left_images_paths))
    print(sum(mpjpe_tp_arr) / len(mpjpe_tp_arr))
    print(sum(mpjpe_mhp_arr) / len(mpjpe_mhp_arr))
    print(sum(depth_error_tp_arr) / len(depth_error_tp_arr))
    print(sum(depth_error_mhp_arr) / len(depth_error_mhp_arr))