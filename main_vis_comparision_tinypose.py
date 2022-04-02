import copy
from email.mime import image
import glob
import os
from pathlib import Path
import time

import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import median_filter

from detector import build_default_detector
# from yamppe3d import get_root_pose_net
from mobile_human_pose import MobileHumanPose
from mobile_human_pose_rootnet import MobileHumanPoseRootnet
from utils_pose_estimation import draw_skeleton, pixel2cam, vis_3d_multiple_skeleton
from utils_camera import get_realsense_perspective_matrix, merge_keypoints, gather_depth_by_idx
from hourglass_pose import HourglassModel, skeleton
from ppdet_deploy.keypoint_infer import PredictConfig_KeyPoint, KeyPoint_Detector
from ppdet_deploy.det_keypoint_unite_infer import predict_with_given_det
from detector import build_default_detector


def build_keypoint_detector_input(bboxes):
    num_person = len(bboxes)
    res = {"boxes": np.zeros((num_person, 6), dtype=bboxes.dtype), "boxes_num": np.array([num_person])}

    res["boxes"][:, 1] = bboxes[:, -1]
    res["boxes"][:, 2:] = bboxes[:, :-1]

    return res


if __name__ == "__main__":
    coco_skeleton = np.array([[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
            [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]])-1

    images_folder = "data/0317/2p"
    rectified_images_paths = sorted(glob.glob(os.path.join(images_folder, "*rectified*.png")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))
    origin_images_paths = sorted(glob.glob(os.path.join(images_folder, "*original*.png")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))
    depths_npy_folder = "data/0317/2p"
    depths_npy_paths = sorted(glob.glob(os.path.join(depths_npy_folder, "depth*.npy")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))

    output_dir = "output_0317_selected"
    os.makedirs(output_dir, exist_ok=True)

    # src = np.float32([[202,266], [204,410], [432,407], [427,260]])
    # dst = np.float32([[169,305], [165,423], [353,438], [358,311]])
    # perspective_matrix = cv2.getPerspectiveTransform(src, dst)

    detector = build_default_detector("nanodet-m.yml", "nanodet_m.ckpt", "cuda:0")
    # pose_mhp = MobileHumanPose("mobile_human_pose_working_well_256x256.onnx")
    pose_mhp = MobileHumanPoseRootnet("mobile_human_pose_working_well_256x256.onnx", "3dmppe_rootnet_snapshot_18.pth.tar")
    model_dir = "/home/tclab/bangwhe/PaddleDetection/output_inference/tinypose_256x192"
    pred_config = PredictConfig_KeyPoint(model_dir = "/home/tclab/bangwhe/PaddleDetection/output_inference/tinypose_256x192")
    pose_tp = KeyPoint_Detector(pred_config, model_dir, device="gpu", run_mode="paddle", use_dark=True)

    cam_focal = pose_mhp.focal
    cam_princpt = pose_mhp.princpt

    bbox_threshold = 0.5
    run_model = True
    alpha = 0.7
    all_angle = False

    # video_resolution = (640*3, 480)
    video_resolution = (480*2, 640)
    video_fps = 30
    video_cc = cv2.VideoWriter_fourcc(*"mp4v")
    video_filepath = f"{output_dir}.mp4"
    video_writer = cv2.VideoWriter(video_filepath, video_cc, video_fps, video_resolution)

    seletced_images = [31, 64, 69, 95, 100, 107, 123, 128, 131, 134, 139, 145, 148]
    # seletced_images = [69, 95, 123, 139]
    pbar = tqdm(seletced_images)
    for i in pbar:
        depth_image = np.load(depths_npy_paths[i])
        vis_z = np.zeros(depth_image.shape,dtype='uint8')
        cv2.normalize(np.abs(depth_image),vis_z,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        vis_z = np.abs(255-vis_z)
        depth_colormap = cv2.applyColorMap(vis_z,cv2.COLORMAP_TURBO)

        depth_image = median_filter(depth_image, 7)
        # depth_image = cv2.bilateralFilter(depth_image, 15, 75, 75)
        depth_image[depth_image >= 6] = 6
        rectified_image = cv2.imread(rectified_images_paths[i])
        origin_image = cv2.imread(origin_images_paths[i])

        meta, bboxes = detector.inference(origin_image)
        vis_image = detector.visualize(bboxes[0], meta, detector.cfg.class_names, bbox_threshold)
        filtered_bbox = np.array(bboxes[0][0])
        filtered_bbox = filtered_bbox[filtered_bbox[:, -1] > bbox_threshold]

        if len(filtered_bbox) > 0:
            output_pose_2d, output_pose_3d, output_scores = pose_mhp(origin_image, filtered_bbox)
            vis_image_mhp = pose_mhp.visualize(origin_image, output_pose_2d, output_scores)

            tp_pose_result = predict_with_given_det(rectified_image, build_keypoint_detector_input(filtered_bbox), pose_tp, 1, bbox_threshold, 0, False)
            tp_pose_2d = np.array(tp_pose_result["keypoint"][0])[:, :, :2]
            tp_pose_2d = merge_keypoints(tp_pose_2d, "coco")
            vis_image_tp = HourglassModel.visualize(rectified_image, tp_pose_2d)

            # no empty hole
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=50), cv2.COLORMAP_TURBO)
            cv2.imwrite("depth_colormap.jpg", depth_colormap)

            # tp_depth_idx = cv2.perspectiveTransform(tp_pose_2d, perspective_matrix).astype(int)
            tp_depth_idx = tp_pose_2d.copy().astype(int)
            tp_depth = gather_depth_by_idx(depth_image, tp_depth_idx) * 1000

            if i == 139:
                tp_depth[np.where(tp_depth == 6000)] = 1756.5604248046875

            # avoid inf depth
            if np.where(tp_depth == 5000)[0].shape[0] == 0 or True:
                tp_pose_3d = np.concatenate((tp_pose_2d, tp_depth[..., np.newaxis]), axis=2).copy()
                for j in range(tp_pose_3d.shape[0]):
                        tp_pose_3d[j] = pixel2cam(tp_pose_3d[j], cam_focal, cam_princpt)

                view_output_dir = f"{output_dir}/{i}/tp"
                os.makedirs(view_output_dir, exist_ok=True)
                tp_3d_image = vis_3d_multiple_skeleton(tp_pose_3d, np.ones_like(tp_pose_3d), skeleton, model_type="tp", 
                                                        output_dir=view_output_dir, all_angle=all_angle)
                vis_image_tp_3d = vis_image_tp.copy()
                # vis_image_tp_3d = cv2.addWeighted(vis_image_tp_3d, alpha, depth_colormap, 1 - alpha, 0.0)
                vis_image_tp_3d = cv2.resize(vis_image_tp_3d, (1600, 1200))
                depth_colormap = HourglassModel.visualize(depth_colormap, tp_pose_2d)
                tp_3d_image = np.hstack((tp_3d_image, vis_image_tp_3d, cv2.resize(depth_colormap, (1600, 1200))))
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

        pose_mhp_2d = merge_keypoints(output_pose_2d, "muco")
        vis_image_mhp = HourglassModel.visualize(vis_image, pose_mhp_2d)
        cv2.putText(vis_image_mhp, "MobileHumanPose", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        pose_tp_2d = tp_pose_2d
        vis_image_tp = vis_image_tp
        cv2.putText(vis_image_tp, "Ours Hourglass", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

        video_writer.write(np.hstack((vis_image_mhp, vis_image_tp)))

        num_person = len(filtered_bbox) if len(bboxes[0][0]) > 0 else 0
        # print(f" pipeline time: {end - start:.3f}s | persons: {num_person}")

        pbar.set_description_str(f"{num_person} person{'s' if num_person > 1 else ''}")
    
    video_writer.release()