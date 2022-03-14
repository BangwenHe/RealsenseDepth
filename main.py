## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import time

import pyrealsense2 as rs
import numpy as np
import cv2
import copy
import mmcv
from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)

from detector import build_default_detector
from yamppe3d import get_root_pose_net
from mobile_human_pose import MobileHumanPose
from utils_pose_estimation import draw_skeleton
from utils_camera import get_realsense_perspective_matrix, perspective_transform, gather_depth_by_idx


def main():
    # Configure depth and color streams
    pipeline = rs.pipeline(rs.context())
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    print(device_product_line)

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    perspective_matrix = get_realsense_perspective_matrix()

    # build detector
    detector = build_default_detector("nanodet-m.yml", "nanodet_m.ckpt", "cuda:0")
    # posenet = get_root_pose_net(
    #     r"C:\Users\Bangwen\PycharmProjects\moon_pose_estimation_setup\snapshot_18.pth.tar",
    #     r"C:\Users\Bangwen\PycharmProjects\moon_pose_estimation_setup\snapshot_24.pth.tar")
    pose_mhp = MobileHumanPose(
        r"C:\Users\Bangwen\PycharmProjects\ONNX-Mobile-Human-Pose-3D\models\mobile_human_pose_working_well_256x256.onnx")

    pose_hrhrnet = init_pose_model(r"C:\Users\Bangwen\PycharmProjects\mmpose\configs\body\2d_kpt_sview_rgb_img\associative_embedding\coco\higherhrnet_w32_coco_512x512.py",
                              "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512-8ae85183_20200713.pth",
                              device="cuda:0")

    bbox_threshold = 0.5
    run_model = True

    # Start streaming
    pipeline.start(config)
    depths = []

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if run_model:
                start = time.time()

                meta, bboxes = detector.inference(color_image)
                vis_image = detector.visualize(bboxes[0], meta, detector.cfg.class_names, bbox_threshold)

                if len(bboxes[0][0]) > 0:
                    filtered_bbox = np.array(bboxes[0][0])
                    filtered_bbox = filtered_bbox[filtered_bbox[:, -1] > bbox_threshold]
                    # root_depth_list, output_pose_2d, output_pose_3d = posenet.inference(color_image, filtered_bbox, bbox_threshold)
                    # vis_image = posenet.visualize(vis_image, output_pose_2d)

                    for i, bbox in enumerate(filtered_bbox):
                        # Draw the estimated pose
                        keypoints, pose_3d, scores = pose_mhp(color_image, bbox, 1)
                        vis_image_mhp = draw_skeleton(vis_image, keypoints, bbox[:2], scores)

                    pose_results, returned_outputs = inference_bottom_up_pose_model(
                        pose_hrhrnet, color_image)
                    vis_image_gt = vis_pose_result(pose_hrhrnet, vis_image, pose_results)

                    gt_2d = np.array([res["keypoints"] for res in pose_results])
                    # gt_2d_depth_idx = cv2.perspectiveTransform(gt_2d[:, :, :2], perspective_matrix).astype(int)
                    gt_2d_depth_idx = perspective_transform(perspective_matrix, gt_2d[:, :, :2]).astype(int)
                    gt_2d_depth = gather_depth_by_idx(depth_image, gt_2d_depth_idx)

                end = time.time()
                num_person = len(filtered_bbox) if len(bboxes[0][0]) > 0 else 0
                print(f" pipeline time: {end - start:.3f}s | persons: {num_person}")
            else:
                vis_image = color_image

            # add depths into list
            depths.append(copy.deepcopy(depth_image))

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_TURBO)
            vis_image = cv2.warpPerspective(vis_image, perspective_matrix, (640, 480))

            depth_color_image = cv2.addWeighted(vis_image, 0.7, depth_colormap, 0.3, 0.0)
            # hybrid_image = cv2.resize(hybrid_image, (960, 720))
            hybrid_image = np.hstack([depth_color_image, vis_image_mhp, vis_image_gt])

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', hybrid_image)
            cv2.waitKey(1)

    finally:
        # np.save("test.npy", np.asarray(depths))

        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    main()