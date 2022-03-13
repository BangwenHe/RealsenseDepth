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

# Configure depth and color streams
pipeline = rs.pipeline()
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

src = np.float32([[74, 109], [37, 654], [841, 128], [833, 706]]) / 1.5
dst = np.float32([[201, 206], [177, 541], [691, 217], [677, 578]]) / 1.5
perspective_matrix = cv2.getPerspectiveTransform(src, dst)

# build detector
detector = build_default_detector("nanodet-m.yml", "nanodet_m.ckpt", "cuda:0")
# posenet = get_root_pose_net(
#     r"C:\Users\Bangwen\PycharmProjects\moon_pose_estimation_setup\snapshot_18.pth.tar",
#     r"C:\Users\Bangwen\PycharmProjects\moon_pose_estimation_setup\snapshot_24.pth.tar")
pose_estimator = MobileHumanPose(
    r"C:\Users\Bangwen\PycharmProjects\ONNX-Mobile-Human-Pose-3D\models\mobile_human_pose_working_well_256x256.onnx")

hrhrnet = init_pose_model(r"C:\Users\Bangwen\PycharmProjects\mmpose\configs\body\2d_kpt_sview_rgb_img\associative_embedding\coco\higherhrnet_w32_coco_512x512.py",
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

                # for i, bbox in enumerate(filtered_bbox):
                #     # Draw the estimated pose
                #     keypoints, pose_3d, person_heatmap, scores = pose_estimator(color_image, bbox, 1)
                #     vis_image = draw_skeleton(vis_image, keypoints, bbox[:2], scores)

                pose_results, returned_outputs = inference_bottom_up_pose_model(
                    hrhrnet, color_image)
                vis_image = vis_pose_result(hrhrnet, vis_image, pose_results)

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

        hybrid_image = cv2.addWeighted(vis_image, 0.7, depth_colormap, 0.3, 0.0)
        hybrid_image = cv2.resize(hybrid_image, (960, 720))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', hybrid_image)
        cv2.waitKey(1)

finally:
    # np.save("test.npy", np.asarray(depths))

    # Stop streaming
    pipeline.stop()
