import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    parser = argparse.ArgumentParser(description="capture rgb image and depth data from realsense camera")

    parser.add_argument("--video_filepath", type=str, required=True, help="path to save rgb video")
    parser.add_argument("--npy_filepath", type=str, required=True, help="path to save depth npy data")
    parser.add_argument("--num_frames", type=int, default=1000, help="number of capture frames")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_filepath = args.video_filepath
    npy_filepath = args.npy_filepath
    os.makedirs(Path(video_filepath).parent, exist_ok=True)
    os.makedirs(Path(npy_filepath).parent, exist_ok=True)

    assert not os.path.exists(video_filepath), f"{os.path.abspath(video_filepath)} already exists"
    assert not os.path.exists(npy_filepath), f"{os.path.abspath(npy_filepath)} already exists"

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

    src = np.float32([[74, 109], [37, 654], [841, 128], [833, 706]]) / 1.5
    dst = np.float32([[201, 206], [177, 541], [691, 217], [677, 578]]) / 1.5
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)

    # Start streaming
    cfg = pipeline.start(config)
    depths = []

    streams = cfg.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = streams.get_intrinsics()

    # Print information about both cameras
    print("Intrinsics:",  intrinsics)

    video_resolution = (640, 480)
    video_fps = 30
    video_cc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_filepath, video_cc, video_fps, video_resolution)

    i = 0
    max_frames = args.num_frames

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
            # add depths into list
            depths.append(depth_image.copy())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_TURBO)
            vis_image = cv2.warpPerspective(color_image, perspective_matrix, video_resolution)

            hybrid_image = cv2.addWeighted(vis_image, 0.7, depth_colormap, 0.3, 0.0)
            hybrid_image = cv2.resize(hybrid_image, video_resolution)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', hybrid_image)
            cv2.waitKey(1)

            i += 1
            video_writer.write(color_image)
            print(f"{i} / {max_frames}", end="\r")
            if i > max_frames:
                np.save(npy_filepath, np.stack(depths))
                break

    finally:
        # np.save("test.npy", np.asarray(depths))

        # Stop streaming
        pipeline.stop()