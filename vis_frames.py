import cv2
import os
import glob

from matplotlib.pyplot import get
import numpy as np

from utils_camera import get_realsense_perspective_matrix, get_realsense_affine_matrix


if __name__ == "__main__":
    # frames = np.load("data/offline_vertical2.npy")
    # video_cap = cv2.VideoCapture("data/offline_vertical2.mp4")
    # frames = np.load("data/offline_2p.npy")
    # video_cap = cv2.VideoCapture("data/offline_2p.mp4")
    images_folder = "data/0317/2p"
    rectified_images_paths = sorted(glob.glob(os.path.join(images_folder, "*rectified*.png")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))
    origin_images_paths = sorted(glob.glob(os.path.join(images_folder, "*original*.png")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))
    depths_npy_folder = "data/0317/2p"
    depths_npy_paths = sorted(glob.glob(os.path.join(images_folder, "depth*.npy")), key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[-1]))


    is_vertical = True
    perspective_matrix = get_realsense_affine_matrix(is_vertical=is_vertical)
    resolution = (480, 640) if is_vertical else (640, 480)

    alpha = 0.9

    for i in range(len(rectified_images_paths)):
        bgr_image = cv2.imread(rectified_images_paths[i])
        depth_map = np.load(depths_npy_paths[i])

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=50), cv2.COLORMAP_TURBO)
        # bgr_image = cv2.warpPerspective(bgr_image, perspective_matrix, resolution)
        # bgr_image = cv2.warpAffine(bgr_image, perspective_matrix, resolution)
        hybrid_image = depth_colormap
        # hybrid_image = cv2.addWeighted(depth_colormap, alpha, bgr_image, 1 - alpha, 0.0)

        # cv2.imshow("depth", hybrid_image)
        # cv2.waitKey(0)

        # 7~15 24~34 42~69 74~121
        cv2.imwrite(f"depth_maps/{i:03d}.jpg", hybrid_image)
