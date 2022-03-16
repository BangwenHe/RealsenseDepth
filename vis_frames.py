import cv2
from matplotlib.pyplot import get
import numpy as np

from utils_camera import get_realsense_perspective_matrix, get_realsense_affine_matrix


if __name__ == "__main__":
    # frames = np.load("data/offline_vertical2.npy")
    # video_cap = cv2.VideoCapture("data/offline_vertical2.mp4")
    frames = np.load("data/offline_2p.npy")
    video_cap = cv2.VideoCapture("data/offline_2p.mp4")

    is_vertical = True
    perspective_matrix = get_realsense_affine_matrix(is_vertical=is_vertical)
    resolution = (480, 640) if is_vertical else (640, 480)

    alpha = 0.3

    for i in range(len(frames)):
        _, bgr_image = video_cap.read()
        depth_map = frames[i]

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_TURBO)
        # bgr_image = cv2.warpPerspective(bgr_image, perspective_matrix, resolution)
        bgr_image = cv2.warpAffine(bgr_image, perspective_matrix, resolution)
        hybrid_image = cv2.addWeighted(depth_colormap, alpha, bgr_image, 1 - alpha, 0.0)

        cv2.imshow("depth", hybrid_image)
        cv2.waitKey(0)
