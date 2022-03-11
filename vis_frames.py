import cv2
import numpy as np


if __name__ == "__main__":
    frames = np.load("4person.npy")

    for i in range(len(frames)):
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(frames[i], alpha=0.03), cv2.COLORMAP_TURBO)
        cv2.imshow("depth", depth_colormap)
        cv2.waitKey(30)