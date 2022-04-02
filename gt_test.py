import glob
import os

import cv2
import numpy as np
from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
    
if __name__ == "__main__":
    pose_hrhrnet = init_pose_model("/home/tclab/bangwhe/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py",
                              "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth",
                              device="cuda:0")
    images_paths = glob.glob(os.path.join("accuracy_test_data/exp1/depth_camera/", "capture*.jpg"))
    images_paths = sorted(images_paths, key=lambda path: int(os.path.split(path)[-1].split(".")[0].split("_")[-1]))

    for idx, image_path in enumerate(images_paths):
        image = cv2.imread(image_path)
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_hrhrnet, image)
        vis_image_gt = vis_pose_result(pose_hrhrnet, image, pose_results)

        cv2.imwrite(f"debug2/gt_debug_{idx}.jpg", vis_image_gt)