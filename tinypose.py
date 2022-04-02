import cv2
import numpy as np

from ppdet_deploy.keypoint_infer import PredictConfig_KeyPoint, KeyPoint_Detector
from ppdet_deploy.det_keypoint_unite_infer import predict_with_given_det
from detector import build_default_detector
from ppdet_deploy.visualize import draw_pose


def build_keypoint_detector_input(bboxes):
    num_person = len(bboxes)
    res = {"boxes": np.zeros((num_person, 6), dtype=bboxes.dtype), "boxes_num": np.array([num_person])}

    res["boxes"][:, 1] = bboxes[:, -1]
    res["boxes"][:, 2:] = bboxes[:, :-1]

    return res


if __name__ == "__main__":
    model_dir = "/home/tclab/bangwhe/PaddleDetection/output_inference/tinypose_256x192"

    pred_config = PredictConfig_KeyPoint(model_dir = "/home/tclab/bangwhe/PaddleDetection/output_inference/tinypose_256x192")
    pose_detector = KeyPoint_Detector(pred_config, model_dir, device="gpu", run_mode="paddle", use_dark=True)

    # results = detector.predict(["data/0317/2p/left_rectified_17.png"], 0)
    # print(results)

    frame = cv2.imread(f"data/0317/2p/left_rectified_31.png")
    detector_image = frame.copy()
    hg_image = frame.copy()

    detector = build_default_detector("nanodet-m.yml", "nanodet_m.ckpt", "cuda:0")
    meta, bboxes = detector.inference(detector_image)
    vis_image = detector.visualize(bboxes[0], meta, detector.cfg.class_names, 0.3)
    filtered_bbox = np.array(bboxes[0][0])
    filtered_bbox = filtered_bbox[filtered_bbox[:, -1] > 0.4]

    print(filtered_bbox.shape)

    keypoint_res = predict_with_given_det(
        hg_image, build_keypoint_detector_input(filtered_bbox), pose_detector, 1,
        0.4, 0, False)

    vis_image = draw_pose(hg_image, keypoint_res, returnimg=True)

    print(np.array(keypoint_res["keypoint"][0]).shape)
    cv2.imwrite("frame.jpg", frame)
    cv2.imwrite("tinypose.jpg", vis_image)