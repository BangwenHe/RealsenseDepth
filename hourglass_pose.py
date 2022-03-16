import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
import tensorflow as tf

from utils_pose_estimation import crop_image
from detector import build_default_detector
from rootnet3d.common.utils.pose_utils import process_bbox
from rootnet3d.data.dataset import generate_patch_image


joint_num = 14
skeleton = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (2, 8), (8, 9), (9, 10), (5, 11), (11, 12), (12, 13)]
cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, joint_num + 2)]
colors_cv = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
colors_plt = [np.array((c[0], c[1], c[2])) for c in colors]


class HourglassModel(object):
    def __init__(self, tflite_filepath):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_filepath)
        self.interpreter.allocate_tensors()

        self.get_input_details()
        self.get_output_details()

        self.expand_ratio = 1

    def __call__(self, img, bboxes):
        num_person = len(bboxes)

        output_pose_2d = np.zeros((num_person, self.output_channels, 2))
        for i in range(num_person):
            input_img, processed_bbox = self.preprocess(img, bboxes[i], idx=i)
            
            self.interpreter.set_tensor(self.input_details[0]["index"], input_img)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

            pose_2d = self.postprocess(processed_bbox, output_data)
            output_pose_2d[i] = pose_2d
        
        return output_pose_2d

    def process_bbox(self, bbox):
        input_aspect = self.input_width / self.input_height

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        center_x = w / 2 + bbox[0]
        center_y = h / 2 + bbox[1]

        if abs(input_aspect - w / h) > 0.5:
            if w / input_aspect < h:
                w = h * (input_aspect - 0.3)
            elif w / input_aspect > h:
                h = w / (input_aspect - 0.3)

        processed_bbox = np.zeros_like(bbox)
        processed_bbox[0] = max(center_x - w / 2, 0)
        processed_bbox[1] = max(center_y - h / 2, 0)
        processed_bbox[2] = min(center_x + w / 2, 480)
        processed_bbox[3] = min(center_y + h / 2, 640)
        
        return processed_bbox

    def preprocess(self, img, bbox, idx=0):
        """
        resize image
        """
        processed_bbox = self.process_bbox(bbox)
        input_img = crop_image(img, processed_bbox)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # black_img = np.zeros((self.input_height, self.input_width), dtype=np.uint8)
        # h, w, _ = input_img.shape
        # black_img[0:h, 0:w] = input_img
        # input_img = black_img

        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        cv2.imwrite(f"crop{idx}.jpg", cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))

        input_img = input_img.astype(np.float32)
        # normalize
        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[np.newaxis, np.newaxis, :]
        # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[np.newaxis, np.newaxis, :]
        # input_img -= mean
        # input_img /= std

        input_img = input_img[np.newaxis, :]  # NHWC

        return input_img, processed_bbox

    def postprocess(self, bbox, heatmaps):
        # (1, 48, 48, 14) -> nxyc
        heat_y = heatmaps.sum(axis=2)
        pose_y = np.argmax(heat_y, axis=1)

        heat_x = heatmaps.sum(axis=1)
        pose_x = np.argmax(heat_x, axis=1)

        pose = np.concatenate((pose_x, pose_y))

        # resize coordinate back
        pose[0] = pose[0] * (bbox[2] - bbox[0]) / self.output_width + bbox[0]
        pose[1] = pose[1] * (bbox[3] - bbox[1]) / self.output_height + bbox[1]

        # tranpose from 2x14 to 14x2
        return pose.T

    def get_input_details(self):
        self.input_details = self.interpreter.get_input_details()
        self.input_channels = self.input_details[0]['shape'][3]
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

    def get_output_details(self):
        self.output_details = self.interpreter.get_output_details()
        self.output_channels = self.output_details[0]['shape'][3]
        self.output_height = self.output_details[0]['shape'][1]
        self.output_width = self.output_details[0]['shape'][2]

    @staticmethod
    def visualize(original_img, pose_2d):
        vis_img = original_img.copy()

        for j in range(len(pose_2d)):
            for i, segment in enumerate(skeleton):
                point1_id = segment[0]
                point2_id = segment[1]

                point1 = (int(pose_2d[j][point1_id, 0]), int(pose_2d[j][point1_id, 1]))
                point2 = (int(pose_2d[j][point2_id, 0]), int(pose_2d[j][point2_id, 1]))

                vis_img = cv2.line(vis_img, point1, point2,colors_cv[i], thickness=3, lineType=cv2.LINE_AA)
                cv2.circle(vis_img, point1, radius=5, color=colors_cv[i], thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(vis_img, point2, radius=5, color=colors_cv[i], thickness=-1, lineType=cv2.LINE_AA)

        return vis_img


if __name__ == "__main__":
    model = HourglassModel("model.tflite")
    print(model.input_details, model.output_details)

    import cv2
    video_cap = cv2.VideoCapture("data/offline_2p.mp4")
    # video_cap = cv2.VideoCapture("data/offline_square.mp4")
    # video_cap = cv2.VideoCapture("data/offline_far.mp4")
    # video_cap = cv2.VideoCapture("data/offline_sit.mp4")
    video_cap.set(1, 40)
    _, frame = video_cap.read()

    detector_image = frame.copy()
    hg_image = frame.copy()

    detector = build_default_detector("nanodet-m.yml", "nanodet_m.ckpt", "cuda:0")
    meta, bboxes = detector.inference(detector_image)
    vis_image = detector.visualize(bboxes[0], meta, detector.cfg.class_names, 0.5)
    filtered_bbox = np.array(bboxes[0][0])
    filtered_bbox = filtered_bbox[filtered_bbox[:, -1] > 0.5]

    pose_2d = model(hg_image, filtered_bbox)
    print(pose_2d.shape)
    vis_image = model.visualize(vis_image, pose_2d)

    cv2.imwrite("frame.jpg", frame)
    cv2.imwrite("ours.jpg", vis_image)