import sys
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from scipy.special import softmax
from imread_from_url import imread_from_url

from utils_pose_estimation import pixel2cam, crop_image, joint_num, draw_skeleton, draw_heatmap, \
    vis_3d_multiple_skeleton


class MobileHumanPose():
    def __init__(self, model_path, focal_length=(1500, 1500), principal_points=(1280 / 2, 720 / 2)):
        self.focal_length = list(focal_length)
        self.principal_points = list(principal_points)

        # Initialize model
        self.model = self.initialize_model(model_path)

    def __call__(self, image, bbox, abs_depth=1.0):
        return self.estimate_pose(image, bbox, abs_depth)

    def initialize_model(self, model_path):
        # use gpu for inference
        if ort.get_device() == "GPU":
            self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        else:
            self.session = ort.InferenceSession(model_path)

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def estimate_pose(self, image, bbox, abs_depth=1000):
        input_tensor = self.prepare_input(image, bbox)

        output = self.inference(input_tensor)

        keypoints = self.process_output(output, abs_depth, bbox)

        return keypoints

    def prepare_input(self, image, bbox):
        img = crop_image(image, bbox)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.img_height, self.img_width, self.img_channels = img.shape
        principal_points = [self.img_width / 2, self.img_height / 2]

        img_input = cv2.resize(img, (self.input_width, self.input_height))

        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]

        return img_input.astype(np.float32)

    def inference(self, input_tensor):
        output = self.session.run(self.output_names, {self.input_name: input_tensor})[0]

        return np.squeeze(output)

    def process_output(self, output, abs_depth, bbox):
        heatmaps = output.reshape((-1, joint_num, self.output_depth * self.output_height * self.output_width))
        heatmaps = softmax(heatmaps, 2)

        scores = np.squeeze(np.max(heatmaps, 2))  # Ref: https://github.com/mks0601/3DMPPE_POSENET_RELEASE/issues/47

        heatmaps = heatmaps.reshape((-1, joint_num, self.output_depth, self.output_height, self.output_width))

        accu_x = heatmaps.sum(axis=(2, 3))
        accu_y = heatmaps.sum(axis=(2, 4))
        accu_z = heatmaps.sum(axis=(3, 4))

        accu_x = accu_x * np.arange(self.output_width, dtype=np.float32)
        accu_y = accu_y * np.arange(self.output_height, dtype=np.float32)
        accu_z = accu_z * np.arange(self.output_depth, dtype=np.float32)

        accu_x = accu_x.sum(axis=2, keepdims=True)
        accu_y = accu_y.sum(axis=2, keepdims=True)
        accu_z = accu_z.sum(axis=2, keepdims=True)

        scores2 = []
        for i in range(joint_num):
            scores2.append(heatmaps.sum(axis=2)[0, i, int(accu_y[0, i, 0]), int(accu_x[0, i, 0])])

        accu_x = accu_x / self.output_width
        accu_y = accu_y / self.output_height
        accu_z = accu_z / self.output_depth * 2 - 1

        coord_out = np.squeeze(np.concatenate((accu_x, accu_y, accu_z), axis=2))

        pose_2d = coord_out[:, :2]
        pose_2d[:, 0] = pose_2d[:, 0] * self.img_width + bbox[0]
        pose_2d[:, 1] = pose_2d[:, 1] * self.img_height + bbox[1]

        joint_depth = coord_out[:, 2] * 1000 + abs_depth

        pose_3d = pixel2cam(pose_2d, joint_depth, self.focal_length, self.principal_points)

        # Calculate the joint heatmap
        person_heatmap = cv2.resize(np.sqrt(heatmaps.sum(axis=(1, 2))[0, :, :]), (self.img_width, self.img_height))

        return pose_2d, pose_3d, person_heatmap, scores

    def get_input_details(self):
        self.input_name = self.session.get_inputs()[0].name

        self.input_shape = self.session.get_inputs()[0].shape
        self.channels = self.input_shape[1]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()

        self.output_names = []
        self.output_names.append(model_outputs[0].name)

        self.output_shape = model_outputs[0].shape
        self.output_depth = self.output_shape[1] // joint_num
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3]

