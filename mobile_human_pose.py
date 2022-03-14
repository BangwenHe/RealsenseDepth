from ntpath import join
import sys
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from scipy.special import softmax
from imread_from_url import imread_from_url

from utils_pose_estimation import pixel2cam, crop_image, joint_num, draw_skeleton, draw_heatmap, \
    vis_3d_multiple_skeleton
from rootnet3d.common.utils.pose_utils import process_bbox
from rootnet3d.data.dataset import generate_patch_image


class MobileHumanPose():
    def __init__(self, model_path, focal=None, princpt=None):
        if focal is None:
            self.focal = [604.607, 604.648]

        if princpt is None:
            self.princpt = [321.509, 239.94]

        # Initialize model
        self.original_img_height, self.original_img_width = (480, 640)
        self.model = self.initialize_model(model_path)
        
        # obtain from original config file
        self.depth_dim = 64
        self.bbox_3d_shape = (2000, 2000, 2000)

    # def __call__(self, image, bbox, abs_depth=1.0):
    #     return self.estimate_pose(image, bbox, abs_depth)

    def __call__(self, image, bboxes):
        return self.estimate_pose_3d(image, bboxes)

    def initialize_model(self, model_path):
        # use gpu for inference if available
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

    def soft_argmax(self, output):
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

        coord_out = np.squeeze(np.concatenate((accu_x, accu_y, accu_z), axis=2))

        return coord_out, scores

    def process_output(self, output, abs_depth, bbox):
        coord_out, scores = self.soft_argmax(output)

        accu_x = coord_out[:, 0] / self.output_width
        accu_y = coord_out[:, 1] / self.output_height
        accu_z = coord_out[:, 2] / self.output_depth * 2 - 1

        coord_out = np.squeeze(np.concatenate((accu_x, accu_y, accu_z), axis=2))

        pose_2d = coord_out[:, :2]
        pose_2d[:, 0] = pose_2d[:, 0] * self.img_width + bbox[0]
        pose_2d[:, 1] = pose_2d[:, 1] * self.img_height + bbox[1]

        joint_depth = coord_out[:, 2] * 1000 + abs_depth

        pose_3d = pixel2cam(pose_2d, joint_depth, self.focal_length, self.principal_points)

        return pose_2d, pose_3d, scores

    def estimate_pose_3d(self, img, bboxes):
        bboxes = bboxes[:, 0:4]
        original_img = img.copy()

        person_num = len(bboxes)
        output_pose_2d = np.zeros((person_num, joint_num, 2))
        output_pose_3d = np.zeros((person_num, joint_num, 3))
        output_scores = np.zeros((person_num, joint_num))

        for n in range(person_num):
            bbox = process_bbox(np.array(bboxes[n]), self.original_img_width, self.original_img_height)
            img, img2bb_trans = generate_patch_image(original_img, bbox, False, 0.0)
            img_tensor = self.transform_img(img)

            output = self.inference(img_tensor)
            pose_3d, scores = self.soft_argmax(output)

            # inverse affine transform to get original coordinate
            pose_3d[:,0] = pose_3d[:,0] / self.output_width * self.input_width
            pose_3d[:,1] = pose_3d[:,1] / self.output_height * self.input_height
            pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
            pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            output_pose_2d[n] = pose_3d[:,:2].copy()

            # get root-relative continous depth
            pose_3d[:,2] = (pose_3d[:,2] / self.depth_dim * 2 - 1) * (self.bbox_3d_shape[0]/2)
            pose_3d = pixel2cam(pose_3d, self.focal, self.princpt)
            output_pose_3d[n] = pose_3d.copy()
            output_scores[n] = scores.copy()

        return output_pose_2d, output_pose_3d, output_scores

    def transform_img(self, img: np.ndarray):
        # to tensor
        img_tensor = img.transpose((2, 0, 1))[np.newaxis, :]

        # normalize
        mean = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis]
        std = np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis]
        img_tensor -= mean
        img_tensor /= std

        return img_tensor

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

    def visualize(self, img, pose_2d, pose_score):
        vis_img = img.copy()

        for pose, score in zip(pose_2d, pose_score):
            vis_img = draw_skeleton(vis_img, pose, score)
        
        return vis_img