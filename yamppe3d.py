"""
use distribution package of https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE and https://github.com/mks0601/3DMPPE_POSENET_RELEASE
to inference.
"""

import os

import cv2
import math
import numpy as np
import torch
from torch.nn.parallel.data_parallel import DataParallel
import torchvision.transforms as T

from rootnet3d.main.model import get_root_net
from rootnet3d.main.config import cfg as rootnet_cfg
from rootnet3d.common.utils.pose_utils import process_bbox
from rootnet3d.data.dataset import generate_patch_image

from posenet3d.main.model import get_pose_net
from posenet3d.main.config import cfg as posenet_cfg
from posenet3d.common.utils.pose_utils import pixel2cam
from posenet3d.common.utils.vis import vis_keypoints

# MuCo joint set
joint_num = 21
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip',
               'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand',
               'R_Toe', 'L_Toe')
flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20))
skeleton = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13),
            (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))


class RootPoseNet(object):
    def __init__(self, rootnet, posenet):
        self.rootnet = rootnet
        self.posenet = posenet

        self.transform = torch.nn.Sequential(
            T.Normalize(mean=rootnet_cfg.pixel_mean, std=rootnet_cfg.pixel_std))
        # self.transform = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize(mean=rootnet_cfg.pixel_mean, std=rootnet_cfg.pixel_std)
        # ])

        self.original_img_height, self.original_img_width = (480, 640)
        self.focal = [678, 678]
        self.princpt = [318, 228]

    def preprocess_bboxes(self, bboxes, bbox_threshold):
        bboxes = np.array(bboxes)
        bboxes = bboxes[bboxes[:, -1] > bbox_threshold]
        bboxes[:, 2:4] -= bboxes[:, 0:2]
        return bboxes

    def inference(self, img, bboxes, bbox_threshold):
        # bboxes: nx5, xyxyc
        bboxes = self.preprocess_bboxes(bboxes, bbox_threshold)
        bboxes = bboxes[:, 0:4]
        original_img = img.copy()

        person_num = len(bboxes)
        root_depth_list = np.zeros(person_num)
        output_pose_2d = np.zeros((person_num, joint_num, 2))
        output_pose_3d = np.zeros((person_num, joint_num, 3))

        for n in range(person_num):
            bbox = process_bbox(np.array(bboxes[n]), self.original_img_width, self.original_img_height)
            img, img2bb_trans = generate_patch_image(original_img, bbox, False, 0.0)
            img = self.transform(torch.from_numpy(img).cuda().permute(2, 0, 1).unsqueeze(0))
            # img = self.transform(img)[None, :, :, :]
            k_value = np.array(
                [math.sqrt(rootnet_cfg.bbox_real[0] * rootnet_cfg.bbox_real[1] * self.focal[0] * self.focal[1] / (
                        bbox[2] * bbox[3]))]).astype(np.float32)
            k_value = torch.FloatTensor(np.array([k_value])).cuda()[None, :]

            # forward
            with torch.no_grad():
                root_3d = self.rootnet(img, k_value)  # x,y: pixel, z: root-relative depth (mm)

            with torch.no_grad():
                pose_3d = self.posenet(img)  # x,y: pixel, z: root-relative depth (mm)

            img = img[0].cpu().numpy()
            root_3d = root_3d[0].cpu().numpy()
            root_depth_list[n] = root_3d[2]

            # inverse affine transform (restore the crop and resize)
            pose_3d = pose_3d[0].cpu().numpy()
            pose_3d[:, 0] = pose_3d[:, 0] / posenet_cfg.output_shape[1] * posenet_cfg.input_shape[1]
            pose_3d[:, 1] = pose_3d[:, 1] / posenet_cfg.output_shape[0] * posenet_cfg.input_shape[0]
            pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
            pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose((1, 0))).transpose((1, 0))[:, :2]
            output_pose_2d[n] = pose_3d[:, :2]

            # root-relative discrete depth -> absolute continuous depth
            pose_3d[:, 2] = (pose_3d[:, 2] / posenet_cfg.depth_dim * 2 - 1) * (posenet_cfg.bbox_3d_shape[0] / 2) + \
                            root_depth_list[n]
            pose_3d = pixel2cam(pose_3d, self.focal, self.princpt)
            output_pose_3d[n] = pose_3d

        return root_depth_list, output_pose_2d, output_pose_3d

    def visualize(self, original_img: np.ndarray, poses_2d: np.ndarray):
        person_num = poses_2d.shape[0]
        vis_img = original_img.copy()
        for n in range(person_num):
            vis_kps = np.zeros((3, joint_num))
            vis_kps[0, :] = poses_2d[n][:, 0]
            vis_kps[1, :] = poses_2d[n][:, 1]
            vis_kps[2, :] = 1
            vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
        return vis_img


def get_root_pose_net(rootnet_ckpt, posenet_ckpt):
    rootnet = get_root_net(rootnet_cfg, False)
    posenet = get_pose_net(posenet_cfg, False, 21)

    rootnet = DataParallel(rootnet).cuda()
    ckpt = torch.load(rootnet_ckpt)
    rootnet.load_state_dict(ckpt['network'])
    rootnet.eval()

    posenet = DataParallel(posenet).cuda()
    ckpt = torch.load(posenet_ckpt)
    posenet.load_state_dict(ckpt['network'])
    posenet.eval()

    model = RootPoseNet(rootnet, posenet)

    return model
