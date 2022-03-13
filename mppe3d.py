import os

import cv2
import numpy as np
import onnxruntime as rt


def process_bbox(bbox, width, height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = 1
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox


def generate_patch_image(cvimg, bbox):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    input_length = 256
    half_length = input_length / 2
    src = np.array([[bb_c_x, bb_c_y], [bb_width, bb_c_y], [bb_c_x, bb_height]])
    dst = np.array([[half_length, half_length], [input_length, half_length], [half_length, input_length]])
    trans = cv2.getAffineTransform(src, dst)
    img_patch = cv2.warpAffine(img, trans, (input_length, input_length), flags=cv2.INTER_LINEAR)

    img_patch = img_patch[:, :, ::-1].copy()
    img_patch = img_patch.astype(np.float32)

    return img_patch, trans


class RootNet(object):
    def __init__(self, onnx_filepath):
        assert os.path.exists(onnx_filepath), f"{onnx_filepath} not exists"
        self.sess = rt.InferenceSession(onnx_filepath)
        outputs = self.sess.get_outputs()
        self.output_names = list(map(lambda output: output.name, outputs))

        self.bbox_real = (2000, 2000)
        self.focal = [1500, 1500]
        self.k_divisor = self.bbox_real[0] * self.bbox_real[1] * self.focal[0] * self.focal[1]

    def inference_single_person(self, crop_img, k_value):
        root_3d = self.sess.run(self.output_names, {"img": crop_img, "k_value": k_value})
        return root_3d

    def inference(self, img: np.ndarray, bboxes):
        num_person = len(bboxes)
        original_img_height, original_img_width = img.shape[:2]

        for i in range(num_person):
            bbox = process_bbox(bboxes[i], original_img_height, original_img_width)
            img, trans = generate_patch_image(img, bbox)
            k_value = np.sqrt(self.k_divisor / (bbox[2]*bbox[3]))

            root_3d = self.inference_single_person(img, k_value)
