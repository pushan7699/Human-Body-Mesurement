import os
import sys
import tarfile
import argparse
from six.moves import urllib

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

from demo import main
from src.tf_smpl.batch_smpl import SMPL

# =======================
# HEIGHT & WEIGHT UTILS
# =======================

def estimate_height_cm(joints3d):
    """
    joints3d: (N, 3) SMPL joints in meters
    """
    y_coords = joints3d[:, 1]
    height_m = y_coords.max() - y_coords.min()
    return height_m * 100.0


def estimate_body_volume(verts, faces):
    """
    verts: (V, 3)
    faces: (F, 3)
    """
    volume = 0.0
    for f in faces:
        v1, v2, v3 = verts[f]
        volume += np.dot(np.cross(v1, v2), v3)
    return abs(volume) / 6.0


def estimate_weight_kg(volume_m3):
    BODY_DENSITY = 985  # kg/m^3 (average human density)
    return volume_m3 * BODY_DENSITY


# =======================
# DEEPLAB MODEL
# =======================

class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None

        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]}
        )
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


# =======================
# ARGUMENTS
# =======================

parser = argparse.ArgumentParser(description='Human Height & Weight Estimation')
parser.add_argument('-i', '--input', required=True, help='Input image path')
args = parser.parse_args()

# =======================
# DOWNLOAD DEEPLAB MODEL
# =======================

MODEL_NAME = 'xception_coco_voctrainval'
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URL = 'deeplabv3_pascal_trainval_2018_01_04.tar.gz'

model_dir = 'deeplab_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

download_path = os.path.join(model_dir, _MODEL_URL)
if not os.path.exists(download_path):
    print("Downloading DeepLab model...")
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URL, download_path)

MODEL = DeepLabModel(download_path)
print("DeepLab model loaded")

# =======================
# BACKGROUND REMOVAL
# =======================

image = Image.open(args.input)
resized_im, seg = MODEL.run(image)

seg = cv2.resize(seg.astype(np.uint8), image.size)
mask = (seg == 15).astype(np.uint8) * 255  # person class

img = np.array(image)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

bg_removed = cv2.bitwise_and(img, img, mask=mask)

# =======================
# HMR / SMPL INFERENCE
# =======================

joints, verts, cams, joints3d = main(bg_removed, None, None)

# =======================
# HEIGHT & WEIGHT
# =======================

height_cm = estimate_height_cm(joints3d[0])

smpl = SMPL('models/neutral_smpl_with_cocoplus_reg.pkl')
volume_m3 = estimate_body_volume(verts[0], smpl.faces)
weight_kg = estimate_weight_kg(volume_m3)

# =======================
# OUTPUT
# =======================

print("\n========== ESTIMATION RESULTS ==========")
print(f"Estimated Height : {height_cm:.2f} cm")
print(f"Estimated Weight : {weight_kg:.2f} kg (estimated)")
print("=======================================\n")
