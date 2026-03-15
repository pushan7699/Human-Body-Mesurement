from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from src.util import image as img_util
from src.RunModel import RunModel

# --------------------------------------------------
# GLOBAL MODEL (LOAD ONCE)
# --------------------------------------------------
sess = tf.Session()
model = RunModel(sess=sess)

# --------------------------------------------------
# IMAGE PREPROCESSING (ROBUST & FORGIVING)
# --------------------------------------------------
def preprocess_image_array(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    # Center square crop (VERY IMPORTANT)
    side = min(h, w)
    start_x = (w - side) // 2
    start_y = (h - side) // 2
    img = img[start_y:start_y+side, start_x:start_x+side]

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = 2 * (img - 0.5)

    return img


# --------------------------------------------------
# MAIN (USED BY WEBCAM / IMAGE)
# --------------------------------------------------
def main(img_bgr, json_path=None, _=None):
    """
    img_bgr: person image (full or cropped)
    RETURNS:
        joints, verts, cams, joints3d
    """

    input_img = preprocess_image_array(img_bgr)
    input_img = np.expand_dims(input_img, 0)

    joints, verts, cams, joints3d, _ = model.predict(
        input_img, get_theta=True
    )

    # 🔥 DO NOT REJECT — JUST WARN
    if joints3d is None:
        print("⚠️ Model returned no joints")
        return None, None, None, None

    print("✅ Pose estimated (no hard rejection)")

    return joints, verts, cams, joints3d
