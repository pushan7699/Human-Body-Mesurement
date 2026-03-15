"""
TF util operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# =====================================================
# KEYPOINT L1 LOSS (TF1 SAFE VERSION)
# =====================================================
def keypoint_l1_loss(kp_gt, kp_pred, scale=1., name=None):
    """
    computes: Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint_l1_loss", [kp_gt, kp_pred]):

        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 2))

        # visibility mask
        vis = tf.expand_dims(tf.cast(kp_gt[:, 2], tf.float32), 1)

        # manual L1 (avoid tf.losses.absolute_difference)
        diff = tf.abs(kp_gt[:, :2] - kp_pred)
        weighted = diff * vis

        res = tf.reduce_mean(weighted)

        return res


# =====================================================
# 3D PARAM LOSS (TF1 SAFE VERSION)
# =====================================================
def compute_3d_loss(params_pred, params_gt, has_gt3d):
    """
    Computes L2 loss between predicted and GT 3D params
    Inputs:
      params_pred: N x {226, 42}
      params_gt:   N x {226, 42}
      has_gt3d:    N x 1 tf.float32 {0.,1.}
    """

    with tf.name_scope("3d_loss", [params_pred, params_gt, has_gt3d]):

        weights = tf.expand_dims(tf.cast(has_gt3d, tf.float32), 1)

        # manual MSE (avoid tf.losses.mean_squared_error)
        diff = params_gt - params_pred
        sq = tf.square(diff)
        weighted = sq * weights

        res = tf.reduce_mean(weighted) * 0.5

        return res


# =====================================================
# ALIGN BY PELVIS (UNCHANGED)
# =====================================================
def align_by_pelvis(joints):
    """
    Assumes joints is N x 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    with tf.name_scope("align_by_pelvis", [joints]):
        left_id = 3
        right_id = 2
        pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.
        return joints - tf.expand_dims(pelvis, axis=1)