"""
Tensorflow SMPL implementation as batch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle as pickle

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation


def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class SMPL(object):
    def __init__(self, pkl_path, joint_type='cocoplus', dtype=tf.float32):
        with open(pkl_path, 'rb') as f:
            dd = pickle.load(f, encoding="latin-1")

        # Mean template vertices
        self.v_template = tf.Variable(
            undo_chumpy(dd['v_template']),
            name='v_template',
            dtype=dtype,
            trainable=False)

        # Size of mesh [Number of vertices, 3]
        self.size = [int(self.v_template.shape[0]), 3]

        # number of betas (shape coeffs)
        self.num_betas = dd['shapedirs'].shape[-1]

        # shapedirs: convert to [num_betas, verts*3]
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = tf.Variable(
            shapedir, name='shapedirs', dtype=dtype, trainable=False)

        # Regressor for joint locations given shape - (verts x 24)
        self.J_regressor = tf.Variable(
            dd['J_regressor'].T.todense(),
            name="J_regressor",
            dtype=dtype,
            trainable=False)

        # posedirs -> [207, verts*3]
        num_pose_basis = dd['posedirs'].shape[-1]
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = tf.Variable(
            posedirs, name='posedirs', dtype=dtype, trainable=False)

        # parents
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights: verts x 24
        self.weights = tf.Variable(
            undo_chumpy(dd['weights']),
            name='lbs_weights',
            dtype=dtype,
            trainable=False)

        # joint regressor
        if 'cocoplus_regressor' in dd:
            self.joint_regressor = tf.Variable(
                dd['cocoplus_regressor'].T.todense(),
                name="cocoplus_regressor",
                dtype=dtype,
                trainable=False)
        else:
            print("Warning: cocoplus_regressor not found in SMPL model, using J_regressor as fallback")
            self.joint_regressor = tf.Variable(
                dd['J_regressor'].T.todense(),
                name="joint_regressor",
                dtype=dtype,
                trainable=False)

        if joint_type == 'lsp':
            self.joint_regressor = self.joint_regressor[:, :14]

        if joint_type not in ['cocoplus', 'lsp']:
            print('BAD!! Unknown joint type: %s' % joint_type)
            import ipdb; ipdb.set_trace()

    def __call__(self, beta, theta, get_skin=False, name=None):
        with tf.name_scope(name or "smpl_main", values=[beta, theta]):
            # dynamic batch size (Tensor)
            batch_size = tf.shape(beta)[0]

            # 1. Add shape blend shapes
            # 1. Add shape blend shapes
            # shapedirs currently [num_betas, verts*3]
            shapedirs_reshaped = tf.reshape(self.shapedirs, [-1, self.num_betas])

            # Fix dimension mismatch between beta and shapedirs
            num_betas_input = tf.shape(beta)[1]
            num_betas_model = self.num_betas

            # Ensure beta matches the model's expected number of betas
            beta_adjusted = tf.cond(
                tf.equal(num_betas_input, num_betas_model),
                lambda: beta,
                lambda: tf.cond(
                    tf.less(num_betas_input, num_betas_model),
                    lambda: tf.concat(
                        [beta, tf.zeros([tf.shape(beta)[0], num_betas_model - num_betas_input], dtype=beta.dtype)],
                        axis=1),
                    lambda: beta[:, :num_betas_model]
                )
            )

            v_shaped = tf.reshape(
                tf.matmul(beta_adjusted, tf.transpose(shapedirs_reshaped), name='shape_bs'),
                [-1, self.size[0], self.size[1]]) + self.v_template
            # 2. Infer shape-dependent joint locations.
            Jx = tf.matmul(v_shaped[:, :, 0], self.J_regressor)
            Jy = tf.matmul(v_shaped[:, :, 1], self.J_regressor)
            Jz = tf.matmul(v_shaped[:, :, 2], self.J_regressor)
            J = tf.stack([Jx, Jy, Jz], axis=2)

            # 3. Add pose blend shapes
            Rs = tf.reshape(
                batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, 24, 3, 3])

            with tf.name_scope("lrotmin"):
                # Make identity explicit with same dtype and shape broadcast behavior
                eye3 = tf.eye(3, dtype=Rs.dtype)
                # Rs[:,1:,:,:] has shape [batch,23,3,3], subtract identity via broadcast
                pose_feature = tf.reshape(Rs[:, 1:, :, :] - eye3, [-1, 207])

            v_posed = tf.reshape(
                tf.matmul(pose_feature, self.posedirs),
                [-1, self.size[0], self.size[1]]) + v_shaped

            #4. Get the global joint location
            self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)

            # 5. Do skinning:
            # W is [batch_size, verts, 24]
            # self.weights is [verts, 24]; tile it for batch
            W = tf.reshape(tf.tile(self.weights, [batch_size, 1]), [batch_size, -1, 24])

            # Reshape A: [batch, 24, 4, 4] -> [batch, 24, 16]
            A_reshaped = tf.reshape(A, [batch_size, 24, 16])

            # Apply skinning weights: result [batch, verts, 16]
            T = tf.matmul(W, A_reshaped)

            # Reshape back to [batch, verts, 4, 4]
            T = tf.reshape(T, [batch_size, -1, 4, 4])

            # Build homogeneous vertices: use dynamic vertex count
            num_verts = tf.shape(v_posed)[1]
            ones = tf.ones([batch_size, num_verts, 1], dtype=v_posed.dtype)
            v_posed_homo = tf.concat([v_posed, ones], axis=2)

            # Multiply T (4x4) with v_posed_homo (4x1)
            v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))  # [batch, verts, 4, 1]
            verts = v_homo[:, :, :3, 0]

            # Regress joints from verts
            joint_x = tf.matmul(verts[:, :, 0], self.joint_regressor)
            joint_y = tf.matmul(verts[:, :, 1], self.joint_regressor)
            joint_z = tf.matmul(verts[:, :, 2], self.joint_regressor)
            joints = tf.stack([joint_x, joint_y, joint_z], axis=2)

            if get_skin:
                return verts, joints, Rs
            else:
                return joints
