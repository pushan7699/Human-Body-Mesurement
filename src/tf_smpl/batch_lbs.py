

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def batch_skew(vec, batch_size=None):
    """
    vec is N x 3, batch_size is int
    returns N x 3 x 3. Skew-symmetric version of each matrix.
    """
    with tf.name_scope("batch_skew"):
        if batch_size is None:
            batch_size = vec.shape.as_list()[0]
        col_inds = tf.constant([1, 2, 3, 5, 6, 7])
        indices = tf.reshape(
            tf.reshape(tf.range(0, batch_size) * 9, [-1, 1]) + col_inds,
            [-1, 1])
        updates = tf.reshape(
            tf.stack(
                [
                    -vec[:, 2], vec[:, 1],
                    vec[:, 2], -vec[:, 0],
                    -vec[:, 1], vec[:, 0]
                ],
                axis=1), [-1])
        out_shape = [batch_size * 9]
        res = tf.scatter_nd(indices, updates, out_shape)
        res = tf.reshape(res, [batch_size, 3, 3])
        return res


def batch_rodrigues(theta, name=None):
    """
    Theta is N x 3
    Returns rotation matrices for each axis-angle vector.
    """
    with tf.name_scope(name or "batch_rodrigues"):
        batch_size = theta.shape.as_list()[0]

        angle = tf.expand_dims(tf.norm(theta + 1e-8, axis=1), -1)
        r = tf.expand_dims(tf.div(theta, angle), -1)
        angle = tf.expand_dims(angle, -1)
        cos = tf.cos(angle)
        sin = tf.sin(angle)

        outer = tf.matmul(r, r, transpose_b=True, name="outer")
        eyes = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
        R = cos * eyes + (1 - cos) * outer + sin * batch_skew(
            r, batch_size=batch_size)
        return R


def batch_lrotmin(theta, name=None):
    """
    Equation 9 in the SMPL paper.
    Computes (R - I) for each local joint rotation.
    """
    with tf.name_scope(name or "batch_lrotmin"):
        theta = theta[:, 3:]  # ignore global rotation
        Rs = batch_rodrigues(tf.reshape(theta, [-1, 3]))
        lrotmin = tf.reshape(Rs - tf.eye(3), [-1, 207])
        return lrotmin


def make_A(R, t, name=None):
    """
    Construct homogeneous transformation matrices A from rotation (R) and translation (t).
    R: [batch_size, 3, 3]
    t: [batch_size, 3, 1]
    """
    with tf.name_scope(name or "Make_A"):
        batch_size = tf.shape(t)[0]

        # Ensure R has batch dimension
        R = R + tf.zeros([batch_size, 1, 1])  # broadcasting trick

        # Build homogeneous coordinates
        R_homo = tf.pad(R, [[0, 0], [0, 1], [0, 0]])  # add a 4th row
        t_homo = tf.concat([t, tf.ones([batch_size, 1, 1])], axis=1)  # add homogeneous translation

        A = tf.concat([R_homo, t_homo], axis=2)
        return A


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):

    with tf.name_scope("batch_forward_kinematics"):
        N = tf.shape(Rs)[0]

        # Optionally rotate the SMPL base coordinate frame
        if rotate_base:
            print('Flipping the SMPL coordinate frame!!!!')
            rot_x = tf.constant(
                [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=Rs.dtype)
            rot_x = tf.reshape(tf.tile(rot_x, [N, 1]), [N, 3, 3])
            root_rotation = tf.matmul(Rs[:, 0, :, :], rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]

        # Expand Js for broadcasting (N x 24 x 3 x 1)
        Js = tf.expand_dims(Js, -1)

        # Compute the transformation for the root joint
        root_J = Js[:, 0, :, :]  # (N x 3 x 1)
        A0 = make_A(root_rotation, root_J)

        # Store transformations in a list
        results = [A0]

        # Iterate through other joints
        for i in range(1, 24):
            j_parent = parent[i]
            t_rel = Js[:, i, :, :] - Js[:, j_parent, :, :]  # relative translation
            A_i = tf.matmul(results[j_parent], make_A(Rs[:, i, :, :], t_rel))
            results.append(A_i)

        # Stack all transformations
        results = tf.stack(results, axis=1)

        # Compute new joint locations (the translation part of each transform)
        new_J = results[:, :, :3, 3]

        # Compute relative transformation A for LBS (subtract rest pose)
        batch_size = tf.shape(Js)[0]  # ✅ FIXED
        Js_w0 = tf.concat([Js, tf.zeros([batch_size, 24, 1, 1])], axis=2)
        init_bone = tf.matmul(results, Js_w0)
        init_bone = tf.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
        A = results - init_bone

        return new_J, A
