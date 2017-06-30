import os
import tensorflow as tf
from tensorflow.python.framework import ops

fm_ops = tf.load_op_library(os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'lib', 'libfast_tffm.so'
))


@ops.RegisterGradient('FmScorer')
def _fm_scorer_grad(op, pred_grad, reg_grad):
    feature_ids = op.inputs[0]
    feature_params = op.inputs[1]
    feature_vals = op.inputs[2]
    feature_poses = op.inputs[3]
    factor_lambda = op.inputs[4]
    bias_lambda = op.inputs[5]
    with ops.control_dependencies([pred_grad.op, reg_grad.op]):
        return (
            None,
            fm_ops.fm_grad(
                feature_ids, feature_params, feature_vals, feature_poses,
                factor_lambda, bias_lambda, pred_grad, reg_grad
            ),
            None, None, None, None
        )


def fm_parser(data_strings, vocab_size, hash_feature_id=False):
    data_strings = tf.convert_to_tensor(
        data_strings, dtype=tf.string, name='data_strings'
    )
    return fm_ops.fm_parser(
        data_strings, vocab_size, hash_feature_id
    )


def fm_scorer(feature_ids, feature_params, feature_vals, feature_poses,
              factor_lambda, bias_lambda):
    feature_ids = tf.convert_to_tensor(
        feature_ids, dtype=tf.int32, name='feature_ids'
    )
    feature_params = tf.convert_to_tensor(
        feature_params, dtype=tf.float32, name='feature_params'
    )
    feature_vals = tf.convert_to_tensor(
        feature_vals, dtype=tf.float32, name='feature_vals'
    )
    feature_poses = tf.convert_to_tensor(
        feature_poses, dtype=tf.int32, name='feature_poses'
    )
    return fm_ops.fm_scorer(
        feature_ids, feature_params, feature_vals, feature_poses,
        factor_lambda, bias_lambda
    )
