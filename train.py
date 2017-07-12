from __future__ import print_function
import glob
import time
import sys
import ConfigParser
import tensorflow as tf
from py.fm_ops import fm_ops
import time


class ModelSpecs(object):
        vocabulary_size = 8000000
        vocabulary_block_num = 100
        factor_num = 100
        hash_feature_id = False
        log_file = './log'
        batch_size = 50000
        init_value_range = 0.01
        factor_lambda = 0
        bias_lambda = 0
        thread_num = 10
        epoch_num = 2
        shuffle_batch_threads_num = 1
        min_after_dequeue = 100
        learning_rate = 0.01
        adagrad_initial_accumulator = 0.1
        loss_type = 'mse'
        train_files = './data/train_*'
        weight_files = './data/weight_*'


def read_my_file_format(train_file_queue, weight_file_queue, model_specs):

    train_reader = tf.TextLineReader()
    weight_reader = tf.TextLineReader()

    _, train_line = train_reader.read(train_file_queue)
    weight = tf.constant(1.0, dtype=tf.float32)
    if (weight_file_queue is not None):
        _, weight_line = weight_reader.read(weight_file_queue)
        weight = tf.string_to_number(weight_line, out_type=tf.float32)
    label, feature_ids, feature_vals = fm_ops.fm_parser(
        train_line, model_specs.vocabulary_size, model_specs.hash_feature_id)
    feature_ids = tf.reshape(feature_ids, [-1, 1])
    sparse_features = tf.SparseTensor(
        feature_ids, feature_vals, [model_specs.vocabulary_size])
    label.set_shape([])
    return label, weight, sparse_features


def input_pipeline(train_files, weight_files, model_specs):
    seed = time.time()
    train_file_queue = tf.train.string_input_producer(
        train_files, num_epochs=model_specs.num_epochs, shuffle=True, seed=seed)
    weight_file_queue = tf.train.string_input_producer(
        weight_files, num_epochs=model_specs.num_epochs, shuffle=True, seed=seed)

    label, weight, sparse_features = read_my_file_format(
        train_file_queue, weight_file_queue, model_specs)

    # Batching Examples
    capacity = model_specs.min_after_dequeue + 3 * model_specs.batch_size
    labels_batch, weights_batch, sparse_features_batch = tf.train.shuffle_batch(
        [label, weight, sparse_features], batch_size=model_specs.batch_size, capacity=capacity, min_after_dequeue=model_specs.min_after_dequeue, num_threads=model_specs.num_threads)
    return labels_batch, weights_batch, sparse_features_batch


def train(train_files, weight_files, model_specs):

    with tf.Graph().as_default():
        vocab_blocks = []
        vocab_size_per_block = model_specs.vocabulary_size / \
            model_specs.vocabulary_block_num + 1
        init_value_range = model_specs.init_value_range

        for i in range(model_specs.vocabulary_block_num):
            vocab_blocks.append(
                tf.Variable(
                    tf.random_uniform(
                        [
                            vocab_size_per_block,
                            model_specs.factor_num + 1],
                        -init_value_range,
                        init_value_range),
                    name='vocab_block_%d' % i))

        labels_batch, weights_batch, sparse_features_batch = input_pipeline(
            train_files, weight_files, model_specs)

        inds = sparse_features_batch.indices
        feature_vals = sparse_features_batch.values

        ori_ids, feature_ids = tf.unique(
            tf.squeeze(tf.slice(inds, [0, 1], [-1, 1])))
        feature_poses = tf.concat([[0], tf.cumsum(tf.bincount(
            tf.cast(tf.slice(inds, [0, 0], [-1, 1]), tf.int32)))], 0)

        local_params = tf.nn.embedding_lookup(vocab_blocks, ori_ids)

        pred_score, reg_score = fm_ops.fm_scorer(
            feature_ids, local_params, feature_vals, feature_poses, model_specs.factor_lambda, model_specs.bias_lambda)

        if model_specs.loss_type == 'logistic':
            loss = tf.reduce_mean(weights * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pred_score, labels=labels_batch)) / model_specs.batch_size
        elif model_specs.loss_type == 'mse':
            #loss = tf.losses.mean_squared_error(labels_batch, pred_score, weights_batch)
            loss = tf.reduce_mean(
                weights_batch *
                tf.square(
                    pred_score -
                    labels_batch))
        else:
            loss = None

        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdagradOptimizer(
            model_specs.learning_rate,
            model_specs.adagrad_init_accumulator)
        train_op = optimizer.minimize(
            loss + reg_score / model_specs.batch_size,
            global_step=global_step)

        sv = tf.train.Supervisor(logdir=model_specs.log_file)
        with sv.managed_session() as sess:
            while not sv.should_stop():
                _, loss_value, step_num = sess.run(
                    [train_op, loss, global_step])
                print('-- Global Step: %d; Avg loss: %.5f;' % (step_num, loss_value))


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Missing config file path")
        exit()

    GENERAL_SECTION = 'General'
    TRAIN_SECTION = 'Train'
    PREDICT_SECTION = 'Predict'
    CLUSTER_SPEC_SECTION = 'ClusterSpec'
    STR_DELIMITER = ','

    config = ConfigParser.ConfigParser()
    _ = config.read(sys.argv[1])
    model_specs = ModelSpecs()

    def read_config(section, option, not_null=True):
        if not config.has_option(section, option):
            if not_null:
                raise ValueError("%s is undefined." % option)
            else:
                return None
        else:
            value = config.get(section, option)
            print('  {0} = {1}'.format(option, value))
            return value

    def read_strs_config(section, option, not_null=True):
        val = read_config(section, option, not_null)
        if val is not None:
            return [s.strip() for s in val.split(STR_DELIMITER)]
        return None

    print('Config: ')
    model_specs.vocabulary_size = int(read_config(
        GENERAL_SECTION, 'vocabulary_size'))
    model_specs.vocabulary_block_num = int(read_config(
        GENERAL_SECTION, 'vocabulary_block_num'))
    model_specs.factor_num = int(read_config(
        GENERAL_SECTION, 'factor_num'))
    model_specs.hash_feature_id = read_config(
        GENERAL_SECTION, 'hash_feature_id').strip().lower() == 'true'
    model_specs.log_file = read_config(GENERAL_SECTION, 'log_file')

    model_specs.batch_size = int(read_config(TRAIN_SECTION, 'batch_size'))
    model_specs.init_value_range = float(
        read_config(TRAIN_SECTION, 'init_value_range'))
    model_specs.factor_lambda = float(
        read_config(TRAIN_SECTION, 'factor_lambda'))
    model_specs.bias_lambda = float(read_config(TRAIN_SECTION, 'bias_lambda'))
    model_specs.num_threads = int(read_config(TRAIN_SECTION, 'thread_num'))
    model_specs.num_epochs = int(read_config(TRAIN_SECTION, 'epoch_num'))
    model_specs.shuffle_batch_threads_num = int(
        read_config(TRAIN_SECTION, 'shuffle_batch_threads_num'))
    model_specs.min_after_dequeue = int(
        read_config(TRAIN_SECTION, 'min_after_dequeue'))
    model_specs.learning_rate = float(
        read_config(TRAIN_SECTION, 'learning_rate'))
    model_specs.adagrad_init_accumulator = float(
        read_config(TRAIN_SECTION, 'adagrad.initial_accumulator'))
    model_specs.loss_type = read_config(
        TRAIN_SECTION, 'loss_type').strip().lower()
    if model_specs.loss_type not in ['logistic', 'mse']:
        raise ValueError('Unsupported loss type: %s' % loss_type)

    train_files = read_strs_config(TRAIN_SECTION, 'train_files')
    train_files = sorted(sum((glob.glob(f) for f in train_files), []))
    weight_files = read_strs_config(TRAIN_SECTION, 'weight_files', False)
    if weight_files is not None:
        if not isinstance(weight_files, list):
            weight_files = [weight_files]
        weight_files = sorted(sum((glob.glob(f) for f in weight_files), []))
    if len(train_files) != len(weight_files):
        raise ValueError(
            'The numbers of train files and weight files do not match.')

    train(train_files, weight_files, model_specs)


if __name__ == '__main__':
    print('starting...')
    main()
