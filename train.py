import glob
import time
import sys
import ConfigParser
import tensorflow as tf
from py.fm_ops import fm_ops
import time


class ModelSpecs:
    pass


def read_my_file_format(train_file_queue, weight_file_queue, model_specs):

    train_reader = tf.TextLineReader()
    weight_reader = tf.TextLineReader()

    _, train_line = train_reader.read(train_file_queue)
    weight = tf.constant(1.0, dtype=tf.float32)
    if (weight_file_queue is not None):
        _, weight_line = weight_reader.read(weight_file_queue)
        weight = tf.string_to_number(weight_line, out_type=tf.float32)
    label, feature_ids, feature_vals = fm_ops.fm_parser(
        train_line, ModelSpecs.vocabulary_size, ModelSpecs.hash_feature_id)
    feature_ids = tf.reshape(feature_ids, [-1, 1])
    sparse_features = tf.SparseTensor(
        feature_ids, feature_vals, [ModelSpecs.vocabulary_size])
    label.set_shape([])
    return label, weight, sparse_features


def input_pipeline(train_files, weight_files, model_specs):
    seed = time.time()
    train_file_queue = tf.train.string_input_producer(
        train_files, num_epochs=ModelSpecs.num_epochs, shuffle=True, seed=seed)
    weight_file_queue = tf.train.string_input_producer(
        weight_files, num_epochs=ModelSpecs.num_epochs, shuffle=True, seed=seed)

    label, weight, sparse_features = read_my_file_format(
        train_file_queue, weight_file_queue, model_specs)

    # Batching Examples
    capacity = ModelSpecs.min_after_dequeue + 3 * ModelSpecs.batch_size
    labels_batch, weights_batch, sparse_features_batch = tf.train.shuffle_batch(
        [label, weight, sparse_features], batch_size=ModelSpecs.batch_size, capacity=capacity, min_after_dequeue=ModelSpecs.min_after_dequeue, num_threads=ModelSpecs.num_threads)
    return labels_batch, weights_batch, sparse_features_batch


def train(train_files, weight_files, model_specs):

    with tf.Graph().as_default():
        vocab_blocks = []
        vocab_size_per_block = ModelSpecs.vocabulary_size / \
            ModelSpecs.vocabulary_block_num + 1
        init_value_range = ModelSpecs.init_value_range

        for i in range(ModelSpecs.vocabulary_block_num):
            vocab_blocks.append(
                tf.Variable(
                    tf.random_uniform(
                        [
                            vocab_size_per_block,
                            ModelSpecs.factor_num + 1],
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
            feature_ids, local_params, feature_vals, feature_poses, ModelSpecs.factor_lambda, ModelSpecs.bias_lambda)

        if ModelSpecs.loss_type == 'logistic':
            loss = tf.reduce_sum(weights * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pred_score, labels=labels_batch)) / ModelSpecs.batch_size
        elif ModelSpecs.loss_type == 'mse':
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
            ModelSpecs.learning_rate,
            ModelSpecs.adagrad_init_accumulator)
        train_op = optimizer.minimize(
            loss + reg_score / ModelSpecs.batch_size,
            global_step=global_step)

        sv = tf.train.Supervisor(logdir='log')
        with sv.managed_session() as sess:
            while not sv.should_stop():
                sess.run(train_op)
                print '-- Global Step: %d; Avg loss: %.5f;' % (global_step.eval(session=sess), loss.eval(session=sess))


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
    model_specs = {}

    def read_config(section, option, not_null=True):
        if not config.has_option(section, option):
            if not_null:
                raise ValueError("%s is undefined." % option)
            else:
                return None
        else:
            value = config.get(section, option)
            print '  {0} = {1}'.format(option, value)
            return value

    def read_strs_config(section, option, not_null=True):
        val = read_config(section, option, not_null)
        if val is not None:
            return [s.strip() for s in val.split(STR_DELIMITER)]
        return None

    print 'Config: '
    ModelSpecs.factor_num = int(read_config(
        GENERAL_SECTION, 'factor_num'))
    ModelSpecs.vocabulary_size = int(read_config(
        GENERAL_SECTION, 'vocabulary_size'))
    ModelSpecs.vocabulary_block_num = int(read_config(
        GENERAL_SECTION, 'vocabulary_block_num'))
    ModelSpecs.hash_feature_id = read_config(
        GENERAL_SECTION, 'hash_feature_id').strip().lower() == 'true'

    ModelSpecs.batch_size = int(read_config(TRAIN_SECTION, 'batch_size'))
    ModelSpecs.init_value_range = float(
        read_config(TRAIN_SECTION, 'init_value_range'))
    ModelSpecs.factor_lambda = float(
        read_config(TRAIN_SECTION, 'factor_lambda'))
    ModelSpecs.bias_lambda = float(read_config(TRAIN_SECTION, 'bias_lambda'))
    ModelSpecs.num_threads = int(read_config(TRAIN_SECTION, 'thread_num'))
    ModelSpecs.num_epochs = int(read_config(TRAIN_SECTION, 'epoch_num'))
    ModelSpecs.shuffle_batch_threads_num = int(
        read_config(TRAIN_SECTION, 'shuffle_batch_threads_num'))
    ModelSpecs.min_after_dequeue = int(
        read_config(TRAIN_SECTION, 'min_after_dequeue'))
    ModelSpecs.learning_rate = float(
        read_config(TRAIN_SECTION, 'learning_rate'))
    ModelSpecs.adagrad_init_accumulator = float(
        read_config(TRAIN_SECTION, 'adagrad.initial_accumulator'))
    ModelSpecs.loss_type = read_config(
        TRAIN_SECTION, 'loss_type').strip().lower()
    if ModelSpecs.loss_type not in ['logistic', 'mse']:
        raise ValueError('Unsupported loss type: %s' % loss_type)

    train_files = read_strs_config(TRAIN_SECTION, 'train_files')
    train_files = sum((glob.glob(f) for f in train_files), [])
    weight_files = read_strs_config(TRAIN_SECTION, 'weight_files', False)
    if weight_files is not None:
        if not isinstance(weight_files, list):
            weight_files = [weight_files]
        weight_files = sum((glob.glob(f) for f in weight_files), [])
    if len(train_files) != len(weight_files):
        raise ValueError(
            'The numbers of train files and weight files do not match.')

    train(train_files, weight_files, model_specs)


if __name__ == '__main__':
    print('starting...')
    main()
