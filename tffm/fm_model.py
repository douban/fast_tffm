from __future__ import print_function
import time
import glob
import ConfigParser
import tensorflow as tf
from tffm.fm_ops import fm_parser, fm_scorer


class Model(object):
    vocabulary_size = 8000000
    vocabulary_block_num = 100
    factor_num = 10
    hash_feature_id = False
    log_dir = None
    batch_size = 50000
    init_value_range = 0.01
    factor_lambda = 0
    bias_lambda = 0
    learning_rate = 0.01
    adagrad_initial_accumulator = 0.1
    num_epochs = 10
    loss_type = 'mse'
    save_summaries_steps = 100
    queue_size = 10000
    shuffle_threads = 1
    ratio = 4
    train_files = []
    weight_files = []
    ops = []
    ops_names = []

    def _shuffle_input(self,
                       thread_idx,
                       train_file_queue,
                       weight_file_queue,
                       ex_q):
        with tf.name_scope("shuffled_%s" % (thread_idx,)):
            train_reader = tf.TextLineReader()
            weight_reader = tf.TextLineReader()
            _, data_lines = train_reader.read_up_to(
                train_file_queue, self.batch_size)
            _, weight_lines = weight_reader.read_up_to(
                weight_file_queue, self.batch_size)

            min_after_dequeue = 3 * self.batch_size
            capacity = int(min_after_dequeue + self.batch_size * 1.5)
            data_lines_batch, weight_lines_batch = tf.train.shuffle_batch(
                [data_lines, weight_lines], self.batch_size, capacity,
                min_after_dequeue, enqueue_many=True
            )

            weights = tf.string_to_number(weight_lines_batch, tf.float32)
            labels, sizes, feature_ids, feature_vals = fm_parser(
                data_lines_batch, self.vocabulary_size
            )
            ori_ids, feature_ids = tf.unique(feature_ids)
            feature_poses = tf.concat([[0], tf.cumsum(sizes)], 0)

            enq = ex_q.enqueue([
                labels, weights, feature_ids, ori_ids,
                feature_vals, feature_poses
            ])
            return [enq] * self.ratio

    def _input_pipeline(self):
        seed = time.time()

        train_file_queue = tf.train.string_input_producer(
            self.train_files,
            num_epochs=self.num_epochs,
            shared_name="train_file_queue",
            shuffle=True,
            seed=seed
        )
        weight_file_queue = tf.train.string_input_producer(
            self.weight_files,
            num_epochs=self.num_epochs,
            shared_name="weight_file_queue",
            shuffle=True,
            seed=seed
        )

        example_queue = tf.FIFOQueue(
            self.queue_size,
            [tf.float32, tf.float32, tf.int32, tf.int64, tf.float32, tf.int32]
        )
        enqueue_ops = sum(
            (
                self._shuffle_input(i, train_file_queue, weight_file_queue,
                                    example_queue)
                for i in range(self.shuffle_threads)
            ),
            []
        )
        tf.train.add_queue_runner(
            tf.train.QueueRunner(example_queue, enqueue_ops)
        )

        (
            labels, weights, feature_ids, ori_ids, feature_vals, feature_poses
        ) = example_queue.dequeue()

        exq_size = example_queue.size()

        return (
            exq_size, labels, weights, feature_ids, ori_ids,
            feature_vals, feature_poses
        )

    def build_graph(self, monitor, trace):
        vocab_blocks = []
        vocab_size_per_block = (
            self.vocabulary_size / self.vocabulary_block_num + 1
        )
        init_value_range = self.init_value_range

        for i in range(self.vocabulary_block_num):
            vocab_blocks.append(
                tf.Variable(
                    tf.random_uniform(
                        [vocab_size_per_block, self.factor_num + 1],
                        -init_value_range, init_value_range
                    ),
                    name='vocab_block_%d' % i
                )
            )

        (
            exq_size, labels, weights, feature_ids, ori_ids,
            feature_vals, feature_poses
        ) = self._input_pipeline()

        local_params = tf.nn.embedding_lookup(vocab_blocks, ori_ids)

        pred_score, reg_score = fm_scorer(
            feature_ids, local_params, feature_vals, feature_poses,
            self.factor_lambda, self.bias_lambda
        )

        if self.loss_type == 'logistic':
            loss = tf.reduce_mean(
                weights * tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=pred_score, labels=labels
                )
            ) / self.batch_size
        elif self.loss_type == 'mse':
            loss = tf.reduce_mean(
                weights * tf.square(pred_score - labels))
        else:
            # should never be here
            assert False

        tf.summary.scalar('loss', loss)
        global_step = tf.contrib.framework.get_or_create_global_step()
        optimizer = tf.train.AdagradOptimizer(
            self.learning_rate,
            self.adagrad_init_accumulator
        )
        train_op = optimizer.minimize(
            loss + reg_score / self.batch_size,
            global_step=global_step
        )

        self.ops_names = ["train_op", "loss", "step_num"]
        self.ops = [train_op, loss, global_step]
        if monitor:
            shuffleq_sizes = [
                "shuffled_%s/shuffle_batch/"
                "random_shuffle_queue_Size:0" %
                i for i in range(
                    self.shuffle_threads)]
            self.ops_names.append("exq_size")
            self.ops.append(exq_size)
            self.ops_names.extend(shuffleq_sizes)
            self.ops.extend(shuffleq_sizes)

    def _get_config(self, config_file):
        GENERAL_SECTION = 'General'
        TRAIN_SECTION = 'Train'
        STR_DELIMITER = ','

        config = ConfigParser.ConfigParser()
        config.read(config_file)

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
        self.vocabulary_size = int(read_config(
            GENERAL_SECTION, 'vocabulary_size'))
        self.vocabulary_block_num = int(read_config(
            GENERAL_SECTION, 'vocabulary_block_num'))
        self.factor_num = int(read_config(
            GENERAL_SECTION, 'factor_num'))
        self.hash_feature_id = read_config(
            GENERAL_SECTION, 'hash_feature_id').strip().lower() == 'true'
        self.log_dir = read_config(
            GENERAL_SECTION, 'log_dir', not_null=False
        )
        if config.has_option(GENERAL_SECTION, 'save_summaries_steps'):
            self.save_summaries_steps = int(read_config(
                GENERAL_SECTION, 'save_summaries_steps'))

        self.batch_size = int(read_config(TRAIN_SECTION, 'batch_size'))
        self.init_value_range = float(
            read_config(TRAIN_SECTION, 'init_value_range'))
        self.factor_lambda = float(
            read_config(TRAIN_SECTION, 'factor_lambda'))
        self.bias_lambda = float(read_config(TRAIN_SECTION, 'bias_lambda'))
        self.num_epochs = int(read_config(TRAIN_SECTION, 'epoch_num'))
        self.learning_rate = float(
            read_config(TRAIN_SECTION, 'learning_rate'))
        self.adagrad_init_accumulator = float(
            read_config(TRAIN_SECTION, 'adagrad.initial_accumulator'))
        self.loss_type = read_config(
            TRAIN_SECTION, 'loss_type').strip().lower()
        if self.loss_type not in ['logistic', 'mse']:
            raise ValueError('Unsupported loss type: %s' % self.loss_type)

        if config.has_option(TRAIN_SECTION, 'queue_size'):
            self.queue_size = int(read_config(TRAIN_SECTION, 'queue_size'))
        if config.has_option(TRAIN_SECTION, 'shuffle_threads'):
            self.shuffle_threads = int(
                read_config(TRAIN_SECTION, 'shuffle_threads')
            )
        if config.has_option(TRAIN_SECTION, 'ratio'):
            self.ratio = int(read_config(TRAIN_SECTION, 'ratio'))

        train_files = read_strs_config(TRAIN_SECTION, 'train_files')
        self.train_files = sorted(sum((glob.glob(f) for f in train_files), []))
        weight_files = read_strs_config(TRAIN_SECTION, 'weight_files', False)
        if weight_files is not None:
            if not isinstance(weight_files, list):
                weight_files = [weight_files]
            self.weight_files = sorted(
                sum((glob.glob(f) for f in weight_files), []))
            if len(train_files) != len(weight_files):
                raise ValueError(
                    'The numbers of train files'
                    'and weight files do not match.')

    def __init__(self, config_file):
        self._get_config(config_file)
