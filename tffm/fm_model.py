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
    save_steps = 100
    save_summaries_steps = 100
    queue_size = 10000
    shuffle_threads = 1
    train_files = []
    weight_files = []
    predict_files = []
    validation_data_files = []
    validation_weight_files = []
    validation_data = None

    def _shuffle_input(self,
                       thread_idx,
                       train_file_queue,
                       weight_file_queue,
                       ex_q):
        with tf.name_scope("shuffled_%s" % (thread_idx,)):
            train_reader = tf.TextLineReader()
            _, data_lines = train_reader.read_up_to(
                train_file_queue, self.batch_size
            )

            min_after_dequeue = 3 * self.batch_size
            capacity = int(min_after_dequeue + self.batch_size * 1.5)

            if weight_file_queue is not None:
                weight_reader = tf.TextLineReader()
                _, weight_lines = weight_reader.read_up_to(
                    weight_file_queue, self.batch_size
                )

                data_lines_batch, weight_lines_batch = tf.train.shuffle_batch(
                    [data_lines, weight_lines], self.batch_size, capacity,
                    min_after_dequeue, enqueue_many=True,
                    allow_smaller_final_batch=True
                )

                weights = tf.string_to_number(weight_lines_batch, tf.float32)
            else:
                data_lines_batch = tf.train.shuffle_batch(
                    [data_lines], self.batch_size, capacity,
                    min_after_dequeue, enqueue_many=True,
                    allow_smaller_final_batch=True
                )
                weights = tf.ones(tf.shape(data_lines_batch), tf.float32)

            labels, sizes, feature_ids, feature_vals = fm_parser(
                data_lines_batch, self.vocabulary_size
            )
            ori_ids, feature_ids = tf.unique(feature_ids)
            feature_poses = tf.concat([[0], tf.cumsum(sizes)], 0)

            enq = ex_q.enqueue([
                labels, weights, feature_ids, ori_ids,
                feature_vals, feature_poses
            ])
            return enq

    def _input_pipeline(self):
        seed = time.time()

        train_file_queue = tf.train.string_input_producer(
            self.train_files,
            num_epochs=self.num_epochs,
            shared_name="train_file_queue",
            shuffle=True,
            seed=seed
        )

        if len(self.weight_files) == 0:
            weight_file_queue = None
        else:
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
        enqueue_ops = [
            self._shuffle_input(
                i, train_file_queue, weight_file_queue, example_queue
            )
            for i in range(self.shuffle_threads)
        ]
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

    def _pred_op(self, vocab_blocks, task, data_file):
        data_file_queue = tf.train.string_input_producer(
            data_file,
            shared_name=task + "_file_queue"
        )

        example_queue = tf.FIFOQueue(
            self.queue_size,
            [tf.float32, tf.float32, tf.int32, tf.int64, tf.float32, tf.int32]
        )
        enqueue_ops = [
            self._shuffle_input(
                i, data_file_queue, None, example_queue
            )
            for i in range(self.shuffle_threads)
        ]
        tf.train.add_queue_runner(
            tf.train.QueueRunner(example_queue, enqueue_ops)
        )

        (
            labels, weights, feature_ids, ori_ids, feature_vals, feature_poses
        ) = example_queue.dequeue()

        local_params = tf.nn.embedding_lookup(vocab_blocks, ori_ids)

        pred_score, reg_score = fm_scorer(
            feature_ids, local_params, feature_vals, feature_poses,
            self.factor_lambda, self.bias_lambda
        )

        return labels, pred_score

    def load_validation_data(self):
        if len(self.validation_data_files) == 0:
            return

        data_lines = []
        for data_file in self.validation_data_files:
            with open(data_file) as f:
                data_lines.extend(f.readlines())
        data_lines = [l.strip('\n') for l in data_lines]

        weight_lines = []
        for weight_file in self.validation_weight_files:
            with open(weight_file) as f:
                weight_lines.extend(f.readlines())
        weight_lines = [l.strip('\n') for l in weight_lines]

        labels, sizes, feature_ids, feature_vals = fm_parser(
            tf.constant(data_lines, dtype=tf.string), self.vocabulary_size
        )
        ori_ids, feature_ids = tf.unique(feature_ids)
        feature_poses = tf.concat([[0], tf.cumsum(sizes)], 0)

        if len(weight_lines) == 0:
            weights = tf.ones(tf.shape(labels), tf.float32)
        else:
            weights = tf.string_to_number(
                tf.constant(
                    weight_lines,
                    dtype=tf.string),
                tf.float32)

        self.validation_data = dict(zip(
            [
                'labels', 'weights', 'feature_ids',
                'ori_ids', 'feature_vals', 'feature_poses'
            ],
            [
                labels, weights, feature_ids,
                ori_ids, feature_vals, feature_poses
            ]))

    def build_graph(self, monitor, trace):
        self.load_validation_data()
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

        self.pred_op = self._pred_op(
            vocab_blocks, 'predict', self.predict_files)

        if self.validation_data is not None:
            local_params = tf.nn.embedding_lookup(
                vocab_blocks, self.validation_data['ori_ids'])
            v_pred_score, _ = fm_scorer(
                self.validation_data['feature_ids'],
                local_params,
                self.validation_data['feature_vals'],
                self.validation_data['feature_poses'],
                self.factor_lambda, self.bias_lambda
            )

        if self.loss_type == 'logistic':
            loss = tf.reduce_mean(
                weights * tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=pred_score, labels=labels
                )
            )
            if not len(self.validation_data_files) == 0:
                self.valid_op = tf.reduce_mean(
                    self.validation_data['weights'] *
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=v_pred_score,
                        labels=self.validation_data['labels']))

        elif self.loss_type == 'mse':
            loss = tf.reduce_mean(
                weights * tf.square(pred_score - labels))

            if not len(self.validation_data_files) == 0:
                self.valid_op = tf.reduce_mean(
                    self.validation_data['weights'] * tf.square(
                        v_pred_score - self.validation_data['labels']
                    )
                )
        else:
            # should never be here
            assert False

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('exq_size', exq_size)
        global_step = tf.contrib.framework.get_or_create_global_step()
        optimizer = tf.train.AdagradOptimizer(
            self.learning_rate,
            self.adagrad_init_accumulator
        )
        train_op = optimizer.minimize(
            loss + reg_score / self.batch_size,
            global_step=global_step
        )

        self.ops = {
            'train_op': train_op,
            'loss': loss,
            'step_num': global_step
        }

        if monitor:
            shuffleq_sizes = [
                "shuffled_%s/shuffle_batch/"
                "random_shuffle_queue_Size:0" %
                i for i in range(
                    self.shuffle_threads)]
            self.ops['exq_size'] = exq_size
            self.ops['shuffleq_sizes'] = shuffleq_sizes

        self.saver = tf.train.Saver()

    def _get_config(self, config_file):
        GENERAL_SECTION = 'General'
        TRAIN_SECTION = 'Train'
        PREDICT_SECTION = 'Predict'
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
        self.model_file = read_config(
            GENERAL_SECTION, 'model_file', not_null=False
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
        self.save_steps = int(
            read_config(
                TRAIN_SECTION,
                'save_steps',
                not_null=False))
        self.tolerance = float(read_config(TRAIN_SECTION, 'tolerance'))

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

        validation_data_files = read_strs_config(
            TRAIN_SECTION, 'validation_files', False)
        if validation_data_files is not None:
            if not isinstance(validation_data_files, list):
                validation_data_files = [validation_data_files]
            self.validation_data_files = sorted(
                sum((glob.glob(f) for f in validation_data_files), []))
        validation_weight_files = read_strs_config(
            TRAIN_SECTION, 'validation_weight_files', False)
        if validation_weight_files is not None:
            if not isinstance(validation_weight_files, list):
                validation_weight_files = [validation_weight_files]
            self.validation_weight_files = sorted(
                sum((glob.glob(f) for f in validation_weight_files), []))
            if len(validation_data_files) != len(validation_weight_files):
                raise ValueError(
                    'The numbers of validation data files'
                    'and validation weight files do not match.')

        predict_files = read_strs_config(
            PREDICT_SECTION, 'predict_files', False)
        self.predict_files = sorted(
            sum((glob.glob(f) for f in predict_files), []))
        self.score_path = read_config(
            PREDICT_SECTION, 'score_path', not_null=False
        )

    def __init__(self, config_file):
        self._get_config(config_file)
