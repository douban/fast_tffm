from __future__ import print_function
import time
import tensorflow as tf
from tffm.fm_ops import fm_parser, fm_scorer
from tensorflow.python.client import timeline


class Model(object):
    vocabulary_size = 8000000
    vocabulary_block_num = 100
    factor_num = 10
    hash_feature_id = False
    log_file = './log'
    batch_size = 50000
    init_value_range = 0.01
    factor_lambda = 0
    bias_lambda = 0
    learning_rate = 0.01
    adagrad_initial_accumulator = 0.1
    num_epochs = 10
    loss_type = 'mse'
    queue_size = 10000
    shuffle_threads = 1
    ratio = 4
    ps_hosts = None
    worker_hosts = None
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

    def _input_pipeline(self, train_files, weight_files):
        seed = time.time()

        train_file_queue = tf.train.string_input_producer(
            train_files,
            num_epochs=self.num_epochs,
            shared_name="train_file_queue",
            shuffle=True,
            seed=seed
        )
        weight_file_queue = tf.train.string_input_producer(
            weight_files,
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

    def build_graph(
            self,
            train_files,
            weight_files,
            trace,
            monitor,
            job_name,
            task_index,
            cluster,
            server):

        if job_name == "ps":
            server.join()
        elif cluster is None or job_name == "worker":
            with tf.device(tf.train.replica_device_setter(
                    cluster=cluster)):
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
                ) = self._input_pipeline(train_files, weight_files)

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

    def train(self, sess, monitor, trace):
        min_after_dequeue = 3 * self.batch_size
        capacity = int(min_after_dequeue + self.batch_size * 1.5)
        run_metadata = tf.RunMetadata()

        ttotal = 0
        step_num = None
        while not sess.should_stop():
            cur = time.time()
            if step_num is None and trace:
                ops_res = sess.run(
                    self.ops,
                    options=tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE
                    ),
                    run_metadata=run_metadata
                )
            else:
                ops_res = sess.run(self.ops)

            res_dict = dict(zip(self.ops_names, ops_res))
            tend = time.time()
            ttotal = ttotal + tend - cur

            if monitor:
                print(
                    'speed:',
                    self.batch_size / (tend - cur),
                    'shuffle_queue: %.2f%%' %
                    (max((sum(res_dict[q] for q in
                              self.ops_names[-(self.shuffle_threads):]) -
                          self.shuffle_threads *
                          min_after_dequeue) * 100.0 /
                         (capacity * self.shuffle_threads),
                         0)),
                    'example_queue: %.2f%%' %
                    (res_dict['exq_size'] * 100.0 /
                     self.queue_size))

            print(
                '-- Global Step: %d; Avg loss: %.5f;' % (
                    res_dict['step_num'], res_dict['loss']
                )
            )

        print(
            'Average speed: ',
            res_dict['step_num'] * self.batch_size / ttotal,
            ' examples/s'
        )

        if trace is not None:
            if not trace.endswith('.json'):
                trace += '.json'
            with open(trace, 'w') as trace_file:
                timeline_info = timeline.Timeline(
                    step_stats=run_metadata.step_stats)
                trace_file.write(timeline_info.generate_chrome_trace_format())
