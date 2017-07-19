from __future__ import print_function
import glob
import time
import ConfigParser
import tensorflow as tf
from tffm.fm_ops import fm_parser, fm_scorer
from tensorflow.python.client import timeline
import argparse


class ModelSpecs(object):
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


def shuffle_input(
        thread_idx,
        model_specs,
        train_file_queue,
        weight_file_queue,
        ex_q):
    with tf.name_scope("shuffled_%s" % (thread_idx,)):
        train_reader = tf.TextLineReader()
        weight_reader = tf.TextLineReader()
        _, data_lines = train_reader.read_up_to(
            train_file_queue, model_specs.batch_size)
        _, weight_lines = weight_reader.read_up_to(
            weight_file_queue, model_specs.batch_size)

        min_after_dequeue = 3 * model_specs.batch_size
        capacity = int(min_after_dequeue + model_specs.batch_size * 1.5)
        data_lines_batch, weight_lines_batch = tf.train.shuffle_batch(
            [data_lines, weight_lines], model_specs.batch_size, capacity,
            min_after_dequeue, enqueue_many=True
        )

        weights = tf.string_to_number(weight_lines_batch, tf.float32)
        labels, sizes, feature_ids, feature_vals = fm_parser(
            data_lines_batch, model_specs.vocabulary_size
        )
        ori_ids, feature_ids = tf.unique(feature_ids)
        feature_poses = tf.concat([[0], tf.cumsum(sizes)], 0)

        enq = ex_q.enqueue([
            labels, weights, feature_ids, ori_ids, feature_vals, feature_poses
        ])
        return [enq] * model_specs.ratio


def input_pipeline(train_files, weight_files, model_specs):
    seed = time.time()

    train_file_queue = tf.train.string_input_producer(
        train_files,
        num_epochs=model_specs.num_epochs,
        shared_name="train_file_queue",
        shuffle=True,
        seed=seed
    )
    weight_file_queue = tf.train.string_input_producer(
        weight_files,
        num_epochs=model_specs.num_epochs,
        shared_name="weight_file_queue",
        shuffle=True,
        seed=seed
    )

    example_queue = tf.FIFOQueue(
        model_specs.queue_size,
        [tf.float32, tf.float32, tf.int32, tf.int64, tf.float32, tf.int32]
    )
    enqueue_ops = sum(
        (
            shuffle_input(i, model_specs, train_file_queue, weight_file_queue,
                          example_queue)
            for i in range(model_specs.shuffle_threads)
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


def train(train_files, weight_files, model_specs, trace, monitor):

    with tf.Graph().as_default():
        vocab_blocks = []
        vocab_size_per_block = (
            model_specs.vocabulary_size / model_specs.vocabulary_block_num + 1
        )
        init_value_range = model_specs.init_value_range

        for i in range(model_specs.vocabulary_block_num):
            vocab_blocks.append(
                tf.Variable(
                    tf.random_uniform(
                        [vocab_size_per_block, model_specs.factor_num + 1],
                        -init_value_range, init_value_range
                    ),
                    name='vocab_block_%d' % i
                )
            )

        (
            exq_size, labels, weights, feature_ids, ori_ids,
            feature_vals, feature_poses
        ) = input_pipeline(train_files, weight_files, model_specs)

        local_params = tf.nn.embedding_lookup(vocab_blocks, ori_ids)

        pred_score, reg_score = fm_scorer(
            feature_ids, local_params, feature_vals, feature_poses,
            model_specs.factor_lambda, model_specs.bias_lambda
        )

        if model_specs.loss_type == 'logistic':
            loss = tf.reduce_mean(
                weights * tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=pred_score, labels=labels
                )
            ) / model_specs.batch_size
        elif model_specs.loss_type == 'mse':
            loss = tf.reduce_mean(weights * tf.square(pred_score - labels))
        else:
            # should never be here
            assert False

        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdagradOptimizer(
            model_specs.learning_rate,
            model_specs.adagrad_init_accumulator
        )
        train_op = optimizer.minimize(
            loss + reg_score / model_specs.batch_size,
            global_step=global_step
        )

        min_after_dequeue = 3 * model_specs.batch_size
        capacity = int(min_after_dequeue + model_specs.batch_size * 1.5)

        run_metadata = tf.RunMetadata()
        sv = tf.train.Supervisor(saver=None)
        step_num = None
        ttotal = 0
        ops_names = ["train_op", "loss", "step_num"]
        ops = [train_op, loss, global_step]
        if monitor:
            shuffleq_sizes = [
                'shuffled_%s/shuffle_batch/random_shuffle_queue_Size:0' %
                i for i in range(
                    model_specs.shuffle_threads)]
            ops_names.append("exq_size")
            ops.append(exq_size)
            ops_names.extend(shuffleq_sizes)
            ops.extend(shuffleq_sizes)

        with sv.managed_session() as sess:
            while not sv.should_stop():
                cur = time.time()
                if step_num is None and trace:
                    ops_res = sess.run(
                        ops,
                        options=tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE
                        ),
                        run_metadata=run_metadata
                    )
                else:
                    ops_res = sess.run(ops)

                res_dict = dict(zip(ops_names, ops_res))
                tend = time.time()
                ttotal = ttotal + tend - cur

                if monitor:
                    print(
                        'speed:',
                        model_specs.batch_size / (tend - cur),
                        'shuffle_queue: %.2f%%' %
                        (max((sum(res_dict[q] for q in shuffleq_sizes) -
                              model_specs.shuffle_threads *
                              min_after_dequeue) * 100.0 /
                             (capacity *
                              model_specs.shuffle_threads), 0)),
                        'example_queue: %.2f%%' %
                        (res_dict['exq_size'] * 100.0 /
                         model_specs.queue_size))

                print(
                    '-- Global Step: %d; Avg loss: %.5f;' % (
                        res_dict['step_num'], res_dict['loss']
                    )
                )

        print(
            'Average speed: ',
            res_dict['step_num'] * model_specs.batch_size / ttotal,
            ' examples/s'
        )

        if trace is not None:
            if not trace.endswith('.json'):
                trace += '.json'
            with open(trace, 'w') as trace_file:
                timeline_info = timeline.Timeline(
                    step_stats=run_metadata.step_stats)
                trace_file.write(timeline_info.generate_chrome_trace_format())


def dist_train(train_files, weight_files, model_specs, trace, monitor, job_name, task_index):
    cluster = tf.train.ClusterSpec({"ps": model_specs.ps_hosts, "worker": model_specs.worker_hosts})
    server = tf.train.Server(cluster,
        job_name=job_name,
        task_index=task_index)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            #worker_device="/job:worker/task:%d" % task_index,
            cluster=cluster)):
            vocab_blocks = []
            vocab_size_per_block = (
                model_specs.vocabulary_size / model_specs.vocabulary_block_num + 1
            )
            init_value_range = model_specs.init_value_range

            for i in range(model_specs.vocabulary_block_num):
                vocab_blocks.append(
                    tf.Variable(
                        tf.random_uniform(
                            [vocab_size_per_block, model_specs.factor_num + 1],
                            -init_value_range, init_value_range
                        ),
                        name='vocab_block_%d' % i
                    )
                )

            (
                exq_size, labels, weights, feature_ids, ori_ids,
                feature_vals, feature_poses
            ) = input_pipeline(train_files, weight_files, model_specs)

            local_params = tf.nn.embedding_lookup(vocab_blocks, ori_ids)

            pred_score, reg_score = fm_scorer(
                feature_ids, local_params, feature_vals, feature_poses,
                model_specs.factor_lambda, model_specs.bias_lambda
            )

            if model_specs.loss_type == 'logistic':
                loss = tf.reduce_mean(
                    weights * tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=pred_score, labels=labels
                    )
                ) / model_specs.batch_size
            elif model_specs.loss_type == 'mse':
                loss = tf.reduce_mean(weights * tf.square(pred_score - labels))
            else:
                # should never be here
                assert False

            global_step = tf.Variable(0, name='global_step', trainable=False)
            #global_step = tf.contrib.framework.get_or_create_global_step()
            optimizer = tf.train.AdagradOptimizer(
                model_specs.learning_rate,
                model_specs.adagrad_init_accumulator
            )
            train_op = optimizer.minimize(
                loss + reg_score / model_specs.batch_size,
                global_step=global_step
            )

            min_after_dequeue = 3 * model_specs.batch_size
            capacity = int(min_after_dequeue + model_specs.batch_size * 1.5)

            run_metadata = tf.RunMetadata()
            step_num = None
            ttotal = 0
            ops_names = ["train_op", "loss", "step_num"]
            ops = [train_op, loss, global_step]
            if monitor:
                shuffleq_sizes = [
                    'shuffled_%s/shuffle_batch/random_shuffle_queue_Size:0' %
                    i for i in range(
                        model_specs.shuffle_threads)]
                ops_names.append("exq_size")
                ops.append(exq_size)
                ops_names.extend(shuffleq_sizes)
                ops.extend(shuffleq_sizes)

            
            #init_op = tf.global_variables_initializer()
            #TODO: make last_step a param
            # The StopAtStepHook handles stopping after running given steps.
            hooks=[tf.train.StopAtStepHook(last_step=1000)]

            with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=(task_index == 0),
                hooks=hooks) as mon_sess:
                
                while not mon_sess.should_stop():
                    #mon_sess.run(init_op)
                    res = mon_sess.run(ops)
                    print("step_num: ", res[2], ", loss: ", res[1])
                

def get_config(config_file):
    GENERAL_SECTION = 'General'
    TRAIN_SECTION = 'Train'
    CLUSTER_SPEC_SECTION = 'ClusterSpec'
    STR_DELIMITER = ','

    config = ConfigParser.ConfigParser()
    config.read(config_file)
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
    model_specs.num_epochs = int(read_config(TRAIN_SECTION, 'epoch_num'))
    model_specs.learning_rate = float(
        read_config(TRAIN_SECTION, 'learning_rate'))
    model_specs.adagrad_init_accumulator = float(
        read_config(TRAIN_SECTION, 'adagrad.initial_accumulator'))
    model_specs.loss_type = read_config(
        TRAIN_SECTION, 'loss_type').strip().lower()
    if model_specs.loss_type not in ['logistic', 'mse']:
        raise ValueError('Unsupported loss type: %s' % model_specs.loss_type)

    if config.has_option(TRAIN_SECTION, 'queue_size'):
        model_specs.queue_size = int(read_config(TRAIN_SECTION, 'queue_size'))
    if config.has_option(TRAIN_SECTION, 'shuffle_threads'):
        model_specs.shuffle_threads = int(
            read_config(TRAIN_SECTION, 'shuffle_threads')
        )
    if config.has_option(TRAIN_SECTION, 'ratio'):
        model_specs.ratio = int(read_config(TRAIN_SECTION, 'ratio'))

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
    
    model_specs.ps_hosts = read_strs_config(CLUSTER_SPEC_SECTION, 'ps_hosts')
    model_specs.worker_hosts = read_strs_config(CLUSTER_SPEC_SECTION, 'worker_hosts')
    
    return train_files, weight_files, model_specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("job_name", type=str)
    parser.add_argument("task_index", type =int)

    parser.add_argument(
        "-t",
        "--trace",
        help="Stores graph info and runtime stats, generates a timeline file")
    parser.add_argument(
        "-m,",
        "--monitor",
        action="store_true",
        help="Prints execution speed to screen")
    args = parser.parse_args()

    train_files, weight_files, model_specs = get_config(args.config_file)


    #train(train_files, weight_files, model_specs, args.trace, args.monitor)
    dist_train(train_files, weight_files, model_specs, args.trace, args.monitor, args.job_name, args.task_index)

if __name__ == '__main__':
    print('starting...')
    main()
