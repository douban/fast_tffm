from __future__ import print_function
import tensorflow as tf
import argparse
from tffm.fm_model import Model
from tensorflow.python.client import timeline
import time
import os


def predict(model, sess):
    tf.train.start_queue_runners(sess)
    with open(model.score_path, 'w') as f:
        while not sess.should_stop():
            pred_score = sess.run(model.pred_op)
            for score in pred_score:
                f.write(str(score) + '\n')


def train(model, sess, monitor, trace):
    min_after_dequeue = 3 * model.batch_size
    capacity = int(min_after_dequeue + model.batch_size * 1.5)
    run_metadata = tf.RunMetadata()

    tf.train.start_queue_runners(sess)
    st = time.time()
    step_num = None
    end_session = False
    while not (sess.should_stop() or end_session):
        cur = time.time()
        options_ = None
        run_metadata_ = None
        if step_num is None and trace:
            options_ = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE
            )
            run_metadata_ = run_metadata

        try:
            res_dict = sess.run(
                model.ops, options=options_, run_metadata=run_metadata_
            )
        except tf.errors.OutOfRangeError:
            break

        
        step_num = res_dict['step_num']
        tend = time.time()

        if monitor:
            _min = model.shuffle_threads * min_after_dequeue
            shuffleq_size = max(sum(res_dict['shuffleq_sizes']) - _min, 0)
            print(
                'speed:', model.batch_size / (tend - cur),
                'shuffle_queue: %.2f%%' % (
                    shuffleq_size * 100.0 / (capacity * model.shuffle_threads),
                ),
                'example_queue: %.2f%%' % (
                    res_dict['exq_size'] * 100.0 / model.queue_size
                )
            )

        loss = res_dict['loss']
        print('-- Global Step: %d; Avg loss: %.5f;' % (step_num, loss))
        
        if model.validation_data is not None and step_num is not None and step_num % model.save_steps == 0:
            v_loss = sess.run(model.valid_op)
            print('validation loss at step %d: %.8f'% (step_num, v_loss))
            if v_loss < model.tolerance:
                print('Loss on validation data set is below tolerance.'
                        'Training completed.')
                end_session = True

    total = time.time() - st
    print('Average speed: ', step_num * model.batch_size / total, ' ex/s')

    

    if trace is not None:
        if not trace.endswith('.json'):
            trace += '.json'
        with open(trace, 'w') as trace_file:
            timeline_info = timeline.Timeline(
                step_stats=run_metadata.step_stats)
            trace_file.write(timeline_info.generate_chrome_trace_format())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=['train', 'predict'])
    parser.add_argument("config_file", type=str)
    parser.add_argument(
        "--dist_train",
        nargs=4,
        default=None,
        help="--dist_train job_name task_index ps_hosts worker_hosts")

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

    model = Model(args.config_file)
    if args.task == 'predict' and model.log_dir is None:
        print("Missing log directory. Must include a checkpoint file.")
        os.exit(1)

    cluster = None
    master = ''
    worker_device = '/job:worker'
    is_chief = True
    log_dir = model.log_dir
    if args.dist_train is not None:
        ps_hosts = args.dist_train[2].split(',')
        worker_hosts = args.dist_train[3].split(',')
        cluster = tf.train.ClusterSpec(
            {"ps": ps_hosts, "worker": worker_hosts})
        server = tf.train.Server(cluster,
                                 job_name=args.dist_train[0],
                                 task_index=int(args.dist_train[1]))
        if args.dist_train[0] == "ps":
            server.join()
        else:
            assert args.dist_train[0] == 'worker'
            master = server.target
            worker_device = "/job:worker/task:%d" % int(args.dist_train[1])
            is_chief = (int(args.dist_train[1]) == 0)
            if not is_chief:
                log_dir = None

    with tf.device(tf.train.replica_device_setter(
            worker_device=worker_device,
            cluster=cluster)):
        model.build_graph(args.monitor, args.trace)


    saver_hook = tf.train.CheckpointSaverHook(model.log_dir, save_steps=model.save_steps)
    with tf.train.MonitoredTrainingSession(
        master=master, is_chief=is_chief, checkpoint_dir=log_dir,
        save_summaries_steps=model.save_summaries_steps,
        hooks = [saver_hook]
    ) as mon_sess:
        print("========", args.task, "========")
        if args.task == 'train':
            train(model, mon_sess, args.monitor, args.trace)
        else:
            predict(model, mon_sess)


if __name__ == '__main__':
    main()
