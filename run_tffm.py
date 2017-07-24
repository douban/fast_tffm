from __future__ import print_function
import tensorflow as tf
import argparse
from fm_model import Model
from tensorflow.python.client import timeline
import time


def train(model, sess, monitor, trace):
    min_after_dequeue = 3 * model.batch_size
    capacity = int(min_after_dequeue + model.batch_size * 1.5)
    run_metadata = tf.RunMetadata()

    ttotal = 0
    step_num = None
    while not sess.should_stop():
        cur = time.time()
        if step_num is None and trace:
            ops_res = sess.run(
                model.ops,
                options=tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE
                ),
                run_metadata=run_metadata
            )
        else:
            ops_res = sess.run(model.ops)

        res_dict = dict(zip(model.ops_names, ops_res))
        tend = time.time()
        ttotal = ttotal + tend - cur

        if monitor:
            print(
                'speed:',
                model.batch_size / (tend - cur),
                'shuffle_queue: %.2f%%' %
                (max((sum(res_dict[q] for q in
                          model.ops_names[-(model.shuffle_threads):]) -
                      model.shuffle_threads *
                      min_after_dequeue) * 100.0 /
                     (capacity * model.shuffle_threads),
                     0)),
                'example_queue: %.2f%%' %
                (res_dict['exq_size'] * 100.0 /
                 model.queue_size))

        print(
            '-- Global Step: %d; Avg loss: %.5f;' % (
                res_dict['step_num'], res_dict['loss']
            )
        )

    print(
        'Average speed: ',
        res_dict['step_num'] * model.batch_size / ttotal,
        ' examples/s'
    )

    if trace is not None:
        if not trace.endswith('.json'):
            trace += '.json'
        with open(trace, 'w') as trace_file:
            timeline_info = timeline.Timeline(
                step_stats=run_metadata.step_stats)
            trace_file.write(timeline_info.generate_chrome_trace_format())


def main():
    parser = argparse.ArgumentParser()
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
    ps_hosts = args.dist_train[2].split(',')
    worker_hosts = args.dist_train[3].split(',')
    if args.dist_train is not None:
        cluster = tf.train.ClusterSpec(
            {"ps": ps_hosts, "worker": worker_hosts})
        server = tf.train.Server(cluster,
                                 job_name=args.dist_train[0],
                                 task_index=int(args.dist_train[1]))
        if args.dist_train[0] == "ps":
            server.join()
        elif args.dist_train[0] == "worker":
            with tf.device(tf.train.replica_device_setter(
                    cluster=cluster)):
                model.build_graph(args.monitor, args.trace)

            mon_sess = tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=(int(args.dist_train[1]) == 0))
            train(model, mon_sess, args.monitor, args.trace)
    else:
        model.build_graph(args.monitor, args.trace)
        mon_sess = tf.train.MonitoredTrainingSession()
        train(model, mon_sess, args.monitor, args.trace)


if __name__ == '__main__':
    print('starting...')
    main()
