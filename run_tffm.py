from __future__ import print_function
import tensorflow as tf
import argparse
from tffm.fm_model import Model
from tensorflow.python.client import timeline
import time
import os


def predict(model, sess):
    res = dict(zip(model.predict_files, model.pred_ops))
    for data_file, pred_op in res.items():
        with open(data_file + '_score', 'w') as f:
            pred_score = sess.run(pred_op)
            for score in pred_score:
                f.write(str(score) + '\n')
    print('Done. Scores saved to same directory as predict files')


def train(model, sess, monitor, trace):
    min_after_dequeue = 3 * model.batch_size
    capacity = int(min_after_dequeue + model.batch_size * 1.5)
    run_metadata = tf.RunMetadata()

    tf.train.start_queue_runners(sess)
    st = time.time()
    start_step = None
    end_session = False
    while not (sess.should_stop() or end_session):
        cur = time.time()
        options_ = None
        run_metadata_ = None
        if start_step is None and trace:
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
        if start_step is None:
            start_step = step_num - 1
            prev_step = start_step
        tend = time.time()

        if monitor:
            _min = model.shuffle_threads * min_after_dequeue
            shuffleq_size = max(sum(res_dict['shuffleq_sizes']) - _min, 0)
            print(
                'speed:', model.batch_size / (tend - cur) * (step_num - prev_step),
                'shuffle_queue: %.2f%%' % (
                    shuffleq_size * 100.0 / (capacity * model.shuffle_threads),
                ),
                'example_queue: %.2f%%' % (
                    res_dict['exq_size'] * 100.0 / model.queue_size
                )
            )

        prev_step = step_num
        loss = res_dict['loss']
        print('-- Global Step: %d; Avg loss: %.5f;' % (step_num, loss))

        if (model.validation_data is not None
                and step_num is not None
                and step_num % model.save_steps == 0):
            v_loss = sess.run(model.valid_op)
            print('validation loss at step %d: %.8f' % (step_num, v_loss))
            if v_loss < model.tolerance:
                print('Loss on validation data set is below tolerance. '
                      'Training completed.')
                end_session = True

    total = time.time() - st
    print('Average speed: ', (step_num - start_step)
          * model.batch_size / total, ' ex/s')
    print('Model saved to ', model.log_dir)

    if trace is not None:
        if not trace.endswith('.json'):
            trace += '.json'
        with open(trace, 'w') as trace_file:
            timeline_info = timeline.Timeline(
                step_stats=run_metadata.step_stats)
            trace_file.write(timeline_info.generate_chrome_trace_format())


def generate_saved_model(model, export_path):
    ckpt = tf.train.get_checkpoint_state(model.log_dir)
    with tf.Session() as sess:
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        tensor_info_data_lines = tf.saved_model.utils.build_tensor_info(
            model.data_lines_serving)
        tensor_info_pred_score = tf.saved_model.utils.build_tensor_info(
            model.pred_score_serving)

        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'data_lines': tensor_info_data_lines},
                outputs={'scores': tensor_info_pred_score},
                method_name=method_name))
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants
                .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature,
            })
        builder.save()

        print('Done exporting!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=['train', 'predict', 'generate'])
    parser.add_argument("config_file", type=str)
    parser.add_argument(
        "--dist",
        nargs=4,
        metavar=('JOB_NAME', 'TASK_INDEX', 'PS_HOSTS', 'WORKER_HOSTS'),
        default=None,
        help="For distributed training or prediction"
    )

    parser.add_argument(
        "--protocol",
        default='grpc',
        help="protocol for distributed training"
    )

    parser.add_argument(
        "-t",
        "--trace",
        metavar='OUTPUT_FILE_NAME',
        help="Stores graph info and runtime stats, generates a timeline file")
    parser.add_argument(
        "-m,",
        "--monitor",
        action="store_true",
        help="Prints execution speed to screen")
    parser.add_argument(
        "--export_path",
        help="Specifies the location to which the model is to be exported.")
    args = parser.parse_args()

    model = Model(args.config_file)
    if args.task == 'predict' and model.log_dir is None:
        print("Missing log directory. Must include a checkpoint file.")
        os._exit(1)

    cluster = None
    master = ''
    worker_device = '/job:worker'
    is_chief = True
    log_dir = model.log_dir
    if args.dist is not None:
        ps_hosts = args.dist[2].split(',')
        worker_hosts = args.dist[3].split(',')
        cluster = tf.train.ClusterSpec(
            {"ps": ps_hosts, "worker": worker_hosts})
        server = tf.train.Server(cluster,
                                 job_name=args.dist[0],
                                 task_index=int(args.dist[1]),
                                 protocol=args.protocol)
        if args.dist[0] == "ps":
            server.join()
        else:
            assert args.dist[0] == 'worker'
            master = server.target
            worker_device = "/job:worker/task:%d" % int(args.dist[1])
            is_chief = (int(args.dist[1]) == 0)
            if not is_chief:
                log_dir = None
    if args.task == 'generate':
        if args.export_path is None:
            print("Export path is not specified. Use --export_path.")
            os._exit(2)

        model.build_serving_graph()
        generate_saved_model(model, args.export_path)
    else:
        with tf.device(tf.train.replica_device_setter(
                worker_device=worker_device,
                cluster=cluster)):
            model.build_graph(args.monitor, args.trace)

        hooks = []
        if model.log_dir is not None:
            hooks.append(tf.train.CheckpointSaverHook(
                model.log_dir, save_steps=model.save_steps))
        with tf.train.MonitoredTrainingSession(
            master=master, is_chief=is_chief, checkpoint_dir=log_dir,
            save_summaries_steps=model.save_summaries_steps,
            hooks=hooks
        ) as mon_sess:
            print("========", args.task, "========")
            if args.task == 'train':
                train(model, mon_sess, args.monitor, args.trace)
            elif args.task == 'predict':
                predict(model, mon_sess)


if __name__ == '__main__':
    main()
