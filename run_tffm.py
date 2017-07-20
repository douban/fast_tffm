from __future__ import print_function
import glob
import ConfigParser
import tensorflow as tf
import argparse
from fm_model import Model


def get_config(config_file):
    GENERAL_SECTION = 'General'
    TRAIN_SECTION = 'Train'
    CLUSTER_SPEC_SECTION = 'ClusterSpec'
    STR_DELIMITER = ','

    config = ConfigParser.ConfigParser()
    config.read(config_file)
    model = Model()

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
    model.vocabulary_size = int(read_config(
        GENERAL_SECTION, 'vocabulary_size'))
    model.vocabulary_block_num = int(read_config(
        GENERAL_SECTION, 'vocabulary_block_num'))
    model.factor_num = int(read_config(
        GENERAL_SECTION, 'factor_num'))
    model.hash_feature_id = read_config(
        GENERAL_SECTION, 'hash_feature_id').strip().lower() == 'true'
    model.log_file = read_config(GENERAL_SECTION, 'log_file')

    model.batch_size = int(read_config(TRAIN_SECTION, 'batch_size'))
    model.init_value_range = float(
        read_config(TRAIN_SECTION, 'init_value_range'))
    model.factor_lambda = float(
        read_config(TRAIN_SECTION, 'factor_lambda'))
    model.bias_lambda = float(read_config(TRAIN_SECTION, 'bias_lambda'))
    model.num_epochs = int(read_config(TRAIN_SECTION, 'epoch_num'))
    model.learning_rate = float(
        read_config(TRAIN_SECTION, 'learning_rate'))
    model.adagrad_init_accumulator = float(
        read_config(TRAIN_SECTION, 'adagrad.initial_accumulator'))
    model.loss_type = read_config(
        TRAIN_SECTION, 'loss_type').strip().lower()
    if model.loss_type not in ['logistic', 'mse']:
        raise ValueError('Unsupported loss type: %s' % model.loss_type)

    if config.has_option(TRAIN_SECTION, 'queue_size'):
        model.queue_size = int(read_config(TRAIN_SECTION, 'queue_size'))
    if config.has_option(TRAIN_SECTION, 'shuffle_threads'):
        model.shuffle_threads = int(
            read_config(TRAIN_SECTION, 'shuffle_threads')
        )
    if config.has_option(TRAIN_SECTION, 'ratio'):
        model.ratio = int(read_config(TRAIN_SECTION, 'ratio'))

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

    if config.has_option(CLUSTER_SPEC_SECTION, 'ps_hosts'):
        model.ps_hosts = read_strs_config(CLUSTER_SPEC_SECTION, 'ps_hosts')
    if config.has_option(CLUSTER_SPEC_SECTION, 'worker_hosts'):
        model.worker_hosts = read_strs_config(
            CLUSTER_SPEC_SECTION, 'worker_hosts')

    return train_files, weight_files, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument(
        "--dist_train",
        nargs=2,
        default=None,
        help="--dist_train job_name task_index")

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

    train_files, weight_files, model = get_config(args.config_file)
    if args.dist_train is not None:
        cluster = tf.train.ClusterSpec(
            {"ps": model.ps_hosts, "worker": model.worker_hosts})
        server = tf.train.Server(cluster,
                                 job_name=args.dist_train[0],
                                 task_index=int(args.dist_train[1]))
        model.build_graph(train_files,
                          weight_files,
                          args.trace,
                          args.monitor,
                          args.dist_train[0],
                          int(args.dist_train[1]),
                          cluster,
                          server)
        mon_sess = tf.train.MonitoredTrainingSession(
            master=server.target,
            is_chief=(int(args.dist_train[1]) == 0))
    else:
        model.build_graph(
            train_files,
            weight_files,
            args.trace,
            args.monitor,
            None,
            None,
            None,
            None)
        mon_sess = tf.train.MonitoredTrainingSession()

    model.train(mon_sess, args.monitor, args.trace)


if __name__ == '__main__':
    print('starting...')
    main()
