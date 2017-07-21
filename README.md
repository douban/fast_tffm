# Tensorflow-based Distributed Factorization Machine
An efficient distributed factoriazation machine implementation based on tensorflow (cpu only).

1. Support both multi-thread local machine training and distributed training.
2. Can easily benefit from numerous implementations of operators in tensorflow, e.g., different optimizors, loss functions.
3. Customized c++ operators, significantly faster than pure python implementations. Comparable performance (actually faster according to my benchmark) with pure c++ implementation.

## Quick Start
### Build
```
python setup.py build_ext -i
```
### Local Training
```
python run_tffm.py train sample.cfg [-m] [-t trace_file_name]
```
for CPU version, use
```
CUDA_VISIBLE_DEVICES=-1 python ...
```

### Distributed Training
Open 4 command line windows. Run the following commands on each window to start 2 parameter servers and 2 workers.
```
python run_tffm.py sample.cfg --dist_train ps 0
python run_tffm.py sample.cfg --dist_train ps 1
python run_tffm.py sample.cfg --dist_train worker 0
python run_tffm.py sample.cfg --dist_train worker 1
```
### Local Prediction (not supported by most recent update)
```
python fast_tffm.py predict sample.cfg
```
### Distributed Prediction (not supported by most recent update)
Open 4 command line windows. Run the following commands on each window to start 2 parameter servers and 2 workers.
```
python fast_tffm.py dist_predict sample.cfg ps 0
python fast_tffm.py dist_predict sample.cfg ps 1
python fast_tffm.py dist_predict sample.cfg worker 0
python fast_tffm.py dist_predict sample.cfg worker 1
```
## Benchmark

1. Local Mode. Training speed compared with difacto using the same configuration

  + *Configuration*: 36672494 training examples, 10 threads, factor_num = 8, batch_size = 10000, epoch_num = 1, vocabulary_size = 40000000
  + **Difacto**: 337 seconds. 108820 examples / second.
  + **FastTffm**: 157 seconds. 233582 examples / second.
  
2. Distriubuted Mode. (I did not find other open source projects which support distributed training. Difacto claims so, but their distributed mode is not implemeted yet)
  + *Configuration*: 36672494 training examples, 10 threads, factor_num = 8, batch_size = 10000, epoch_num = 1, vocabulary_size = 40000000
  + *Cluster*: 1 ps, 4 workers.
  + **FastTffm**: 49 seconds. 748418 examples / second.
  
## Input Data Format
1. Data File
  ```
  <label> <fid_0>[:<fval_0>] [<fid_1>[:<fval_1>] ...]
  ```
  \<label\>: 0 or 1 if loss_type = logistic; any real number if loss_type = mse.

  \<fid_k\>: An integer if hash_feature_id = False; Arbitrary string if hash_feature_id = True

  \<fval_k\>: Any real number. Default value 1.0 if omitted.

2. Weight File
  Should have the same line number with the corresponding data file. Each line contains one real number.

Check the data/weight files in the data folder for details. The data files are sampled from [criteo lab dataset](http://labs.criteo.com/tag/dataset/).

## Run with TFMesos (not supported by most recent update)

```
tfrun -w 4 -s 1 -m ${MESOS_MASTER} python fast_tffm.py dist_train sample.cfg {job_name} {task_index} {ps_hosts} {worker_hosts}
```

## Tensorboard Visualization (not supported by most recent update)

Set the empty log directory path in `sample.cfg`. The default log saving frequency is 10 files per save. 

Saving content includes the recent 10 files' RMSE and total RMSE of training and validation data.

Find the directory of your tensorflow then use the following command to activate tensorboard:
```
python .../site-packages/tensorflow/tensorboard/tensorboard.py --logdir==your_log_dir
```
The default port is 6060.
