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
python run_tffm.py train sample.cfg --dist ps 0 localhost:2333,localhost:2334 localhost:2335,localhost:2336
python run_tffm.py train sample.cfg --dist ps 1 localhost:2333,localhost:2334 localhost:2335,localhost:2336
python run_tffm.py train sample.cfg --dist worker 0 localhost:2333,localhost:2334 localhost:2335,localhost:2336
python run_tffm.py train sample.cfg --dist worker 1 localhost:2333,localhost:2334 localhost:2335,localhost:2336
```
### Local Prediction
```
python run_tffm.py predict sample.cfg
```
### Distributed Prediction (not supported by most recent update)
Open 4 command line windows. Run the following commands on each window to start 2 parameter servers and 2 workers.
```
python run_tffm.py predict sample.cfg --dist ps 0 localhost:2333,localhost:2334 localhost:2335,localhost:2336
python run_tffm.py predict sample.cfg --dist ps 1 localhost:2333,localhost:2334 localhost:2335,localhost:2336
python run_tffm.py predict sample.cfg --dist worker 0 localhost:2333,localhost:2334 localhost:2335,localhost:2336
python run_tffm.py predict sample.cfg --dist worker 1 localhost:2333,localhost:2334 localhost:2335,localhost:2336
```
 
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

## Run with TFMesos

```
tfrun -w 4 -s 1 -m ${MESOS_MASTER} -- python run_tffm.py [train, predict] sample.cfg --dist_train {job_name} {task_index} {ps_hosts} {worker_hosts}
```
## Export Model to Saved_Model_CLI

To generate a new model (export path must not be a pre-existing directory):
```
python run_tffm.py generate sample.cfg --export_path saved_model
```

To use the model for prediction:

```
saved_model_cli run --dir /home2/libingqing/fast_tffm/saved_model_simplified/ --tag_set serve --signature_def serving_default --inputs data_lines=data.npy --outdir=./scores
```
For detailed instructions on using SavedModel CLI, refer to [TensorFlow official documentation.](https://www.tensorflow.org/programmers_guide/saved_model_cli) 

## Tensorboard Visualization

Set the empty log directory path in `sample.cfg`. The default log saving frequency is 10 global steps per save. 

Saving content includes RMSE and total RMSE of training and validation data over the last 10 steps.

Find the directory of your tensorflow then use the following command to activate tensorboard:
```
python .../site-packages/tensorflow/tensorboard/tensorboard.py --logdir==your_log_dir
```
The default port is 6060.
