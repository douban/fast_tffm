#define EIGEN_USE_GPU

#include "fm_grad_op.h"
#include <iostream>

typedef Eigen::GpuDevice GPUDevice;

namespace fm {
template struct generator::FactorSumGenerator<GPUDevice>;
template struct generator::ParamGradGenerator<GPUDevice>;
template struct functor::FmGrad<GPUDevice>;
}

// vim: ts=2:sts=2:sw=2
