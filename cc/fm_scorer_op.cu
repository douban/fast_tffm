#define EIGEN_USE_GPU

#include "fm_scorer_op.h"

typedef Eigen::GpuDevice GPUDevice;

namespace fm {
  template struct generator::FeatureRankGenerator<GPUDevice>;
  template struct generator::BiasGenerator<GPUDevice>;
  template struct functor::FmScorer<GPUDevice>;
}

// vim: ts=2:sts=2:sw=2
