#define EIGEN_USE_THREADS

#include "fm_scorer_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <ctime>
#include <iostream>

REGISTER_OP("FmScorer")
    .Input("feature_ids: int32")
    .Input("feature_params: float32")
    .Input("feature_vals: float32")
    .Input("feature_poses: int32")
    .Input("factor_lambda: float")
    .Input("bias_lambda: float")
    .Output("pred_score: float32")
    .Output("reg_score: float32");

using namespace tensorflow;
using fm::functor::FmScorer;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
class FmScorerOp : public OpKernel {
 public:
  explicit FmScorerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* feature_ids_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("feature_ids", &feature_ids_tensor));
    const Tensor* feature_params_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("feature_params", &feature_params_tensor));
    const Tensor* feature_vals_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("feature_vals", &feature_vals_tensor));
    const Tensor* feature_poses_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("feature_poses", &feature_poses_tensor));
    const Tensor* factor_lambda_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("factor_lambda", &factor_lambda_tensor));
    const Tensor* bias_lambda_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("bias_lambda", &bias_lambda_tensor));

    auto feature_ids = feature_ids_tensor->flat<int32>();
    auto feature_params = feature_params_tensor->flat<float>();
    auto feature_vals = feature_vals_tensor->flat<float>();
    auto feature_poses = feature_poses_tensor->flat<int32>();
    auto factor_lambda = factor_lambda_tensor->scalar<float>();
    auto bias_lambda = bias_lambda_tensor->scalar<float>();
    
    int64 batch_size = feature_poses.size() - 1;
    int64 factor_num = feature_params_tensor->dim_size(1) - 1;

    Tensor* pred_score_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("pred_score", TensorShape({batch_size}), &pred_score_tensor));
    auto pred_score = pred_score_tensor->flat<float>();
    Tensor* reg_score_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("reg_score", TensorShape({}), &reg_score_tensor));
    auto reg_score = reg_score_tensor->scalar<float>();

    Tensor feature_ranks_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, TensorShape({2, batch_size, factor_num}), &feature_ranks_tensor));
    auto feature_ranks = feature_ranks_tensor.tensor<float, 3>();

    Tensor biases_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, TensorShape({3, batch_size}), &biases_tensor));
    auto biases = biases_tensor.matrix<float>();

    FmScorer<Device> fm_scorer;
    fm_scorer(
      ctx->eigen_device<Device>(),
      feature_ids, feature_params, feature_vals, feature_poses,
      factor_lambda, bias_lambda, batch_size, factor_num,
      biases, feature_ranks, pred_score, reg_score
    );
  }
};

#ifdef WITH_CUDA
namespace fm {
namespace functor {
  template<>
  void FmScorer<GPUDevice>::operator() (
    const GPUDevice& d,
    const typename TTypes<int32>::ConstTensor& feature_ids,
    const typename TTypes<float>::ConstTensor& feature_params,
    const typename TTypes<float>::ConstTensor& feature_vals,
    const typename TTypes<int32>::ConstTensor& feature_poses,
    const typename TTypes<float>::ConstScalar& factor_lambda,
    const typename TTypes<float>::ConstScalar& bias_lambda,
    const int64 batch_size,
    const int64 factor_num,
    typename TTypes<float, 2>::Tensor& biases,
    typename TTypes<float, 3>::Tensor& feature_ranks,
    typename TTypes<float>::Tensor& pred_score,
    typename TTypes<float>::Scalar& reg_score
  ) const;
extern template struct FmScorer<GPUDevice>;

}
}
#endif

REGISTER_KERNEL_BUILDER(Name("FmScorer").Device(DEVICE_CPU), FmScorerOp<CPUDevice>);
#ifdef WITH_CUDA
REGISTER_KERNEL_BUILDER(Name("FmScorer").Device(DEVICE_GPU), FmScorerOp<GPUDevice>);
#endif

// vim: ts=2:sts=2:sw=2
