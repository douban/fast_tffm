/*
 * fm_grad_op.cc
 *
 *  Created on: Sep 5, 2016
 *      Author: mianwei
 */

#define EIGEN_USE_THREADS

#include "fm_grad_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

REGISTER_OP("FmGrad")
    .Input("feature_ids: int32")
    .Input("feature_params: float32")
    .Input("feature_vals: float32")
    .Input("feature_poses: int32")
    .Input("factor_lambda: float32")
    .Input("bias_lambda: float32")
    .Input("pred_grad: float32")
    .Input("reg_grad: float32")
    .Output("params_grad: float32");

using namespace tensorflow;
using fm::generator::FactorSumGenerator;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace fm {

namespace functor {

template<>
void FmGrad<CPUDevice>::operator() (
  const CPUDevice& d,
  const typename TTypes<int32>::ConstTensor& feature_ids,
  const typename TTypes<float, 2>::ConstTensor& feature_params,
  const typename TTypes<float>::ConstTensor& feature_vals,
  const typename TTypes<int32>::ConstTensor& feature_poses,
  const typename TTypes<float>::ConstScalar& factor_lambda,
  const typename TTypes<float>::ConstScalar& bias_lambda,
  const typename TTypes<float>::ConstTensor& pred_grad,
  const typename TTypes<float>::ConstScalar& reg_grad,
  const int64 batch_size,
  const int64 factor_num,
  typename TTypes<float, 2>::Tensor& factor_sum,
  typename TTypes<float>::Tensor& scratch,
  typename TTypes<float, 2>::Tensor& param_grad
) const {
  FactorSumGenerator<CPUDevice> factor_sum_gen(
    feature_poses, feature_ids, feature_vals, feature_params, factor_num
  );
  factor_sum = factor_sum.generate(factor_sum_gen);

  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = feature_poses(i); j < feature_poses(i + 1); ++j) {
      auto fid = feature_ids(j);
      auto fval = feature_vals(j);
      for (size_t k = 0; k < factor_num + 1; ++k) {
        float t = feature_params(fid, k);
        if (k == 0) {
          param_grad(fid, k) += pred_grad(i) * fval + reg_grad() * bias_lambda() * t;
        } else {
          param_grad(fid, k) += reg_grad() * factor_lambda() * t - pred_grad(i) * fval * fval * t;
          param_grad(fid, k) += pred_grad(i) * fval * factor_sum(i, k - 1);
        }
      }
    }
  }
}

#ifdef WITH_CUDA
template<>
void FmGrad<GPUDevice>::operator() (
  const GPUDevice& d,
  const typename TTypes<int32>::ConstTensor& feature_ids,
  const typename TTypes<float, 2>::ConstTensor& feature_params,
  const typename TTypes<float>::ConstTensor& feature_vals,
  const typename TTypes<int32>::ConstTensor& feature_poses,
  const typename TTypes<float>::ConstScalar& factor_lambda,
  const typename TTypes<float>::ConstScalar& bias_lambda,
  const typename TTypes<float>::ConstTensor& pred_grad,
  const typename TTypes<float>::ConstScalar& reg_grad,
  const int64 batch_size,
  const int64 factor_num,
  typename TTypes<float, 2>::Tensor& factor_sum,
  typename TTypes<float>::Tensor& scratch,
  typename TTypes<float, 2>::Tensor& param_grad
) const;
extern template struct FmGrad<GPUDevice>;
#endif

}

}

template<typename Device>
class FmGradOp : public OpKernel {
 public:
  explicit FmGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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
    const Tensor* pred_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("pred_grad", &pred_grad_tensor));
    const Tensor* reg_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("reg_grad", &reg_grad_tensor));

    auto feature_ids = feature_ids_tensor->flat<int32>();
    auto feature_params = feature_params_tensor->matrix<float>();
    auto feature_vals = feature_vals_tensor->flat<float>();
    auto feature_poses = feature_poses_tensor->flat<int32>();
    auto factor_lambda = factor_lambda_tensor->scalar<float>();
    auto bias_lambda = bias_lambda_tensor->scalar<float>();
    auto pred_grad = pred_grad_tensor->flat<float>();
    auto reg_grad = reg_grad_tensor->scalar<float>();

    int64 batch_size = feature_poses.size() - 1;
    int64 factor_num = feature_params_tensor->dim_size(1) - 1;

    Tensor* param_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("params_grad", feature_params_tensor->shape(), &param_grad_tensor));
    auto param_grad = param_grad_tensor->matrix<float>();

    Tensor factor_sum_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, TensorShape({batch_size, factor_num}), &factor_sum_tensor));
    auto factor_sum = factor_sum_tensor.matrix<float>();

    Tensor scratch_tensor;
    if (std::is_same<Device, GPUDevice>::value) {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, TensorShape({factor_num}), &scratch_tensor));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, TensorShape({0}), &scratch_tensor));
    }
    auto scratch = scratch_tensor.flat<float>();

    fm::functor::FmGrad<Device> functor;
    functor(
      ctx->eigen_device<Device>(),
      feature_ids, feature_params, feature_vals, feature_poses,
      factor_lambda, bias_lambda, pred_grad, reg_grad,
      batch_size, factor_num, factor_sum, scratch, param_grad
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("FmGrad").Device(DEVICE_CPU), FmGradOp<CPUDevice>);
#ifdef WITH_CUDA
REGISTER_KERNEL_BUILDER(Name("FmGrad").Device(DEVICE_GPU), FmGradOp<GPUDevice>);
#endif

// vim: ts=2:sts=2:sw=2
