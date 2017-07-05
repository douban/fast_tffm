#include "tensorflow/core/framework/tensor.h"

namespace {
  EIGEN_ALWAYS_INLINE
  void __atomicAdd(float* target, float inc) {
    int32_t* _target = reinterpret_cast<int32_t*>(target);
    int32_t oldval = *_target;
    int32_t newval;
    float tmp;
    do {
      tmp = *reinterpret_cast<float*>(&oldval) + inc;
      newval = *reinterpret_cast<int32_t*>(&tmp);
      oldval = __sync_val_compare_and_swap(_target, oldval, newval);
    } while(oldval != newval);
  }
}

namespace fm {

namespace generator {
  using namespace tensorflow;

  template<typename Device>
  struct FactorSumGenerator {
    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
    FactorSumGenerator(
      const typename TTypes<int32>::ConstTensor& feature_poses_,
      const typename TTypes<int32>::ConstTensor& feature_ids_,
      const typename TTypes<float>::ConstTensor& feature_vals_,
      const typename TTypes<float, 2>::ConstTensor& feature_params_,
      const int64 factor_num_
    ) : feature_poses(feature_poses_), feature_ids(feature_ids_),
    feature_vals(feature_vals_), feature_params(feature_params_),
    factor_num(factor_num_)
    {}

    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
    float operator()(const Eigen::array<Eigen::DenseIndex, 2>& coeff) const {
      auto i = coeff[0];
      auto k = coeff[1];
      float r = 0;
      for (size_t j = feature_poses(i); j < feature_poses(i + 1); ++j) {
        int32 fid = feature_ids(j);
        float fval = feature_vals(j);
        float t = feature_params(fid, k + 1);
        r += fval * t;
      }
      return r;
    }

  private:
    const typename TTypes<int32>::ConstTensor feature_poses;
    const typename TTypes<int32>::ConstTensor feature_ids;
    const typename TTypes<float>::ConstTensor feature_vals;
    const typename TTypes<float, 2>::ConstTensor feature_params;
    const int64 factor_num;
  };

  template<typename Device>
  struct ParamGradGenerator {
    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
    ParamGradGenerator(
      const typename TTypes<float, 2>::Tensor& param_grad_,
      const typename TTypes<int32>::ConstTensor& feature_poses_,
      const typename TTypes<int32>::ConstTensor& feature_ids_,
      const typename TTypes<float>::ConstTensor& feature_vals_,
      const typename TTypes<float, 2>::ConstTensor& feature_params_,
      const typename TTypes<float>::ConstTensor& pred_grad_,
      const typename TTypes<float>::ConstScalar& reg_grad_,
      const typename TTypes<float>::ConstScalar& factor_lambda_,
      const typename TTypes<float>::ConstScalar& bias_lambda_,
      const typename TTypes<float, 2>::Tensor& factor_sum_,
      const int64 factor_num_
    ) : param_grad(param_grad_), feature_poses(feature_poses_), feature_ids(feature_ids_),
    feature_vals(feature_vals_), feature_params(feature_params_),
    pred_grad(pred_grad_), reg_grad(reg_grad_), factor_lambda(factor_lambda_), bias_lambda(bias_lambda_),
    factor_sum(factor_sum_), factor_num(factor_num_)
    {}

    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
    float operator()(const Eigen::array<Eigen::DenseIndex, 2>& coeff) const {
      auto i = coeff[0];
      auto k = coeff[1];
      for (size_t j = feature_poses(i); j < feature_poses(i + 1); ++j) {
        auto fid = feature_ids(j);
        auto fval = feature_vals(j);
        float t = feature_params(fid, k);
        if (k == 0) {
          float v = pred_grad(i) * fval + reg_grad() * bias_lambda() * t;
#ifdef __CUDA_ARCH__
          atomicAdd(&param_grad(fid, k), v);
#else
          __atomicAdd(&param_grad(fid, k), v);
#endif
        } else {
          float v = reg_grad() * factor_lambda() * t - pred_grad(i) * fval * fval * t \
                    + pred_grad(i) * fval * factor_sum(i, k - 1);
#ifdef __CUDA_ARCH__
          atomicAdd(&param_grad(fid, k), v);
#else
          __atomicAdd(&param_grad(fid, k), v);
#endif
        }
      }
      return 0;
    }

  private:
    mutable typename TTypes<float, 2>::Tensor param_grad;
    const typename TTypes<int32>::ConstTensor feature_poses;
    const typename TTypes<int32>::ConstTensor feature_ids;
    const typename TTypes<float>::ConstTensor feature_vals;
    const typename TTypes<float, 2>::ConstTensor feature_params;
    const typename TTypes<float>::ConstTensor pred_grad;
    const typename TTypes<float>::ConstScalar reg_grad;
    const typename TTypes<float>::ConstScalar factor_lambda;
    const typename TTypes<float>::ConstScalar bias_lambda;
    const typename TTypes<float, 2>::Tensor factor_sum;
    const int64 factor_num;
  };
}

namespace functor {
  using namespace tensorflow;

  template<typename Device>
  struct FmGrad {
    EIGEN_ALWAYS_INLINE
    void operator() (
      const Device& d,
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
      generator::FactorSumGenerator<Device> factor_sum_gen(
        feature_poses, feature_ids, feature_vals, feature_params, factor_num
      );
      factor_sum.device(d) = factor_sum.generate(factor_sum_gen);

      generator::ParamGradGenerator<Device> param_grad_gen(
          param_grad, feature_poses, feature_ids, feature_vals, feature_params,
          pred_grad, reg_grad, factor_lambda, bias_lambda, factor_sum, factor_num
      );

      param_grad.device(d) = param_grad.constant(0);

      Eigen::Tensor<int64, 2>::Dimensions matrix_1_by_factor_num{{ 1, factor_num + 1}};
      Eigen::array<int64, 2> batch_size_by_1{{ batch_size, 1 }};
      Eigen::array<int, 1> reduce_on_rows{{ 0 }}; 

      scratch.device(d) = scratch.reshape(matrix_1_by_factor_num).broadcast(batch_size_by_1).generate(param_grad_gen).sum(reduce_on_rows);
    }
  };
}

}

// vim: ts=2:sts=2:sw=2
