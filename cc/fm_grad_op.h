#include "tensorflow/core/framework/tensor.h"

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
    ) const;
  };
}

}

// vim: ts=2:sts=2:sw=2
