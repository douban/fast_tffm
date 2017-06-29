#include "tensorflow/core/framework/tensor.h"

namespace fm {

namespace generator {
  using namespace tensorflow;

  template <typename Device>
  struct FeatureRankGenerator {
    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
    FeatureRankGenerator(
      const typename TTypes<int32>::ConstTensor& feature_poses_,
      const typename TTypes<int32>::ConstTensor& feature_ids_,
      const typename TTypes<float>::ConstTensor& feature_vals_,
      const typename TTypes<float>::ConstTensor& feature_params_,
      const int64 factor_num_
    ) : feature_poses(feature_poses_), feature_ids(feature_ids_),
    feature_vals(feature_vals_), feature_params(feature_params_),
    factor_num(factor_num_)
    {}

    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
    float operator()(const Eigen::array<Eigen::DenseIndex, 3>& coeff) const {
      auto p = coeff[0];
      auto i = coeff[1];
      auto k = coeff[2];
      float r = 0;
      for (size_t j = feature_poses(i); j < feature_poses(i + 1); ++j) {
        int32 fid = feature_ids(j);
        float fval = feature_vals(j);
        size_t param_offset = fid * (factor_num + 1);
        float t = feature_params(param_offset + k + 1);
        r += powf(fval * t, p + 1);
      }
      return r;
    }

  private:
    const typename TTypes<int32>::ConstTensor feature_poses;
    const typename TTypes<int32>::ConstTensor feature_ids;
    const typename TTypes<float>::ConstTensor feature_vals;
    const typename TTypes<float>::ConstTensor feature_params;
    const int64 factor_num;
  };

  template <typename Device>
  struct BiasGenerator {
    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
    BiasGenerator(
      const typename TTypes<int32>::ConstTensor& feature_poses_,
      const typename TTypes<int32>::ConstTensor& feature_ids_,
      const typename TTypes<float>::ConstTensor& feature_vals_,
      const typename TTypes<float>::ConstTensor& feature_params_,
      const int64 factor_num_
    ) : feature_poses(feature_poses_), feature_ids(feature_ids_),
    feature_vals(feature_vals_), feature_params(feature_params_),
    factor_num(factor_num_)
    {}

    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
    float operator()(const Eigen::array<Eigen::DenseIndex, 2>& coeff) const {
      auto p = coeff[0];
      auto i = coeff[1];
      float r = 0;
      for (size_t j = feature_poses(i); j < feature_poses(i + 1); ++j) {
        int32 fid = feature_ids(j);
        float fval = feature_vals(j);
        size_t param_offset = fid * (factor_num + 1);
        float bias = feature_params(param_offset);
        switch(p) {
          case 0:
            r += fval * bias;
            break;
          case 1:
            r += bias * bias;
            break;
          default:
            for (size_t k = 0; k < factor_num; ++k) {
              auto t = feature_params(param_offset + k + 1);
              r += t * t;
            }
        }

      }
      return r;
    }

  private:
    const typename TTypes<int32>::ConstTensor feature_poses;
    const typename TTypes<int32>::ConstTensor feature_ids;
    const typename TTypes<float>::ConstTensor feature_vals;
    const typename TTypes<float>::ConstTensor feature_params;
    const int64 factor_num;
  };

}

namespace functor {
  using namespace tensorflow;

  template <typename Device>
  struct FmScorer {
    EIGEN_ALWAYS_INLINE
    void operator() (
      const Device& d,
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
    ) const {
      using generator::FeatureRankGenerator;
      using generator::BiasGenerator;

      BiasGenerator<Device> bias_gen(
        feature_poses, feature_ids, feature_vals, feature_params, factor_num
      );
      biases.device(d) = biases.generate(bias_gen);

      FeatureRankGenerator<Device> feature_rank_gen(
        feature_poses, feature_ids, feature_vals, feature_params, factor_num
      );
      feature_ranks.device(d) = feature_ranks.generate(feature_rank_gen);

      auto rank_1 = feature_ranks.chip(0, 0);
      auto rank_2 = feature_ranks.chip(1, 0);
      pred_score.device(d) = biases.chip(0, 0) \
                             - 0.5 * rank_2.sum(Eigen::array<int, 1>({1})) \
                             + (0.5 * rank_1 * rank_1).sum(Eigen::array<int, 1>({1}));

      reg_score.device(d) = 0.5 * factor_lambda * biases.chip(2, 0).sum() + 0.5 * bias_lambda * biases.chip(1, 0).sum();
    };
  };
}
}

// vim: ts=2:sts=2:sw=2
