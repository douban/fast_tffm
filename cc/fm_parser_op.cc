#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <ctime>
#include <cstdio>
#include <fstream>

REGISTER_OP("FmParser")
  .Input("data_strings: string")
  .Output("labels: float32")
  .Output("sizes: int32")
  .Output("feature_ids: int64")
  .Output("feature_vals: float32")
  .Attr("vocab_size: int")
  .Attr("hash_feature_id: bool = false");

#define MAX_FEATURE_ID_LENGTH 100

using namespace tensorflow;

class FmParserOp : public OpKernel {
  public:

    explicit FmParserOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_size", &vocab_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("hash_feature_id", &hash_feature_id_));
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor* data_strings_tensor;
      OP_REQUIRES_OK(ctx, ctx->input("data_strings", &data_strings_tensor));
      auto batch_size = data_strings_tensor->dim_size(0);
      auto data_strings = data_strings_tensor->flat<string>();

      std::vector<float> labels;
      std::vector<int32> sizes;
      std::vector<int64> feature_ids;
      std::vector<float> feature_vals;
      for(int32 i=0; i < batch_size; ++i) {
        const string& data_string = data_strings(i);
        ParseLine(
          ctx, data_string, hash_feature_id_, vocab_size_,
          labels, sizes, feature_ids, feature_vals
        );
      }

      AllocateTensorForVector<float>(ctx, "labels", labels);
      AllocateTensorForVector<int32>(ctx, "sizes", sizes);
      AllocateTensorForVector<int64>(ctx, "feature_ids", feature_ids);
      AllocateTensorForVector<float>(ctx, "feature_vals", feature_vals);
    }


  private:
    int64 vocab_size_;
    bool hash_feature_id_;

    void ParseLine(
      OpKernelContext* ctx, const std::string& line,
      bool& hash_feature_id, int64& vocab_size, std::vector<float>& labels,
      std::vector<int32>& sizes, std::vector<int64>& feature_ids,
      std::vector<float>& feature_vals
    ) {
      const char* p = line.c_str();
      char* nextptr;
      float fv;
      int64 ori_id;
      int32 cnt;
      fv = strtof(p, &nextptr);
      OP_REQUIRES(ctx, p != nextptr,
          errors::InvalidArgument("Label could not be read in example: ", line));
      labels.push_back(fv);
      p = nextptr;

      for (cnt = 0; *p != '\0'; ++cnt) {
        OP_REQUIRES(ctx, *p == ' ',
          errors::InvalidArgument("Invalid format in example: ", line));
        ++ p;
        if (*p == '\0') {
          break;
        }
        if (hash_feature_id) {
          for(nextptr = (char*)p; *nextptr != ' ' || *nextptr != ':' || *nextptr != '\0'; ++nextptr) ;
          ori_id = Hash64(p, nextptr - p) % vocab_size;
        } else {
          ori_id = strtoll(p, &nextptr, 10);
          OP_REQUIRES(ctx, p != nextptr,
            errors::InvalidArgument("Invalid format in example: ", line));
          OP_REQUIRES(ctx, ori_id >= 0 && ori_id < vocab_size,
            errors::InvalidArgument(
              "Invalid feature id. Should be in range [0, vocabulary_size).", line
            )
          )
        }
        p = nextptr;
        if (*p == ':') {
          p += 1;
          fv = strtof(p, &nextptr);
          OP_REQUIRES(ctx, p != nextptr,
            errors::InvalidArgument("Invalid feature value. ", line))
          p = nextptr;
        } else {
          fv = 1;
        }
        feature_ids.push_back(ori_id);
        feature_vals.push_back(fv);
      }
      sizes.push_back(cnt);
    }


    template<typename T>
      void AllocateTensorForVector(OpKernelContext* ctx, const string& name, const std::vector<T>& data) {
        Tensor* tensor;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(name, TensorShape({static_cast<int64>(data.size())}), &tensor));
        auto tensor_data = tensor->flat<T>();
        for (size_t i = 0; i < data.size(); ++i) {
          tensor_data(i) = data[i];
        }
      }
};

REGISTER_KERNEL_BUILDER(Name("FmParser").Device(DEVICE_CPU), FmParserOp);
