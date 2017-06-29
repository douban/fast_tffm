#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <ctime>
#include <cstdio>
#include <fstream>

REGISTER_OP("FmParser")
    .Input("data_string: string")
    .Output("label: float32")
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
    const Tensor* data_string_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("data_string", &data_string_tensor));
    auto data_string = data_string_tensor->scalar<string>()(); 
 
    float label_val;
    std::vector<int64> feature_ids;
    std::vector<float> feature_vals;
    ParseLine(ctx, data_string, hash_feature_id_, vocab_size_, label_val, feature_ids, feature_vals);

    Tensor* label_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("label", TensorShape({}), &label_tensor));
    auto label = label_tensor->scalar<float>();
    label() = label_val;
    

    AllocateTensorForVector<int64>(ctx, "feature_ids", feature_ids);
    AllocateTensorForVector<float>(ctx, "feature_vals", feature_vals); 
  }


 private:
  int64 vocab_size_;
  bool hash_feature_id_;
  
  void ParseLine(OpKernelContext* ctx, std::basic_string<char>& line, bool& hash_feature_id, int64& vocab_size, float& label, std::vector<int64>& feature_ids, std::vector<float>& feature_vals) {
    const char* p = line.c_str();
    float fv;
    int64 ori_id;
    int offset;
    OP_REQUIRES(ctx, sscanf(p, "%f%n", &fv, &offset) == 1,
            errors::InvalidArgument("Label could not be read in example: ", line));
    label = fv;
    p += offset;

    size_t read_size;
    char ori_id_str[MAX_FEATURE_ID_LENGTH];
    char* err;
    while (true) {
      if (sscanf(p, " %[^: ]%n", ori_id_str, &offset) != 1) break;
      if (hash_feature_id) {
        ori_id = Hash64(ori_id_str, strlen(ori_id_str));
      } else {
        ori_id = strtol(ori_id_str, &err, 10);
        OP_REQUIRES(ctx, *err == 0, errors::InvalidArgument("Invalid feature id ", ori_id_str, ". Set hash_feature_id = True?"))
      }
      ori_id = labs(ori_id % vocab_size);
      p += offset;
      if (*p == ':') {
        OP_REQUIRES(ctx, sscanf(p, ":%f%n", &fv, &offset) == 1, errors::InvalidArgument("Invalid feature value: ", ori_id_str))
        p += offset;
      } else {
        fv = 1;
      }
      feature_ids.push_back(ori_id);
      feature_vals.push_back(fv);
    }
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
