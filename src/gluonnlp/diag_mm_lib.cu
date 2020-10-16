#include <iostream>
#include <cassert>
#include <string>
#include "lib_api.h"

const int NumThreadPerBlock = 256;
const int MaxGridNum = 65535;

using namespace mxnet::ext;

/*

Phase 1: scores = F.sw_atten_score(query, key, dilation, w=w, symmetric=symmetric)

--- Input:
query's shape  (batch_size, seq_length, num_heads, num_hidden)
key's shape    (batch_size, seq_length, num_heads, num_hidden)
dilation's shape  (num_heads,)
w is the one-sided length of the sliding window
--- Output:
score's shape  (batch_size, seq_length, num_heads, w + w + 1) if symmetric else
               (batch_size, seq_length, num_heads, w + 1)

For example, when seq_len = 6, w = 2, symmetric = True, the score will be

  0 0 x x x
  0 x x x x
  x x x x x
  x x x x x
  x x x x 0
  x x x 0 0

when seq_len = 6, w = 2, symmetric = False, the score will be

  0 0 x
  0 x x
  x x x
  x x x
  x x x
  x x x


Phase 2: context_vec = F.sw_atten_context(score, value, dilation, w=w, symmetric=symmetric)

--- Input:
value's shape  (batch_size, seq_length, num_heads, num_hidden)
--- Output:
context_vec's shape  (batch_size, seq_length, num_heads, num_hidden)

*/



__global__ void diag_mm_gpu(float *out, float *lhs, float *rhs, int64_t *dilation,
                            int64_t batch_size, int64_t seq_length, int64_t num_heads,
                            int64_t out_last_dim, int64_t lhs_last_dim, int64_t w,
                            int64_t w_right, int64_t N, bool diagonal_lhs, bool transpose_lhs) {
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
    if (tid >= N) return;
    out[tid] = 0;
    int64_t stride = seq_length * num_heads * out_last_dim;
    int64_t idx_0 = tid / stride; // batch idx
    int64_t tmp = tid % stride;
    stride = num_heads * out_last_dim;
    int64_t idx_1 = tmp / stride; // sequence idx
    tmp = tmp % stride;
    int64_t idx_2 = tmp / out_last_dim; // head idx
    int64_t idx_3 = tmp % out_last_dim; // window idx or hidden feature idx

    if (!diagonal_lhs) {
      int64_t tmp_idx = idx_1 + dilation[idx_2] * (idx_3 - w);
      if (tmp_idx >= seq_length || tmp_idx < 0) continue;
      for (int64_t i = 0; i < lhs_last_dim; i++) {
        int64_t lhs_idx = idx_0 * (seq_length * num_heads * lhs_last_dim) + \
          idx_1 * (num_heads * lhs_last_dim) + idx_2 * lhs_last_dim + i;
        int64_t rhs_idx = idx_0 * (seq_length * num_heads * lhs_last_dim) + \
          tmp_idx * (num_heads * lhs_last_dim) + idx_2 * lhs_last_dim + i;
        out[tid] += lhs[lhs_idx] * rhs[rhs_idx];
      }
    } else {
      if (!transpose_lhs) {
        for (int64_t i = 0; i < lhs_last_dim; i++) {
          int64_t tmp_idx = idx_1 + dilation[idx_2] * (i - w);
          if (tmp_idx >= seq_length || tmp_idx < 0) continue;
          int64_t lhs_idx = idx_0 * (seq_length * num_heads * lhs_last_dim) + \
            idx_1 * (num_heads * lhs_last_dim) + idx_2 * lhs_last_dim + i;
          int64_t rhs_idx = idx_0 * (seq_length * num_heads * out_last_dim) + \
            tmp_idx * (num_heads * out_last_dim) + idx_2 * out_last_dim + idx_3;
          out[tid] += lhs[lhs_idx] * rhs[rhs_idx];
        }
      } else {
        for (int64_t i = 0; i < lhs_last_dim; i++) {
          int64_t tmp_idx = idx_1 + dilation[idx_2] * (i - w_right);
          if (tmp_idx >= seq_length || tmp_idx < 0) continue;
          int64_t lhs_idx = idx_0 * (seq_length * num_heads * lhs_last_dim) + \
            tmp_idx * (num_heads * lhs_last_dim) + idx_2 * lhs_last_dim + ((w_right + w) - i);
          int64_t rhs_idx = idx_0 * (seq_length * num_heads * out_last_dim) + \
            tmp_idx * (num_heads * out_last_dim) + idx_2 * out_last_dim + idx_3;
          out[tid] += lhs[lhs_idx] * rhs[rhs_idx];
        }
      }
    }
  }
}


void diag_mm_impl(MXTensor& out, MXTensor& lhs, MXTensor& rhs, MXTensor& dilation,
                  bool diagonal_lhs, bool transpose_lhs, int64_t w,
                  int64_t w_right, const OpResource& res) {
  float* lhs_data = lhs.data<float>();
  float* rhs_data = rhs.data<float>();
  int64_t* dilation_data = dilation.data<int64_t>();
  float* out_data = out.data<float>();

  mx_stream_t cuda_stream = res.get_cuda_stream();

  int64_t batch_size = lhs.shape[0];
  int64_t seq_length = lhs.shape[1];
  int64_t num_heads = lhs.shape[2];
  int64_t lhs_last_dim = lhs.shape[3];
  int64_t num_hidden = rhs.shape[3];
  int64_t out_last_dim = out.shape[3];
  int N = out.size();
  int ngrid = std::min(MaxGridNum, (N + NumThreadPerBlock - 1) / NumThreadPerBlock);
  diag_mm_gpu<<<ngrid, NumThreadPerBlock, 0, cuda_stream>>>(out_data, lhs_data,
    rhs_data, dilation_data, batch_size, seq_length, num_heads, out_last_dim, lhs_last_dim,
    w, w_right, N, diagonal_lhs, transpose_lhs);
}


void parse_str_attrs(const std::unordered_map<std::string, std::string>& attrs,
                     bool* symmetric, unsigned int* w) {
  auto it = attrs.find("w");
  assert(it != attrs.end());
  *w = std::stoi(it->second);

  *symmetric = true;
  it = attrs.find("symmetric");
  if (it != attrs.end() && it->second == "False") {
    *symmetric = false;
  }

}


MXReturnValue score_forwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                               std::vector<MXTensor>* inputs,
                               std::vector<MXTensor>* outputs,
                               const OpResource& res) {
  bool symmetric = true;
  unsigned int w = 0;
  parse_str_attrs(attrs, &symmetric, &w);
  int64_t w_right = symmetric ? w : 0;
  assert(inputs->at(2).dtype == kInt64);
  // score = matmul(query, key.T)
  diag_mm_impl(outputs->at(0), inputs->at(0), inputs->at(1), inputs->at(2),
    false, false, w, w_right, res);

  return MX_SUCCESS;
}


MXReturnValue score_backwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                                std::vector<MXTensor>* inputs,
                                std::vector<MXTensor>* outputs,
                                const OpResource& res) {
  bool symmetric = true;
  unsigned int w = 0;
  parse_str_attrs(attrs, &symmetric, &w);
  int64_t w_right = symmetric ? w : 0;
  // dilation
  assert(inputs->at(3).dtype == kInt64);
  // grad_query = matmul(grad_score, key)
  diag_mm_impl(outputs->at(0), inputs->at(0), inputs->at(2), inputs->at(3),
               true, false, w, w_right, res);
  // grad_key = matmul(grad_score.T, query)
  diag_mm_impl(outputs->at(1), inputs->at(0), inputs->at(1), inputs->at(3),
               true, true, w, w_right, res);

  return MX_SUCCESS;
}


MXReturnValue context_forwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                                 std::vector<MXTensor>* inputs,
                                 std::vector<MXTensor>* outputs,
                                 const OpResource& res) {
  bool symmetric = true;
  unsigned int w = 0;
  parse_str_attrs(attrs, &symmetric, &w);
  int64_t w_right = symmetric ? w : 0;

  assert(inputs->at(2).dtype == kInt64);
  // context_vec = matmul(score, value)
  diag_mm_impl(outputs->at(0), inputs->at(0), inputs->at(1), inputs->at(2),
    true, false, w, w_right, res);

  return MX_SUCCESS;
}


MXReturnValue context_backwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                                  std::vector<MXTensor>* inputs,
                                  std::vector<MXTensor>* outputs,
                                  const OpResource& res) {
  bool symmetric = true;
  unsigned int w = 0;
  parse_str_attrs(attrs, &symmetric, &w);
  int64_t w_right = symmetric ? w : 0;
  // dilation
  assert(inputs->at(3).dtype == kInt64);
  // grad_score = matmul(grad_context, value.T)
  diag_mm_impl(outputs->at(0), inputs->at(0), inputs->at(2), inputs->at(3),
               false, false, w, w_right, res);
  // grad_value = matmul(score.T, grad_context)
  diag_mm_impl(outputs->at(1), inputs->at(1), inputs->at(0), inputs->at(3),
               true, true, w, w_right, res);

  return MX_SUCCESS;
}


MXReturnValue parseAttrs(const std::unordered_map<std::string, std::string>& attrs,
                         int* num_in, int* num_out) {
  *num_in = 3;
  *num_out = 1;
  return MX_SUCCESS;
}


MXReturnValue inferType(const std::unordered_map<std::string, std::string>& attrs,
                        std::vector<int>* intypes,
                        std::vector<int>* outtypes) {
  assert(intypes->at(0) == kFloat32);
  outtypes->at(0) = intypes->at(0);
  return MX_SUCCESS;
}


MXReturnValue score_inferShape(const std::unordered_map<std::string, std::string>& attrs,
                               std::vector<std::vector<unsigned int>>* inshapes,
                               std::vector<std::vector<unsigned int>>* outshapes) {
  unsigned int batch_size = inshapes->at(0)[0];
  unsigned int seq_length = inshapes->at(0)[1];
  unsigned int num_heads = inshapes->at(0)[2];
  unsigned int lhs_last_dim = inshapes->at(0)[3];
  unsigned int num_hidden = inshapes->at(1)[3];
  assert(lhs_last_dim == num_hidden);
  assert(inshapes->at(2)[0] == num_heads);

  bool symmetric = true;
  unsigned int w = 0;
  parse_str_attrs(attrs, &symmetric, &w);
  unsigned int w_len = symmetric ? (w + w + 1) : (w + 1);

  outshapes->at(0) = {batch_size, seq_length, num_heads, w_len};

  return MX_SUCCESS;
}


MXReturnValue context_inferShape(const std::unordered_map<std::string, std::string>& attrs,
                                 std::vector<std::vector<unsigned int>>* inshapes,
                                 std::vector<std::vector<unsigned int>>* outshapes) {
  unsigned int batch_size = inshapes->at(0)[0];
  unsigned int seq_length = inshapes->at(0)[1];
  unsigned int num_heads = inshapes->at(0)[2];
  unsigned int lhs_last_dim = inshapes->at(0)[3];
  unsigned int num_hidden = inshapes->at(1)[3];
  assert(inshapes->at(2)[0] == num_heads);

  bool symmetric = true;
  unsigned int w = 0;
  parse_str_attrs(attrs, &symmetric, &w);
  unsigned int w_len = symmetric ? (2 * w + 1) : (w + 1);
  assert(lhs_last_dim == w_len);

  outshapes->at(0) = {batch_size, seq_length, num_heads, num_hidden};

  return MX_SUCCESS;
}



REGISTER_OP(sw_atten_score)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(score_inferShape)
.setForward(score_forwardGPU, "gpu")
.setBackward(score_backwardGPU, "gpu");


REGISTER_OP(sw_atten_context)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(context_inferShape)
.setForward(context_forwardGPU, "gpu")
.setBackward(context_backwardGPU, "gpu");


/*

mask = F.mask_like(score, dilation, w=w, symmetric=symmetric)

mask's shape : score.shape

For example, when seq_len = 6, w = 2, symmetric = True, the mask will be

  0 0 1 1 1
  0 x 1 1 1
  1 1 1 1 1
  1 1 1 1 1
  1 1 1 1 0
  1 1 1 0 0

when seq_len = 6, w = 2, symmetric = False, the mask will be

  0 0 1
  0 1 1
  1 1 1
  1 1 1
  1 1 1
  1 1 1

*/

/*
__global__ void mask_fwd_gpu(float *out, int64_t *dilation, int64_t *val_length, bool symmetric,
                             int64_t w, int64_t seq_length, int64_t num_heads, int64_t N) {
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
    if (tid >= N) return;
    out[tid] = 1;
    int64_t w_len = symmetric ? (w + w + 1) : (w + 1);
    int64_t idx_0 = tid / (seq_length * num_heads * w_len); // batch idx
    int64_t tmp = tid % (seq_length * num_heads * w_len);
    int64_t idx_1 = tmp / (num_heads * w_len); // sequence idx
    tmp = tmp % (num_heads * w_len);
    int64_t idx_2 = tmp / w_len; // head idx
    int64_t idx_3 = tmp % w_len; // win idx

    bool is_zero = idx_3 < (w - idx_1/dilation[idx_2]) || idx_1 >= val_length[idx_0] \
      || (symmetric && (w_len-idx_3-1) < (w - (val_length[idx_0]-idx_1-1)/dilation[idx_2]));
    if (is_zero) out[tid] = 0;
  }
}
*/


__global__ void mask_fwd_gpu(float *out, int64_t *dilation, int64_t *val_length, bool symmetric,
                             int64_t w, int64_t seq_length, int64_t num_heads, int64_t N) {
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
    if (tid >= N) return;
    out[tid] = 1;
    int64_t w_len = symmetric ? (w + w + 1) : (w + 1);
    int64_t idx_0 = tid / (seq_length * num_heads * (w_len+1)); // batch idx
    int64_t tmp = tid % (seq_length * num_heads * (w_len+1));
    int64_t idx_1 = tmp / (num_heads * (w_len+1)); // sequence idx
    tmp = tmp % (num_heads * (w_len+1));
    int64_t idx_2 = tmp / (w_len+1); // head idx
    int64_t idx_3 = tmp % (w_len+1); // win idx

    if (idx_3 == w_len) continue;

    bool is_zero = idx_3 < (w - idx_1/dilation[idx_2]) || idx_1 >= val_length[idx_0] \
      || (symmetric && (w_len-idx_3-1) < (w - (val_length[idx_0]-idx_1-1)/dilation[idx_2]));
    is_zero = is_zero || idx_1 == (w - idx_3) * dilation[idx_2];
    if (is_zero) out[tid] = 0;
  }
}


__global__ void mask_bwd_gpu(float *out, int64_t N) {
  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
    if (tid >= N) return;
    out[tid] = 0;
  }
}


MXReturnValue mask_forwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                              std::vector<MXTensor>* inputs,
                              std::vector<MXTensor>* outputs,
                              const OpResource& res) {
  bool symmetric = true;
  unsigned int w = 0;
  parse_str_attrs(attrs, &symmetric, &w);
  assert(inputs->at(1).dtype == kInt64);
  assert(inputs->at(2).dtype == kInt64);

  float* out = outputs->at(0).data<float>();
  int64_t* dilation = inputs->at(1).data<int64_t>();
  int64_t* val_length = inputs->at(2).data<int64_t>();

  mx_stream_t cuda_stream = res.get_cuda_stream();

  int64_t seq_length = inputs->at(0).shape[1];
  int64_t num_heads = inputs->at(0).shape[2];
  int N = outputs->at(0).size();
  int ngrid = std::min(MaxGridNum, (N + NumThreadPerBlock - 1) / NumThreadPerBlock);
  mask_fwd_gpu<<<ngrid, NumThreadPerBlock, 0, cuda_stream>>>(out, dilation, val_length,
    symmetric, w, seq_length, num_heads, N);

  return MX_SUCCESS;
}


MXReturnValue mask_backwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                               std::vector<MXTensor>* inputs,
                               std::vector<MXTensor>* outputs,
                               const OpResource& res) {
  float* out = outputs->at(0).data<float>();
  mx_stream_t cuda_stream = res.get_cuda_stream();
  int N = outputs->at(0).size();
  int ngrid = std::min(MaxGridNum, (N + NumThreadPerBlock - 1) / NumThreadPerBlock);
  mask_bwd_gpu<<<ngrid, NumThreadPerBlock, 0, cuda_stream>>>(out, N);
  return MX_SUCCESS;
}


MXReturnValue mask_inferShape(const std::unordered_map<std::string, std::string>& attrs,
                              std::vector<std::vector<unsigned int>>* inshapes,
                              std::vector<std::vector<unsigned int>>* outshapes) {
  //outshapes->at(0) = inshapes->at(0);
  outshapes->at(0) = {inshapes->at(0)[0], inshapes->at(0)[1],
                      inshapes->at(0)[2], 1 + inshapes->at(0)[3]};
  return MX_SUCCESS;
}


MXReturnValue mask_parseAttrs(const std::unordered_map<std::string, std::string>& attrs,
                              int* num_in, int* num_out) {
  *num_in = 3;
  *num_out = 1;
  return MX_SUCCESS;
}


REGISTER_OP(mask_like)
.setParseAttrs(mask_parseAttrs)
.setInferType(inferType)
.setInferShape(mask_inferShape)
.setForward(mask_forwardGPU, "gpu")
.setBackward(mask_backwardGPU, "gpu");



MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}



