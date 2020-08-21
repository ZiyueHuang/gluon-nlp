#include <iostream>
#include <cassert>
#include <string>
#include "lib_api.h"

#define NumThreadPerBlock 256 // mxnet recommended cuda thread number per block


/*

Phase 1: scores = F.my_diag_mm(query, key, dilation, w=w, diagonal_lhs=False,
                               transpose_lhs=False, symmetric=symmetric)
--- Input:
query's shape  (batch_size, seq_length, num_heads, num_hidden)
key's shape    (batch_size, seq_length, num_heads, num_hidden)
dilation's shape  (num_heads,)
w is the one-sided length of the sliding window
--- Output:
score's shape  (batch_size, seq_length, num_heads, w + w + 1) if symmetric else
               (batch_size, seq_length, num_heads, w + 1)



Phase 2: context_vec = F.my_diag_mm(score, value, dilation, w=w, diagonal_lhs=True,
                                    transpose_lhs=False, symmetric=symmetric)
--- Input:
value's shape  (batch_size, seq_length, num_heads, num_hidden)
--- Output:
context_vec's shape  (batch_size, seq_length, num_heads, num_hidden)

*/



__global__ void diag_mm_gpu(float *out, float *lhs, float *rhs, int64_t *dilation,
                            int64_t batch_size, int64_t seq_length, int64_t num_heads,
                            int64_t out_last_dim, int64_t lhs_last_dim, int64_t w,
                            int64_t w_right, int64_t N, bool diagonal_lhs, bool transpose_lhs) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
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
    if (tmp_idx >= seq_length || tmp_idx < 0) return;
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
  int64_t N = out.size();
  int num_block = (N + NumThreadPerBlock - 1) / NumThreadPerBlock;
  diag_mm_gpu<<<num_block, NumThreadPerBlock, 0, cuda_stream>>>(out_data, lhs_data,
    rhs_data, dilation_data, batch_size, seq_length, num_heads, out_last_dim, lhs_last_dim,
    w, w_right, N, diagonal_lhs, transpose_lhs);
}

void parse_str_attrs(const std::unordered_map<std::string, std::string>& attrs,
                     bool* diagonal_lhs, bool* transpose_lhs, bool* symmetric,
                     unsigned int* w) {
  *diagonal_lhs = false;
  *transpose_lhs = false;
  *symmetric = true;
  auto it = attrs.find("diagonal_lhs");
  if (it != attrs.end() && it->second == "True") {
    *diagonal_lhs = true;
  }
  it = attrs.find("transpose_lhs");
  if (it != attrs.end() && it->second == "True") {
    *transpose_lhs = true;
  }
  it = attrs.find("w");
  assert(it != attrs.end());
  *w = std::stoi(it->second);
  it = attrs.find("symmetric");
  if (it != attrs.end() && it->second == "False") {
    *symmetric = false;
  }

}

MXReturnValue diag_mm_forwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                                 std::vector<MXTensor>* inputs,
                                 std::vector<MXTensor>* outputs,
                                 const OpResource& res) {
  bool diagonal_lhs = false;
  bool transpose_lhs = false;
  bool symmetric = true;
  unsigned int w = 0;
  parse_str_attrs(attrs, &diagonal_lhs, &transpose_lhs, &symmetric, &w);
  int64_t w_right = symmetric ? w : 0;

  assert(inputs->at(2).dtype == kInt64);

  diag_mm_impl(outputs->at(0), inputs->at(0), inputs->at(1), inputs->at(2),
    diagonal_lhs, transpose_lhs, w, w_right, res);

  return MX_SUCCESS;
}

MXReturnValue diag_mm_backwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                                  std::vector<MXTensor>* inputs,
                                  std::vector<MXTensor>* outputs,
                                  const OpResource& res) {
  bool diagonal_lhs = false;
  bool transpose_lhs = false;
  bool symmetric = true;
  unsigned int w = 0;
  parse_str_attrs(attrs, &diagonal_lhs, &transpose_lhs, &symmetric, &w);
  int64_t w_right = symmetric ? w : 0;

  assert(inputs->at(3).dtype == kInt64);

  diag_mm_impl(outputs->at(0), inputs->at(0), inputs->at(2), inputs->at(3),
               !diagonal_lhs, false, w, w_right, res);

  if (diagonal_lhs) {
    diag_mm_impl(outputs->at(1), inputs->at(1), inputs->at(0), inputs->at(3),
                 true, true, w, w_right, res);
  } else {
    diag_mm_impl(outputs->at(1), inputs->at(0), inputs->at(1), inputs->at(3),
                 true, true, w, w_right, res);
  }

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
  outtypes->at(0) = intypes->at(0);
  return MX_SUCCESS;
}

MXReturnValue inferShape(const std::unordered_map<std::string, std::string>& attrs,
                         std::vector<std::vector<unsigned int>>* inshapes,
                         std::vector<std::vector<unsigned int>>* outshapes) {
  unsigned int batch_size = inshapes->at(0)[0];
  unsigned int seq_length = inshapes->at(0)[1];
  unsigned int num_heads = inshapes->at(0)[2];
  unsigned int lhs_last_dim = inshapes->at(0)[3];
  unsigned int num_hidden = inshapes->at(1)[3];
  assert(inshapes->at(2)[0] == num_heads);

  bool diagonal_lhs = false;
  bool transpose_lhs = false;
  bool symmetric = true;
  unsigned int w = 0;
  parse_str_attrs(attrs, &diagonal_lhs, &transpose_lhs, &symmetric, &w);
  unsigned int w_len = symmetric ? (2 * w + 1) : (w + 1);

  if (diagonal_lhs) {
    assert(lhs_last_dim == w_len);
    outshapes->at(0) = {batch_size, seq_length, num_heads, num_hidden};
  } else {
    assert(lhs_last_dim == num_hidden);
    assert(!transpose_lhs);
    outshapes->at(0) = {batch_size, seq_length, num_heads, w_len};
  }

  return MX_SUCCESS;
}


REGISTER_OP(my_diag_mm)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setForward(diag_mm_forwardGPU, "gpu")
.setBackward(diag_mm_backwardGPU, "gpu");


MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}



