#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
namespace {

constexpr int64_t kLiveSlot = -2;

// cuMemcpy3DBatchAsync was added after the CUDA 12.4 toolkit used to build this
// extension. Resolve its stable CUDA 12.8 driver ABI at runtime so newer
// drivers can use it without making the extension require a newer toolkit.
constexpr int kMemcpyOperandPointer = 0x1;
constexpr int kMemcpySrcAccessOrderStream = 0x1;
constexpr unsigned int kMemcpyPreferOverlapWithCompute = 0x1;

struct KvMemcpy3DPointerOperand {
  CUdeviceptr ptr;
  size_t row_length;
  size_t layer_height;
  CUmemLocation location_hint;
};

struct KvMemcpy3DArrayOperand {
  CUarray array;
  size_t offset_x;
  size_t offset_y;
  size_t offset_z;
};

struct KvMemcpy3DOperand {
  int type;
  union {
    KvMemcpy3DPointerOperand ptr;
    KvMemcpy3DArrayOperand array;
  } op;
};

struct KvMemcpy3DExtent {
  size_t width;
  size_t height;
  size_t depth;
};

struct KvMemcpy3DBatchOp {
  KvMemcpy3DOperand src;
  KvMemcpy3DOperand dst;
  KvMemcpy3DExtent extent;
  int src_access_order;
  unsigned int flags;
};

using KvMemcpy3DBatchAsyncFn = CUresult(CUDAAPI *)(
    size_t,
    KvMemcpy3DBatchOp*,
    size_t*,
    unsigned long long,
    CUstream);

static_assert(sizeof(KvMemcpy3DOperand) == 40, "Unexpected CUDA batch operand ABI");
static_assert(sizeof(KvMemcpy3DBatchOp) == 112, "Unexpected CUDA batch operation ABI");

KvMemcpy3DBatchAsyncFn resolve_memcpy_3d_batch_async() {
  static std::once_flag once;
  static KvMemcpy3DBatchAsyncFn fn = nullptr;
  std::call_once(once, [] {
    void* handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      fn = reinterpret_cast<KvMemcpy3DBatchAsyncFn>(
          dlsym(handle, "cuMemcpy3DBatchAsync"));
    }
  });
  return fn;
}

KvMemcpy3DOperand make_batch_pointer_operand(
    const void* ptr,
    size_t row_length,
    size_t layer_height,
    CUmemLocationType location_type,
    int location_id) {
  KvMemcpy3DOperand operand{};
  operand.type = kMemcpyOperandPointer;
  operand.op.ptr.ptr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
  operand.op.ptr.row_length = row_length;
  operand.op.ptr.layer_height = layer_height;
  operand.op.ptr.location_hint.type = location_type;
  operand.op.ptr.location_hint.id = location_id;
  return operand;
}

void append_batch_2d_copy(
    std::vector<KvMemcpy3DBatchOp>& ops,
    const void* src,
    size_t src_pitch,
    void* dst,
    size_t dst_pitch,
    size_t width_bytes,
    size_t height,
    CUmemLocationType src_location_type,
    int src_location_id,
    CUmemLocationType dst_location_type,
    int dst_location_id,
    unsigned int flags = kMemcpyPreferOverlapWithCompute) {
  KvMemcpy3DBatchOp op{};
  op.src = make_batch_pointer_operand(
      src, src_pitch, height, src_location_type, src_location_id);
  op.dst = make_batch_pointer_operand(
      dst, dst_pitch, height, dst_location_type, dst_location_id);
  op.extent = {width_bytes, height, 1};
  op.src_access_order = kMemcpySrcAccessOrderStream;
  op.flags = flags;
  ops.push_back(op);
}

void ensure_current_cuda_context_for_driver_batch(int device, const char* label) {
  static thread_local int initialized_device = -1;
  if (initialized_device == device) {
    return;
  }
  cudaError_t runtime_result = cudaSetDevice(device);
  TORCH_CHECK(runtime_result == cudaSuccess, "cudaSetDevice before ", label, " failed: ",
              cudaGetErrorString(runtime_result));
  runtime_result = cudaFree(nullptr);
  TORCH_CHECK(runtime_result == cudaSuccess, "CUDA context initialization before ", label, " failed: ",
              cudaGetErrorString(runtime_result));
  initialized_device = device;
}

bool cpu_kv_is_head_major(const torch::Tensor& cpu_k, int64_t num_heads) {
  return cpu_k.dim() == 4 && cpu_k.size(1) == num_heads && cpu_k.size(2) != num_heads;
}

int64_t infer_gpu_slot_head_count(
    const torch::Tensor& gpu_k,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  TORCH_CHECK(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");
  const int64_t tokens_per_head = frame_seqlen * gpu_cache_frames;
  TORCH_CHECK(tokens_per_head > 0, "invalid GPU cache token shape");
  TORCH_CHECK(gpu_k.dim() == 4 && gpu_k.size(2) == 1, "gpu KV must be [B,Tslot,1,D]");
  TORCH_CHECK(gpu_k.size(1) % tokens_per_head == 0, "gpu KV slot token dimension mismatch");
  return gpu_k.size(1) / tokens_per_head;
}

bool kv_sparse_tiled_materialize_enabled() {
  const char* value = std::getenv("HORIZONKV_TILED_MATERIALIZE");
  if (value == nullptr) {
    return false;
  }
  return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0 ||
         std::strcmp(value, "on") == 0;
}

template <typename scalar_t>
__global__ void kv_sparse_frame_materialize_kernel(
    const scalar_t* __restrict__ gpu_k,
    const scalar_t* __restrict__ gpu_v,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    const int64_t* __restrict__ head_indices,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ slot_indices,
    const scalar_t* __restrict__ live_k,
    const scalar_t* __restrict__ live_v,
    int64_t B,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t head_count,
    int64_t n_selected,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t gpu_cache_tokens,
    int64_t live_len,
    int64_t live_start_index,
    bool has_live,
    int64_t total_elements) {
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total_elements;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t t = linear;
    const int64_t d = t % head_dim;
    t /= head_dim;
    const int64_t token_in_frame = t % frame_seqlen;
    t /= frame_seqlen;
    const int64_t col = t % n_selected;
    t /= n_selected;
    const int64_t row = t % head_count;
    const int64_t b = t / head_count;

    const int64_t pair_idx = row * n_selected + col;
    const int64_t head_id = head_indices[row];
    const int64_t frame_id = frame_indices[pair_idx];
    const int64_t out_token = col * frame_seqlen + token_in_frame;
    const int64_t out_offset =
        ((b * head_count + row) * (n_selected * frame_seqlen) + out_token) * head_dim + d;

    scalar_t k_value = scalar_t(0);
    scalar_t v_value = scalar_t(0);

    if (head_id >= 0 && head_id < num_heads && frame_id >= 0) {
      const int64_t frame_start = frame_id * frame_seqlen;
      if (frame_start < total_tokens) {
        const int64_t remaining = total_tokens - frame_start;
        const int64_t frame_len = frame_seqlen < remaining ? frame_seqlen : remaining;
        if (frame_len > 0) {
          const int64_t src_in_frame =
              token_in_frame < frame_len ? token_in_frame : frame_len - 1;
          const int64_t slot = slot_indices[pair_idx];
          if (slot == kLiveSlot && has_live) {
            const int64_t live_pos = frame_start - live_start_index + src_in_frame;
            if (live_pos >= 0 && live_pos < live_len) {
              const int64_t live_offset =
                  ((b * live_len + live_pos) * num_heads + head_id) * head_dim + d;
              k_value = live_k[live_offset];
              v_value = live_v[live_offset];
            }
          } else if (slot >= 0) {
            const int64_t slot_token =
                (head_id * gpu_cache_frames + slot) * frame_seqlen + src_in_frame;
            const int64_t gpu_offset =
                ((b * gpu_cache_tokens + slot_token) * head_dim) + d;
            k_value = gpu_k[gpu_offset];
            v_value = gpu_v[gpu_offset];
          }
        }
      }
    }

    k_out[out_offset] = k_value;
    v_out[out_offset] = v_value;
  }
}

template <typename scalar_t>
__global__ void kv_sparse_frame_materialize_tiled_kernel(
    const scalar_t* __restrict__ gpu_k,
    const scalar_t* __restrict__ gpu_v,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    const int64_t* __restrict__ head_indices,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ slot_indices,
    const scalar_t* __restrict__ live_k,
    const scalar_t* __restrict__ live_v,
    int64_t B,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t head_count,
    int64_t n_selected,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t gpu_cache_tokens,
    int64_t live_len,
    int64_t live_start_index,
    bool has_live,
    int64_t elems_per_segment) {
  const int64_t pair_linear = static_cast<int64_t>(blockIdx.x);
  int64_t pair = pair_linear;
  const int64_t col = pair % n_selected;
  pair /= n_selected;
  const int64_t row = pair % head_count;
  const int64_t b = pair / head_count;
  const int64_t elem = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (elem >= elems_per_segment || b >= B) {
    return;
  }

  const int64_t token_in_frame = elem / head_dim;
  const int64_t d = elem - token_in_frame * head_dim;
  const int64_t pair_idx = row * n_selected + col;
  const int64_t head_id = head_indices[row];
  const int64_t frame_id = frame_indices[pair_idx];
  const int64_t out_base =
      ((b * head_count + row) * (n_selected * frame_seqlen) + col * frame_seqlen) * head_dim;
  const int64_t out_offset = out_base + elem;

  scalar_t k_value = scalar_t(0);
  scalar_t v_value = scalar_t(0);
  if (head_id >= 0 && head_id < num_heads && frame_id >= 0) {
    const int64_t frame_start = frame_id * frame_seqlen;
    if (frame_start < total_tokens) {
      const int64_t remaining = total_tokens - frame_start;
      const int64_t frame_len = frame_seqlen < remaining ? frame_seqlen : remaining;
      if (frame_len > 0) {
        const int64_t src_in_frame = token_in_frame < frame_len ? token_in_frame : frame_len - 1;
        const int64_t slot = slot_indices[pair_idx];
        if (slot == kLiveSlot && has_live) {
          const int64_t live_pos = frame_start - live_start_index + src_in_frame;
          if (live_pos >= 0 && live_pos < live_len) {
            const int64_t live_offset =
                ((b * live_len + live_pos) * num_heads + head_id) * head_dim + d;
            k_value = live_k[live_offset];
            v_value = live_v[live_offset];
          }
        } else if (slot >= 0) {
          const int64_t slot_token =
              (head_id * gpu_cache_frames + slot) * frame_seqlen + src_in_frame;
          const int64_t gpu_offset =
              ((b * gpu_cache_tokens + slot_token) * head_dim) + d;
          k_value = gpu_k[gpu_offset];
          v_value = gpu_v[gpu_offset];
        }
      }
    }
  }

  k_out[out_offset] = k_value;
  v_out[out_offset] = v_value;
}

template <typename scalar_t>
__global__ void kv_sparse_frame_materialize_varlen_tiled_kernel(
    const scalar_t* __restrict__ gpu_k,
    const scalar_t* __restrict__ gpu_v,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    const int64_t* __restrict__ head_indices,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ slot_indices,
    const int32_t* __restrict__ cu_seqlens_k,
    const scalar_t* __restrict__ live_k,
    const scalar_t* __restrict__ live_v,
    int64_t B,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t head_count,
    int64_t n_selected,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t gpu_cache_tokens,
    int64_t live_len,
    int64_t live_start_index,
    bool has_live,
    int64_t elems_per_segment) {
  const int64_t pair_linear = static_cast<int64_t>(blockIdx.x);
  int64_t pair = pair_linear;
  const int64_t col = pair % n_selected;
  pair /= n_selected;
  const int64_t row = pair % head_count;
  const int64_t b = pair / head_count;
  const int64_t elem = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (elem >= elems_per_segment || b >= B) {
    return;
  }

  const int64_t pair_idx = row * n_selected + col;
  const int64_t head_id = head_indices[row];
  const int64_t frame_id = frame_indices[pair_idx];
  if (head_id < 0 || head_id >= num_heads || frame_id < 0) {
    return;
  }

  const int64_t row_linear = b * head_count + row;
  const int64_t row_start = static_cast<int64_t>(cu_seqlens_k[row_linear]);
  const int64_t row_end = static_cast<int64_t>(cu_seqlens_k[row_linear + 1]);
  const int64_t dst_token = col * frame_seqlen + elem / head_dim;
  if (row_start + dst_token >= row_end) {
    return;
  }

  const int64_t token_in_frame = elem / head_dim;
  const int64_t d = elem - token_in_frame * head_dim;
  const int64_t frame_start = frame_id * frame_seqlen;
  if (frame_start >= total_tokens) {
    return;
  }
  const int64_t remaining = total_tokens - frame_start;
  const int64_t frame_len = frame_seqlen < remaining ? frame_seqlen : remaining;
  if (frame_len <= 0) {
    return;
  }
  const int64_t src_in_frame = token_in_frame < frame_len ? token_in_frame : frame_len - 1;
  const int64_t slot = slot_indices[pair_idx];

  scalar_t k_value = scalar_t(0);
  scalar_t v_value = scalar_t(0);
  if (slot == kLiveSlot && has_live) {
    const int64_t live_pos = frame_start - live_start_index + src_in_frame;
    if (live_pos >= 0 && live_pos < live_len) {
      const int64_t live_offset =
          ((b * live_len + live_pos) * num_heads + head_id) * head_dim + d;
      k_value = live_k[live_offset];
      v_value = live_v[live_offset];
    }
  } else if (slot >= 0) {
    const int64_t slot_token =
        (head_id * gpu_cache_frames + slot) * frame_seqlen + src_in_frame;
    const int64_t gpu_offset =
        ((b * gpu_cache_tokens + slot_token) * head_dim) + d;
    k_value = gpu_k[gpu_offset];
    v_value = gpu_v[gpu_offset];
  }

  const int64_t out_offset = (row_start + dst_token) * head_dim + d;
  k_out[out_offset] = k_value;
  v_out[out_offset] = v_value;
}

__forceinline__ __device__ float kv_sparse_warp_sum(float value) {
  constexpr unsigned int kMask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kMask, value, offset);
  }
  return value;
}

template <typename scalar_t>
__global__ void kv_sparse_cache_direct_attention_warp_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ gpu_k,
    const scalar_t* __restrict__ gpu_v,
    scalar_t* __restrict__ out,
    const int64_t* __restrict__ head_indices,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ slot_indices,
    const scalar_t* __restrict__ live_k,
    const scalar_t* __restrict__ live_v,
    int64_t B,
    int64_t Lq,
    int64_t head_count,
    int64_t n_selected,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t gpu_cache_tokens,
    int64_t live_len,
    int64_t live_start_index,
    bool has_live,
    float scale) {
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = 4;
  constexpr int kMaxHeadDim = 256;
  constexpr int kMaxDimsPerLane = kMaxHeadDim / kWarpSize;
  constexpr unsigned int kMask = 0xffffffffu;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  if (warp >= kWarpsPerBlock) {
    return;
  }

  const int64_t query_linear =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int64_t total_queries = B * head_count * Lq;
  if (query_linear >= total_queries) {
    return;
  }

  const int64_t q_pos = query_linear % Lq;
  const int64_t q_row = query_linear / Lq;
  const int64_t row = q_row % head_count;
  const int64_t b = q_row / head_count;
  const int64_t head_id = head_indices[row];

  float q_vals[kMaxDimsPerLane];
  float acc_vals[kMaxDimsPerLane];
  int dim_vals[kMaxDimsPerLane];
  int dim_count = 0;
  const int64_t q_base = (q_row * Lq + q_pos) * head_dim;
  for (int64_t d = lane; d < head_dim && dim_count < kMaxDimsPerLane; d += kWarpSize) {
    dim_vals[dim_count] = static_cast<int>(d);
    q_vals[dim_count] = static_cast<float>(q[q_base + d]);
    acc_vals[dim_count] = 0.0f;
    ++dim_count;
  }

  float m = -3.4028234663852886e38f;
  float l = 0.0f;
  int valid_tokens = 0;
  const bool valid_head = head_id >= 0 && head_id < num_heads;

  if (valid_head) {
    for (int64_t col = 0; col < n_selected; ++col) {
      const int64_t pair_idx = row * n_selected + col;
      const int64_t frame_id = frame_indices[pair_idx];
      const int64_t slot = slot_indices[pair_idx];
      if (frame_id < 0) {
        continue;
      }
      const int64_t frame_start = frame_id * frame_seqlen;
      if (frame_start >= total_tokens) {
        continue;
      }
      const int64_t remaining = total_tokens - frame_start;
      const int64_t frame_len = frame_seqlen < remaining ? frame_seqlen : remaining;
      if (frame_len <= 0) {
        continue;
      }
      const bool use_live = slot == kLiveSlot && has_live;
      const bool use_slot = slot >= 0 && slot < gpu_cache_frames;
      if (!use_live && !use_slot) {
        continue;
      }

      for (int64_t token_in_frame = 0; token_in_frame < frame_seqlen; ++token_in_frame) {
        const int64_t src_in_frame =
            token_in_frame < frame_len ? token_in_frame : frame_len - 1;
        int64_t k_base = -1;
        int64_t v_base = -1;
        if (use_live) {
          const int64_t live_pos = frame_start - live_start_index + src_in_frame;
          if (live_pos >= 0 && live_pos < live_len) {
            k_base = ((b * live_len + live_pos) * num_heads + head_id) * head_dim;
            v_base = k_base;
          }
        } else {
          const int64_t slot_token =
              (head_id * gpu_cache_frames + slot) * frame_seqlen + src_in_frame;
          if (slot_token >= 0 && slot_token < gpu_cache_tokens) {
            k_base = (b * gpu_cache_tokens + slot_token) * head_dim;
            v_base = k_base;
          }
        }

        float dot = 0.0f;
        if (k_base >= 0) {
          const scalar_t* k_src = use_live ? live_k : gpu_k;
          for (int i = 0; i < dim_count; ++i) {
            dot += q_vals[i] * static_cast<float>(k_src[k_base + dim_vals[i]]);
          }
        }
        dot = kv_sparse_warp_sum(dot);
        const float score = __shfl_sync(kMask, dot, 0) * scale;
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);
        const float beta = expf(score - m_new);
        for (int i = 0; i < dim_count; ++i) {
          float v_value = 0.0f;
          if (v_base >= 0) {
            if (use_live) {
              v_value = static_cast<float>(live_v[v_base + dim_vals[i]]);
            } else {
              v_value = static_cast<float>(gpu_v[v_base + dim_vals[i]]);
            }
          }
          acc_vals[i] = acc_vals[i] * alpha + v_value * beta;
        }
        l = l * alpha + beta;
        m = m_new;
        ++valid_tokens;
      }
    }
  }

  const int64_t out_base = q_base;
  const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
  for (int i = 0; i < dim_count; ++i) {
    const float value = valid_tokens > 0 ? acc_vals[i] * inv_l : 0.0f;
    out[out_base + dim_vals[i]] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void kv_sparse_cache_direct_attention_block_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ gpu_k,
    const scalar_t* __restrict__ gpu_v,
    scalar_t* __restrict__ out,
    const int64_t* __restrict__ head_indices,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ slot_indices,
    const scalar_t* __restrict__ live_k,
    const scalar_t* __restrict__ live_v,
    int64_t B,
    int64_t Lq,
    int64_t head_count,
    int64_t n_selected,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t gpu_cache_tokens,
    int64_t live_len,
    int64_t live_start_index,
    bool has_live,
    float scale) {
  constexpr int kBlockM = 8;
  constexpr int kBlockN = 32;
  constexpr int kMaxHeadDim = 256;
  constexpr unsigned int kMask = 0xffffffffu;
  constexpr float kNegInf = -3.4028234663852886e38f;

  __shared__ scalar_t q_s[kBlockM * kMaxHeadDim];
  __shared__ scalar_t k_s[kBlockN * kMaxHeadDim];
  __shared__ scalar_t v_s[kBlockN * kMaxHeadDim];
  __shared__ float acc_s[kBlockM * kMaxHeadDim];
  __shared__ float score_s[kBlockM * kBlockN];
  __shared__ float m_s[kBlockM];
  __shared__ float l_s[kBlockM];
  __shared__ float alpha_s[kBlockM];
  __shared__ int valid_key_s[kBlockN];
  __shared__ int live_key_s[kBlockN];
  __shared__ int64_t key_base_s[kBlockN];

  const int64_t q_tile_start = static_cast<int64_t>(blockIdx.x) * kBlockM;
  const int64_t row_linear = static_cast<int64_t>(blockIdx.y);
  if (row_linear >= B * head_count) {
    return;
  }
  const int64_t row = row_linear % head_count;
  const int64_t b = row_linear / head_count;
  const int64_t q_row = row_linear;
  const int64_t head_id = head_indices[row];
  const bool valid_head = head_id >= 0 && head_id < num_heads;
  const int64_t total_key_tokens = n_selected * frame_seqlen;

  for (int idx = threadIdx.x; idx < kBlockM * head_dim; idx += blockDim.x) {
    const int mq = idx / head_dim;
    const int d = idx - mq * head_dim;
    const int64_t q_pos = q_tile_start + mq;
    scalar_t value = scalar_t(0);
    if (q_pos < Lq) {
      value = q[(q_row * Lq + q_pos) * head_dim + d];
    }
    q_s[idx] = value;
    acc_s[idx] = 0.0f;
  }
  if (threadIdx.x < kBlockM) {
    m_s[threadIdx.x] = kNegInf;
    l_s[threadIdx.x] = 0.0f;
    alpha_s[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (int64_t key_tile_start = 0; key_tile_start < total_key_tokens; key_tile_start += kBlockN) {
    if (threadIdx.x < kBlockN) {
      const int n = threadIdx.x;
      const int64_t key_token = key_tile_start + n;
      int valid = 0;
      int is_live = 0;
      int64_t key_base = -1;
      if (valid_head && key_token < total_key_tokens) {
        const int64_t col = key_token / frame_seqlen;
        const int64_t token_in_frame = key_token - col * frame_seqlen;
        const int64_t pair_idx = row * n_selected + col;
        const int64_t frame_id = frame_indices[pair_idx];
        const int64_t slot = slot_indices[pair_idx];
        if (frame_id >= 0) {
          const int64_t frame_start = frame_id * frame_seqlen;
          if (frame_start < total_tokens) {
            const int64_t remaining = total_tokens - frame_start;
            const int64_t frame_len = frame_seqlen < remaining ? frame_seqlen : remaining;
            if (frame_len > 0) {
              const int64_t src_in_frame =
                  token_in_frame < frame_len ? token_in_frame : frame_len - 1;
              if (slot == kLiveSlot && has_live) {
                const int64_t live_pos = frame_start - live_start_index + src_in_frame;
                valid = 1;
                is_live = 1;
                if (live_pos >= 0 && live_pos < live_len) {
                  key_base = ((b * live_len + live_pos) * num_heads + head_id) * head_dim;
                }
              } else if (slot >= 0 && slot < gpu_cache_frames) {
                const int64_t slot_token =
                    (head_id * gpu_cache_frames + slot) * frame_seqlen + src_in_frame;
                valid = 1;
                if (slot_token >= 0 && slot_token < gpu_cache_tokens) {
                  key_base = (b * gpu_cache_tokens + slot_token) * head_dim;
                }
              }
            }
          }
        }
      }
      valid_key_s[n] = valid;
      live_key_s[n] = is_live;
      key_base_s[n] = key_base;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kBlockN * head_dim; idx += blockDim.x) {
      const int n = idx / head_dim;
      const int d = idx - n * head_dim;
      scalar_t k_value = scalar_t(0);
      scalar_t v_value = scalar_t(0);
      if (valid_key_s[n] && key_base_s[n] >= 0) {
        if (live_key_s[n]) {
          k_value = live_k[key_base_s[n] + d];
          v_value = live_v[key_base_s[n] + d];
        } else {
          k_value = gpu_k[key_base_s[n] + d];
          v_value = gpu_v[key_base_s[n] + d];
        }
      }
      k_s[idx] = k_value;
      v_s[idx] = v_value;
    }
    __syncthreads();

    if (threadIdx.x < kBlockM * kBlockN) {
      const int mq = threadIdx.x / kBlockN;
      const int n = threadIdx.x - mq * kBlockN;
      const int64_t q_pos = q_tile_start + mq;
      float score = kNegInf;
      if (q_pos < Lq && valid_key_s[n]) {
        float dot = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
          dot += static_cast<float>(q_s[mq * head_dim + d]) *
                 static_cast<float>(k_s[n * head_dim + d]);
        }
        score = dot * scale;
      }
      score_s[threadIdx.x] = score;
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp < kBlockM) {
      const int mq = warp;
      const int64_t q_pos = q_tile_start + mq;
      float score = (lane < kBlockN && q_pos < Lq) ? score_s[mq * kBlockN + lane] : kNegInf;
      float tile_max = score;
      for (int offset = 16; offset > 0; offset >>= 1) {
        tile_max = fmaxf(tile_max, __shfl_down_sync(kMask, tile_max, offset));
      }
      tile_max = __shfl_sync(kMask, tile_max, 0);
      const float old_m = m_s[mq];
      const float new_m = fmaxf(old_m, tile_max);
      const float beta = expf(score - new_m);
      float tile_sum = (lane < kBlockN && q_pos < Lq) ? beta : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        tile_sum += __shfl_down_sync(kMask, tile_sum, offset);
      }
      if (lane == 0) {
        const float old_l = l_s[mq];
        const float alpha = expf(old_m - new_m);
        alpha_s[mq] = alpha;
        m_s[mq] = new_m;
        l_s[mq] = old_l * alpha + tile_sum;
      }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kBlockM * head_dim; idx += blockDim.x) {
      const int mq = idx / head_dim;
      const int d = idx - mq * head_dim;
      const int64_t q_pos = q_tile_start + mq;
      if (q_pos >= Lq) {
        continue;
      }
      float value = acc_s[idx] * alpha_s[mq];
      const float row_m = m_s[mq];
      for (int n = 0; n < kBlockN; ++n) {
        const float weight = expf(score_s[mq * kBlockN + n] - row_m);
        value += weight * static_cast<float>(v_s[n * head_dim + d]);
      }
      acc_s[idx] = value;
    }
    __syncthreads();
  }

  for (int idx = threadIdx.x; idx < kBlockM * head_dim; idx += blockDim.x) {
    const int mq = idx / head_dim;
    const int d = idx - mq * head_dim;
    const int64_t q_pos = q_tile_start + mq;
    if (q_pos >= Lq) {
      continue;
    }
    const float denom = l_s[mq];
    const float value = denom > 0.0f ? acc_s[idx] / denom : 0.0f;
    out[(q_row * Lq + q_pos) * head_dim + d] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__global__ void kv_sparse_cache_direct_attention_warp_tiled_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ gpu_k,
    const scalar_t* __restrict__ gpu_v,
    scalar_t* __restrict__ out,
    const int64_t* __restrict__ head_indices,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ slot_indices,
    const scalar_t* __restrict__ live_k,
    const scalar_t* __restrict__ live_v,
    int64_t B,
    int64_t Lq,
    int64_t head_count,
    int64_t n_selected,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t gpu_cache_tokens,
    int64_t live_len,
    int64_t live_start_index,
    bool has_live,
    float scale) {
  constexpr int kWarpSize = 32;
  constexpr int kBlockM = 8;
  constexpr int kBlockN = 32;
  constexpr int kMaxHeadDim = 256;
  constexpr int kMaxDimsPerLane = kMaxHeadDim / kWarpSize;
  constexpr unsigned int kMask = 0xffffffffu;

  __shared__ scalar_t k_s[kBlockN * kMaxHeadDim];
  __shared__ scalar_t v_s[kBlockN * kMaxHeadDim];
  __shared__ int valid_key_s[kBlockN];
  __shared__ int live_key_s[kBlockN];
  __shared__ int64_t key_base_s[kBlockN];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  if (warp >= kBlockM) {
    return;
  }

  const int64_t q_tile_start = static_cast<int64_t>(blockIdx.x) * kBlockM;
  const int64_t row_linear = static_cast<int64_t>(blockIdx.y);
  if (row_linear >= B * head_count) {
    return;
  }
  const int64_t row = row_linear % head_count;
  const int64_t b = row_linear / head_count;
  const int64_t q_pos = q_tile_start + warp;
  const int64_t q_row = row_linear;
  const int64_t head_id = head_indices[row];
  const bool valid_head = head_id >= 0 && head_id < num_heads;
  const bool valid_query = valid_head && q_pos < Lq;
  const int64_t total_key_tokens = n_selected * frame_seqlen;

  float q_vals[kMaxDimsPerLane];
  float acc_vals[kMaxDimsPerLane];
  int dim_vals[kMaxDimsPerLane];
  int dim_count = 0;
  if (valid_query) {
    const int64_t q_base = (q_row * Lq + q_pos) * head_dim;
    for (int64_t d = lane; d < head_dim && dim_count < kMaxDimsPerLane; d += kWarpSize) {
      dim_vals[dim_count] = static_cast<int>(d);
      q_vals[dim_count] = static_cast<float>(q[q_base + d]);
      acc_vals[dim_count] = 0.0f;
      ++dim_count;
    }
  }

  float m = -3.4028234663852886e38f;
  float l = 0.0f;

  for (int64_t key_tile_start = 0; key_tile_start < total_key_tokens; key_tile_start += kBlockN) {
    if (threadIdx.x < kBlockN) {
      const int n = threadIdx.x;
      const int64_t key_token = key_tile_start + n;
      int valid = 0;
      int is_live = 0;
      int64_t key_base = -1;
      if (valid_head && key_token < total_key_tokens) {
        const int64_t col = key_token / frame_seqlen;
        const int64_t token_in_frame = key_token - col * frame_seqlen;
        const int64_t pair_idx = row * n_selected + col;
        const int64_t frame_id = frame_indices[pair_idx];
        const int64_t slot = slot_indices[pair_idx];
        if (frame_id >= 0) {
          const int64_t frame_start = frame_id * frame_seqlen;
          if (frame_start < total_tokens) {
            const int64_t remaining = total_tokens - frame_start;
            const int64_t frame_len = frame_seqlen < remaining ? frame_seqlen : remaining;
            if (frame_len > 0) {
              const int64_t src_in_frame =
                  token_in_frame < frame_len ? token_in_frame : frame_len - 1;
              if (slot == kLiveSlot && has_live) {
                const int64_t live_pos = frame_start - live_start_index + src_in_frame;
                valid = 1;
                is_live = 1;
                if (live_pos >= 0 && live_pos < live_len) {
                  key_base = ((b * live_len + live_pos) * num_heads + head_id) * head_dim;
                }
              } else if (slot >= 0 && slot < gpu_cache_frames) {
                const int64_t slot_token =
                    (head_id * gpu_cache_frames + slot) * frame_seqlen + src_in_frame;
                valid = 1;
                if (slot_token >= 0 && slot_token < gpu_cache_tokens) {
                  key_base = (b * gpu_cache_tokens + slot_token) * head_dim;
                }
              }
            }
          }
        }
      }
      valid_key_s[n] = valid;
      live_key_s[n] = is_live;
      key_base_s[n] = key_base;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kBlockN * head_dim; idx += blockDim.x) {
      const int n = idx / head_dim;
      const int d = idx - n * head_dim;
      scalar_t k_value = scalar_t(0);
      scalar_t v_value = scalar_t(0);
      if (valid_key_s[n] && key_base_s[n] >= 0) {
        if (live_key_s[n]) {
          k_value = live_k[key_base_s[n] + d];
          v_value = live_v[key_base_s[n] + d];
        } else {
          k_value = gpu_k[key_base_s[n] + d];
          v_value = gpu_v[key_base_s[n] + d];
        }
      }
      k_s[idx] = k_value;
      v_s[idx] = v_value;
    }
    __syncthreads();

    if (valid_query) {
      for (int n = 0; n < kBlockN; ++n) {
        if (!valid_key_s[n]) {
          continue;
        }
        float dot = 0.0f;
        for (int i = 0; i < dim_count; ++i) {
          dot += q_vals[i] * static_cast<float>(k_s[n * head_dim + dim_vals[i]]);
        }
        dot = kv_sparse_warp_sum(dot);
        const float score = __shfl_sync(kMask, dot, 0) * scale;
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);
        const float beta = expf(score - m_new);
        for (int i = 0; i < dim_count; ++i) {
          const float v_value = static_cast<float>(v_s[n * head_dim + dim_vals[i]]);
          acc_vals[i] = acc_vals[i] * alpha + v_value * beta;
        }
        l = l * alpha + beta;
        m = m_new;
      }
    }
    __syncthreads();
  }

  if (valid_query) {
    const int64_t out_base = (q_row * Lq + q_pos) * head_dim;
    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (int i = 0; i < dim_count; ++i) {
      out[out_base + dim_vals[i]] = static_cast<scalar_t>(acc_vals[i] * inv_l);
    }
  }
}


__global__ void kv_sparse_lookup_frame_slots_kernel(
    const int64_t* __restrict__ head_indices,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ frame_to_slot,
    int64_t* __restrict__ slot_indices,
    int32_t* __restrict__ miss_flag,
    int64_t head_count,
    int64_t n_selected,
    int64_t frame_capacity,
    int64_t total_tokens,
    int64_t frame_seqlen,
    int64_t num_heads,
    int64_t gpu_cache_frames,
    int64_t live_start_index,
    int64_t live_len,
    bool has_live) {
  const int64_t total = head_count * n_selected;
  const int64_t live_end_index = live_start_index + live_len;
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = linear / n_selected;
    const int64_t head_id = head_indices[row];
    const int64_t frame_id = frame_indices[linear];
    int64_t slot = -3;

    if (head_id >= 0 && head_id < num_heads && frame_id >= 0) {
      const int64_t frame_start = frame_id * frame_seqlen;
      if (frame_start < total_tokens) {
        const int64_t remaining = total_tokens - frame_start;
        const int64_t frame_len = frame_seqlen < remaining ? frame_seqlen : remaining;
        if (frame_len > 0) {
          const int64_t frame_end = frame_start + frame_len;
          const bool is_live =
              has_live && frame_start >= live_start_index && frame_end <= live_end_index;
          if (is_live) {
            slot = kLiveSlot;
          } else if (frame_id < frame_capacity) {
            const int64_t looked_up = frame_to_slot[head_id * frame_capacity + frame_id];
            if (looked_up >= 0 && looked_up < gpu_cache_frames) {
              slot = looked_up;
            } else {
              atomicExch(miss_flag, 1);
            }
          } else {
            atomicExch(miss_flag, 1);
          }
        }
      }
    }

    slot_indices[linear] = slot;
  }
}


__global__ void kv_sparse_update_frame_slot_map_kernel(
    int64_t* __restrict__ frame_to_slot,
    const int64_t* __restrict__ head_ids,
    const int64_t* __restrict__ frame_ids,
    const int64_t* __restrict__ slot_ids,
    const int64_t* __restrict__ evicted_frame_ids,
    int64_t n,
    int64_t num_heads,
    int64_t frame_capacity) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t head_id = head_ids[i];
    if (head_id < 0 || head_id >= num_heads) {
      continue;
    }
    const int64_t evicted = evicted_frame_ids[i];
    if (evicted >= 0 && evicted < frame_capacity) {
      frame_to_slot[head_id * frame_capacity + evicted] = -1;
    }
    const int64_t frame_id = frame_ids[i];
    const int64_t slot = slot_ids[i];
    if (frame_id >= 0 && frame_id < frame_capacity && slot >= 0) {
      frame_to_slot[head_id * frame_capacity + frame_id] = slot;
    }
  }
}


template <typename scalar_t>
__global__ void kv_sparse_copy_frames_to_slots_zero_copy_kernel(
    const scalar_t* __restrict__ cpu_k,
    const scalar_t* __restrict__ cpu_v,
    scalar_t* __restrict__ gpu_k,
    scalar_t* __restrict__ gpu_v,
    const int64_t* __restrict__ head_ids,
    const int64_t* __restrict__ frame_ids,
    const int64_t* __restrict__ slot_ids,
    int64_t n,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t D,
    int64_t gpu_total_tokens,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t total_elements) {
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total_elements;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t t = linear;
    const int64_t d = t % D;
    t /= D;
    const int64_t token_in_frame = t % frame_seqlen;
    t /= frame_seqlen;
    const int64_t b = t % B;
    const int64_t item = t / B;
    if (item >= n) {
      continue;
    }
    const int64_t head_id = head_ids[item];
    const int64_t frame_id = frame_ids[item];
    const int64_t slot = slot_ids[item];
    if (head_id < 0 || head_id >= H || frame_id < 0 || slot < 0 || slot >= gpu_cache_frames) {
      continue;
    }
    const int64_t src_token = frame_id * frame_seqlen + token_in_frame;
    if (src_token >= T) {
      continue;
    }
    const int64_t src_offset = ((b * T + src_token) * H + head_id) * D + d;
    const int64_t dst_token = (head_id * gpu_cache_frames + slot) * frame_seqlen + token_in_frame;
    const int64_t dst_offset = (b * gpu_total_tokens + dst_token) * D + d;
    gpu_k[dst_offset] = cpu_k[src_offset];
    gpu_v[dst_offset] = cpu_v[src_offset];
  }
}

template <typename scalar_t>
__global__ void kv_sparse_copy_frame_plan_to_slots_zero_copy_kernel(
    const scalar_t* __restrict__ cpu_k,
    const scalar_t* __restrict__ cpu_v,
    scalar_t* __restrict__ gpu_k,
    scalar_t* __restrict__ gpu_v,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ slot_plan,
    const int64_t* __restrict__ copy_plan,
    int64_t pair_count,
    int64_t n_selected,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t D,
    int64_t gpu_total_tokens,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  for (int64_t pair = blockIdx.x; pair < pair_count; pair += gridDim.x) {
    if (copy_plan[pair] == 0) {
      continue;
    }
    const int64_t head_id = pair / n_selected;
    const int64_t frame_id = frame_indices[pair];
    const int64_t slot = slot_plan[pair];
    if (head_id < 0 || head_id >= H || frame_id < 0 || slot < 0 || slot >= gpu_cache_frames) {
      continue;
    }
    const int64_t frame_start = frame_id * frame_seqlen;
    if (frame_start >= T) {
      continue;
    }
    const int64_t remaining = T - frame_start;
    const int64_t frame_len = frame_seqlen < remaining ? frame_seqlen : remaining;
    if (frame_len <= 0) {
      continue;
    }
    const int64_t elements_per_pair = B * frame_len * D;
    for (int64_t elem = threadIdx.x; elem < elements_per_pair; elem += blockDim.x) {
      int64_t t = elem;
      const int64_t d = t % D;
      t /= D;
      const int64_t token_in_frame = t % frame_len;
      const int64_t b = t / frame_len;
      const int64_t src_token = frame_start + token_in_frame;
      const int64_t src_offset = ((b * T + src_token) * H + head_id) * D + d;
      const int64_t dst_token = (head_id * gpu_cache_frames + slot) * frame_seqlen + token_in_frame;
      const int64_t dst_offset = (b * gpu_total_tokens + dst_token) * D + d;
      gpu_k[dst_offset] = cpu_k[src_offset];
      gpu_v[dst_offset] = cpu_v[src_offset];
    }
  }
}

template <typename scalar_t>
__global__ void kv_sparse_scatter_staging_to_slots_kernel(
    const scalar_t* __restrict__ staging_k,
    const scalar_t* __restrict__ staging_v,
    scalar_t* __restrict__ gpu_k,
    scalar_t* __restrict__ gpu_v,
    const int64_t* __restrict__ head_ids,
    const int64_t* __restrict__ slot_ids,
    const int64_t* __restrict__ src_rows,
    const int64_t* __restrict__ src_cols,
    int64_t n,
    int64_t B,
    int64_t active_rows,
    int64_t staging_token_count,
    int64_t D,
    int64_t gpu_total_tokens,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t total_elements) {
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total_elements;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t t = linear;
    const int64_t d = t % D;
    t /= D;
    const int64_t token_in_frame = t % frame_seqlen;
    t /= frame_seqlen;
    const int64_t b = t % B;
    const int64_t item = t / B;
    if (item >= n) {
      continue;
    }
    const int64_t head_id = head_ids[item];
    const int64_t slot = slot_ids[item];
    const int64_t src_row = src_rows[item];
    const int64_t src_col = src_cols[item];
    if (head_id < 0 || slot < 0 || slot >= gpu_cache_frames ||
        src_row < 0 || src_row >= active_rows || src_col < 0) {
      continue;
    }
    const int64_t src_token = src_col * frame_seqlen + token_in_frame;
    if (src_token >= staging_token_count) {
      continue;
    }
    const int64_t folded_src_row = b * active_rows + src_row;
    const int64_t src_offset = (folded_src_row * staging_token_count + src_token) * D + d;
    const int64_t dst_token = (head_id * gpu_cache_frames + slot) * frame_seqlen + token_in_frame;
    if (dst_token >= gpu_total_tokens) {
      continue;
    }
    const int64_t dst_offset = (b * gpu_total_tokens + dst_token) * D + d;
    gpu_k[dst_offset] = staging_k[src_offset];
    gpu_v[dst_offset] = staging_v[src_offset];
  }
}


template <typename scalar_t>
__global__ void pooled_qk_q_mean_kernel(
    const scalar_t* __restrict__ q,
    const int64_t* __restrict__ head_indices,
    float* __restrict__ q_mean,
    int64_t B,
    int64_t Lq,
    int64_t H,
    int64_t D,
    int64_t Hs,
    int64_t frame_seqlen,
    int64_t qs0,
    int64_t qs1,
    int64_t qs2,
    int64_t qs3) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t total = B * Hs * D;
  if (linear >= total) {
    return;
  }
  int64_t t = linear;
  const int64_t d = t % D;
  t /= D;
  const int64_t row = t % Hs;
  const int64_t b = t / Hs;
  const int64_t head_id = head_indices[row];
  float acc = 0.0f;
  if (head_id >= 0 && head_id < H) {
    const bool simple_mean = frame_seqlen <= 1 || (frame_seqlen > 0 && Lq % frame_seqlen == 0);
    const int64_t full_blocks = simple_mean ? 0 : Lq / frame_seqlen;
    const int64_t rest = simple_mean ? 0 : Lq - full_blocks * frame_seqlen;
    const int64_t pooled_blocks = simple_mean ? 1 : full_blocks + (rest > 0 ? 1 : 0);
    const int64_t full_tokens = full_blocks * frame_seqlen;
    for (int64_t pos = threadIdx.x; pos < Lq; pos += blockDim.x) {
      const int64_t off = b * qs0 + pos * qs1 + head_id * qs2 + d * qs3;
      const float value = static_cast<float>(q[off]);
      if (simple_mean) {
        acc += value / static_cast<float>(Lq);
      } else if (pos < full_tokens) {
        acc += value / static_cast<float>(frame_seqlen * pooled_blocks);
      } else {
        acc += value / static_cast<float>(rest * pooled_blocks);
      }
    }
  }
  __shared__ float smem[256];
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    q_mean[(b * Hs + row) * D + d] = smem[0];
  }
}

template <typename scalar_t>
__global__ void pooled_qk_score_kernel(
    const float* __restrict__ q_mean,
    const scalar_t* __restrict__ frame_k,
    const int64_t* __restrict__ head_indices,
    float* __restrict__ scores,
    int64_t B,
    int64_t F,
    int64_t H,
    int64_t D,
    int64_t Hs,
    int64_t fs0,
    int64_t fs1,
    int64_t fs2,
    int64_t fs3) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t total = Hs * F;
  if (linear >= total) {
    return;
  }
  const int64_t f = linear % F;
  const int64_t row = linear / F;
  const int64_t head_id = head_indices[row];
  float acc = 0.0f;
  if (head_id >= 0 && head_id < H) {
    const int64_t total_bd = B * D;
    for (int64_t idx = threadIdx.x; idx < total_bd; idx += blockDim.x) {
      const int64_t d = idx % D;
      const int64_t b = idx / D;
      const int64_t off = b * fs0 + f * fs1 + head_id * fs2 + d * fs3;
      acc += q_mean[(b * Hs + row) * D + d] * static_cast<float>(frame_k[off]);
    }
  }
  __shared__ float smem[256];
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float scale = rsqrtf(static_cast<float>(D > 0 ? D : 1));
  const float denom = static_cast<float>(B > 0 ? B : 1);
  if (threadIdx.x == 0) {
    scores[row * F + f] = smem[0] * scale / denom;
  }
}

__device__ bool selected_contains_sorted(const int64_t* values, int count, int64_t value) {
  int64_t lo = 0;
  int64_t hi = count;
  while (lo < hi) {
    const int64_t mid = lo + ((hi - lo) >> 1);
    if (values[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < count && values[lo] == value;
}

__device__ void selected_insert_sorted(int64_t* values, int count, int64_t value) {
  int pos = count;
  while (pos > 0 && values[pos - 1] > value) {
    values[pos] = values[pos - 1];
    --pos;
  }
  values[pos] = value;
}

__global__ void pooled_qk_topk_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ compulsory_indices,
    int64_t* __restrict__ out,
    int64_t Hs,
    int64_t F,
    int64_t compulsory_count,
    int64_t max_kv_frames,
    int64_t out_cols) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= Hs) {
    return;
  }

  constexpr int kMaxOut = 256;
  if (F <= max_kv_frames) {
    if (threadIdx.x == 0) {
      for (int64_t i = 0; i < out_cols; ++i) {
        out[row * out_cols + i] = i < F ? i : -1;
      }
    }
    return;
  }

  __shared__ int64_t selected[kMaxOut];
  __shared__ int selected_count;
  __shared__ int64_t best_frame_sh[256];
  __shared__ float best_score_sh[256];

  if (threadIdx.x == 0) {
    selected_count = 0;
    for (int64_t i = 0; i < compulsory_count && selected_count < kMaxOut; ++i) {
      const int64_t frame = compulsory_indices[i];
      if (frame >= 0 && frame < F &&
          !selected_contains_sorted(selected, selected_count, frame)) {
        selected_insert_sorted(selected, selected_count, frame);
        ++selected_count;
      }
    }
  }
  __syncthreads();

  const int target = static_cast<int>(max_kv_frames < kMaxOut ? max_kv_frames : kMaxOut);
  while (selected_count < target) {
    int64_t local_best_frame = -1;
    float local_best_score = -3.4028234663852886e38f;
    for (int64_t f = threadIdx.x; f < F; f += blockDim.x) {
      if (selected_contains_sorted(selected, selected_count, f)) {
        continue;
      }
      const float score = scores[row * F + f];
      if (local_best_frame < 0 || score > local_best_score) {
        local_best_score = score;
        local_best_frame = f;
      }
    }
    best_frame_sh[threadIdx.x] = local_best_frame;
    best_score_sh[threadIdx.x] = local_best_score;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        const float other_score = best_score_sh[threadIdx.x + offset];
        const int64_t other_frame = best_frame_sh[threadIdx.x + offset];
        if (other_frame >= 0 &&
            (best_frame_sh[threadIdx.x] < 0 || other_score > best_score_sh[threadIdx.x])) {
          best_score_sh[threadIdx.x] = other_score;
          best_frame_sh[threadIdx.x] = other_frame;
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const int64_t best_frame = best_frame_sh[0];
      if (best_frame >= 0) {
        selected_insert_sorted(selected, selected_count, best_frame);
        ++selected_count;
      }
    }
    __syncthreads();
    if (best_frame_sh[0] < 0) {
      break;
    }
  }

  if (threadIdx.x == 0) {
    for (int64_t i = 0; i < out_cols; ++i) {
      out[row * out_cols + i] = i < selected_count ? selected[i] : -1;
    }
  }
}

void check_tensor(bool cond, const char* msg) {
  TORCH_CHECK(cond, msg);
}

int64_t copy_frame_to_slot_unchecked(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    int64_t head_id,
    int64_t frame_id,
    int64_t slot,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    cudaStream_t stream) {
  const int64_t B = cpu_k.size(0);
  const int64_t H = infer_gpu_slot_head_count(gpu_k, frame_seqlen, gpu_cache_frames);
  const bool head_major = cpu_kv_is_head_major(cpu_k, H);
  const int64_t T = head_major ? cpu_k.size(2) : cpu_k.size(1);
  const int64_t D = cpu_k.size(3);

  check_tensor(cpu_k.size(0) == gpu_k.size(0) && cpu_k.size(3) == gpu_k.size(3),
               "cpu/gpu KV shape mismatch");
  check_tensor((head_major ? cpu_k.size(1) : cpu_k.size(2)) == H,
               "cpu KV head dimension mismatch");
  check_tensor(head_id >= 0 && head_id < H, "head_id out of range");
  check_tensor(frame_id >= 0, "frame_id must be non-negative");
  check_tensor(slot >= 0 && slot < gpu_cache_frames, "slot out of range");

  const int64_t start = frame_id * frame_seqlen;
  if (start >= T) {
    return 0;
  }
  const int64_t copy_tokens = std::min<int64_t>(frame_seqlen, T - start);
  if (copy_tokens <= 0) {
    return 0;
  }

  const size_t elem_size = static_cast<size_t>(cpu_k.element_size());
  const size_t width_bytes = static_cast<size_t>(D) * elem_size;
  const size_t src_pitch = head_major
      ? static_cast<size_t>(copy_tokens * D) * elem_size
      : static_cast<size_t>(H * D) * elem_size;
  const size_t dst_pitch = head_major
      ? static_cast<size_t>(copy_tokens * D) * elem_size
      : static_cast<size_t>(D) * elem_size;
  const size_t copy_width = head_major
      ? static_cast<size_t>(copy_tokens * D) * elem_size
      : width_bytes;
  const size_t copy_height = head_major ? 1 : static_cast<size_t>(copy_tokens);
  const int64_t gpu_cache_tokens = gpu_k.size(1);
  const int64_t slot_start = (head_id * gpu_cache_frames + slot) * frame_seqlen;

  const char* cpu_k_ptr = static_cast<const char*>(cpu_k.data_ptr());
  const char* cpu_v_ptr = static_cast<const char*>(cpu_v.data_ptr());
  char* gpu_k_ptr = static_cast<char*>(gpu_k.data_ptr());
  char* gpu_v_ptr = static_cast<char*>(gpu_v.data_ptr());
  for (int64_t b = 0; b < B; ++b) {
    const int64_t src_elem = head_major
        ? (b * H * T + head_id * T + start) * D
        : ((b * T + start) * H + head_id) * D;
    const int64_t dst_elem = (b * gpu_cache_tokens + slot_start) * D;
    const char* src_k = cpu_k_ptr + src_elem * static_cast<int64_t>(elem_size);
    const char* src_v = cpu_v_ptr + src_elem * static_cast<int64_t>(elem_size);
    char* dst_k = gpu_k_ptr + dst_elem * static_cast<int64_t>(elem_size);
    char* dst_v = gpu_v_ptr + dst_elem * static_cast<int64_t>(elem_size);
    cudaError_t err = cudaMemcpy2DAsync(
        dst_k,
        dst_pitch,
        src_k,
        src_pitch,
        copy_width,
        copy_height,
        cudaMemcpyHostToDevice,
        stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync K failed: ", cudaGetErrorString(err));
    err = cudaMemcpy2DAsync(
        dst_v,
        dst_pitch,
        src_v,
        src_pitch,
        copy_width,
        copy_height,
        cudaMemcpyHostToDevice,
        stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync V failed: ", cudaGetErrorString(err));
  }
  return copy_tokens;
}

int64_t copy_frame_run_to_slots_unchecked(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    int64_t head_id,
    int64_t start_frame_id,
    int64_t start_slot,
    int64_t run_frames,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    cudaStream_t stream) {
  const int64_t B = cpu_k.size(0);
  const int64_t H = infer_gpu_slot_head_count(gpu_k, frame_seqlen, gpu_cache_frames);
  const bool head_major = cpu_kv_is_head_major(cpu_k, H);
  const int64_t T = head_major ? cpu_k.size(2) : cpu_k.size(1);
  const int64_t D = cpu_k.size(3);

  check_tensor(cpu_k.size(0) == gpu_k.size(0) && cpu_k.size(3) == gpu_k.size(3),
               "cpu/gpu KV shape mismatch");
  check_tensor((head_major ? cpu_k.size(1) : cpu_k.size(2)) == H,
               "cpu KV head dimension mismatch");
  check_tensor(head_id >= 0 && head_id < H, "head_id out of range");
  check_tensor(start_frame_id >= 0, "start_frame_id must be non-negative");
  check_tensor(start_slot >= 0 && start_slot < gpu_cache_frames, "start_slot out of range");
  check_tensor(run_frames > 0, "run_frames must be positive");

  const int64_t start = start_frame_id * frame_seqlen;
  if (start >= T) {
    return 0;
  }

  int64_t capped_run_frames = run_frames;
  const int64_t max_slot_frames = gpu_cache_frames - start_slot;
  if (capped_run_frames > max_slot_frames) {
    capped_run_frames = max_slot_frames;
  }
  const int64_t max_token_frames = (T - start + frame_seqlen - 1) / frame_seqlen;
  if (capped_run_frames > max_token_frames) {
    capped_run_frames = max_token_frames;
  }
  if (capped_run_frames <= 0) {
    return 0;
  }

  const int64_t total_copy_tokens = std::min<int64_t>(capped_run_frames * frame_seqlen, T - start);
  if (total_copy_tokens <= 0) {
    return 0;
  }

  const size_t elem_size = static_cast<size_t>(cpu_k.element_size());
  const size_t width_bytes = static_cast<size_t>(D) * elem_size;
  const size_t src_pitch = head_major
      ? static_cast<size_t>(total_copy_tokens * D) * elem_size
      : static_cast<size_t>(H * D) * elem_size;
  const size_t dst_pitch = head_major
      ? static_cast<size_t>(total_copy_tokens * D) * elem_size
      : static_cast<size_t>(D) * elem_size;
  const size_t copy_width = head_major
      ? static_cast<size_t>(total_copy_tokens * D) * elem_size
      : width_bytes;
  const size_t copy_height = head_major ? 1 : static_cast<size_t>(total_copy_tokens);
  const int64_t gpu_cache_tokens = gpu_k.size(1);
  const int64_t slot_start = (head_id * gpu_cache_frames + start_slot) * frame_seqlen;

  const char* cpu_k_ptr = static_cast<const char*>(cpu_k.data_ptr());
  const char* cpu_v_ptr = static_cast<const char*>(cpu_v.data_ptr());
  char* gpu_k_ptr = static_cast<char*>(gpu_k.data_ptr());
  char* gpu_v_ptr = static_cast<char*>(gpu_v.data_ptr());
  for (int64_t b = 0; b < B; ++b) {
    const int64_t src_elem = head_major
        ? (b * H * T + head_id * T + start) * D
        : ((b * T + start) * H + head_id) * D;
    const int64_t dst_elem = (b * gpu_cache_tokens + slot_start) * D;
    const char* src_k = cpu_k_ptr + src_elem * static_cast<int64_t>(elem_size);
    const char* src_v = cpu_v_ptr + src_elem * static_cast<int64_t>(elem_size);
    char* dst_k = gpu_k_ptr + dst_elem * static_cast<int64_t>(elem_size);
    char* dst_v = gpu_v_ptr + dst_elem * static_cast<int64_t>(elem_size);
    cudaError_t err = cudaMemcpy2DAsync(
        dst_k,
        dst_pitch,
        src_k,
        src_pitch,
        copy_width,
        copy_height,
        cudaMemcpyHostToDevice,
        stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync coalesced K failed: ", cudaGetErrorString(err));
    err = cudaMemcpy2DAsync(
        dst_v,
        dst_pitch,
        src_v,
        src_pitch,
        copy_width,
        copy_height,
        cudaMemcpyHostToDevice,
        stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync coalesced V failed: ", cudaGetErrorString(err));
  }
  return capped_run_frames;
}

int64_t copy_live_run_to_temp_slots_unchecked(
    torch::Tensor live_k,
    torch::Tensor live_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    int64_t head_id,
    int64_t src_token_start,
    int64_t dst_temp_slot_start,
    int64_t token_len,
    int64_t frame_seqlen,
    int64_t gpu_live_temp_frames,
    int64_t dst_token_base,
    cudaStream_t stream) {
  const int64_t B = live_k.size(0);
  const int64_t T = live_k.size(1);
  const int64_t H = live_k.size(2);
  const int64_t D = live_k.size(3);

  check_tensor(head_id >= 0 && head_id < H, "head_id out of range");
  check_tensor(src_token_start >= 0, "src_token_start must be non-negative");
  check_tensor(dst_temp_slot_start >= 0 && dst_temp_slot_start < gpu_live_temp_frames,
               "dst_temp_slot_start out of range");
  check_tensor(token_len > 0, "token_len must be positive");
  check_tensor(frame_seqlen > 0, "frame_seqlen must be positive");
  check_tensor(gpu_live_temp_frames > 0, "gpu_live_temp_frames must be positive");
  check_tensor(dst_token_base >= 0, "dst_token_base must be non-negative");

  if (src_token_start >= T) {
    return 0;
  }

  int64_t copy_tokens = std::min<int64_t>(token_len, T - src_token_start);
  const int64_t max_slot_tokens = (gpu_live_temp_frames - dst_temp_slot_start) * frame_seqlen;
  copy_tokens = std::min<int64_t>(copy_tokens, max_slot_tokens);
  if (copy_tokens <= 0) {
    return 0;
  }

  const int64_t gpu_total_tokens = gpu_k.size(1);
  const int64_t dst_token_start =
      dst_token_base + (head_id * gpu_live_temp_frames + dst_temp_slot_start) * frame_seqlen;
  check_tensor(dst_token_start >= 0 && dst_token_start + copy_tokens <= gpu_total_tokens,
               "destination live temp slot range out of bounds");

  const size_t elem_size = static_cast<size_t>(live_k.element_size());
  const size_t width_bytes = static_cast<size_t>(D) * elem_size;
  const size_t src_pitch = static_cast<size_t>(H * D) * elem_size;
  const size_t dst_pitch = static_cast<size_t>(D) * elem_size;

  const char* live_k_ptr = static_cast<const char*>(live_k.data_ptr());
  const char* live_v_ptr = static_cast<const char*>(live_v.data_ptr());
  char* gpu_k_ptr = static_cast<char*>(gpu_k.data_ptr());
  char* gpu_v_ptr = static_cast<char*>(gpu_v.data_ptr());

  for (int64_t b = 0; b < B; ++b) {
    const int64_t src_elem = ((b * T + src_token_start) * H + head_id) * D;
    const int64_t dst_elem = (b * gpu_total_tokens + dst_token_start) * D;
    const char* src_k = live_k_ptr + src_elem * static_cast<int64_t>(elem_size);
    const char* src_v = live_v_ptr + src_elem * static_cast<int64_t>(elem_size);
    char* dst_k = gpu_k_ptr + dst_elem * static_cast<int64_t>(elem_size);
    char* dst_v = gpu_v_ptr + dst_elem * static_cast<int64_t>(elem_size);
    cudaError_t err = cudaMemcpy2DAsync(
        dst_k,
        dst_pitch,
        src_k,
        src_pitch,
        width_bytes,
        static_cast<size_t>(copy_tokens),
        cudaMemcpyDeviceToDevice,
        stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync live K failed: ", cudaGetErrorString(err));
    err = cudaMemcpy2DAsync(
        dst_v,
        dst_pitch,
        src_v,
        src_pitch,
        width_bytes,
        static_cast<size_t>(copy_tokens),
        cudaMemcpyDeviceToDevice,
        stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync live V failed: ", cudaGetErrorString(err));
  }
  return copy_tokens;
}

}  // namespace

void kv_sparse_frame_materialize_cuda(
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor k_out,
    torch::Tensor v_out,
    torch::Tensor head_indices,
    torch::Tensor frame_indices,
    torch::Tensor slot_indices,
    torch::Tensor live_k,
    torch::Tensor live_v,
    int64_t frame_seqlen,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t gpu_cache_frames,
    int64_t live_start_index,
    bool has_live) {
  check_tensor(k_out.is_cuda() && v_out.is_cuda(), "k_out/v_out must be CUDA tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(head_indices.is_cuda() && frame_indices.is_cuda() && slot_indices.is_cuda(),
               "indices must be CUDA tensors");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu KV must be contiguous");
  check_tensor(k_out.is_contiguous() && v_out.is_contiguous(), "outputs must be contiguous");
  check_tensor(head_indices.is_contiguous() && frame_indices.is_contiguous() && slot_indices.is_contiguous(),
               "indices must be contiguous");
  check_tensor(gpu_k.scalar_type() == gpu_v.scalar_type(), "gpu KV dtype mismatch");
  check_tensor(gpu_k.scalar_type() == k_out.scalar_type(), "gpu/output dtype mismatch");
  check_tensor(gpu_k.scalar_type() == v_out.scalar_type(), "gpu/output dtype mismatch");
  check_tensor(frame_indices.dim() == 2, "frame_indices must be [head_count,n_selected]");
  check_tensor(slot_indices.sizes() == frame_indices.sizes(), "slot_indices shape mismatch");
  check_tensor(head_indices.dim() == 1 && head_indices.size(0) == frame_indices.size(0),
               "head_indices shape mismatch");
  check_tensor(head_indices.scalar_type() == at::kLong, "head_indices must be int64");
  check_tensor(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  check_tensor(slot_indices.scalar_type() == at::kLong, "slot_indices must be int64");

  const int64_t B = gpu_k.size(0);
  const int64_t D = gpu_k.size(3);
  const int64_t head_count = frame_indices.size(0);
  const int64_t n_selected = frame_indices.size(1);
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");
  check_tensor(num_heads > 0, "num_heads must be positive");
  check_tensor(k_out.size(0) == B * head_count, "k_out batch/head dimension mismatch");
  check_tensor(v_out.sizes() == k_out.sizes(), "v_out shape mismatch");
  check_tensor(k_out.size(1) == n_selected * frame_seqlen, "output token dimension mismatch");
  check_tensor(k_out.size(2) == 1 && k_out.size(3) == D, "output head/head_dim mismatch");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(gpu_k.size(2) == 1 && gpu_k.size(3) == D, "gpu KV shape mismatch");
  if (has_live) {
    check_tensor(live_k.is_cuda() && live_v.is_cuda(), "live KV must be CUDA tensors");
    check_tensor(live_k.is_contiguous() && live_v.is_contiguous(), "live KV must be contiguous");
    check_tensor(live_k.scalar_type() == gpu_k.scalar_type(), "live/gpu dtype mismatch");
    check_tensor(live_v.scalar_type() == gpu_k.scalar_type(), "live/gpu dtype mismatch");
    check_tensor(live_k.dim() == 4 && live_v.sizes() == live_k.sizes(), "live KV must be [B,T,H,D]");
    check_tensor(live_k.size(0) == B && live_k.size(2) == num_heads && live_k.size(3) == D,
                 "live KV shape mismatch");
  }

  const int64_t total_elements = B * head_count * n_selected * frame_seqlen * D;
  if (total_elements <= 0) {
    return;
  }
  const int64_t gpu_cache_tokens = gpu_k.size(1);
  const int64_t live_len = has_live ? live_k.size(1) : 0;

  const c10::cuda::CUDAGuard device_guard(k_out.device());
  const int threads = 256;
  const bool use_tiled_materialize = kv_sparse_tiled_materialize_enabled();
  const int64_t blocks64 = (total_elements + threads - 1) / threads;
  const int blocks = static_cast<int>(std::min<int64_t>(blocks64, 65535));
  const int64_t pair_count = B * head_count * n_selected;
  const int64_t elems_per_segment = frame_seqlen * D;
  const int64_t tiles64 = (elems_per_segment + threads - 1) / threads;
  const dim3 tiled_blocks(
      static_cast<unsigned int>(pair_count),
      static_cast<unsigned int>(tiles64));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gpu_k.scalar_type(),
                                  "kv_sparse_frame_materialize_cuda", [&] {
    if (use_tiled_materialize) {
      kv_sparse_frame_materialize_tiled_kernel<scalar_t><<<tiled_blocks, threads, 0, stream>>>(
          gpu_k.data_ptr<scalar_t>(),
          gpu_v.data_ptr<scalar_t>(),
          k_out.data_ptr<scalar_t>(),
          v_out.data_ptr<scalar_t>(),
          head_indices.data_ptr<int64_t>(),
          frame_indices.data_ptr<int64_t>(),
          slot_indices.data_ptr<int64_t>(),
          has_live ? live_k.data_ptr<scalar_t>() : nullptr,
          has_live ? live_v.data_ptr<scalar_t>() : nullptr,
          B,
          total_tokens,
          num_heads,
          D,
          head_count,
          n_selected,
          frame_seqlen,
          gpu_cache_frames,
          gpu_cache_tokens,
          live_len,
          live_start_index,
          has_live,
          elems_per_segment);
    } else {
      kv_sparse_frame_materialize_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          gpu_k.data_ptr<scalar_t>(),
          gpu_v.data_ptr<scalar_t>(),
          k_out.data_ptr<scalar_t>(),
          v_out.data_ptr<scalar_t>(),
          head_indices.data_ptr<int64_t>(),
          frame_indices.data_ptr<int64_t>(),
          slot_indices.data_ptr<int64_t>(),
          has_live ? live_k.data_ptr<scalar_t>() : nullptr,
          has_live ? live_v.data_ptr<scalar_t>() : nullptr,
          B,
          total_tokens,
          num_heads,
          D,
          head_count,
          n_selected,
          frame_seqlen,
          gpu_cache_frames,
          gpu_cache_tokens,
          live_len,
          live_start_index,
          has_live,
          total_elements);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void kv_sparse_frame_materialize_varlen_cuda(
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor k_out,
    torch::Tensor v_out,
    torch::Tensor head_indices,
    torch::Tensor frame_indices,
    torch::Tensor slot_indices,
    torch::Tensor cu_seqlens_k,
    torch::Tensor live_k,
    torch::Tensor live_v,
    int64_t frame_seqlen,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t gpu_cache_frames,
    int64_t live_start_index,
    bool has_live) {
  check_tensor(k_out.is_cuda() && v_out.is_cuda(), "k_out/v_out must be CUDA tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(head_indices.is_cuda() && frame_indices.is_cuda() && slot_indices.is_cuda(),
               "indices must be CUDA tensors");
  check_tensor(cu_seqlens_k.is_cuda(), "cu_seqlens_k must be CUDA tensor");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu KV must be contiguous");
  check_tensor(k_out.is_contiguous() && v_out.is_contiguous(), "outputs must be contiguous");
  check_tensor(head_indices.is_contiguous() && frame_indices.is_contiguous() && slot_indices.is_contiguous(),
               "indices must be contiguous");
  check_tensor(cu_seqlens_k.is_contiguous(), "cu_seqlens_k must be contiguous");
  check_tensor(gpu_k.scalar_type() == gpu_v.scalar_type(), "gpu KV dtype mismatch");
  check_tensor(gpu_k.scalar_type() == k_out.scalar_type(), "gpu/output dtype mismatch");
  check_tensor(gpu_k.scalar_type() == v_out.scalar_type(), "gpu/output dtype mismatch");
  check_tensor(frame_indices.dim() == 2, "frame_indices must be [head_count,n_selected]");
  check_tensor(slot_indices.sizes() == frame_indices.sizes(), "slot_indices shape mismatch");
  check_tensor(head_indices.dim() == 1 && head_indices.size(0) == frame_indices.size(0),
               "head_indices shape mismatch");
  check_tensor(cu_seqlens_k.dim() == 1, "cu_seqlens_k must be 1D");
  check_tensor(head_indices.scalar_type() == at::kLong, "head_indices must be int64");
  check_tensor(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  check_tensor(slot_indices.scalar_type() == at::kLong, "slot_indices must be int64");
  check_tensor(cu_seqlens_k.scalar_type() == at::kInt, "cu_seqlens_k must be int32");

  const int64_t B = gpu_k.size(0);
  const int64_t D = gpu_k.size(3);
  const int64_t head_count = frame_indices.size(0);
  const int64_t n_selected = frame_indices.size(1);
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");
  check_tensor(num_heads > 0, "num_heads must be positive");
  check_tensor(cu_seqlens_k.size(0) == B * head_count + 1, "cu_seqlens_k shape mismatch");
  check_tensor(k_out.dim() == 3 && v_out.sizes() == k_out.sizes(), "outputs must be [T,1,D]");
  check_tensor(k_out.size(1) == 1 && k_out.size(2) == D, "output head/head_dim mismatch");
  check_tensor(k_out.size(0) > 0, "output token capacity must be positive");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(gpu_k.size(2) == 1 && gpu_k.size(3) == D, "gpu KV shape mismatch");
  if (has_live) {
    check_tensor(live_k.is_cuda() && live_v.is_cuda(), "live KV must be CUDA tensors");
    check_tensor(live_k.is_contiguous() && live_v.is_contiguous(), "live KV must be contiguous");
    check_tensor(live_k.scalar_type() == gpu_k.scalar_type(), "live/gpu dtype mismatch");
    check_tensor(live_v.scalar_type() == gpu_k.scalar_type(), "live/gpu dtype mismatch");
    check_tensor(live_k.dim() == 4 && live_v.sizes() == live_k.sizes(), "live KV must be [B,T,H,D]");
    check_tensor(live_k.size(0) == B && live_k.size(2) == num_heads && live_k.size(3) == D,
                 "live KV shape mismatch");
  }

  const int64_t total_elements = B * head_count * n_selected * frame_seqlen * D;
  if (total_elements <= 0) {
    return;
  }
  const int64_t gpu_cache_tokens = gpu_k.size(1);
  const int64_t live_len = has_live ? live_k.size(1) : 0;

  const c10::cuda::CUDAGuard device_guard(k_out.device());
  const int threads = 256;
  const int64_t pair_count = B * head_count * n_selected;
  const int64_t elems_per_segment = frame_seqlen * D;
  const int64_t tiles64 = (elems_per_segment + threads - 1) / threads;
  const dim3 tiled_blocks(
      static_cast<unsigned int>(pair_count),
      static_cast<unsigned int>(tiles64));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gpu_k.scalar_type(),
                                  "kv_sparse_frame_materialize_varlen_cuda", [&] {
    kv_sparse_frame_materialize_varlen_tiled_kernel<scalar_t><<<tiled_blocks, threads, 0, stream>>>(
        gpu_k.data_ptr<scalar_t>(),
        gpu_v.data_ptr<scalar_t>(),
        k_out.data_ptr<scalar_t>(),
        v_out.data_ptr<scalar_t>(),
        head_indices.data_ptr<int64_t>(),
        frame_indices.data_ptr<int64_t>(),
        slot_indices.data_ptr<int64_t>(),
        cu_seqlens_k.data_ptr<int32_t>(),
        has_live ? live_k.data_ptr<scalar_t>() : nullptr,
        has_live ? live_v.data_ptr<scalar_t>() : nullptr,
        B,
        total_tokens,
        num_heads,
        D,
        head_count,
        n_selected,
        frame_seqlen,
        gpu_cache_frames,
        gpu_cache_tokens,
        live_len,
        live_start_index,
        has_live,
        elems_per_segment);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor kv_sparse_cache_direct_attention_cuda(
    torch::Tensor q,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor head_indices,
    torch::Tensor frame_indices,
    torch::Tensor slot_indices,
    torch::Tensor live_k,
    torch::Tensor live_v,
    int64_t frame_seqlen,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t gpu_cache_frames,
    int64_t live_start_index,
    bool has_live,
    double softmax_scale) {
  check_tensor(q.is_cuda(), "q must be a CUDA tensor");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(head_indices.is_cuda() && frame_indices.is_cuda() && slot_indices.is_cuda(),
               "indices must be CUDA tensors");
  check_tensor(q.is_contiguous(), "q must be contiguous");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu KV must be contiguous");
  check_tensor(head_indices.is_contiguous() && frame_indices.is_contiguous() && slot_indices.is_contiguous(),
               "indices must be contiguous");
  check_tensor(q.device() == gpu_k.device() && gpu_v.device() == gpu_k.device(),
               "q/gpu KV device mismatch");
  check_tensor(head_indices.device() == gpu_k.device() && frame_indices.device() == gpu_k.device() &&
                   slot_indices.device() == gpu_k.device(),
               "index/gpu KV device mismatch");
  check_tensor(q.scalar_type() == gpu_k.scalar_type(), "q/gpu KV dtype mismatch");
  check_tensor(gpu_k.scalar_type() == gpu_v.scalar_type(), "gpu KV dtype mismatch");
  check_tensor(head_indices.scalar_type() == at::kLong, "head_indices must be int64");
  check_tensor(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  check_tensor(slot_indices.scalar_type() == at::kLong, "slot_indices must be int64");
  check_tensor(q.dim() == 4 && q.size(2) == 1, "q must be [B*heads,Lq,1,D]");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(gpu_k.size(2) == 1, "gpu KV must have singleton head dimension");
  check_tensor(frame_indices.dim() == 2, "frame_indices must be [head_count,n_selected]");
  check_tensor(slot_indices.sizes() == frame_indices.sizes(), "slot_indices shape mismatch");
  check_tensor(head_indices.dim() == 1 && head_indices.size(0) == frame_indices.size(0),
               "head_indices shape mismatch");
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");
  check_tensor(total_tokens > 0 && num_heads > 0, "token/head counts must be positive");

  const int64_t B = gpu_k.size(0);
  const int64_t gpu_cache_tokens = gpu_k.size(1);
  const int64_t head_dim = q.size(3);
  const int64_t head_count = head_indices.size(0);
  const int64_t Lq = q.size(1);
  const int64_t n_selected = frame_indices.size(1);
  check_tensor(gpu_k.size(0) == B && gpu_k.size(3) == head_dim, "gpu KV shape mismatch");
  check_tensor(q.size(0) == B * head_count, "q batch/head dimension mismatch");
  check_tensor(head_dim > 0 && head_dim <= 256, "cache-direct attention supports 1 <= D <= 256");
  check_tensor(gpu_cache_tokens >= num_heads * gpu_cache_frames * frame_seqlen,
               "gpu cache token dimension too small");
  if (has_live) {
    check_tensor(live_k.is_cuda() && live_v.is_cuda(), "live KV must be CUDA tensors");
    check_tensor(live_k.is_contiguous() && live_v.is_contiguous(), "live KV must be contiguous");
    check_tensor(live_k.device() == gpu_k.device() && live_v.device() == gpu_k.device(),
                 "live/gpu KV device mismatch");
    check_tensor(live_k.scalar_type() == q.scalar_type(), "live/q dtype mismatch");
    check_tensor(live_v.scalar_type() == q.scalar_type(), "live/q dtype mismatch");
    check_tensor(live_k.dim() == 4 && live_v.sizes() == live_k.sizes(), "live KV must be [B,T,H,D]");
    check_tensor(live_k.size(0) == B && live_k.size(2) == num_heads && live_k.size(3) == head_dim,
                 "live KV shape mismatch");
  }

  auto out = torch::empty_like(q);
  const int64_t total_queries = B * head_count * Lq;
  if (total_queries <= 0 || n_selected <= 0) {
    out.zero_();
    return out;
  }

  const c10::cuda::CUDAGuard device_guard(q.device());
  const float scale = softmax_scale > 0.0
      ? static_cast<float>(softmax_scale)
      : 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t live_len = has_live ? live_k.size(1) : 0;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) {
    constexpr int threads = 256;
    constexpr int block_m = 8;
    const dim3 blocks(
        static_cast<unsigned int>((Lq + block_m - 1) / block_m),
        static_cast<unsigned int>(B * head_count));
    if (q.scalar_type() == at::kHalf) {
      kv_sparse_cache_direct_attention_warp_tiled_kernel<at::Half><<<blocks, threads, 0, stream>>>(
          q.data_ptr<at::Half>(),
          gpu_k.data_ptr<at::Half>(),
          gpu_v.data_ptr<at::Half>(),
          out.data_ptr<at::Half>(),
          head_indices.data_ptr<int64_t>(),
          frame_indices.data_ptr<int64_t>(),
          slot_indices.data_ptr<int64_t>(),
          has_live ? live_k.data_ptr<at::Half>() : nullptr,
          has_live ? live_v.data_ptr<at::Half>() : nullptr,
          B,
          Lq,
          head_count,
          n_selected,
          total_tokens,
          num_heads,
          head_dim,
          frame_seqlen,
          gpu_cache_frames,
          gpu_cache_tokens,
          live_len,
          live_start_index,
          has_live,
          scale);
    } else {
      kv_sparse_cache_direct_attention_warp_tiled_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
          q.data_ptr<at::BFloat16>(),
          gpu_k.data_ptr<at::BFloat16>(),
          gpu_v.data_ptr<at::BFloat16>(),
          out.data_ptr<at::BFloat16>(),
          head_indices.data_ptr<int64_t>(),
          frame_indices.data_ptr<int64_t>(),
          slot_indices.data_ptr<int64_t>(),
          has_live ? live_k.data_ptr<at::BFloat16>() : nullptr,
          has_live ? live_v.data_ptr<at::BFloat16>() : nullptr,
          B,
          Lq,
          head_count,
          n_selected,
          total_tokens,
          num_heads,
          head_dim,
          frame_seqlen,
          gpu_cache_frames,
          gpu_cache_tokens,
          live_len,
          live_start_index,
          has_live,
          scale);
    }
  } else {
    constexpr int threads = 128;
    constexpr int warps_per_block = threads / 32;
    const int blocks = static_cast<int>(
        std::min<int64_t>((total_queries + warps_per_block - 1) / warps_per_block, 2147483647LL));
    AT_DISPATCH_FLOATING_TYPES(q.scalar_type(),
                               "kv_sparse_cache_direct_attention_cuda_warp_fallback", [&] {
      kv_sparse_cache_direct_attention_warp_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        q.data_ptr<scalar_t>(),
        gpu_k.data_ptr<scalar_t>(),
        gpu_v.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        head_indices.data_ptr<int64_t>(),
        frame_indices.data_ptr<int64_t>(),
        slot_indices.data_ptr<int64_t>(),
        has_live ? live_k.data_ptr<scalar_t>() : nullptr,
        has_live ? live_v.data_ptr<scalar_t>() : nullptr,
        B,
        Lq,
        head_count,
        n_selected,
        total_tokens,
        num_heads,
        head_dim,
        frame_seqlen,
        gpu_cache_frames,
        gpu_cache_tokens,
        live_len,
        live_start_index,
        has_live,
        scale);
    });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


std::vector<torch::Tensor> kv_sparse_lookup_frame_slots_cuda(
    torch::Tensor head_indices,
    torch::Tensor frame_indices,
    torch::Tensor frame_to_slot,
    int64_t total_tokens,
    int64_t frame_seqlen,
    int64_t num_heads,
    int64_t gpu_cache_frames,
    int64_t live_start_index,
    int64_t live_len,
    bool has_live) {
  check_tensor(head_indices.is_cuda() && frame_indices.is_cuda() && frame_to_slot.is_cuda(),
               "lookup tensors must be CUDA tensors");
  check_tensor(head_indices.is_contiguous() && frame_indices.is_contiguous() &&
                   frame_to_slot.is_contiguous(),
               "lookup tensors must be contiguous");
  check_tensor(head_indices.scalar_type() == at::kLong, "head_indices must be int64");
  check_tensor(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  check_tensor(frame_to_slot.scalar_type() == at::kLong, "frame_to_slot must be int64");
  check_tensor(head_indices.dim() == 1, "head_indices must be 1D");
  check_tensor(frame_indices.dim() == 2, "frame_indices must be 2D");
  check_tensor(frame_to_slot.dim() == 2, "frame_to_slot must be 2D");
  check_tensor(frame_indices.size(0) == head_indices.size(0), "lookup shape mismatch");
  check_tensor(frame_to_slot.size(0) >= num_heads, "frame_to_slot head dimension too small");
  check_tensor(frame_seqlen > 0, "frame_seqlen must be positive");
  check_tensor(num_heads > 0 && gpu_cache_frames > 0, "head/cache sizes must be positive");
  check_tensor(frame_to_slot.device() == frame_indices.device(), "lookup device mismatch");
  check_tensor(head_indices.device() == frame_indices.device(), "lookup device mismatch");

  const int64_t head_count = head_indices.size(0);
  const int64_t n_selected = frame_indices.size(1);
  auto slot_indices = torch::full_like(frame_indices, static_cast<int64_t>(-3));
  auto miss_flag = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(frame_indices.device()));
  if (head_count <= 0 || n_selected <= 0 || total_tokens <= 0) {
    return {slot_indices, miss_flag};
  }

  const c10::cuda::CUDAGuard device_guard(frame_indices.device());
  const int threads = 256;
  const int64_t total = head_count * n_selected;
  const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kv_sparse_lookup_frame_slots_kernel<<<blocks, threads, 0, stream>>>(
      head_indices.data_ptr<int64_t>(),
      frame_indices.data_ptr<int64_t>(),
      frame_to_slot.data_ptr<int64_t>(),
      slot_indices.data_ptr<int64_t>(),
      miss_flag.data_ptr<int32_t>(),
      head_count,
      n_selected,
      frame_to_slot.size(1),
      total_tokens,
      frame_seqlen,
      num_heads,
      gpu_cache_frames,
      live_start_index,
      live_len,
      has_live);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {slot_indices, miss_flag};
}


void kv_sparse_update_frame_slot_map_cuda(
    torch::Tensor frame_to_slot,
    torch::Tensor head_ids,
    torch::Tensor frame_ids,
    torch::Tensor slot_ids,
    torch::Tensor evicted_frame_ids) {
  check_tensor(frame_to_slot.is_cuda() && head_ids.is_cuda() && frame_ids.is_cuda() &&
                   slot_ids.is_cuda() && evicted_frame_ids.is_cuda(),
               "slot metadata update tensors must be CUDA tensors");
  check_tensor(frame_to_slot.is_contiguous() && head_ids.is_contiguous() &&
                   frame_ids.is_contiguous() && slot_ids.is_contiguous() &&
                   evicted_frame_ids.is_contiguous(),
               "slot metadata update tensors must be contiguous");
  check_tensor(frame_to_slot.scalar_type() == at::kLong, "frame_to_slot must be int64");
  check_tensor(head_ids.scalar_type() == at::kLong, "head_ids must be int64");
  check_tensor(frame_ids.scalar_type() == at::kLong, "frame_ids must be int64");
  check_tensor(slot_ids.scalar_type() == at::kLong, "slot_ids must be int64");
  check_tensor(evicted_frame_ids.scalar_type() == at::kLong, "evicted_frame_ids must be int64");
  check_tensor(frame_to_slot.dim() == 2, "frame_to_slot must be 2D");
  check_tensor(head_ids.dim() == 1 && frame_ids.sizes() == head_ids.sizes() &&
                   slot_ids.sizes() == head_ids.sizes() &&
                   evicted_frame_ids.sizes() == head_ids.sizes(),
               "slot metadata update tensors must have shape [N]");
  check_tensor(head_ids.device() == frame_to_slot.device() &&
                   frame_ids.device() == frame_to_slot.device() &&
                   slot_ids.device() == frame_to_slot.device() &&
                   evicted_frame_ids.device() == frame_to_slot.device(),
               "slot metadata update device mismatch");

  const int64_t n = head_ids.size(0);
  if (n <= 0) {
    return;
  }
  const c10::cuda::CUDAGuard device_guard(frame_to_slot.device());
  const int threads = 128;
  const int blocks = static_cast<int>(std::min<int64_t>((n + threads - 1) / threads, 65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kv_sparse_update_frame_slot_map_kernel<<<blocks, threads, 0, stream>>>(
      frame_to_slot.data_ptr<int64_t>(),
      head_ids.data_ptr<int64_t>(),
      frame_ids.data_ptr<int64_t>(),
      slot_ids.data_ptr<int64_t>(),
      evicted_frame_ids.data_ptr<int64_t>(),
      n,
      frame_to_slot.size(0),
      frame_to_slot.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


torch::Tensor kv_sparse_select_pooled_qk_topk_cuda(
    torch::Tensor q,
    torch::Tensor frame_k,
    torch::Tensor head_indices,
    torch::Tensor compulsory_indices,
    int64_t frame_seqlen,
    int64_t max_kv_frames) {
  check_tensor(q.is_cuda() && frame_k.is_cuda(), "q/frame_k must be CUDA tensors");
  check_tensor(head_indices.is_cuda() && compulsory_indices.is_cuda(),
               "head_indices/compulsory_indices must be CUDA tensors");
  check_tensor(q.dim() == 4, "q must be [B,Lq,H,D]");
  check_tensor(frame_k.dim() == 4, "frame_k must be [B,F,H,D]");
  check_tensor(head_indices.dim() == 1, "head_indices must be 1D");
  check_tensor(compulsory_indices.dim() == 1, "compulsory_indices must be 1D");
  check_tensor(head_indices.scalar_type() == at::kLong, "head_indices must be int64");
  check_tensor(compulsory_indices.scalar_type() == at::kLong, "compulsory_indices must be int64");
  check_tensor(q.scalar_type() == frame_k.scalar_type(), "q/frame_k dtype mismatch");
  check_tensor(q.size(0) == frame_k.size(0), "q/frame_k batch mismatch");
  check_tensor(q.size(2) == frame_k.size(2), "q/frame_k head mismatch");
  check_tensor(q.size(3) == frame_k.size(3), "q/frame_k head_dim mismatch");
  check_tensor(frame_seqlen >= 0, "frame_seqlen must be non-negative");
  check_tensor(max_kv_frames >= 0, "max_kv_frames must be non-negative");

  const int64_t B = q.size(0);
  const int64_t Lq = q.size(1);
  const int64_t H = q.size(2);
  const int64_t D = q.size(3);
  const int64_t F = frame_k.size(1);
  const int64_t Hs = head_indices.size(0);
  const int64_t compulsory_count = compulsory_indices.size(0);
  if (B <= 0 || Lq <= 0 || H <= 0 || D <= 0 || F <= 0 || Hs <= 0) {
    return torch::empty({Hs, 0}, torch::TensorOptions().dtype(torch::kLong).device(q.device()));
  }

  int64_t out_cols = 0;
  if (F <= max_kv_frames) {
    out_cols = F;
  } else if (compulsory_count >= max_kv_frames) {
    out_cols = compulsory_count;
  } else {
    out_cols = max_kv_frames;
  }
  check_tensor(out_cols <= 256, "fused topK output supports at most 256 selected frames");

  const c10::cuda::CUDAGuard device_guard(q.device());
  head_indices = head_indices.contiguous();
  compulsory_indices = compulsory_indices.contiguous();
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
  auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(q.device());
  auto q_mean = torch::empty({B, Hs, D}, float_opts);
  auto scores = torch::empty({Hs, F}, float_opts);
  auto out = torch::empty({Hs, out_cols}, long_opts);

  const int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t q_total = B * Hs * D;
  const int q_blocks = static_cast<int>(q_total);
  const int64_t score_total = Hs * F;
  const int score_blocks = static_cast<int>(score_total);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(),
                                  "kv_sparse_select_pooled_qk_topk_cuda", [&] {
    pooled_qk_q_mean_kernel<scalar_t><<<q_blocks, threads, 0, stream>>>(
        q.data_ptr<scalar_t>(),
        head_indices.data_ptr<int64_t>(),
        q_mean.data_ptr<float>(),
        B,
        Lq,
        H,
        D,
        Hs,
        frame_seqlen,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3));
    pooled_qk_score_kernel<scalar_t><<<score_blocks, threads, 0, stream>>>(
        q_mean.data_ptr<float>(),
        frame_k.data_ptr<scalar_t>(),
        head_indices.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        B,
        F,
        H,
        D,
        Hs,
        frame_k.stride(0),
        frame_k.stride(1),
        frame_k.stride(2),
        frame_k.stride(3));
  });
  pooled_qk_topk_kernel<<<static_cast<unsigned int>(Hs), threads, 0, stream>>>(
      scores.data_ptr<float>(),
      compulsory_indices.data_ptr<int64_t>(),
      out.data_ptr<int64_t>(),
      Hs,
      F,
      compulsory_count,
      max_kv_frames,
      out_cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int64_t kv_sparse_copy_frame_to_slot_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    int64_t head_id,
    int64_t frame_id,
    int64_t slot,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  check_tensor(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu_k/gpu_v must be contiguous");
  check_tensor(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned for async HtoD copy");
  check_tensor(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_k.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_v.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be 4D");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");

  const int64_t B = cpu_k.size(0);
  const int64_t H = infer_gpu_slot_head_count(gpu_k, frame_seqlen, gpu_cache_frames);
  const bool head_major = cpu_kv_is_head_major(cpu_k, H);
  const int64_t T = head_major ? cpu_k.size(2) : cpu_k.size(1);
  const int64_t D = cpu_k.size(3);
  check_tensor(gpu_k.size(0) == B && gpu_k.size(2) == 1 && gpu_k.size(3) == D,
               "gpu KV shape mismatch");
  check_tensor((head_major ? cpu_k.size(1) : cpu_k.size(2)) == H,
               "cpu KV head dimension mismatch");
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");
  check_tensor(head_id >= 0 && head_id < H, "head_id out of range");
  check_tensor(frame_id >= 0, "frame_id must be non-negative");
  check_tensor(slot >= 0 && slot < gpu_cache_frames, "slot out of range");

  const int64_t start = frame_id * frame_seqlen;
  if (start >= T) {
    return 0;
  }
  const int64_t slot_len = std::min<int64_t>(frame_seqlen, T - start);
  if (slot_len <= 0) {
    return 0;
  }

  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return copy_frame_to_slot_unchecked(
      cpu_k,
      cpu_v,
      gpu_k,
      gpu_v,
      head_id,
      frame_id,
      slot,
      frame_seqlen,
      gpu_cache_frames,
      stream);
}

int64_t kv_sparse_copy_frames_to_slots_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor head_ids,
    torch::Tensor frame_ids,
    torch::Tensor slot_ids,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  check_tensor(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu_k/gpu_v must be contiguous");
  check_tensor(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned for async HtoD copy");
  check_tensor(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_k.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_v.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be 4D");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(head_ids.device().is_cpu() && frame_ids.device().is_cpu() && slot_ids.device().is_cpu(),
               "copy plan tensors must be CPU tensors");
  check_tensor(head_ids.is_contiguous() && frame_ids.is_contiguous() && slot_ids.is_contiguous(),
               "copy plan tensors must be contiguous");
  check_tensor(head_ids.scalar_type() == at::kLong, "head_ids must be int64");
  check_tensor(frame_ids.scalar_type() == at::kLong, "frame_ids must be int64");
  check_tensor(slot_ids.scalar_type() == at::kLong, "slot_ids must be int64");
  check_tensor(head_ids.dim() == 1 && frame_ids.sizes() == head_ids.sizes() && slot_ids.sizes() == head_ids.sizes(),
               "copy plan tensors must have shape [N]");

  const int64_t B = cpu_k.size(0);
  const int64_t H = infer_gpu_slot_head_count(gpu_k, frame_seqlen, gpu_cache_frames);
  const bool head_major = cpu_kv_is_head_major(cpu_k, H);
  const int64_t T = head_major ? cpu_k.size(2) : cpu_k.size(1);
  const int64_t D = cpu_k.size(3);
  (void)B;
  check_tensor(gpu_k.size(0) == B && gpu_k.size(2) == 1 && gpu_k.size(3) == D,
               "gpu KV shape mismatch");
  check_tensor((head_major ? cpu_k.size(1) : cpu_k.size(2)) == H,
               "cpu KV head dimension mismatch");
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");

  const int64_t n = head_ids.size(0);
  const int64_t* head_ptr = head_ids.data_ptr<int64_t>();
  const int64_t* frame_ptr = frame_ids.data_ptr<int64_t>();
  const int64_t* slot_ptr = slot_ids.data_ptr<int64_t>();
  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t copied = 0;
  int64_t i = 0;
  while (i < n) {
    const int64_t head_id = head_ptr[i];
    const int64_t frame_id = frame_ptr[i];
    const int64_t slot = slot_ptr[i];
    check_tensor(head_id >= 0 && head_id < H, "head_id out of range");
    check_tensor(frame_id >= 0, "frame_id must be non-negative");
    check_tensor(slot >= 0 && slot < gpu_cache_frames, "slot out of range");

    int64_t run_len = 1;
    while (i + run_len < n) {
      const int64_t next_head_id = head_ptr[i + run_len];
      const int64_t next_frame_id = frame_ptr[i + run_len];
      const int64_t next_slot = slot_ptr[i + run_len];
      check_tensor(next_head_id >= 0 && next_head_id < H, "head_id out of range");
      check_tensor(next_frame_id >= 0, "frame_id must be non-negative");
      check_tensor(next_slot >= 0 && next_slot < gpu_cache_frames, "slot out of range");

      if (next_head_id != head_id) {
        break;
      }
      if (next_frame_id != frame_id + run_len) {
        break;
      }
      if (next_slot != slot + run_len) {
        break;
      }
      const int64_t expected_frame_start = next_frame_id * frame_seqlen;
      if (expected_frame_start >= T) {
        break;
      }
      const int64_t prior_frame_start = (next_frame_id - 1) * frame_seqlen;
      if (prior_frame_start + frame_seqlen > T) {
        break;
      }
      run_len += 1;
    }

    if (run_len > 1) {
      const int64_t frames_copied = copy_frame_run_to_slots_unchecked(
          cpu_k,
          cpu_v,
          gpu_k,
          gpu_v,
          head_id,
          frame_id,
          slot,
          run_len,
          frame_seqlen,
          gpu_cache_frames,
          stream);
      copied += frames_copied;
      i += run_len;
      continue;
    }

    int64_t len = copy_frame_to_slot_unchecked(
        cpu_k,
        cpu_v,
        gpu_k,
        gpu_v,
        head_id,
        frame_id,
        slot,
        frame_seqlen,
        gpu_cache_frames,
        stream);
    if (len > 0) {
      copied += 1;
    }
    i += 1;
  }
  return copied;
}

int64_t kv_sparse_copy_frame_plan_to_slots_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor frame_indices,
    torch::Tensor slot_plan,
    torch::Tensor copy_plan,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  check_tensor(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu_k/gpu_v must be contiguous");
  check_tensor(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned for async HtoD copy");
  check_tensor(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_k.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_v.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be 4D");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(frame_indices.device().is_cpu() && slot_plan.device().is_cpu() && copy_plan.device().is_cpu(),
               "copy plan tensors must be CPU tensors");
  check_tensor(frame_indices.is_contiguous() && slot_plan.is_contiguous() && copy_plan.is_contiguous(),
               "copy plan tensors must be contiguous");
  check_tensor(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  check_tensor(slot_plan.scalar_type() == at::kLong, "slot_plan must be int64");
  check_tensor(copy_plan.scalar_type() == at::kLong, "copy_plan must be int64");
  check_tensor(frame_indices.dim() == 2 && slot_plan.sizes() == frame_indices.sizes() &&
                   copy_plan.sizes() == frame_indices.sizes(),
               "copy plan tensors must have shape [H, selected]");

  const int64_t B = cpu_k.size(0);
  const int64_t H = infer_gpu_slot_head_count(gpu_k, frame_seqlen, gpu_cache_frames);
  const bool head_major = cpu_kv_is_head_major(cpu_k, H);
  const int64_t T = head_major ? cpu_k.size(2) : cpu_k.size(1);
  const int64_t D = cpu_k.size(3);
  (void)B;
  check_tensor(gpu_k.size(0) == B && gpu_k.size(2) == 1 && gpu_k.size(3) == D,
               "gpu KV shape mismatch");
  check_tensor((head_major ? cpu_k.size(1) : cpu_k.size(2)) == H,
               "cpu KV head dimension mismatch");
  check_tensor(frame_indices.size(0) == H, "copy plan head dimension mismatch");
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");

  const int64_t n_selected = frame_indices.size(1);
  const int64_t* frame_ptr = frame_indices.data_ptr<int64_t>();
  const int64_t* slot_ptr = slot_plan.data_ptr<int64_t>();
  const int64_t* copy_ptr = copy_plan.data_ptr<int64_t>();
  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t copied = 0;
  for (int64_t head_id = 0; head_id < H; ++head_id) {
    int64_t col = 0;
    while (col < n_selected) {
      const int64_t offset = head_id * n_selected + col;
      if (copy_ptr[offset] == 0) {
        col += 1;
        continue;
      }
      const int64_t frame_id = frame_ptr[offset];
      const int64_t slot = slot_ptr[offset];
      check_tensor(frame_id >= 0, "frame_id must be non-negative");
      check_tensor(slot >= 0 && slot < gpu_cache_frames, "slot out of range");

      int64_t run_len = 1;
      while (col + run_len < n_selected) {
        const int64_t next_offset = head_id * n_selected + col + run_len;
        if (copy_ptr[next_offset] == 0) {
          break;
        }
        const int64_t next_frame_id = frame_ptr[next_offset];
        const int64_t next_slot = slot_ptr[next_offset];
        check_tensor(next_frame_id >= 0, "frame_id must be non-negative");
        check_tensor(next_slot >= 0 && next_slot < gpu_cache_frames, "slot out of range");
        if (next_frame_id != frame_id + run_len || next_slot != slot + run_len) {
          break;
        }
        const int64_t expected_frame_start = next_frame_id * frame_seqlen;
        if (expected_frame_start >= T) {
          break;
        }
        const int64_t prior_frame_start = (next_frame_id - 1) * frame_seqlen;
        if (prior_frame_start + frame_seqlen > T) {
          break;
        }
        run_len += 1;
      }

      if (run_len > 1) {
        const int64_t frames_copied = copy_frame_run_to_slots_unchecked(
            cpu_k,
            cpu_v,
            gpu_k,
            gpu_v,
            head_id,
            frame_id,
            slot,
            run_len,
            frame_seqlen,
            gpu_cache_frames,
            stream);
        copied += frames_copied;
        col += run_len;
        continue;
      }

      int64_t len = copy_frame_to_slot_unchecked(
          cpu_k,
          cpu_v,
          gpu_k,
          gpu_v,
          head_id,
          frame_id,
          slot,
          frame_seqlen,
          gpu_cache_frames,
          stream);
      if (len > 0) {
        copied += 1;
      }
      col += 1;
    }
  }
  return copied;
}

int64_t kv_sparse_copy_frame_plan_to_slots_batched_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor frame_indices,
    torch::Tensor slot_plan,
    torch::Tensor copy_plan,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  check_tensor(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu_k/gpu_v must be contiguous");
  check_tensor(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned for async HtoD copy");
  check_tensor(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_k.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_v.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be [B,T,H,D]");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(frame_indices.device().is_cpu() && slot_plan.device().is_cpu() && copy_plan.device().is_cpu(),
               "copy plan tensors must be CPU tensors");
  check_tensor(frame_indices.is_contiguous() && slot_plan.is_contiguous() && copy_plan.is_contiguous(),
               "copy plan tensors must be contiguous");
  check_tensor(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  check_tensor(slot_plan.scalar_type() == at::kLong, "slot_plan must be int64");
  check_tensor(copy_plan.scalar_type() == at::kLong, "copy_plan must be int64");
  check_tensor(frame_indices.dim() == 2 && slot_plan.sizes() == frame_indices.sizes() &&
                   copy_plan.sizes() == frame_indices.sizes(),
               "copy plan tensors must have shape [H, selected]");

  KvMemcpy3DBatchAsyncFn batch_fn = resolve_memcpy_3d_batch_async();
  if (batch_fn == nullptr) {
    return 0;
  }

  const int64_t B = cpu_k.size(0);
  const int64_t H = frame_indices.size(0);
  const bool head_major = cpu_kv_is_head_major(cpu_k, H);
  const int64_t T = head_major ? cpu_k.size(2) : cpu_k.size(1);
  const int64_t D = cpu_k.size(3);
  check_tensor(gpu_k.size(0) == B && gpu_k.size(2) == 1 && gpu_k.size(3) == D,
               "gpu KV shape mismatch");
  check_tensor((head_major ? cpu_k.size(1) : cpu_k.size(2)) == H, "copy plan head dimension mismatch");
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");

  const int64_t n_selected = frame_indices.size(1);
  const int64_t* frame_ptr = frame_indices.data_ptr<int64_t>();
  const int64_t* slot_ptr = slot_plan.data_ptr<int64_t>();
  const int64_t* copy_ptr = copy_plan.data_ptr<int64_t>();
  const size_t elem_size = static_cast<size_t>(cpu_k.element_size());
  const size_t width_bytes = static_cast<size_t>(D) * elem_size;
  const size_t src_pitch = static_cast<size_t>(H * D) * elem_size;
  const size_t dst_pitch = width_bytes;
  const int64_t src_batch_stride = T * H * D;
  const int64_t gpu_cache_tokens = gpu_k.size(1);
  const char* cpu_k_ptr = static_cast<const char*>(cpu_k.data_ptr());
  const char* cpu_v_ptr = static_cast<const char*>(cpu_v.data_ptr());
  char* gpu_k_ptr = static_cast<char*>(gpu_k.data_ptr());
  char* gpu_v_ptr = static_cast<char*>(gpu_v.data_ptr());

  std::vector<KvMemcpy3DBatchOp> ops;
  ops.reserve(static_cast<size_t>(B * frame_indices.numel() * 2));
  int64_t copied = 0;
  for (int64_t head_id = 0; head_id < H; ++head_id) {
    int64_t col = 0;
    while (col < n_selected) {
      const int64_t offset = head_id * n_selected + col;
      if (copy_ptr[offset] == 0) {
        ++col;
        continue;
      }
      const int64_t frame_id = frame_ptr[offset];
      const int64_t slot = slot_ptr[offset];
      check_tensor(frame_id >= 0, "frame_id must be non-negative");
      check_tensor(slot >= 0 && slot < gpu_cache_frames, "slot out of range");

      int64_t run_len = 1;
      while (col + run_len < n_selected) {
        const int64_t next_offset = offset + run_len;
        if (copy_ptr[next_offset] == 0 ||
            frame_ptr[next_offset] != frame_id + run_len ||
            slot_ptr[next_offset] != slot + run_len) {
          break;
        }
        if ((frame_id + run_len) * frame_seqlen >= T || slot + run_len >= gpu_cache_frames) {
          break;
        }
        ++run_len;
      }

      const int64_t start = frame_id * frame_seqlen;
      const int64_t copy_tokens = std::min<int64_t>(run_len * frame_seqlen, T - start);
      if (copy_tokens <= 0) {
        col += run_len;
        continue;
      }
      for (int64_t b = 0; b < B; ++b) {
        const int64_t src_elem = head_major
            ? b * src_batch_stride + head_id * T * D + start * D
            : b * src_batch_stride + start * H * D + head_id * D;
        const int64_t dst_token = (head_id * gpu_cache_frames + slot) * frame_seqlen;
        const int64_t dst_elem = (b * gpu_cache_tokens + dst_token) * D;
        const size_t copy_width = head_major
            ? static_cast<size_t>(copy_tokens * D) * elem_size
            : width_bytes;
        const size_t copy_height = head_major ? 1 : static_cast<size_t>(copy_tokens);
        const size_t copy_src_pitch = head_major ? copy_width : src_pitch;
        const size_t copy_dst_pitch = head_major ? copy_width : dst_pitch;
        append_batch_2d_copy(
            ops,
            cpu_k_ptr + src_elem * static_cast<int64_t>(elem_size),
            copy_src_pitch,
            gpu_k_ptr + dst_elem * static_cast<int64_t>(elem_size),
            copy_dst_pitch,
            copy_width,
            copy_height,
            CU_MEM_LOCATION_TYPE_HOST,
            0,
            CU_MEM_LOCATION_TYPE_DEVICE,
            gpu_k.get_device());
        append_batch_2d_copy(
            ops,
            cpu_v_ptr + src_elem * static_cast<int64_t>(elem_size),
            copy_src_pitch,
            gpu_v_ptr + dst_elem * static_cast<int64_t>(elem_size),
            copy_dst_pitch,
            copy_width,
            copy_height,
            CU_MEM_LOCATION_TYPE_HOST,
            0,
            CU_MEM_LOCATION_TYPE_DEVICE,
            gpu_k.get_device());
      }
      copied += run_len;
      col += run_len;
    }
  }
  if (ops.empty() || copied <= 0) {
    return 0;
  }

  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  // CUDAGuard may see the worker thread's default device as already selected
  // even though that thread has no current primary context yet.  The driver
  // batch API requires a current context for the stream passed below.
  static thread_local int initialized_device = -1;
  if (initialized_device != gpu_k.get_device()) {
    cudaError_t runtime_result = cudaSetDevice(gpu_k.get_device());
    TORCH_CHECK(
        runtime_result == cudaSuccess,
        "cudaSetDevice before batched H2D failed: ",
        cudaGetErrorString(runtime_result));
    runtime_result = cudaFree(nullptr);
    TORCH_CHECK(
        runtime_result == cudaSuccess,
        "CUDA context initialization before batched H2D failed: ",
        cudaGetErrorString(runtime_result));
    initialized_device = gpu_k.get_device();
  }
  cudaStream_t runtime_stream = at::cuda::getCurrentCUDAStream();
  if (runtime_stream == nullptr) {
    return 0;
  }
  size_t fail_idx = std::numeric_limits<size_t>::max();
  CUresult result = batch_fn(
      ops.size(),
      ops.data(),
      &fail_idx,
      0,
      reinterpret_cast<CUstream>(runtime_stream));
  TORCH_CHECK(
      result == CUDA_SUCCESS,
      "cuMemcpy3DBatchAsync failed with CUresult=",
      static_cast<int>(result),
      ", failing descriptor=",
      fail_idx,
      ", descriptor_count=",
      ops.size());
  return copied;
}

int64_t kv_sparse_copy_frames_to_staging_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor staging_k,
    torch::Tensor staging_v,
    torch::Tensor head_ids,
    torch::Tensor frame_ids,
    int64_t frame_seqlen) {
  check_tensor(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  check_tensor(staging_k.is_cuda() && staging_v.is_cuda(), "staging_k/staging_v must be CUDA tensors");
  check_tensor(staging_v.device() == staging_k.device(), "staging K/V device mismatch");
  check_tensor(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  check_tensor(staging_k.is_contiguous() && staging_v.is_contiguous(),
               "staging_k/staging_v must be contiguous");
  check_tensor(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned for async HtoD copy");
  check_tensor(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  check_tensor(cpu_k.scalar_type() == staging_k.scalar_type(), "cpu/staging dtype mismatch");
  check_tensor(cpu_k.scalar_type() == staging_v.scalar_type(), "cpu/staging dtype mismatch");
  check_tensor(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be [B,T,H,D]");
  check_tensor(staging_k.dim() == 4 && staging_v.sizes() == staging_k.sizes(),
               "staging KV must be [B*rows,Tstage,1,D]");
  check_tensor(head_ids.device().is_cpu() && frame_ids.device().is_cpu(),
               "head_ids/frame_ids must be CPU tensors");
  check_tensor(head_ids.is_contiguous() && frame_ids.is_contiguous(),
               "head_ids/frame_ids must be contiguous");
  check_tensor(head_ids.scalar_type() == at::kLong, "head_ids must be int64");
  check_tensor(frame_ids.scalar_type() == at::kLong, "frame_ids must be int64");
  check_tensor(head_ids.dim() == 1 && frame_ids.dim() == 2, "head_ids must be 1D and frame_ids 2D");
  check_tensor(frame_ids.size(0) == head_ids.size(0), "head/frame row count mismatch");
  check_tensor(frame_seqlen > 0, "frame_seqlen must be positive");

  const int64_t B = cpu_k.size(0);
  const int64_t T = cpu_k.size(1);
  const int64_t H = cpu_k.size(2);
  const int64_t D = cpu_k.size(3);
  const int64_t rows = head_ids.size(0);
  const int64_t max_selected = frame_ids.size(1);
  const int64_t token_count = max_selected * frame_seqlen;
  check_tensor(staging_k.size(0) == B * rows, "staging batch/row dimension mismatch");
  check_tensor(staging_k.size(1) >= token_count, "staging token dimension too small");
  check_tensor(staging_k.size(2) == 1 && staging_k.size(3) == D, "staging KV shape mismatch");
  if (B <= 0 || rows <= 0 || max_selected <= 0 || token_count <= 0) {
    return 0;
  }

  const int64_t* head_ptr = head_ids.data_ptr<int64_t>();
  const int64_t* frame_ptr = frame_ids.data_ptr<int64_t>();
  const size_t elem_size = static_cast<size_t>(cpu_k.element_size());
  const size_t width_bytes = static_cast<size_t>(D) * elem_size;
  const size_t src_pitch = static_cast<size_t>(H * D) * elem_size;
  const size_t dst_pitch = static_cast<size_t>(D) * elem_size;
  const int64_t src_batch_stride = T * H * D;
  const int64_t dst_row_stride = staging_k.size(1) * D;
  const char* cpu_k_ptr = static_cast<const char*>(cpu_k.data_ptr());
  const char* cpu_v_ptr = static_cast<const char*>(cpu_v.data_ptr());
  char* staging_k_ptr = static_cast<char*>(staging_k.data_ptr());
  char* staging_v_ptr = static_cast<char*>(staging_v.data_ptr());

  const c10::cuda::CUDAGuard device_guard(staging_k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t copied = 0;
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t head_id = head_ptr[row];
    check_tensor(head_id >= 0 && head_id < H, "head_id out of range");
    int64_t col = 0;
    while (col < max_selected) {
      const int64_t frame_id = frame_ptr[row * max_selected + col];
      if (frame_id < 0) {
        ++col;
        continue;
      }
      check_tensor(frame_id >= 0, "frame_id must be non-negative");
      const int64_t start = frame_id * frame_seqlen;
      if (start >= T) {
        ++col;
        continue;
      }

      int64_t run_len = 1;
      while (col + run_len < max_selected) {
        const int64_t next_frame_id = frame_ptr[row * max_selected + col + run_len];
        if (next_frame_id != frame_id + run_len) {
          break;
        }
        const int64_t next_start = next_frame_id * frame_seqlen;
        if (next_start >= T) {
          break;
        }
        const int64_t prior_start = (next_frame_id - 1) * frame_seqlen;
        if (prior_start + frame_seqlen > T) {
          break;
        }
        run_len += 1;
      }

      const int64_t total_copy_tokens = std::min<int64_t>(run_len * frame_seqlen, T - start);
      if (total_copy_tokens <= 0) {
        col += run_len;
        continue;
      }
      const int64_t dst_token_start = col * frame_seqlen;
      for (int64_t b = 0; b < B; ++b) {
        const int64_t src_elem = b * src_batch_stride + start * H * D + head_id * D;
        const int64_t dst_row = b * rows + row;
        const int64_t dst_elem = dst_row * dst_row_stride + dst_token_start * D;
        const char* src_k = cpu_k_ptr + src_elem * static_cast<int64_t>(elem_size);
        const char* src_v = cpu_v_ptr + src_elem * static_cast<int64_t>(elem_size);
        char* dst_k = staging_k_ptr + dst_elem * static_cast<int64_t>(elem_size);
        char* dst_v = staging_v_ptr + dst_elem * static_cast<int64_t>(elem_size);
        cudaError_t err = cudaMemcpy2DAsync(
            dst_k,
            dst_pitch,
            src_k,
            src_pitch,
            width_bytes,
            static_cast<size_t>(total_copy_tokens),
            cudaMemcpyHostToDevice,
            stream);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync staging K failed: ", cudaGetErrorString(err));
        err = cudaMemcpy2DAsync(
            dst_v,
            dst_pitch,
            src_v,
            src_pitch,
            width_bytes,
            static_cast<size_t>(total_copy_tokens),
            cudaMemcpyHostToDevice,
            stream);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync staging V failed: ", cudaGetErrorString(err));
      }
      copied += B * run_len;
      col += run_len;
    }
  }
  return copied;
}

int64_t kv_sparse_copy_frames_to_compact_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor compact_k,
    torch::Tensor compact_v,
    torch::Tensor head_ids,
    torch::Tensor frame_ids,
    torch::Tensor cu_seqlens,
    int64_t frame_seqlen,
    bool cpu_head_major) {
  check_tensor(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  check_tensor(compact_k.is_cuda() && compact_v.is_cuda(), "compact_k/compact_v must be CUDA tensors");
  check_tensor(compact_v.device() == compact_k.device(), "compact K/V device mismatch");
  check_tensor(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  check_tensor(compact_k.is_contiguous() && compact_v.is_contiguous(), "compact K/V must be contiguous");
  check_tensor(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned for async HtoD copy");
  check_tensor(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  check_tensor(cpu_k.scalar_type() == compact_k.scalar_type(), "cpu/compact dtype mismatch");
  check_tensor(cpu_k.scalar_type() == compact_v.scalar_type(), "cpu/compact dtype mismatch");
  check_tensor(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be 4D");
  check_tensor(compact_k.dim() == 3 && compact_v.sizes() == compact_k.sizes(),
               "compact KV must be [packed_tokens,1,D]");
  check_tensor(head_ids.device().is_cpu() && frame_ids.device().is_cpu() && cu_seqlens.device().is_cpu(),
               "compact copy plans must be CPU tensors");
  check_tensor(head_ids.is_contiguous() && frame_ids.is_contiguous() && cu_seqlens.is_contiguous(),
               "compact copy plans must be contiguous");
  check_tensor(head_ids.scalar_type() == at::kLong && frame_ids.scalar_type() == at::kLong &&
                   cu_seqlens.scalar_type() == at::kLong,
               "compact copy plans must be int64");
  check_tensor(head_ids.dim() == 1 && frame_ids.dim() == 2, "head_ids must be 1D and frame_ids 2D");
  check_tensor(frame_ids.size(0) == head_ids.size(0), "head/frame row count mismatch");
  check_tensor(frame_seqlen > 0, "frame_seqlen must be positive");

  const int64_t B = cpu_k.size(0);
  const int64_t H = cpu_head_major ? cpu_k.size(1) : cpu_k.size(2);
  const int64_t T = cpu_head_major ? cpu_k.size(2) : cpu_k.size(1);
  const int64_t D = cpu_k.size(3);
  const int64_t rows = head_ids.size(0);
  const int64_t max_selected = frame_ids.size(1);
  check_tensor(H > 0 && T >= 0 && D > 0, "invalid CPU KV shape");
  check_tensor(compact_k.size(1) == 1 && compact_k.size(2) == D, "compact KV shape mismatch");
  check_tensor(cu_seqlens.dim() == 1 && cu_seqlens.size(0) == B * rows + 1,
               "cu_seqlens shape mismatch");
  if (B <= 0 || rows <= 0 || max_selected <= 0 || compact_k.size(0) <= 0) {
    return 0;
  }

  const int64_t* head_ptr = head_ids.data_ptr<int64_t>();
  const int64_t* frame_ptr = frame_ids.data_ptr<int64_t>();
  const int64_t* cu_ptr = cu_seqlens.data_ptr<int64_t>();
  check_tensor(cu_ptr[B * rows] <= compact_k.size(0), "compact KV token capacity too small");

  const size_t elem_size = static_cast<size_t>(cpu_k.element_size());
  const size_t width_bytes = static_cast<size_t>(D) * elem_size;
  const size_t token_major_src_pitch = static_cast<size_t>(H * D) * elem_size;
  const size_t dst_pitch = width_bytes;
  const int64_t src_batch_stride = T * H * D;
  const char* cpu_k_ptr = static_cast<const char*>(cpu_k.data_ptr());
  const char* cpu_v_ptr = static_cast<const char*>(cpu_v.data_ptr());
  char* compact_k_ptr = static_cast<char*>(compact_k.data_ptr());
  char* compact_v_ptr = static_cast<char*>(compact_v.data_ptr());

  const c10::cuda::CUDAGuard device_guard(compact_k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  KvMemcpy3DBatchAsyncFn batch_fn = resolve_memcpy_3d_batch_async();
  const bool use_batch = batch_fn != nullptr && stream != nullptr;
  std::vector<KvMemcpy3DBatchOp> ops;
  if (use_batch) {
    ops.reserve(static_cast<size_t>(B * rows * max_selected * 2));
  }

  int64_t copied = 0;
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t head_id = head_ptr[row];
    check_tensor(head_id >= 0 && head_id < H, "head_id out of range");
    int64_t col = 0;
    while (col < max_selected) {
      const int64_t frame_id = frame_ptr[row * max_selected + col];
      if (frame_id < 0) {
        ++col;
        continue;
      }
      check_tensor(frame_id >= 0, "frame_id must be non-negative");
      const int64_t start = frame_id * frame_seqlen;
      if (start >= T) {
        ++col;
        continue;
      }

      int64_t run_len = 1;
      while (col + run_len < max_selected) {
        const int64_t next_frame_id = frame_ptr[row * max_selected + col + run_len];
        if (next_frame_id != frame_id + run_len) {
          break;
        }
        const int64_t next_start = next_frame_id * frame_seqlen;
        if (next_start >= T) {
          break;
        }
        const int64_t prior_start = (next_frame_id - 1) * frame_seqlen;
        if (prior_start + frame_seqlen > T) {
          break;
        }
        ++run_len;
      }

      const int64_t total_copy_tokens = std::min<int64_t>(run_len * frame_seqlen, T - start);
      if (total_copy_tokens <= 0) {
        col += run_len;
        continue;
      }

      for (int64_t b = 0; b < B; ++b) {
        const int64_t packed_row = b * rows + row;
        const int64_t dst_token_start = cu_ptr[packed_row] + col * frame_seqlen;
        check_tensor(dst_token_start >= cu_ptr[packed_row] &&
                         dst_token_start + total_copy_tokens <= cu_ptr[packed_row + 1],
                     "compact destination span out of range");
        const int64_t src_elem = cpu_head_major
            ? b * src_batch_stride + head_id * T * D + start * D
            : b * src_batch_stride + start * H * D + head_id * D;
        const int64_t dst_elem = dst_token_start * D;
        const size_t copy_width = cpu_head_major
            ? static_cast<size_t>(total_copy_tokens * D) * elem_size
            : width_bytes;
        const size_t copy_height = cpu_head_major ? 1 : static_cast<size_t>(total_copy_tokens);
        const size_t copy_src_pitch = cpu_head_major ? copy_width : token_major_src_pitch;
        const size_t copy_dst_pitch = cpu_head_major ? copy_width : dst_pitch;
        if (use_batch) {
          append_batch_2d_copy(
              ops,
              cpu_k_ptr + src_elem * static_cast<int64_t>(elem_size),
              copy_src_pitch,
              compact_k_ptr + dst_elem * static_cast<int64_t>(elem_size),
              copy_dst_pitch,
              copy_width,
              copy_height,
              CU_MEM_LOCATION_TYPE_HOST,
              0,
              CU_MEM_LOCATION_TYPE_DEVICE,
              compact_k.get_device());
          append_batch_2d_copy(
              ops,
              cpu_v_ptr + src_elem * static_cast<int64_t>(elem_size),
              copy_src_pitch,
              compact_v_ptr + dst_elem * static_cast<int64_t>(elem_size),
              copy_dst_pitch,
              copy_width,
              copy_height,
              CU_MEM_LOCATION_TYPE_HOST,
              0,
              CU_MEM_LOCATION_TYPE_DEVICE,
              compact_k.get_device());
        } else {
          cudaError_t err = cudaMemcpy2DAsync(
              compact_k_ptr + dst_elem * static_cast<int64_t>(elem_size),
              copy_dst_pitch,
              cpu_k_ptr + src_elem * static_cast<int64_t>(elem_size),
              copy_src_pitch,
              copy_width,
              copy_height,
              cudaMemcpyHostToDevice,
              stream);
          TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync compact K failed: ", cudaGetErrorString(err));
          err = cudaMemcpy2DAsync(
              compact_v_ptr + dst_elem * static_cast<int64_t>(elem_size),
              copy_dst_pitch,
              cpu_v_ptr + src_elem * static_cast<int64_t>(elem_size),
              copy_src_pitch,
              copy_width,
              copy_height,
              cudaMemcpyHostToDevice,
              stream);
          TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync compact V failed: ", cudaGetErrorString(err));
        }
      }
      copied += B * run_len;
      col += run_len;
    }
  }
  if (use_batch && !ops.empty()) {
    ensure_current_cuda_context_for_driver_batch(compact_k.get_device(), "compact batched H2D");
    size_t fail_idx = std::numeric_limits<size_t>::max();
    CUresult result = batch_fn(
        ops.size(),
        ops.data(),
        &fail_idx,
        0,
        reinterpret_cast<CUstream>(stream));
    TORCH_CHECK(
        result == CUDA_SUCCESS,
        "cuMemcpy3DBatchAsync compact H2D failed with CUresult=",
        static_cast<int>(result),
        ", failing descriptor=",
        fail_idx,
        ", descriptor_count=",
        ops.size());
  }
  return copied;
}

int64_t kv_sparse_copy_frame_plan_to_slots_zero_copy_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor frame_indices,
    torch::Tensor slot_plan,
    torch::Tensor copy_plan,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t copy_count) {
  check_tensor(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(frame_indices.is_cuda() && slot_plan.is_cuda() && copy_plan.is_cuda(),
               "zero-copy dense plan tensors must be CUDA tensors");
  check_tensor(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu_k/gpu_v must be contiguous");
  check_tensor(frame_indices.is_contiguous() && slot_plan.is_contiguous() && copy_plan.is_contiguous(),
               "zero-copy dense plan tensors must be contiguous");
  check_tensor(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned");
  check_tensor(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_k.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_v.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  check_tensor(slot_plan.scalar_type() == at::kLong, "slot_plan must be int64");
  check_tensor(copy_plan.scalar_type() == at::kLong, "copy_plan must be int64");
  check_tensor(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be [B,T,H,D]");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(frame_indices.dim() == 2 && slot_plan.sizes() == frame_indices.sizes() &&
                   copy_plan.sizes() == frame_indices.sizes(),
               "zero-copy dense plan tensors must have shape [H, selected]");
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");

  if (copy_count <= 0) {
    return 0;
  }
  const int64_t B = cpu_k.size(0);
  const int64_t T = cpu_k.size(1);
  const int64_t H = cpu_k.size(2);
  const int64_t D = cpu_k.size(3);
  const int64_t n_selected = frame_indices.size(1);
  const int64_t pair_count = frame_indices.numel();
  check_tensor(gpu_k.size(0) == B && gpu_k.size(2) == 1 && gpu_k.size(3) == D,
               "gpu KV shape mismatch");
  check_tensor(frame_indices.size(0) == H, "copy plan head dimension mismatch");
  check_tensor(frame_indices.device() == gpu_k.device(), "frame_indices/gpu device mismatch");
  check_tensor(slot_plan.device() == gpu_k.device() && copy_plan.device() == gpu_k.device(),
               "copy plan/gpu device mismatch");
  if (B <= 0 || T <= 0 || H <= 0 || D <= 0 || n_selected <= 0 || pair_count <= 0) {
    return 0;
  }

  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  void* cpu_k_device_ptr = nullptr;
  void* cpu_v_device_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&cpu_k_device_ptr, cpu_k.data_ptr(), 0);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  err = cudaHostGetDevicePointer(&cpu_v_device_ptr, cpu_v.data_ptr(), 0);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }

  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(pair_count, 65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, cpu_k.scalar_type(),
                                  "kv_sparse_copy_frame_plan_to_slots_zero_copy_cuda", [&] {
    kv_sparse_copy_frame_plan_to_slots_zero_copy_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        static_cast<const scalar_t*>(cpu_k_device_ptr),
        static_cast<const scalar_t*>(cpu_v_device_ptr),
        gpu_k.data_ptr<scalar_t>(),
        gpu_v.data_ptr<scalar_t>(),
        frame_indices.data_ptr<int64_t>(),
        slot_plan.data_ptr<int64_t>(),
        copy_plan.data_ptr<int64_t>(),
        pair_count,
        n_selected,
        B,
        T,
        H,
        D,
        gpu_k.size(1),
        frame_seqlen,
        gpu_cache_frames);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return copy_count;
}

int64_t kv_sparse_copy_frames_to_slots_zero_copy_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor head_ids,
    torch::Tensor frame_ids,
    torch::Tensor slot_ids,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  check_tensor(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(head_ids.is_cuda() && frame_ids.is_cuda() && slot_ids.is_cuda(),
               "zero-copy copy plan tensors must be CUDA tensors");
  check_tensor(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu_k/gpu_v must be contiguous");
  check_tensor(head_ids.is_contiguous() && frame_ids.is_contiguous() && slot_ids.is_contiguous(),
               "zero-copy copy plan tensors must be contiguous");
  check_tensor(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned");
  check_tensor(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_k.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(cpu_k.scalar_type() == gpu_v.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(head_ids.scalar_type() == at::kLong, "head_ids must be int64");
  check_tensor(frame_ids.scalar_type() == at::kLong, "frame_ids must be int64");
  check_tensor(slot_ids.scalar_type() == at::kLong, "slot_ids must be int64");
  check_tensor(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be [B,T,H,D]");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(head_ids.dim() == 1 && frame_ids.sizes() == head_ids.sizes() && slot_ids.sizes() == head_ids.sizes(),
               "zero-copy copy plan tensors must have shape [N]");
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");

  const int64_t n = head_ids.size(0);
  if (n <= 0) {
    return 0;
  }
  const int64_t B = cpu_k.size(0);
  const int64_t T = cpu_k.size(1);
  const int64_t H = cpu_k.size(2);
  const int64_t D = cpu_k.size(3);
  check_tensor(gpu_k.size(0) == B && gpu_k.size(2) == 1 && gpu_k.size(3) == D,
               "gpu KV shape mismatch");
  check_tensor(head_ids.device() == gpu_k.device(), "copy plan/gpu device mismatch");

  void* cpu_k_device_ptr = nullptr;
  void* cpu_v_device_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&cpu_k_device_ptr, cpu_k.data_ptr(), 0);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  err = cudaHostGetDevicePointer(&cpu_v_device_ptr, cpu_v.data_ptr(), 0);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }

  const int64_t total_elements = n * B * frame_seqlen * D;
  if (total_elements <= 0) {
    return 0;
  }
  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((total_elements + threads - 1) / threads, 65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, cpu_k.scalar_type(),
                                  "kv_sparse_copy_frames_to_slots_zero_copy_cuda", [&] {
    kv_sparse_copy_frames_to_slots_zero_copy_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        static_cast<const scalar_t*>(cpu_k_device_ptr),
        static_cast<const scalar_t*>(cpu_v_device_ptr),
        gpu_k.data_ptr<scalar_t>(),
        gpu_v.data_ptr<scalar_t>(),
        head_ids.data_ptr<int64_t>(),
        frame_ids.data_ptr<int64_t>(),
        slot_ids.data_ptr<int64_t>(),
        n,
        B,
        T,
        H,
        D,
        gpu_k.size(1),
        frame_seqlen,
        gpu_cache_frames,
        total_elements);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return n;
}

int64_t kv_sparse_scatter_staging_to_slots_cuda(
    torch::Tensor staging_k,
    torch::Tensor staging_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor head_ids,
    torch::Tensor slot_ids,
    torch::Tensor src_rows,
    torch::Tensor src_cols,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  check_tensor(staging_k.is_cuda() && staging_v.is_cuda(), "staging_k/staging_v must be CUDA tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(head_ids.is_cuda() && slot_ids.is_cuda() && src_rows.is_cuda() && src_cols.is_cuda(),
               "scatter plan tensors must be CUDA tensors");
  check_tensor(staging_k.is_contiguous() && staging_v.is_contiguous(), "staging tensors must be contiguous");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu tensors must be contiguous");
  check_tensor(head_ids.is_contiguous() && slot_ids.is_contiguous() &&
                   src_rows.is_contiguous() && src_cols.is_contiguous(),
               "scatter plan tensors must be contiguous");
  check_tensor(staging_k.scalar_type() == staging_v.scalar_type(), "staging dtype mismatch");
  check_tensor(staging_k.scalar_type() == gpu_k.scalar_type(), "staging/gpu dtype mismatch");
  check_tensor(staging_k.scalar_type() == gpu_v.scalar_type(), "staging/gpu dtype mismatch");
  check_tensor(head_ids.scalar_type() == at::kLong && slot_ids.scalar_type() == at::kLong &&
                   src_rows.scalar_type() == at::kLong && src_cols.scalar_type() == at::kLong,
               "scatter plan tensors must be int64");
  check_tensor(staging_k.dim() == 4 && staging_v.sizes() == staging_k.sizes(),
               "staging KV must be [B*rows,Tstage,1,D]");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(staging_k.size(2) == 1 && gpu_k.size(2) == 1, "KV tensors must have singleton head dim");
  check_tensor(head_ids.dim() == 1 && slot_ids.sizes() == head_ids.sizes() &&
                   src_rows.sizes() == head_ids.sizes() && src_cols.sizes() == head_ids.sizes(),
               "scatter plan tensors must have shape [N]");
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");
  check_tensor(staging_k.device() == gpu_k.device(), "staging/gpu device mismatch");
  check_tensor(staging_v.device() == gpu_v.device(), "staging/gpu device mismatch");
  check_tensor(head_ids.device() == gpu_k.device(), "scatter plan/gpu device mismatch");

  const int64_t n = head_ids.size(0);
  if (n <= 0) {
    return 0;
  }
  const int64_t B = gpu_k.size(0);
  const int64_t D = gpu_k.size(3);
  check_tensor(B > 0 && D > 0, "invalid gpu KV shape");
  check_tensor(staging_k.size(0) % B == 0, "staging rows must be divisible by batch size");
  const int64_t active_rows = staging_k.size(0) / B;
  const int64_t staging_token_count = staging_k.size(1);
  check_tensor(active_rows > 0 && staging_token_count >= frame_seqlen, "invalid staging shape");
  check_tensor(staging_k.size(3) == D, "staging/gpu head dim mismatch");
  const int64_t gpu_total_tokens = gpu_k.size(1);
  check_tensor(gpu_total_tokens >= gpu_cache_frames * frame_seqlen,
               "gpu slot cache is smaller than one head cache");

  const int64_t total_elements = n * B * frame_seqlen * D;
  if (total_elements <= 0) {
    return 0;
  }
  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((total_elements + threads - 1) / threads, 65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, staging_k.scalar_type(),
                                  "kv_sparse_scatter_staging_to_slots_cuda", [&] {
    kv_sparse_scatter_staging_to_slots_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        staging_k.data_ptr<scalar_t>(),
        staging_v.data_ptr<scalar_t>(),
        gpu_k.data_ptr<scalar_t>(),
        gpu_v.data_ptr<scalar_t>(),
        head_ids.data_ptr<int64_t>(),
        slot_ids.data_ptr<int64_t>(),
        src_rows.data_ptr<int64_t>(),
        src_cols.data_ptr<int64_t>(),
        n,
        B,
        active_rows,
        staging_token_count,
        D,
        gpu_total_tokens,
        frame_seqlen,
        gpu_cache_frames,
        total_elements);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return n;
}


int64_t kv_sparse_copy_live_runs_to_slots_cuda(
    torch::Tensor live_k,
    torch::Tensor live_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor head_ids,
    torch::Tensor token_starts,
    torch::Tensor slot_ids,
    torch::Tensor token_lens,
    int64_t frame_seqlen,
    int64_t gpu_live_temp_frames,
    int64_t dst_token_base) {
  check_tensor(live_k.is_cuda() && live_v.is_cuda(), "live_k/live_v must be CUDA tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(live_k.is_contiguous() && live_v.is_contiguous(), "live_k/live_v must be contiguous");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu_k/gpu_v must be contiguous");
  check_tensor(live_k.scalar_type() == live_v.scalar_type(), "live KV dtype mismatch");
  check_tensor(live_k.scalar_type() == gpu_k.scalar_type(), "live/gpu dtype mismatch");
  check_tensor(live_k.scalar_type() == gpu_v.scalar_type(), "live/gpu dtype mismatch");
  check_tensor(live_k.dim() == 4 && live_v.sizes() == live_k.sizes(), "live KV must be [B,T,H,D]");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(gpu_k.size(0) == live_k.size(0) && gpu_k.size(2) == 1 && gpu_k.size(3) == live_k.size(3),
               "gpu KV shape mismatch");
  check_tensor(head_ids.device().is_cpu() && token_starts.device().is_cpu() &&
                   slot_ids.device().is_cpu() && token_lens.device().is_cpu(),
               "copy plan tensors must be CPU tensors");
  check_tensor(head_ids.is_contiguous() && token_starts.is_contiguous() &&
                   slot_ids.is_contiguous() && token_lens.is_contiguous(),
               "copy plan tensors must be contiguous");
  check_tensor(head_ids.scalar_type() == at::kLong, "head_ids must be int64");
  check_tensor(token_starts.scalar_type() == at::kLong, "token_starts must be int64");
  check_tensor(slot_ids.scalar_type() == at::kLong, "slot_ids must be int64");
  check_tensor(token_lens.scalar_type() == at::kLong, "token_lens must be int64");
  check_tensor(head_ids.dim() == 1 && token_starts.sizes() == head_ids.sizes() &&
                   slot_ids.sizes() == head_ids.sizes() && token_lens.sizes() == head_ids.sizes(),
               "copy plan tensors must have shape [N]");
  check_tensor(frame_seqlen > 0, "frame_seqlen must be positive");
  check_tensor(gpu_live_temp_frames > 0, "gpu_live_temp_frames must be positive");
  check_tensor(dst_token_base >= 0, "dst_token_base must be non-negative");
  check_tensor(live_k.device() == gpu_k.device(), "live_k and gpu_k must be on same CUDA device");
  check_tensor(live_v.device() == gpu_v.device(), "live_v and gpu_v must be on same CUDA device");

  const int64_t n = head_ids.size(0);
  if (n <= 0) {
    return 0;
  }

  const int64_t* head_ptr = head_ids.data_ptr<int64_t>();
  const int64_t* token_start_ptr = token_starts.data_ptr<int64_t>();
  const int64_t* slot_ptr = slot_ids.data_ptr<int64_t>();
  const int64_t* token_len_ptr = token_lens.data_ptr<int64_t>();

  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t copied = 0;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t token_count = copy_live_run_to_temp_slots_unchecked(
        live_k,
        live_v,
        gpu_k,
        gpu_v,
        head_ptr[i],
        token_start_ptr[i],
        slot_ptr[i],
        token_len_ptr[i],
        frame_seqlen,
        gpu_live_temp_frames,
        dst_token_base,
        stream);
    if (token_count > 0) {
      copied += 1;
    }
  }
  return copied;
}

int64_t kv_sparse_copy_kv_span_to_cpu_cuda(
    torch::Tensor gpu_k_span,
    torch::Tensor gpu_v_span,
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    int64_t start_token) {
  check_tensor(gpu_k_span.is_cuda() && gpu_v_span.is_cuda(), "gpu_k_span/gpu_v_span must be CUDA tensors");
  check_tensor(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  check_tensor(gpu_k_span.is_contiguous() && gpu_v_span.is_contiguous(), "gpu_k_span/gpu_v_span must be contiguous");
  check_tensor(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  check_tensor(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned");
  check_tensor(gpu_k_span.scalar_type() == gpu_v_span.scalar_type(), "gpu span dtype mismatch");
  check_tensor(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  check_tensor(gpu_k_span.scalar_type() == cpu_k.scalar_type(), "cpu/gpu dtype mismatch");
  check_tensor(gpu_k_span.dim() == 4 && gpu_v_span.sizes() == gpu_k_span.sizes(),
               "gpu KV span must be [B,T,H,D]");
  check_tensor(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be 4D");

  const int64_t B = gpu_k_span.size(0);
  const int64_t span_tokens = gpu_k_span.size(1);
  const int64_t H = gpu_k_span.size(2);
  const int64_t D = gpu_k_span.size(3);
  const bool head_major = cpu_kv_is_head_major(cpu_k, H);
  const int64_t cpu_tokens = head_major ? cpu_k.size(2) : cpu_k.size(1);
  check_tensor(cpu_k.size(0) == B && (head_major ? cpu_k.size(1) : cpu_k.size(2)) == H && cpu_k.size(3) == D,
               "cpu KV shape mismatch");
  check_tensor(start_token >= 0, "start_token must be non-negative");
  check_tensor(start_token + span_tokens <= cpu_tokens, "destination token span out of range");
  if (span_tokens <= 0 || B <= 0) {
    return 0;
  }

  const size_t elem_size = static_cast<size_t>(gpu_k_span.element_size());
  const size_t width_bytes = static_cast<size_t>(span_tokens * H * D) * elem_size;
  const size_t src_pitch = width_bytes;
  const size_t dst_pitch = static_cast<size_t>(cpu_tokens * H * D) * elem_size;
  const int64_t dst_token_offset_elems = start_token * H * D;

  const char* gpu_k_ptr = static_cast<const char*>(gpu_k_span.data_ptr());
  const char* gpu_v_ptr = static_cast<const char*>(gpu_v_span.data_ptr());
  char* cpu_k_ptr = static_cast<char*>(cpu_k.data_ptr());
  char* cpu_v_ptr = static_cast<char*>(cpu_v.data_ptr());
  if (!head_major) {
    cpu_k_ptr += dst_token_offset_elems * static_cast<int64_t>(elem_size);
    cpu_v_ptr += dst_token_offset_elems * static_cast<int64_t>(elem_size);
  }

  const c10::cuda::CUDAGuard device_guard(gpu_k_span.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (head_major) {
    KvMemcpy3DBatchAsyncFn batch_fn = resolve_memcpy_3d_batch_async();
    if (batch_fn == nullptr || stream == nullptr) {
      return 0;
    }
    const size_t row_bytes = static_cast<size_t>(D) * elem_size;
    const size_t gpu_token_pitch = static_cast<size_t>(H * D) * elem_size;
    std::vector<KvMemcpy3DBatchOp> ops;
    ops.reserve(static_cast<size_t>(B * H * 2));
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        const int64_t src_elem = ((b * span_tokens) * H + h) * D;
        const int64_t dst_elem = (b * H * cpu_tokens + h * cpu_tokens + start_token) * D;
        append_batch_2d_copy(
            ops,
            gpu_k_ptr + src_elem * static_cast<int64_t>(elem_size),
            gpu_token_pitch,
            cpu_k_ptr + dst_elem * static_cast<int64_t>(elem_size),
            row_bytes,
            row_bytes,
            static_cast<size_t>(span_tokens),
            CU_MEM_LOCATION_TYPE_DEVICE,
            gpu_k_span.get_device(),
            CU_MEM_LOCATION_TYPE_HOST,
            0,
            0);
        append_batch_2d_copy(
            ops,
            gpu_v_ptr + src_elem * static_cast<int64_t>(elem_size),
            gpu_token_pitch,
            cpu_v_ptr + dst_elem * static_cast<int64_t>(elem_size),
            row_bytes,
            row_bytes,
            static_cast<size_t>(span_tokens),
            CU_MEM_LOCATION_TYPE_DEVICE,
            gpu_k_span.get_device(),
            CU_MEM_LOCATION_TYPE_HOST,
            0,
            0);
      }
    }
    size_t fail_idx = std::numeric_limits<size_t>::max();
    CUresult result = batch_fn(
        ops.size(), ops.data(), &fail_idx, 0, reinterpret_cast<CUstream>(stream));
    return result == CUDA_SUCCESS ? span_tokens : 0;
  }
  cudaError_t err = cudaMemcpy2DAsync(
      cpu_k_ptr,
      dst_pitch,
      gpu_k_ptr,
      src_pitch,
      width_bytes,
      static_cast<size_t>(B),
      cudaMemcpyDeviceToHost,
      stream);
  TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync KV->CPU K failed: ", cudaGetErrorString(err));
  err = cudaMemcpy2DAsync(
      cpu_v_ptr,
      dst_pitch,
      gpu_v_ptr,
      src_pitch,
      width_bytes,
      static_cast<size_t>(B),
      cudaMemcpyDeviceToHost,
      stream);
  TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync KV->CPU V failed: ", cudaGetErrorString(err));
  return span_tokens;
}

int64_t kv_sparse_copy_compact_to_slots_d2d_cuda(
    torch::Tensor compact_k,
    torch::Tensor compact_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor head_ids,
    torch::Tensor slot_ids,
    torch::Tensor src_rows,
    torch::Tensor src_cols,
    torch::Tensor cu_seqlens,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  check_tensor(compact_k.is_cuda() && compact_v.is_cuda(), "compact_k/compact_v must be CUDA tensors");
  check_tensor(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  check_tensor(compact_k.device() == gpu_k.device() && compact_v.device() == gpu_v.device(),
               "compact/cache device mismatch");
  check_tensor(compact_k.is_contiguous() && compact_v.is_contiguous(), "compact KV must be contiguous");
  check_tensor(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu KV must be contiguous");
  check_tensor(compact_k.scalar_type() == compact_v.scalar_type(), "compact KV dtype mismatch");
  check_tensor(compact_k.scalar_type() == gpu_k.scalar_type() && compact_k.scalar_type() == gpu_v.scalar_type(),
               "compact/cache dtype mismatch");
  check_tensor(compact_k.dim() == 3 && compact_v.sizes() == compact_k.sizes(),
               "compact KV must be [packed_tokens,1,D]");
  check_tensor(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  check_tensor(compact_k.size(1) == 1 && gpu_k.size(2) == 1 && compact_k.size(2) == gpu_k.size(3),
               "compact/cache shape mismatch");
  check_tensor(head_ids.device().is_cpu() && slot_ids.device().is_cpu() && src_rows.device().is_cpu() &&
                   src_cols.device().is_cpu() && cu_seqlens.device().is_cpu(),
               "D2D copy plans must be CPU tensors");
  check_tensor(head_ids.is_contiguous() && slot_ids.is_contiguous() && src_rows.is_contiguous() &&
                   src_cols.is_contiguous() && cu_seqlens.is_contiguous(),
               "D2D copy plans must be contiguous");
  check_tensor(head_ids.scalar_type() == at::kLong && slot_ids.scalar_type() == at::kLong &&
                   src_rows.scalar_type() == at::kLong && src_cols.scalar_type() == at::kLong &&
                   cu_seqlens.scalar_type() == at::kLong,
               "D2D copy plans must be int64");
  check_tensor(head_ids.dim() == 1 && slot_ids.sizes() == head_ids.sizes() &&
                   src_rows.sizes() == head_ids.sizes() && src_cols.sizes() == head_ids.sizes(),
               "D2D copy plan tensors must have shape [N]");
  check_tensor(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");

  const int64_t n = head_ids.size(0);
  if (n <= 0) {
    return 0;
  }
  const int64_t B = gpu_k.size(0);
  check_tensor(B > 0, "GPU cache batch must be positive");
  const int64_t D = gpu_k.size(3);
  check_tensor(cu_seqlens.dim() == 1 && (cu_seqlens.size(0) - 1) % B == 0,
               "cu_seqlens shape mismatch");
  const int64_t rows = (cu_seqlens.size(0) - 1) / B;
  const auto* head_ptr = head_ids.data_ptr<int64_t>();
  const auto* slot_ptr = slot_ids.data_ptr<int64_t>();
  const auto* row_ptr = src_rows.data_ptr<int64_t>();
  const auto* col_ptr = src_cols.data_ptr<int64_t>();
  const auto* cu_ptr = cu_seqlens.data_ptr<int64_t>();
  const size_t copy_bytes = static_cast<size_t>(frame_seqlen * D) * compact_k.element_size();
  const int64_t gpu_tokens = gpu_k.size(1);
  const char* compact_k_ptr = static_cast<const char*>(compact_k.data_ptr());
  const char* compact_v_ptr = static_cast<const char*>(compact_v.data_ptr());
  char* gpu_k_ptr = static_cast<char*>(gpu_k.data_ptr());
  char* gpu_v_ptr = static_cast<char*>(gpu_v.data_ptr());

  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  KvMemcpy3DBatchAsyncFn batch_fn = resolve_memcpy_3d_batch_async();
  const bool use_batch = batch_fn != nullptr && stream != nullptr;
  std::vector<KvMemcpy3DBatchOp> ops;
  if (use_batch) {
    ops.reserve(static_cast<size_t>(n * B * 2));
  }
  for (int64_t i = 0; i < n; ++i) {
    const int64_t head = head_ptr[i];
    const int64_t slot = slot_ptr[i];
    const int64_t src_row = row_ptr[i];
    const int64_t src_col = col_ptr[i];
    check_tensor(head >= 0 && slot >= 0 && slot < gpu_cache_frames, "invalid destination slot");
    check_tensor(src_row >= 0 && src_row < rows && src_col >= 0, "invalid compact source");
    for (int64_t b = 0; b < B; ++b) {
      const int64_t packed_row = b * rows + src_row;
      const int64_t src_token = cu_ptr[packed_row] + src_col * frame_seqlen;
      check_tensor(src_token >= cu_ptr[packed_row] && src_token + frame_seqlen <= cu_ptr[packed_row + 1],
                   "compact source span out of range");
      const int64_t dst_token = (head * gpu_cache_frames + slot) * frame_seqlen;
      check_tensor(dst_token >= 0 && dst_token + frame_seqlen <= gpu_tokens,
                   "GPU slot destination out of range");
      const int64_t src_elem = src_token * D;
      const int64_t dst_elem = (b * gpu_tokens + dst_token) * D;
      if (use_batch) {
        append_batch_2d_copy(
            ops,
            compact_k_ptr + src_elem * static_cast<int64_t>(compact_k.element_size()),
            copy_bytes,
            gpu_k_ptr + dst_elem * static_cast<int64_t>(gpu_k.element_size()),
            copy_bytes,
            copy_bytes,
            1,
            CU_MEM_LOCATION_TYPE_DEVICE,
            compact_k.get_device(),
            CU_MEM_LOCATION_TYPE_DEVICE,
            gpu_k.get_device());
        append_batch_2d_copy(
            ops,
            compact_v_ptr + src_elem * static_cast<int64_t>(compact_v.element_size()),
            copy_bytes,
            gpu_v_ptr + dst_elem * static_cast<int64_t>(gpu_v.element_size()),
            copy_bytes,
            copy_bytes,
            1,
            CU_MEM_LOCATION_TYPE_DEVICE,
            compact_k.get_device(),
            CU_MEM_LOCATION_TYPE_DEVICE,
            gpu_k.get_device());
      } else {
        cudaError_t err = cudaMemcpyAsync(
            gpu_k_ptr + dst_elem * static_cast<int64_t>(gpu_k.element_size()),
            compact_k_ptr + src_elem * static_cast<int64_t>(compact_k.element_size()),
            copy_bytes,
            cudaMemcpyDeviceToDevice,
            stream);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync compact K slot fill failed: ", cudaGetErrorString(err));
        err = cudaMemcpyAsync(
            gpu_v_ptr + dst_elem * static_cast<int64_t>(gpu_v.element_size()),
            compact_v_ptr + src_elem * static_cast<int64_t>(compact_v.element_size()),
            copy_bytes,
            cudaMemcpyDeviceToDevice,
            stream);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync compact V slot fill failed: ", cudaGetErrorString(err));
      }
    }
  }
  if (use_batch && !ops.empty()) {
    ensure_current_cuda_context_for_driver_batch(gpu_k.get_device(), "compact slot-fill batched D2D");
    size_t fail_idx = std::numeric_limits<size_t>::max();
    CUresult result = batch_fn(
        ops.size(),
        ops.data(),
        &fail_idx,
        0,
        reinterpret_cast<CUstream>(stream));
    TORCH_CHECK(
        result == CUDA_SUCCESS,
        "cuMemcpy3DBatchAsync compact slot fill failed with CUresult=",
        static_cast<int>(result),
        ", failing descriptor=",
        fail_idx,
        ", descriptor_count=",
        ops.size());
  }
  return n;
}
