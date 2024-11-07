#include "pbb_kernels.cuh"

template  <int B, int H, int lgH> __global__ void histogramKernel(uint32_t *d_keys_in, uint32_t *histogram, int Q, int ith_pass, uint32_t num_items) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t block_offset = blockIdx.x * Q * B;
  uint32_t thread_offset = threadIdx.x * Q;

  for (uint32_t i = 0; i < Q; i++) {
    if (gid * Q + i >= num_items) {return;}
    uint32_t block_offset = blockIdx.x * Q * B;
    uint32_t thread_offset = threadIdx.x * Q;
    uint8_t pass = (d_keys_in[block_offset + thread_offset + i] >> (ith_pass * lgH)) & 0xFF; // TODO: CHANGE BACK to 0xFF

    uint32_t element = d_keys_in[block_offset + thread_offset + i];

    uint32_t hist_block_offset = blockIdx.x * H;
    atomicAdd(&histogram[hist_block_offset + pass], 1);
  }
}

template <int TILE> __global__ void transposeKernel(uint32_t *histogram, uint32_t *histogram_tr, int width, int height) {
  __shared__ uint32_t shmem[TILE][TILE+1];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  if (x < width && y < height){
    shmem[threadIdx.y][threadIdx.x] = histogram[y * width + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  if (x < height && y < width){
    histogram_tr[y * height + x] = shmem[threadIdx.x][threadIdx.y];
  }
}

__global__ void scanKernel(uint32_t *histogram, int H, int numBlocks) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid != 0) { return; }

  // TODO: Parallise this.
  for (int i = 1; i < H * numBlocks; i++){
    histogram[i] = histogram[i] + histogram[i-1];
  }

}

__device__ u_int8_t isBitUnset(uint32_t num, int bit){
  return ((num >> bit) & 1 ) ^ 1;
}

__device__ u_int8_t isBitSet(uint32_t num, int bit){
  return ((num >> bit) & 1 );
}


template <int Q, int B> __device__ void partition2(uint32_t vals[Q], int lgH, int outerLoopIndex, int bit, uint32_t final_res[Q*B], uint32_t num_items){
  __shared__ uint32_t isT[B];
  __shared__ uint32_t isF[B];
  uint32_t tfs[Q];
  uint32_t ffs[Q];

  uint32_t isTrg[Q];
  uint32_t acc = 0;
  for (int q = 0; q < Q; q++){
    uint32_t isUnset = 0;
    if (!((blockIdx.x * Q * B) + threadIdx.x * Q + q >= num_items)) {
      isUnset = isBitUnset(vals[q], bit + outerLoopIndex * lgH);
    }
    tfs[q] = isUnset;
    acc += isUnset;
    isTrg[q] = acc;
  }
  __syncthreads();

  isT[threadIdx.x] = isTrg[Q-1];
  __syncthreads();
  uint32_t res1 = scanIncBlock<Add<uint32_t>>(isT, threadIdx.x);
  __syncthreads();
  isT[threadIdx.x] = res1;
  __syncthreads();
  uint32_t thd_prefix = (threadIdx.x == 0) ? 0 : isT[threadIdx.x-1];
  for (int q = 0; q < Q; q++){
    isTrg[q] += thd_prefix;
  }

  uint32_t isFrg[Q];
  acc = 0;
  for (int q = 0; q < Q; q++){
    uint32_t isSet;
    if (!((blockIdx.x * Q * B) + threadIdx.x * Q + q >= num_items)) {
      isSet = isBitSet(vals[q], bit + outerLoopIndex * lgH);
    }
    ffs[q] = isSet;
    acc += isSet;
    isFrg[q] = acc;
  }

  __syncthreads();

  isF[threadIdx.x] = isFrg[Q-1];
  __syncthreads();
  uint32_t res2 = scanIncBlock<Add<uint32_t>>(isF, threadIdx.x);
  __syncthreads();
  isF[threadIdx.x] = res2;
  __syncthreads();
  uint32_t thd_prefix_2 = (threadIdx.x == 0) ? 0 : isF[threadIdx.x-1];
  uint32_t split = isT[B-1];

  for (int q = 0; q < Q; q++){
    isFrg[q] += thd_prefix_2 + split;
  }


  uint32_t inds[Q];
  for (int q = 0; q < Q; q++){
    if ((blockIdx.x * Q * B) + threadIdx.x * Q + q >= num_items) {break;}
    if (tfs[q] == 1){
      inds[q] = isTrg[q]-1;
    }
    else {
      inds[q] = isFrg[q]-1;
    }
  }


  for (int q = 0; q < Q; q++){
    if ((blockIdx.x * Q * B) + threadIdx.x * Q + q >= num_items) {break;}
    uint32_t ind = inds[q];
    uint32_t val = vals[q];
    final_res[ind] = val;
  }
}

template <int Q, int B, int lgH> __global__ void sortTile(uint32_t *keys_in, uint32_t *keys_out, int ith_pass, int N){
  __shared__ uint32_t result[Q*B];

  uint32_t elements[Q];
  for (int q = 0; q < Q; q++){
    if ((blockIdx.x * Q * B) + threadIdx.x * Q + q >= N) {break;}
    elements[q] = keys_in[(blockIdx.x * Q * B) + threadIdx.x * Q + q];
  }

  for (int bit = 0; bit < lgH; bit++){
    partition2<Q,B>(elements, lgH, ith_pass, bit, result, N);
    __syncthreads();
    for (int q = 0; q < Q; q++){
      elements[q] = result[threadIdx.x * Q + q];
    }
    __syncthreads();
  }

  for(int q=0; q<Q; q++) {
    if ((blockIdx.x * Q * B) + threadIdx.x * Q + q >= N) {break;}

    uint32_t elm = elements[q];
    keys_out[(blockIdx.x * Q * B) + threadIdx.x * Q + q] = elm;
  }
}

template <int Q, int B, int H, int lgH> __global__ void scatter(uint32_t *keys, uint32_t *keys_out, uint32_t *histograms, uint32_t* histograms_tst, int ith_pass, int N){
  // use only H elements of shared mem and reuse it for both arrays
  __shared__ uint32_t shmem[H];
  uint32_t *histo_scn_inc = shmem;
  uint32_t *histo_tst_minus_org_minus_scn_exc = shmem;

  uint32_t hist_index        = blockIdx.x * H + threadIdx.x;
  uint32_t histo_org         = histograms[hist_index];
  histo_scn_inc[threadIdx.x] = histo_org;
  __syncthreads();

  uint32_t scanRes = scanIncBlock<Add<uint32_t>>(histo_scn_inc, threadIdx.x);
  __syncthreads();

  histo_scn_inc[threadIdx.x] = scanRes;
  __syncthreads();

  uint32_t scanExcRes = threadIdx.x == 0 ? 0 : histo_scn_inc[threadIdx.x - 1];

  histo_tst_minus_org_minus_scn_exc[threadIdx.x] =
    histograms_tst[hist_index] - // corresponds to histo_tst[threadIdx.x];
    histo_org -                  // corresponds to histo_org[threadIdx.x];
    scanExcRes;                  // corresponds to histo_scn_exc[threadIdx.x];
  __syncthreads();

  uint32_t block_offset = blockIdx.x * Q * B;
  #pragma unroll
  for (int q = 0; q < Q; q++) {
    uint32_t loc_pos = q * blockDim.x + threadIdx.x;

    if (block_offset + loc_pos < N) {
      uint32_t elm = keys[block_offset + loc_pos];
      uint8_t  bin = (elm >> (ith_pass * lgH)) & 0xFF; // NOTE: changed back to 0xFF

      // uint32_t glb_pos = histo_tst[bin] - histo_org[bin] + loc_pos - histo_scn_inc[bin];
      uint32_t glb_pos = loc_pos + histo_tst_minus_org_minus_scn_exc[bin];
      if (glb_pos < N) {
        keys_out[glb_pos] = elm;
      }

    }
  }
}