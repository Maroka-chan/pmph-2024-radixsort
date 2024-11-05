#define lgWARP      5
#define WARP        (1<<lgWARP)

template  <int B, int H> __global__ void histogramKernel(const uint32_t *d_keys_in, uint32_t *histogram, int Q, int ith_pass, uint32_t num_items) {
  __shared__ uint32_t histo_shared[H];
  histo_shared[threadIdx.x] = 0;
  __syncthreads();

  for (uint32_t i = 0; i < Q; i++) {
    if ((blockIdx.x * Q * B) + threadIdx.x * Q + i >= num_items) {return;}
    uint8_t bin = (d_keys_in[(blockIdx.x * Q * B) + threadIdx.x * Q + i] >> (ith_pass * 8)) & 0xFF;
    atomicAdd(&histo_shared[bin], 1);
  }

  uint32_t hist_index = blockIdx.x * H + threadIdx.x;

  histogram[hist_index] = histo_shared[threadIdx.x];
}

template <int TILE> __global__ void transposeKernel(uint32_t *histogram, uint32_t *histogram_tr, int H, int numBlocks) {
  __shared__ uint32_t shmem[TILE][TILE+1];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  if (x < H && y < numBlocks){
    shmem[threadIdx.y][threadIdx.x] = histogram[y * H + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  if (x < numBlocks && y < H){
    histogram_tr[y * numBlocks + x] = shmem[threadIdx.x][threadIdx.y];
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

// Taken from weekly 2
template<class T>
class Add {
  public:
    typedef T InpElTp;
    typedef T RedElTp;
    static const bool commutative = true;
    static __device__ __host__ inline T identInp()                    { return (T)0;    }
    static __device__ __host__ inline T mapFun(const T& el)           { return el;      }
    static __device__ __host__ inline T identity()                    { return (T)0;    }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }

    static __device__ __host__ inline bool equals(const T t1, const T t2) { return (t1 == t2); }
    static __device__ __host__ inline T remVolatile(volatile T& t)    { T res = t; return res; }
};



__device__ u_int8_t isBitUnset(uint32_t num, int bit){
  return ((num >> bit) & 1 ) ^ 1;
}

__device__ u_int8_t isBitSet(uint32_t num, int bit){
  return ((num >> bit) & 1 );
}

// Taken from weekly 2
template<class OP>
__device__ inline typename OP::RedElTp
scanIncWarp( volatile typename OP::RedElTp* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & (WARP-1);

    #pragma unroll
    for (int d=0; d < lgWARP; d++){
        int h = pow(2, d);
        if (lane>=h) {
            ptr[idx] = OP::apply(ptr[idx-h], ptr[idx]);
        }
    }
    return OP::remVolatile(ptr[idx]);
}

// Taken from weekly 2
template<class OP>
__device__ inline typename OP::RedElTp
scanIncBlock(volatile typename OP::RedElTp* ptr, const unsigned int idx) {
    const unsigned int lane   = idx & (WARP-1);
    const unsigned int warpid = idx >> lgWARP;

    // 1. perform scan at warp level. `scanIncWarp` computes its result in-place
    //    and also returns the per-thread result.
    typename OP::RedElTp res = scanIncWarp<OP>(ptr,idx);
    __syncthreads();

    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and
    //   max block size = 32^2 = 1024
    typename OP::RedElTp temp;
    if (lane == (WARP-1)) { temp = OP::remVolatile(ptr[idx]);}
    __syncthreads();
    if (lane == (WARP-1)) { ptr[warpid] = temp; }
    __syncthreads();

    // 3. scan again the first warp.
    if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step.
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }

    return res;
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

template <int Q, int B, int lgH> __global__ void sortTile(uint32_t* keys, int ith_pass, int N){
  __shared__ uint32_t result[Q*B];

  uint32_t elements[Q];
  for (int q = 0; q < Q; q++){
    if ((blockIdx.x * Q * B) + threadIdx.x * Q + q >= N) {break;}
    elements[q] = keys[(blockIdx.x * Q * B) + threadIdx.x * Q + q];
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
    keys[(blockIdx.x * Q * B) + threadIdx.x * Q + q] = elm;
  }
}

template <int Q, int B, int lgH> __global__ void scatter(uint32_t* keys, uint32_t* histograms, uint32_t* histograms_tst, int ith_pass, int N){
  __shared__ uint32_t shmem[Q*B];
  __shared__ uint32_t histo_tst[B];
  __shared__ uint32_t histo_org[B];
  __shared__ uint32_t histo_scn_exc[B];
  __shared__ uint32_t histo_scn_inc[B];

  uint32_t hist_index = blockIdx.x * 256 + threadIdx.x;

  histo_org[threadIdx.x] = histograms[hist_index];
  histo_tst[threadIdx.x] = histograms_tst[hist_index];
  __syncthreads();

  uint32_t scanRes = scanIncBlock<Add<uint32_t>>(histo_org, threadIdx.x);
  __syncthreads();
  histo_scn_inc[threadIdx.x] = scanRes;
  __syncthreads();
  histo_scn_exc[threadIdx.x] = (threadIdx.x == 0) ? 0 : histo_scn_inc[threadIdx.x-1];
  __syncthreads();

  // restore histo_org because scanIncBlock scans in-place
  histo_org[threadIdx.x] = histograms[hist_index];
  __syncthreads();

  uint32_t block_offset = blockIdx.x * Q * B;
  uint32_t thread_offset = threadIdx.x * Q;
  // Copy from global to shared
  for (int i = 0; i < Q; i++) {
    if ((blockIdx.x * Q * B) + threadIdx.x * Q + i >= N) {break;}
    shmem[threadIdx.x * Q + i] = keys[block_offset + thread_offset + i];
  }
  __syncthreads();

  uint32_t elements[Q];
  // Copy from shared to register
  for (int q = 0; q < Q; q++){
    if ((blockIdx.x * Q * B) + threadIdx.x * Q + q >= N) {break;}
    elements[q] = shmem[threadIdx.x * Q + q];
  }

  for(int q=0; q<Q; q++) {
    if ((blockIdx.x * Q * B) + threadIdx.x * Q + q >= N) {break;}
    uint32_t elm = elements[q];
    uint8_t bin = (elm >> (ith_pass * lgH)) & 0xFF;
    // TODO: Unsure about this
    uint32_t loc_pos = q*blockDim.x + threadIdx.x;
    // uint32_t loc_pos = gid * Q + q;
    uint32_t glb_pos = histo_tst[bin] - histo_org[bin] + loc_pos - histo_scn_exc[bin];
    if(glb_pos < N) {
      keys[glb_pos] = elm;
    } 
  }
}
