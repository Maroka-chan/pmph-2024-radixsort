#define lgWARP      8
#define WARP        (1<<lgWARP)

__global__ void histogramKernel(const uint32_t *d_keys_in, uint32_t *histogram, int H, int Q, int B, uint32_t num_items) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (uint32_t i = 0; i < Q; i++) {
    if (gid * Q + i >= num_items) {return;}
    for (uint32_t y = 0; y < 4; y++) {
      uint32_t block_offset = blockIdx.x * Q * B;
      uint32_t thread_offset = threadIdx.x * Q;
      uint32_t pass = d_keys_in[block_offset + thread_offset + i] >> (y * 8);
      pass = pass & 0xFF;


      uint32_t hist_block_offset = blockIdx.x * H;
      atomicAdd(&histogram[hist_block_offset + pass], 1);
    }
  }
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

__global__ void flattenKernel() {

}

__global__ void scanKernel(uint32_t *histogram, int H, int numBlocks) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid != 0) { return; }

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



__device__ int isBitUnset(uint32_t num, int bit){
  return ((num >> bit) & 1 ) ^ 1;
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








template <int Q, int B> __global__ void finalKernel(const uint32_t *d_keys_in, uint32_t num_items, int lgH, int outerLoopIndex){
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Step 1
  __shared__ uint32_t shmem[Q*B];
  __shared__ uint32_t bitRes[B];


  uint32_t block_offset = blockIdx.x * Q * B;
  uint32_t thread_offset = threadIdx.x * Q;

  for (int i = 0; i < Q; i++) {
    if (gid * Q + i >= num_items) {return;}
    shmem[threadIdx.x * Q + i] = d_keys_in[block_offset + thread_offset + i];
  }
  __syncthreads();

  // Step 2
  for (int i = 0; i < lgH; i++){
    uint32_t acc = 0;
    for (int j = 0; j < Q; j++){

      uint32_t isUnset = isBitUnset(shmem[threadIdx.x * Q + j], i + outerLoopIndex * lgH);
      acc += isUnset;

    }





    bitRes[threadIdx.x] = acc;

    if (threadIdx.x == 0 && i == 0 && outerLoopIndex == 0) {
      for (int tmp = 0; tmp < B; tmp++){
        printf("%i: %i \n", tmp, bitRes[tmp]);
      }
    }

    uint32_t res = scanIncBlock<Add<uint32_t>>(bitRes, threadIdx.x);
    __syncthreads();
    bitRes[threadIdx.x] = res;
    __syncthreads();

    if (threadIdx.x == 0 && i == 0 && outerLoopIndex == 0) {
      for (int tmp = 0; tmp < B; tmp++){
        printf("%i: %i \n", tmp, bitRes[tmp]);
      }
    }

  }

  __syncthreads();



  // Step 3

}