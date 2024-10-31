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

__device__ int isBitSet(uint32_t num, int bit){
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


/*
template <int Q, int B> __device__ void partition2(uint32_t *arr, int lgH, int loopIndex){
  uint32_t tfs[B];
  uint32_t isT[Q];
  uint32_t ffs[B];
  uint32_t isF[Q];
  uint32_t inds[Q];
  uint32_t scatter_res[Q];


  for (int q = 0; q < Q; q++){
    tfs[q] = isBitUnset(arr[q], loopIndex * lgH);
  }


  uint32_t temp[B];
  temp[threadIdx.x] = scanIncBlock<Add<uint32_t>>(tfs, threadIdx.x);

  if (threadIdx.x < Q) {
    isT[threadIdx.x] = temp[threadIdx.x];
  }
  int i = isT[Q-1];


  for (int q = 0; q < Q; q++){
    uint32_t tmp = tfs[q];
    if (tmp == 1){
      ffs[q] = 0;
    }
    else {
      ffs[q] = 1;
    }
  }



  temp[threadIdx.x] = scanIncBlock<Add<uint32_t>>(ffs, threadIdx.x);

  if (threadIdx.x < Q) {
    isF[threadIdx.x] = temp[threadIdx.x];
  }


  for (int q = 0; q < Q; q++){
    if (tfs[q] == 1){
      inds[q] = isT[q]-1;
    }
    else {
      inds[q] = isF[q]-1;
    }
  }

  for (int q = 0; q < Q; q++){
    int index = inds[q];
    scatter_res[index] = arr[q];
  }

  //if (threadIdx.x == 0) {
  //  for (int q = 0; q < Q; q++) {
  //    printf("%i \n", scatter_res[q]);
  //  }
  //}



  for (int q = 0; q < Q; q++) {
    arr[q] = scatter_res[q];
  }
}

*/


template <int Q, int B> __device__ void partition2(uint32_t vals[Q], int lgH, int outerLoopIndex, int bit, uint32_t final_res[Q*B], uint32_t num_items){
  __shared__ uint32_t isT[B];
  __shared__ uint32_t isF[B];
  uint32_t tfs[Q];
  uint32_t ffs[Q];

  uint32_t isTrg[Q];
  uint32_t acc = 0;
  for (int q = 0; q < Q; q++){
    uint32_t isUnset = isBitUnset(vals[q], bit + outerLoopIndex * lgH);
    tfs[q] = isUnset;
    acc += isUnset;
    isTrg[q] = acc;
  }
  __syncthreads();

  uint32_t split = isTrg[Q-1];

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
    uint32_t isSet = isBitSet(vals[q], bit + outerLoopIndex * lgH);
    ffs[q] = isSet;
    acc += isSet;
    isFrg[q] = acc;
  }

  __syncthreads();

  isF[threadIdx.x] = isTrg[Q-1];
  __syncthreads();
  uint32_t res2 = scanIncBlock<Add<uint32_t>>(isF, threadIdx.x);
  __syncthreads();
  isF[threadIdx.x] = res2;
  __syncthreads();

  //thd_prefix = (threadIdx.x == 0) ? 0 : isT[threadIdx.x-1];
  for (int q = 0; q < Q; q++){
    isFrg[q] += split;
  }

  //if (threadIdx.x == 0 && blockIdx.x == 0){
  //  for (int j = 0; j < Q; j++){
  //    printf("F: %i T: %i \n", isFrg[j], isTrg[j]);
  //  }
  //}

  __syncthreads();


  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t inds[Q];
  for (int q = 0; q < Q; q++){
    if (gid * Q + q >= num_items) {break;}
    if (tfs[q] == 1){
      inds[q] = isTrg[q]-1;

    }
    else {
      inds[q] = isFrg[q]-1;
    }
  }

  __syncthreads();


  for (int q = 0; q < Q; q++){
    if (gid * Q + q >= num_items) {break;}
    uint32_t ind = inds[q];
    uint32_t val = vals[q];
    final_res[ind] = val;
  }
}



template <int Q, int B> __global__ void finalKernel(uint32_t *d_keys_in, uint32_t *histogramArr, uint32_t num_items, int lgH, int outerLoopIndex, uint32_t *origHist){
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Step 1
  __shared__ uint32_t shmem[Q*B];
  __shared__ uint32_t result[Q*B];


  uint32_t block_offset = blockIdx.x * Q * B;
  uint32_t thread_offset = threadIdx.x * Q;


  __syncthreads();

  // Copy from global to shared
  for (int i = 0; i < Q; i++) {
    if (gid * Q + i >= num_items) {break;}
    shmem[threadIdx.x * Q + i] = d_keys_in[block_offset + thread_offset + i];
  }
  __syncthreads();


  uint32_t elements[Q];
  // Copy from shared to register
  for (int q = 0; q < Q; q++){
    if (gid * Q + q >= num_items) {break;}
    elements[q] = shmem[threadIdx.x * Q + q];
  }


  __syncthreads();
  // Step 2
  for (int bit = 0; bit < lgH; bit++){

    partition2<Q,B>(elements, Q, outerLoopIndex, bit, result, num_items);

    for (int q = 0; q < Q; q++){
      elements[q] = result[threadIdx.x * Q + q];
    }
  }

  __syncthreads();

  // Step 3.1

  __shared__ uint32_t originalHist[B];
  __shared__ uint32_t scannedHist[B];
  __shared__ uint32_t originalScannedHist[B];

  __syncthreads();

  // Copy from global to shared
  originalHist[threadIdx.x] = origHist[B * blockIdx.x + threadIdx.x];
  scannedHist[threadIdx.x] = histogramArr[B * blockIdx.x + threadIdx.x];
  __syncthreads();




  // Step 3.2

  uint32_t scanRes = scanIncBlock<Add<uint32_t>>(originalHist, threadIdx.x);
  __syncthreads();
  originalScannedHist[threadIdx.x] = scanRes;


  // Step 3.3

  __syncthreads();

  for (int q = 0; q < Q; q++){
    if (gid * Q + q >= num_items) {break;}
    uint32_t element = elements[q];
    uint32_t bin = element >> (outerLoopIndex * 8);
    bin = bin & 0xFF;
    uint32_t globalOffset = scannedHist[bin];
    d_keys_in[globalOffset] = element;
  }

}

template <int Q, int B> __global__ void partition2Test(){

  __shared__ uint32_t result[Q*B];
  uint32_t elements[] = {250, 0, 5 ,4 ,2 ,3 ,7 ,8, 10};

  uint32_t num_items = 9;


  for (int bit = 0; bit < 8; bit++){
    partition2<Q,B>(elements, Q, 0, bit, result, num_items);
    __syncthreads();
    for (int q = 0; q < Q; q++){
      elements[q] = result[threadIdx.x * Q + q];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int q = 0; q < Q+3; q++) {
        printf("%u ", elements[q]);
      }
      printf("\n");
    }
    __syncthreads();
  }


}

