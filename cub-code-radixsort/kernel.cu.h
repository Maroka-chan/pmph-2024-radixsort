#define lgWARP      5
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
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint32_t isT[B];
  __shared__ uint32_t isF[B];
  uint32_t tfs[Q];
  uint32_t ffs[Q];

  uint32_t isTrg[Q];
  uint32_t acc = 0;
  for (int q = 0; q < Q; q++){
    uint32_t isUnset = 0;
    if (!(gid * Q + q >= num_items)) {
      isUnset = isBitUnset(vals[q], bit + outerLoopIndex * lgH);
    }
    tfs[q] = isUnset;
    acc += isUnset;
    isTrg[q] = acc;
  }
  __syncthreads();

  // uint32_t split = isTrg[Q-1];

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
    if (!(gid * Q + q >= num_items)) {
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
  // if (threadIdx.x == 0 || threadIdx.x == 1 ){
  //   printf("split: %u, thd_prefix: %u, thd_prefix_2: %u, isTrg[Q-1]: %u, for threadIdx.x: %u\n", split, thd_prefix, thd_prefix_2, isTrg[Q-1], threadIdx.x);
  // }

  for (int q = 0; q < Q; q++){
    isFrg[q] += thd_prefix_2 + split;
  }

  __syncthreads();

  uint32_t inds[Q];
  for (int q = 0; q < Q; q++){
    if (gid * Q + q >= num_items) {break;}
    if (tfs[q] == 1){
      inds[q] = isTrg[q]-1;
      // printf("threadIdx.x: %u, inds[%u]=isTrg[%u]-1=%u\n", threadIdx.x, q, q, inds[q]);
    }
    else {
      inds[q] = isFrg[q]-1;
      // printf("threadIdx.x: %u, inds[%u]=isFrg[%u]-1=%u\n", threadIdx.x, q, q, inds[q]);
    }
  }

  __syncthreads();

  for (int q = 0; q < Q; q++){
    if (gid * Q + q >= num_items) {break;}
    uint32_t ind = inds[q];
    uint32_t val = vals[q];
    if (threadIdx.x == 0 || threadIdx.x == 1) {
    // printf("num items: %u, gid: %u, q: %u\n", num_items, gid, q);
      // printf("threadIdx.x == %u, ind: %u, val: %u\n", threadIdx.x, ind, val);
    }
    //printf("thread: %u, ind %u: %u\n", threadIdx.x, q, inds[q]);
    final_res[ind] = val;
  }
}



template <int Q, int B> __launch_bounds__(B) __global__ void finalKernel(uint32_t *d_keys_in, uint32_t *histogramArr, uint32_t num_items, int lgH, int outerLoopIndex, uint32_t *origHist){
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Step 1
  __shared__ uint32_t shmem[Q*B];
  __shared__ uint32_t result[Q*B];
  __shared__ uint32_t histo_tst[B];
  __shared__ uint32_t histo_org[B];
  __shared__ uint32_t histo_scn_exc[B];
  __shared__ uint32_t histo_scn_inc[B];

  int tid = threadIdx.x;



  uint32_t block_offset = blockIdx.x * Q * B;
  uint32_t thread_offset = threadIdx.x * Q;


  __syncthreads();

  // Copy from global to shared
  for (int i = 0; i < Q; i++) {
    if (gid * Q + i >= num_items) {break;}
    shmem[threadIdx.x * Q + i] = d_keys_in[block_offset + thread_offset + i];
  }
  __syncthreads();

  // if (threadIdx.x == 1) {
  //   for (int i = 0; i < Q*B; i++)
  //   {
  //     printf("elm[%u]: %u\n", i, shmem[i]);
  //   }
  // }
  __syncthreads();

  uint32_t elements[Q];
  // Copy from shared to register
  for (int q = 0; q < Q; q++){
    if (gid * Q + q >= num_items) {break;}
    elements[q] = shmem[threadIdx.x * Q + q];
    // printf("threadIdx.x: %u, register elm[%u]: %u\n", threadIdx.x, q, elements[q]);
  }
  __syncthreads();



  // __syncthreads();
  // Step 2
  for (int bit = 0; bit < lgH; bit++){

    partition2<Q,B>(elements, lgH, outerLoopIndex, bit, result, num_items);

    if (threadIdx.x == 1) {
      for (int i = 0; i < num_items; i++) {
        //printf("thread: %i, result[%i]: %u\n", threadIdx.x, i, result[i]);
        //printf("thread: %i, d_keys_in[%i]: %u\n", threadIdx.x, i, d_keys_in[i]);
        //printf("thread: %i, shmem[10]: %u\n", threadIdx.x, shmem[10]);
      }
    }

    for (int q = 0; q < Q; q++){
      if (gid * Q + q >= num_items) {break;}
      // printf("threadIdx: %u writing result: %u\n", threadIdx.x, result[threadIdx.x * Q + q]);
      elements[q] = result[threadIdx.x * Q + q];

      if (threadIdx.x == 1) {
        // printf("thread: %i, result[%i]: %u\n", threadIdx.x, threadIdx.x * Q + q, result[threadIdx.x * Q + q]);
        // printf("thread: %i, elements[%i]: %u\n", threadIdx.x, q, elements[q]);
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < 2) {
    for (int i = 0; i < num_items; i++) {
      //printf("thread: %i, elements[%i]: %u\n", threadIdx.x, i, elements[i]);
      //printf("thread: %i, d_keys_in[%i]: %u\n", threadIdx.x, i, d_keys_in[i]);
      //printf("thread: %i, shmem[10]: %u\n", threadIdx.x, shmem[10]);
    }
  }

  histo_tst[tid] = 0;
  histo_org[tid] = 0;
  histo_scn_exc[tid] = 0;
  histo_scn_inc[tid] = 0;

  __syncthreads();

  // Step 3.1

  // Copy from global to shared
  //printf("thread: %u, gid: %u, origHist: %u\n", threadIdx.x, gid, origHist[gid]);
  histo_org[threadIdx.x] = origHist[gid];
  histo_tst[threadIdx.x] = histogramArr[gid];

  __syncthreads();


  //printf("histo_org[%u]: %u origHist[%u]: %u\n", threadIdx.x, histo_org[threadIdx.x], gid, origHist[gid]);

  // Step 3.2
  uint32_t scanRes = scanIncBlock<Add<uint32_t>>(histo_org, threadIdx.x);
  __syncthreads();
  histo_scn_inc[threadIdx.x] = scanRes;
  __syncthreads();
  histo_scn_exc[threadIdx.x] = (threadIdx.x == 0) ? 0 : histo_scn_inc[threadIdx.x-1];
  __syncthreads();

  histo_org[threadIdx.x] = origHist[gid];

  __syncthreads();

  //printf("threadIdx: %u, histo_tst: %u, histo_scn_exc: %u\n", threadIdx.x, histo_tst[threadIdx.x], histo_scn_exc[threadIdx.x]);

  // Step 3.3

  for(int q=0; q<Q; q++) {
    if (gid * Q + q >= num_items) {break;}
    uint32_t elm = elements[q];
    uint8_t bin = (elm >> (outerLoopIndex * 8)) & 0xFF;
    // printf("q: %i, gid: %i, This is bin: %i\n", q, gid, bin);
    //uint32_t loc_pos = q*blockDim.x + threadIdx.x;
    uint32_t loc_pos = gid * Q + q;
    uint32_t glb_pos = histo_tst[bin] - histo_org[bin] + loc_pos - histo_scn_exc[bin];
    //printf("elm: %u\n", elm);
    // printf("threadIdx.x: %u, elm: %u, bin: %u -> %u - %u + %u - %u = %u\n", threadIdx.x, elm, bin, histo_tst[bin], histo_org[bin], loc_pos, histo_scn_exc[bin], glb_pos);
    //printf("bin: %u\n", bin);
    //printf("histo_tst: %u\n", histo_tst[bin]);
    //printf("histo_scn_exc: %u\n", histo_scn_exc[bin]);
    if(glb_pos < num_items) {
      d_keys_in[glb_pos] = elm;
    }
  }

}


template <int Q, int B, int lgH> __global__ void partition2Test(){
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ uint32_t result[Q*B];
  const int N = 8;
  uint32_t input[N] = {8,8,6,5,4,3,1,1};
  // 8 = 0000 1000
  // 7 = 0000 0111
  // 6 = 0000 0110
  // 5 = 0000 0101
  // 4 = 0000 0100
  // 3 = 0000 0011
  // 2 = 0000 0010
  // 1 = 0000 0001

  if (threadIdx.x == 0) {
    // printf("Input:\n");
    for (int i = 0; i < N; i++)
    {
      // printf("%i ", input[i]);
    }
    // printf("\n");
    // printf("Results:\n");
  }
  __syncthreads();
  uint32_t elements[Q];
  for (int q = 0; q < Q; q++){
    if (gid * Q + q >= N) {break;}
    elements[q] = input[threadIdx.x * Q + q];
  }
  __syncthreads();

  for (int bit = 0; bit < 8; bit++){
    partition2<Q,B>(elements, lgH, 0, bit, result, N);
    __syncthreads();
    for (int q = 0; q < Q; q++){
      elements[q] = result[threadIdx.x * Q + q];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i = 0; i < N; i++) {
        printf("%u ", result[i]);
      }
      printf("\n");
    }
    __syncthreads();
  }


}

