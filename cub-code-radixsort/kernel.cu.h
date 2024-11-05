#define lgWARP      5
#define WARP        (1<<lgWARP)

template  <int B, int H, int lgH> __global__ void histogramKernel(uint32_t *d_keys_in, uint32_t *histogram, int Q, int ith_pass, uint32_t num_items) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t block_offset = blockIdx.x * Q * B;
  uint32_t thread_offset = threadIdx.x * Q;

  // if (blockIdx.x == 1 && threadIdx.x == 0) {
  //   printf("d_keys_in: ");
  //   for (uint32_t i = 0; i < Q; i++) {
  //     printf("%u ", d_keys_in[block_offset + thread_offset + i]);
  //   }
  //   printf("\n");
  // }

  for (uint32_t i = 0; i < Q; i++) {
    if (gid * Q + i >= num_items) {return;}
    uint32_t block_offset = blockIdx.x * Q * B;
    uint32_t thread_offset = threadIdx.x * Q;
    uint8_t pass = (d_keys_in[block_offset + thread_offset + i] >> (ith_pass * lgH)) & 0xF; // TODO: CHANGE BACK to 0xFF

    uint32_t element = d_keys_in[block_offset + thread_offset + i];

    // if (blockIdx.x == 1 && threadIdx.x == 0) {
    //   printf("element: %u\n", element);
    // }

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

  // if (blockIdx.x == 1 && threadIdx.x == 0) {
  //   printf("Q: %d, B: %d\n", Q, B);
  //   printf("sorttile d_keys_in: ");
  //   for (uint32_t i = 0; i < N; i++) {
  //     printf("%u ", keys[i]);
  //   }
  //   printf("\n");
  // }

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

template <int Q, int B, int H, int lgH> __global__ void scatter(uint32_t* keys, uint32_t* histograms, uint32_t* histograms_tst, int ith_pass, int N){
  __shared__ uint32_t shmem[Q*B];
  __shared__ uint32_t histo_tst[H]; // NOTE: should be H not B (is the same by coincidence)
  __shared__ uint32_t histo_org[H];
  __shared__ uint32_t histo_scn_exc[H];
  __shared__ uint32_t histo_scn_inc[H];

  uint32_t hist_index = blockIdx.x * H + threadIdx.x;

  histo_org[threadIdx.x] = histograms[hist_index];
  histo_tst[threadIdx.x] = histograms_tst[hist_index];
  __syncthreads();

  // if (threadIdx.x == 0) {
  //   printf("histo_org: ");
  //   for (int i = 0; i < H; i++) {
  //     printf("%u ", histo_org[i]);
  //   }
  //   printf("\n");

  //   // printf("histo_tst: ");
  //   // for (int i = 0; i < H; i++) {
  //   //   printf("%u ", histo_tst[i]);
  //   // }
  //   // printf("\n");
  // }
  // __syncthreads();

  uint32_t scanRes = scanIncBlock<Add<uint32_t>>(histo_org, threadIdx.x);
  __syncthreads();
  // for (int i = 0; i < H; i++) {
  //   histo_scn_inc[i] = scanRes;
  // }
  histo_scn_inc[threadIdx.x] = scanRes;
  __syncthreads();
  // for (int i = 0; i < H; i++) {
  //   histo_scn_exc[i] = (i == 0) ? 0 : histo_scn_inc[i-1];
  // }
  histo_scn_exc[threadIdx.x] = (threadIdx.x == 0) ? 0 : histo_scn_inc[threadIdx.x-1];
  __syncthreads();

  // if (blockIdx.x == 1 && threadIdx.x == 0) {
  //   printf("histo_scn_exc: ");
  //   for (int i = 0; i < H; i++) {
  //     printf("%u ", histo_scn_exc[i]);
  //   }
  //   printf("\n");
  // }
  // __syncthreads();

  // restore histo_org because scanIncBlock scans in-place
  histo_org[threadIdx.x] = histograms[hist_index];
  __syncthreads();

  // if (threadIdx.x == 0) {
  //   printf("histo_org: ");
  //   for (int i = 0; i < H; i++) {
  //     printf("%u ", histo_org[i]);
  //   }
  //   printf("\n");
  // }
  // __syncthreads();

  // Copy from global to shared
  for (int i = 0; i < Q; i++) {
    uint32_t loc_pos = i*blockDim.x + threadIdx.x;
    if (blockIdx.x * Q * B + loc_pos >= N) {break;}
    shmem[loc_pos] = keys[blockIdx.x * Q * B + loc_pos];
  }
  __syncthreads();

  // if (threadIdx.x == 0) {
  //   printf("shmem: ");
  //   for (int i = 0; i < Q * B; i++) {
  //     printf("%u ", shmem[i]);
  //   }
  //   printf("\n");
  // }
  // __syncthreads();

  uint32_t elements[Q];
  // Copy from shared to register
  for (int q = 0; q < Q; q++){
    uint32_t loc_pos = q*blockDim.x + threadIdx.x;
    if (blockIdx.x * Q * B + loc_pos >= N) {break;}
    elements[q] = shmem[loc_pos];
  }


  for (int q = 0; q < Q; q++) {
    if ((blockIdx.x * Q * B) + threadIdx.x * Q + q >= N) {break;}
    printf("Iteration: %u, Block: %u, Thread %u: Element %u: %u\n", q, blockIdx.x, threadIdx.x, q, elements[q]);
  }
  __syncthreads();

  for(int q=0; q<Q; q++) {
    uint32_t elm = elements[q];
    uint8_t bin = (elm >> (ith_pass * lgH)) & 0x0F; // TODO: CHANGE BACK to 0xFF
    // TODO: Unsure about this
    uint32_t loc_pos = q*blockDim.x + threadIdx.x;
    if (blockIdx.x * Q * B + loc_pos >= N) {break;}
    // uint32_t loc_pos = gid * Q + q;
    uint32_t glb_pos = histo_tst[bin] - histo_org[bin] + loc_pos - histo_scn_exc[bin];
    if(glb_pos < N) {
      printf("blockIdx.x: %u, threadIdx.x: %u, elm: %u, bin: %u -> %u - %u + %u - %u = %u\n", blockIdx.x, threadIdx.x, elm, bin, histo_tst[bin], histo_org[bin], loc_pos, histo_scn_exc[bin], glb_pos);
      keys[glb_pos] = elm;
    }
  }

  __syncthreads();

  // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   printf("keys: ");
  //   for (uint32_t i = 0; i < N; i++) {
  //     printf("%u ", keys[i]);
  //   }
  //   printf("\n");
  // }
}
