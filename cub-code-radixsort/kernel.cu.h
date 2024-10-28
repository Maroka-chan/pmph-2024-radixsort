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

/*
__global__ void finalKernel(const uint32_t *d_keys_in, int Q, int B, uint32_t num_items, int lgH){
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Step 1
  __shared__ uint32_t shmem[Q*B];

  for (int i = 0; i < Q; i++) {
    if (gid * Q + i >= num_items) {return;}
    uint32_t block_offset = blockIdx.x * Q * B;
    uint32_t thread_offset = threadIdx.x * Q;
    shmem[block_offset + thread_offset + i] = d_keys_in[block_offset + thread_offset + i];
  }
  __syncthreads();

  // Step 2
  for (int i = 0; i < lgH; i++){

  }


  // Step 3

}
*/