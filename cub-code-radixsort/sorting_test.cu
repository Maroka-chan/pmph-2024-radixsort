//#include "../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "cub/cub.cuh"
#include "helper.cu.h"
#include "kernel.cu.h"

// template<class Z>
int validateZ(u_int32_t* A, uint32_t sizeAB) {
    int wrongCounter = 0;
    for(uint32_t i = 1; i < sizeAB; i++) {

      if (A[i-1] > A[i]){
        // printf("INVALID RESULT for i:%d, (A[i-1]=%d > A[i]=%d)\n", i, A[i-1], A[i]);
        wrongCounter++;
      }
    }
    return wrongCounter;
}

void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        uint32_t r = rand();
        data[i] = r;
    }
}

double sortRedByKeyCUB( uint32_t* data_keys_in
                      , uint32_t* data_keys_out
                      , const uint64_t N
) {
    int beg_bit = 0;
    int end_bit = 32;

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaDeviceSynchronize();
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    { // one dry run
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaDeviceSynchronize();
    }
    cudaCheckError();

    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int k=0; k<GPU_RUNS; k++) {
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    cudaFree(tmp_sort_mem);

    return elapsed;
}

// https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh
// Just copy-pasted the params from above (Not sure which ones we need for our implementation)
void radixSortKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    u_int32_t* d_keys_in,
    u_int32_t* d_keys_out,
    u_int64_t num_items,
    int begin_bit,
    int end_bit
) {
    const int B = 256; // CUDA block size
    const int Q = 22; // elements processed by each thread
    const int lgH = 8; // bits sorted at a time
    const int H = 256; // Histogram size

    int numBlocks = (num_items + B * Q - 1) / (B * Q);
    // printf("numBlocks: %u\n", numBlocks);
    int threadsPerBlock = B;

    uint32_t *histogram_res = (uint32_t*) malloc(numBlocks * H * sizeof(uint32_t));
    uint32_t *histogram;
    cudaSucceeded(cudaMalloc((void**) &histogram, numBlocks * H * sizeof(uint32_t)));
    uint32_t *transpose_res;
    cudaSucceeded(cudaMalloc((void**) &transpose_res, numBlocks * H * sizeof(uint32_t)));
    uint32_t *final_transpose_res;
    cudaSucceeded(cudaMalloc((void**) &final_transpose_res, numBlocks * H * sizeof(uint32_t)));

    cudaMemcpy(d_keys_out, d_keys_in, num_items * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    for (int i = 0; i < 4; i++) {
        // Step 1. (Also writes back to global memory)
        sortTile<Q,B,lgH><<<numBlocks, threadsPerBlock>>>(d_keys_out, i, num_items);
        cudaDeviceSynchronize();

        // Step 2.
        histogramKernel<H, B><<<numBlocks, threadsPerBlock>>>(d_keys_out, histogram, Q, i, num_items);
        cudaDeviceSynchronize();

        // Step 3. transpose -> scan -> transpose
        const int TILE = 16;
        int dimy = (numBlocks + TILE - 1) / TILE;
        int dimx = (H + TILE - 1) / TILE;
        dim3 block(TILE, TILE, 1), grid(dimx, dimy, 1);

        transposeKernel<TILE><<<grid, block>>>(histogram, transpose_res, H, numBlocks);
        cudaDeviceSynchronize();
        scanKernel<<<numBlocks, threadsPerBlock>>>(transpose_res, H, numBlocks);
        cudaDeviceSynchronize();
        transposeKernel<TILE><<<grid, block>>>(transpose_res, final_transpose_res, H, numBlocks);
        cudaDeviceSynchronize();

        // Step 4. Scatter sorted tiles to global positions
        scatter<Q,B,lgH><<<numBlocks, threadsPerBlock>>>(d_keys_out, histogram, final_transpose_res, i, num_items);
        cudaDeviceSynchronize();
    }

    cudaFree(histogram);
}



double radixSortBench( uint32_t* data_keys_in
                      , uint32_t* data_keys_out
                      , const uint64_t N
) {
    int beg_bit = 0;
    int end_bit = 32;

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;


    { // sort prelude
        radixSortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    { // one dry run
        radixSortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaDeviceSynchronize();
    }
    cudaCheckError();


    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int k=0; k<GPU_RUNS; k++) {
        radixSortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    cudaFree(tmp_sort_mem);

    return elapsed;
}

int main (int argc, char * argv[]) {
    if (argc != 3) {
        printf("Usage: %s <size-of-array> <baseline>(1 for baseline)\n", argv[0]);
        exit(1);
    }

    const uint32_t N = atoi(argv[1]);
    const uint64_t BASELINE = atoi(argv[2]);

    //Allocate and Initialize Host data with random values
    uint32_t *h_keys = (uint32_t*) malloc(N*sizeof(uint32_t));//{2, 3, 50, 1, 10, 5, 667, 3, 78, 23, 100};
    uint32_t* h_keys_res  = (uint32_t*) malloc(N*sizeof(uint32_t));
    randomInitNat(h_keys, N, N/10);

    //Allocate and Initialize Device data
    uint32_t* d_keys_in;
    uint32_t* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in,  N * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_keys_in, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, N * sizeof(uint32_t)));

    double elapsed;
    if(BASELINE) {
        elapsed = sortRedByKeyCUB( d_keys_in, d_keys_out, N );
    } else {
        elapsed = radixSortBench( d_keys_in, d_keys_out, N );
    }

    cudaMemcpy(h_keys_res, d_keys_out, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();

    int wrongCounter = validateZ(h_keys_res, N);
    printf("wrongCounter: %u\n", wrongCounter);
    bool success = wrongCounter == 0;

    if(BASELINE) {
        printf("Baseline (CUB) Sorting for N=%lu runs in: %.2f us, VALID: %d\n", N, elapsed, success);
    } else {
        printf("Radix Sorting for N=%lu runs in: %.2f us, VALID: %d\n", N, elapsed, success);
    }


    // Cleanup and closing
    cudaFree(d_keys_in); cudaFree(d_keys_out);
    //free(h_keys);
    free(h_keys_res);

    return success ? 0 : 1;
}
