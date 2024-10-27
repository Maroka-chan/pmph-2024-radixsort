//#include "../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "cub/cub.cuh"
#include "helper.cu.h"
#include "kernel.cu.h"

template<class Z>
bool validateZ(Z* A, uint32_t sizeAB) {
    printf("Printing first 100 numbers: \n");
    for(uint32_t i = 1; i < sizeAB; i++) {
      // Sanity check to see that its not all zeros
      if (i < 100)
      {
        //printf("A[i]=%d\n", A[i]);
      }

      if (A[i-1] > A[i]){
        printf("INVALID RESULT for i:%d, (A[i-1]=%d > A[i]=%d)\n", i, A[i-1], A[i]);
        return false;
      }
    }
    return true;
}

void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        //data[i] = r; //% 255;
	data[i] = 16932;
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
    const u_int32_t* d_keys_in,
    u_int32_t* d_keys_out,
    u_int64_t num_items,
    int begin_bit,
    int end_bit
) {
    const int B = 256;
    const int Q = 22;
    const int lgH = 8;
    const int H = pow(2, lgH);

    int numBlocks = 2;
    int threadsPerBlock = B;

    uint32_t *histogram_res = (uint32_t*) malloc(numBlocks * H * sizeof(uint32_t));
    uint32_t *histogram;
    cudaSucceeded(cudaMalloc((void**) &histogram, numBlocks * H * sizeof(uint32_t)));

    // First Kernel ... results in a 2D array.
    // TODO determine input/output for each kernel
    histogramKernel<<<numBlocks, threadsPerBlock>>>(d_keys_in, histogram, H, Q, B, num_items);

    cudaMemcpy(histogram_res, histogram, numBlocks * H * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();
    for (int i = 0; i < numBlocks * H; i++) {
      if (i%H==0) {printf("\n----------------\n"); }
      printf("%5i: %5i ", i, histogram_res[i]);
    }


    transposeKernel<<<numBlocks, threadsPerBlock>>>();
    flattenKernel<<<numBlocks, threadsPerBlock>>>();
    // I suppose this is what he refers to as the last kernel?
    // Should have same configuration as the first Kernel
    scanKernel<<<numBlocks, threadsPerBlock>>>();

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

    return 1;

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
    const uint64_t N = atoi(argv[1]);
    const uint64_t BASELINE = atoi(argv[2]);

    //Allocate and Initialize Host data with random values
    uint32_t* h_keys  = (uint32_t*) malloc(N*sizeof(uint32_t));
    uint32_t* h_keys_res  = (uint32_t*) malloc(N*sizeof(uint32_t));
    randomInitNat(h_keys, N, N/10);

    //uint32_t h_keys[23] = {
    //    16932, 18045, 19213, 20576, 21450, 22134, 23890,
    //    1032, 2457, 3001, 4096, 5123, 6287, 7391, 8542,
    //    9001, 10234, 11875, 12340, 13564, 14789, 15829,0
    //};

    /*
        1032, 2457, 3001, 4096, 5123, 6287, 7391, 8542,
        9001, 10234, 11875, 12340, 13564, 14789, 15829,
        16932, 18045, 19213, 20576, 21450, 22134, 23890
    */

    //for (int i = 0; i < N; i++) {
    //  printf("%i\n", h_keys[i]);
    //}

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

    bool success = validateZ(h_keys_res, N);

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
