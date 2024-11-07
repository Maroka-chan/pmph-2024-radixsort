#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <sys/time.h>
#include <time.h>

#define DEBUG_INFO  false
#define SCAN_INC false

#define lgWARP      5
#define WARP        (1<<lgWARP)

#define NUM_BLOCKS_SCAN     1024
#define ELEMS_PER_THREAD    24

typedef unsigned int uint32_t;
typedef int           int32_t;

uint32_t MAX_HWDTH;
uint32_t MAX_BLOCK;
uint32_t MAX_SHMEM;

cudaDeviceProp prop;

void initHwd() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaGetDeviceProperties(&prop, 0);
    MAX_HWDTH = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    MAX_BLOCK = prop.maxThreadsPerBlock;
    MAX_SHMEM = prop.sharedMemPerBlock;

    if (DEBUG_INFO) {
        printf("Device name: %s\n", prop.name);
        printf("Number of hardware threads: %d\n", MAX_HWDTH);
        printf("Max block size: %d\n", MAX_BLOCK);
        printf("Shared memory size: %d\n", MAX_SHMEM);
        puts("====");
    }
}

// int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
// {
//     unsigned int resolution=1000000;
//     long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
//     result->tv_sec = diff / resolution;
//     result->tv_usec = diff % resolution;
//     return (diff<0);
// }

#endif // CONSTANTS_H
