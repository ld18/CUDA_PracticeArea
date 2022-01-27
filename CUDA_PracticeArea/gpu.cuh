#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

namespace GPU_Simple {

    //Simple base gpu implementations
    __global__ void addUp(int* int_array, const int* int_array_length, int* maxVal);
    int launch_addUp(const int* int_array, const int int_array_length);

    __global__ void getMax(int* int_array, const int* int_array_length, int* maxVal);
    int launch_getMax(const int* int_array, const int int_array_length);

    __global__ void getMovingAvg(const int* int_array, const int* int_array_length, float* array_smooth, const int* avg_legth);
    void launch_getMovingAvg(const int* int_array, const int int_array_length, float* array_smooth, const int avg_legth);

}

namespace GPU_Better {

    //Bettere gpu implementations
    //Much less branch divergence per warp
    __global__ void addUp(int* int_array, const int* int_array_length, int* maxVal);
    int launch_addUp(const int* int_array, const int int_array_length);

    __global__ void getMax(int* int_array, const int* int_array_length, int* maxVal);
    int launch_getMax(const int* int_array, const int int_array_length);

}

namespace GPU_Tiled {

    //Tiled algorithem with shared memory
    //  RTX 2060 sharedMemPerBlock = 49152 bytes
    //  RTX 2060 maxThreadsPerBlock = 1024 threads
    //  Shared memory per Block = ( blockDim.x + avg_legth - 1 ) * sizeof(int) 
    //      ->  ( 1024 + avg_legth - 1 ) * 4 = sharedMemPerBlock
    //         = avg_legth = (sharedMemPerBlock - 4092) / 4
    //      -> use maxThreadsPerBlock since sharedMemPerBlock is not the limitng factor
    //      -> larger avg_legth is better for the shared memory usage  
    __global__ void getMovingAvg(const int* int_array, const int* int_array_length, float* array_smooth, const int* avg_legth);
    void launch_getMovingAvg(const int* int_array, const int int_array_length, float* array_smooth, const int avg_legth);

}