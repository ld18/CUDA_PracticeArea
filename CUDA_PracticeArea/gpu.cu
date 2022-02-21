
#include "gpu.cuh"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include <assert.h>

using namespace std;
__constant__ int int_array_CM[16384]; //Adapt to used datasize

__global__ void GPU_Simple::addUp(int* int_array, const int* int_array_length) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        if (index % (2 * stride) == 0) {
            if (index < *int_array_length && index + stride < *int_array_length) {
                int_array[index] += int_array[index + stride];
            }
        }
    }
}

int GPU_Simple::launch_addUp(const int* int_array, const int int_array_length) {
    int* dev_int_array = nullptr;
    int* dev_int_array_length = nullptr;
    int host_maxVal = 0;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_int_array, int_array_length * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_int_array_length, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_int_array, int_array, int_array_length * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_int_array_length, &int_array_length, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }

    constexpr int blockSize = 1024;
    dim3 threads(blockSize); // number of threads per block
    dim3 grid(ceil((float)int_array_length / blockSize)); // number of blocks in grid

    auto start = std::chrono::high_resolution_clock::now();
    GPU_Simple::addUp << < grid, threads >> > (dev_int_array, dev_int_array_length); // start kernel (executed on device)
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>elapsed = finish - start;
    cout << setprecision(8) << elapsed.count() * 1000;

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code % d after launching addKernel!\n", cudaStatus);
        exit;
    }

    // Copy output vector from GPU buffer to host memory.
    int blockSum = 0;
    for (int i = 0; i < grid.x; i++) {
        cudaStatus = cudaMemcpy(&blockSum, dev_int_array + blockSize * i, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            printf("cudaMemcpy failed!");
            exit;
        }
        host_maxVal += blockSum;
    }

    cudaFree(dev_int_array);
    cudaFree(dev_int_array_length);

    return host_maxVal;
}

__global__ void GPU_Simple::getMax(int* int_array, const int* int_array_length) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        if (index % (2 * stride) == 0) {
            if (index < *int_array_length && index + stride < *int_array_length) {
                int_array[index] = max(int_array[index], int_array[index + stride]);
            }
        }
    }
}

int GPU_Simple::launch_getMax(const int* int_array, const int int_array_length) {
    int* dev_int_array = nullptr;
    int* dev_int_array_length = nullptr;
    int host_maxVal = 0;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_int_array, int_array_length * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_int_array_length, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_int_array, int_array, int_array_length * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_int_array_length, &int_array_length, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }

    constexpr int blockSize = 1024;
    dim3 threads(blockSize); // number of threads per block
    dim3 grid(ceil((float)int_array_length / blockSize)); // number of blocks in grid

    auto start = std::chrono::high_resolution_clock::now();
    GPU_Simple::getMax << < grid, threads >> > (dev_int_array, dev_int_array_length); // start kernel (executed on device)
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>elapsed = finish - start;
    cout << setprecision(8) << elapsed.count() * 1000;

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code % d after launching addKernel!\n", cudaStatus);
        exit;
    }

    // Copy output vector from GPU buffer to host memory.
    int blockMax = 0;
    for (int i = 0; i < grid.x; i++) {
        cudaStatus = cudaMemcpy(&blockMax, dev_int_array + blockSize * i, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            printf("cudaMemcpy failed!");
            exit;
        }
        host_maxVal = max(host_maxVal, blockMax);
    }

    cudaFree(dev_int_array);
    cudaFree(dev_int_array_length);

    return host_maxVal;
}

__global__ void GPU_Simple::getMovingAvg(const int* int_array_length, float* array_smooth, const int* avg_legth) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < (*int_array_length) && index >= (*avg_legth)){
        float movingAvgVal = 0;
        for (int i = 0; i < *avg_legth; i ++) {
            movingAvgVal += int_array_CM[index - i];
        }
        array_smooth[index] =  movingAvgVal / *avg_legth;
    }
}

void GPU_Simple::launch_getMovingAvg(const int* int_array, const int int_array_length, float* array_smooth, const int avg_legth) {
    assert(int_array_length == 16384, "datalenghth is not fit for constant memory");
    int* dev_int_array_length = nullptr;
    float* dev_array_smooth = nullptr;
    int* dev_avg_legth = nullptr;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit;
    }
    //cudaStatus = cudaMalloc((void**)&dev_int_array, int_array_length * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    printf("cudaMalloc failed!");
    //    exit;
    //}
    cudaStatus = cudaMalloc((void**)&dev_int_array_length, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_array_smooth, int_array_length * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_avg_legth, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    ////cudaStatus = cudaMemcpy(dev_int_array, int_array, int_array_length * sizeof(int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpyToSymbol(int_array_CM, int_array, int_array_length * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpyToSymbol failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_int_array_length, &int_array_length, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_avg_legth, &avg_legth, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }

    constexpr int blockSize = 1024;
    dim3 threads(blockSize); // number of threads per block
    dim3 grid(ceil((float)int_array_length / blockSize)); // number of blocks in grid

    auto start = std::chrono::high_resolution_clock::now();
    GPU_Simple::getMovingAvg << < grid, threads >> > (dev_int_array_length, dev_array_smooth, dev_avg_legth); // start kernel (executed on device)
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>elapsed = finish - start;
    cout << setprecision(8) << elapsed.count() * 1000;

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code % d after launching addKernel!\n", cudaStatus);
        exit;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(array_smooth, dev_array_smooth, int_array_length * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }

    cudaFree(dev_int_array_length);
    cudaFree(dev_array_smooth);
    cudaFree(dev_avg_legth);
}

__global__ void GPU_Better::addUp(int* int_array, const int* int_array_length) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int stride = blockDim.x; stride > 1;) {
        __syncthreads();
        stride >> 1;
        if (index < stride) {
            if (index < *int_array_length && index + stride < *int_array_length) {
                int_array[index] += int_array[index + stride];
            }
        }
    }
}

int GPU_Better::launch_addUp(const int* int_array, const int int_array_length) {
    int* dev_int_array = nullptr;
    int* dev_int_array_length = nullptr;
    int host_maxVal = 0;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_int_array, int_array_length * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_int_array_length, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_int_array, int_array, int_array_length * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_int_array_length, &int_array_length, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }

    constexpr int blockSize = 1024;
    dim3 threads(blockSize); // number of threads per block
    dim3 grid(ceil((float)int_array_length / blockSize)); // number of blocks in grid

    auto start = std::chrono::high_resolution_clock::now();
    GPU_Simple::addUp << < grid, threads >> > (dev_int_array, dev_int_array_length); // start kernel (executed on device)
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>elapsed = finish - start;
    cout << setprecision(8) << elapsed.count() * 1000;

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code % d after launching addKernel!\n", cudaStatus);
        exit;
    }

    // Copy output vector from GPU buffer to host memory.
    int blockSum = 0;
    for (int i = 0; i < grid.x; i++) {
        cudaStatus = cudaMemcpy(&blockSum, dev_int_array + blockSize * i, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            printf("cudaMemcpy failed!");
            exit;
        }
        host_maxVal += blockSum;
    }

    cudaFree(dev_int_array);
    cudaFree(dev_int_array_length);

    return host_maxVal;
}

__global__ void GPU_Better::getMax(int* int_array, const int* int_array_length) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int stride = blockDim.x; stride > 1;) {
        __syncthreads();
        stride >> 1;
        if (index < stride) {
            if (index < *int_array_length && index + stride < *int_array_length) {
                int_array[index] = max(int_array[index], int_array[index + stride]);
            }
        }
    }
}

int GPU_Better::launch_getMax(const int* int_array, const int int_array_length) {
    int* dev_int_array = nullptr;
    int* dev_int_array_length = nullptr;
    int host_maxVal = 0;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_int_array, int_array_length * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_int_array_length, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_int_array, int_array, int_array_length * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_int_array_length, &int_array_length, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }

    constexpr int blockSize = 1024;
    dim3 threads(blockSize); // number of threads per block
    dim3 grid(ceil((float)int_array_length / blockSize)); // number of blocks in grid

    auto start = std::chrono::high_resolution_clock::now();
    GPU_Simple::getMax << < grid, threads >> > (dev_int_array, dev_int_array_length); // start kernel (executed on device)
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>elapsed = finish - start;
    cout << setprecision(8) << elapsed.count() * 1000;

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code % d after launching addKernel!\n", cudaStatus);
        exit;
    }

    // Copy output vector from GPU buffer to host memory.
    int blockMax = 0;
    for (int i = 0; i < grid.x; i++) {
        cudaStatus = cudaMemcpy(&blockMax, dev_int_array + blockSize * i, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            printf("cudaMemcpy failed!");
            exit;
        }
        host_maxVal = max(host_maxVal, blockMax);
    }

    cudaFree(dev_int_array);
    cudaFree(dev_int_array_length);

    return host_maxVal;
}

__global__ void GPU_Tiled::getMovingAvg(const int* int_array_length, float* array_smooth, const int* avg_legth) {
    constexpr int tile_Width = 1024;
    constexpr short halo = 64 - 1;
    constexpr int sharedMemoryLength = tile_Width + halo;
    __shared__ int int_array_SM[sharedMemoryLength];

    const short filter_length = *avg_legth;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < (*int_array_length) && index >= filter_length) {
        for (short i = 0; i < ceil(((float)sharedMemoryLength) / blockDim.x); i++) {
            if (threadIdx.x + i * tile_Width < sharedMemoryLength) {
                int_array_SM[threadIdx.x + i * tile_Width] = int_array_CM[index - halo + i * tile_Width];
            }
        }
        __syncthreads();
        float movingAvgVal = 0;
        for (short i = 0; i < filter_length; i++) {
            movingAvgVal += int_array_SM[threadIdx.x + i];
        }
        array_smooth[index] = movingAvgVal / filter_length;
    }
}

void GPU_Tiled::launch_getMovingAvg(const int* int_array, const int int_array_length, float* array_smooth, const int avg_legth) {
    assert(int_array_length == 16384, "datalenghth is not fit for constant memory");
    assert(avg_legth == 64);
    int* dev_int_array_length = nullptr;
    float* dev_array_smooth = nullptr;
    int* dev_avg_legth = nullptr;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit;
    }
    //cudaStatus = cudaMalloc((void**)&dev_int_array, int_array_length * sizeof(int));
    //if (cudaStatus != cudaSuccess) {
    //    printf("cudaMalloc failed!");
    //    exit;
    //}
    cudaStatus = cudaMalloc((void**)&dev_int_array_length, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_array_smooth, int_array_length * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    cudaStatus = cudaMalloc((void**)&dev_avg_legth, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        exit;
    }
    //cudaStatus = cudaMemcpy(dev_int_array, int_array, int_array_length * sizeof(int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpyToSymbol(int_array_CM, int_array, int_array_length * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_int_array_length, &int_array_length, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }
    cudaStatus = cudaMemcpy(dev_avg_legth, &avg_legth, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }

    constexpr int blockSize = 1024;
    dim3 threads(blockSize); // number of threads per block
    dim3 grid(ceil((float)int_array_length / blockSize)); // number of blocks in grid

    auto start = std::chrono::high_resolution_clock::now();
    GPU_Tiled::getMovingAvg << < grid, threads >> > (dev_int_array_length, dev_array_smooth, dev_avg_legth); // start kernel (executed on device)s
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>elapsed = finish - start;
    cout << setprecision(8) << elapsed.count() * 1000;

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code % d after launching addKernel!\n", cudaStatus);
        exit;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(array_smooth, dev_array_smooth, int_array_length * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!");
        exit;
    }

    cudaFree(dev_int_array_length);
    cudaFree(dev_array_smooth);
    cudaFree(dev_avg_legth);
}
