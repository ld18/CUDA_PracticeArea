
#include <chrono> 
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <iterator>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cpu.h"
#include "gpu.cuh"
#include "data.h"

using namespace std;

int main()
{
	int res_int = 0;
	float res_float = 0;
	auto start = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double>elapsed = finish - start;

	cudaDeviceProp devProp;
	cudaError_t cudaStatus = cudaGetDeviceProperties(&devProp, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceProperties failed!");
		exit(1);
	}
	cout << "Cuda Device" << endl << endl;
	printf("name: \t%s \n", devProp.name);
	printf("warpSize: \t%d \n", devProp.warpSize);
	printf("clockRate: \t%d \n", devProp.clockRate);
	printf("concurrentKernels: \t%d \n", devProp.concurrentKernels);
	printf("maxGridSize.x: \t%d \n", devProp.maxGridSize[0]);
	printf("maxGridSize.y: \t%d \n", devProp.maxGridSize[1]);
	printf("maxGridSize.z: \t%d \n", devProp.maxGridSize[2]);
	printf("maxThreadsDim.x: \t%d \n", devProp.maxThreadsDim[0]);
	printf("maxThreadsDim.y: \t%d \n", devProp.maxThreadsDim[1]);
	printf("maxThreadsDim.z: \t%d \n", devProp.maxThreadsDim[2]);
	printf("maxThreadsPerBlock: \t%d \n", devProp.maxThreadsPerBlock);
	printf("sharedMemPerBlock: \t%d \n", devProp.sharedMemPerBlock);

	cout << endl << endl;
	cout << "Started Performance comapre" << endl;
	cout << "Used data is a array of ints (0-111) with the length " << int_array_length << ". " << avg_legth << " values are used for the moving average calculation." << endl;

	cout << endl << endl;
	cout << "Beginning with CPU" << endl;
	res_int = CPU_Baseline::addUp(int_array, int_array_length); //Prime Call: first cuda function call is significant slower then second

	cout << endl << "addUp:  \t\t";
	cout << "\tRuntime ";
	start = std::chrono::high_resolution_clock::now();
	res_int = CPU_Baseline::addUp(int_array, int_array_length);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << elapsed.count() * 1000;
	cout << "\tSum(" << res_int << ")";

	cout << endl << "2 threads addUp:\t";
	cout << "\tRuntime ";
	start = std::chrono::high_resolution_clock::now();
	res_int = CPU_Threaded::launch_addUp(int_array, int_array_length);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << elapsed.count() * 1000;
	cout << "\tSum(" << res_int << ")";

	cout << endl << "getMax: \t";
	cout << "\tRuntime ";
	start = std::chrono::high_resolution_clock::now();
	res_int = CPU_Baseline::getMax(int_array, int_array_length);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << elapsed.count() * 1000;
	cout << "\tMax(" << res_int << ")";

	cout << endl << "getAvg: \t";
	cout << "\tRuntime ";
	start = std::chrono::high_resolution_clock::now();
	res_float = CPU_Baseline::getAvg(int_array, int_array_length);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << elapsed.count() * 1000;
	cout << "\tAvg(" << res_float << ")";

	cout << endl << "getMovingAvg:\t\t";
	cout << "\tRuntime ";
	start = std::chrono::high_resolution_clock::now();
	float array_smooth[int_array_length] = { 0 };
	CPU_Baseline::getMovingAvg(int_array, int_array_length, array_smooth, avg_legth);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << elapsed.count() * 1000;
	cout << setprecision(4) << "\tMovAvg(" << array_smooth[avg_legth - 1] << ", " << array_smooth[avg_legth] << ", " << array_smooth[100] << ", " << array_smooth[1030] << ", " << array_smooth[int_array_length - 10] << ", " << array_smooth[int_array_length - 60] << ", " << array_smooth[int_array_length - 50] << ", " << array_smooth[int_array_length - 1] << ")";

	cout << endl << "2 threads getMovingAvg:\t";
	cout << "\tRuntime ";
	start = std::chrono::high_resolution_clock::now();
	float array_smooth_4[int_array_length] = { 0 };
	CPU_Threaded::launch_getMovingAvg(int_array, int_array_length, array_smooth_4, avg_legth);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << elapsed.count() * 1000;
	cout << setprecision(4) << "\tMovAvg(" << array_smooth_4[avg_legth - 1] << ", " << array_smooth_4[avg_legth] << ", " << array_smooth_4[100] << ", " << array_smooth_4[1030] << ", " << array_smooth_4[int_array_length - 10] << ", " << array_smooth_4[int_array_length - 60] << ", " << array_smooth_4[int_array_length - 50] << ", " << array_smooth_4[int_array_length - 1] << ")";

	cout << endl << endl;
	cout << "Beginning with GPU " << endl;
	cout << "Priming call (because first run takes longer, just ignore) ";
	res_int = GPU_Simple::launch_addUp(int_array, int_array_length); //Prime Call: first cuda function call is significant slower then second
	cout << endl;

	cout << endl << "addUp:  \t";
	cout << "\tKernel_Runtime ";
	start = std::chrono::high_resolution_clock::now();
	res_int = GPU_Simple::launch_addUp(int_array, int_array_length);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << "\tOverall_Runtime " << elapsed.count() * 1000;
	cout << "\tSum(" << res_int << ")";

	cout << endl << "faster addUp:\t";
	cout << "\tKernel_Runtime ";
	start = std::chrono::high_resolution_clock::now();
	res_int = GPU_Better::launch_addUp(int_array, int_array_length);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << "\tOverall_Runtime " << elapsed.count() * 1000;
	cout << "\tSum(" << res_int << ")";

	cout << endl << "getMax: \t";
	cout << "\tKernel_Runtime ";
	start = std::chrono::high_resolution_clock::now();
	res_int = GPU_Simple::launch_getMax(int_array, int_array_length);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << "\tOverall_Runtime " << elapsed.count() * 1000;
	cout << setprecision(4) << "\tMax(" << res_int << ")";
	cout << endl << "faster getMax:\t";
	cout << "\tKernel_Runtime ";
	start = std::chrono::high_resolution_clock::now();
	res_int = GPU_Better::launch_getMax(int_array, int_array_length);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << "\tOverall_Runtime " << elapsed.count() * 1000;
	cout << setprecision(4) << "\tMax(" << res_int << ")";

	cout << endl << "getMovingAvg:\t";
	cout << "\tKernel_Runtime ";
	float array_smooth_2[int_array_length] = { 0 };
	start = std::chrono::high_resolution_clock::now();
	GPU_Simple::launch_getMovingAvg(int_array, int_array_length, array_smooth_2, avg_legth);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << "\tOverallt_Runtime " << elapsed.count() * 1000;
	cout << setprecision(4) << "\tMovAvg(" << array_smooth[avg_legth - 1] << ", " << array_smooth[avg_legth] << ", " << array_smooth[100] << ", " << array_smooth[1030] << ", " << array_smooth[int_array_length - 10] << ", " << array_smooth[int_array_length - 60] << ", " << array_smooth[int_array_length - 50] << ", " << array_smooth[int_array_length - 1] << ")";
	cout << endl << "tiled getMovingAvg:\t";
	cout << "Kernel_Runtime ";
	float array_smooth_3[int_array_length] = { 0 };
	start = std::chrono::high_resolution_clock::now();
	GPU_Tiled::launch_getMovingAvg(int_array, int_array_length, array_smooth_3, avg_legth);
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	cout << setprecision(6) << "\tOverallt_Runtime " << elapsed.count() * 1000;
	cout << setprecision(4) << "\tMovAvg(" << array_smooth[avg_legth - 1] << ", " << array_smooth[avg_legth] << ", " << array_smooth[100] << ", " << array_smooth[1030] << ", " << array_smooth[int_array_length - 10] << ", " << array_smooth[int_array_length - 60] << ", " << array_smooth[int_array_length - 50] << ", " << array_smooth[int_array_length - 1] << ")";


	cout << endl << endl;
}