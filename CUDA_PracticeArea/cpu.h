#pragma once

#include <algorithm>
#include <future>
#include <functional>
#include <algorithm>
#include <vector>

using namespace std;

namespace CPU_Baseline {

	//Vanilla base cpu implementations
	int addUp(const int* int_array, const int& int_array_length) {
		int sumVal = 0;
		for (int i = 0; i < int_array_length; i++) {
			sumVal += int_array[i];
		}
		return sumVal;
	}
	int getMax(const int* int_array, const int& int_array_length) {
		int maxVal = int_array[0];
		for (int i = 1; i < int_array_length; i++) {
			maxVal = max(maxVal, int_array[i]);
		}
		return maxVal;
	}
	float getAvg(const int* int_array, const int& int_array_length) {
		float avgVal = 0;
		for (int i = 0; i < int_array_length; i++) {
			avgVal += int_array[i];
		}
		return avgVal / int_array_length;
	}
	void getMovingAvg(const int* int_array, const int& int_array_length, float* array_smooth, const int& avg_legth) {
		for (int i = 0; i < int_array_length; i++) {
			if (i >= avg_legth) {
				float movingAvgVal = 0;
				for (int c = 0; c < avg_legth; c++) {
					movingAvgVal += int_array[i - c];
				}
				array_smooth[i] = movingAvgVal / avg_legth;
			}
		}
	}
}
namespace CPU_Threaded {
	int launch_addUp(const int* int_array, const int& int_array_length) {
		const int maxNumberOfThreads = max((int)thread::hardware_concurrency(), 4); //can return 0, for some plattforms
		const int numberOfThreads = 2;// maxNumberOfThreads;
		const int chunkSize = int_array_length / numberOfThreads;
		vector<future<int>> results;
		for (int i = 0; i < numberOfThreads - 1; ++i) {
			results.push_back(async(launch::deferred, CPU_Baseline::addUp, int_array + i * chunkSize, chunkSize));
		}
		int sum = CPU_Baseline::addUp(int_array + (numberOfThreads - 1) * chunkSize, chunkSize);
		for (int i = 0; i < int_array_length % numberOfThreads; ++i) {
			sum += int_array[i + chunkSize * numberOfThreads];
		}
		for (int i = 0; i < numberOfThreads - 1; ++i) {
			sum += results[i].get();
		}
		return sum;
	}
	void getMovingAvg(const int* int_array, const int& int_array_length, float* array_smooth, const int& data_position, const int& data_lenght, const int& avg_legth) {
		float movingAvgVal = 0;
		for (int i = 0; i < data_lenght; i++) {
			if (data_position + i >= avg_legth) {
				movingAvgVal = 0;
				for (unsigned short c = 0; c < avg_legth; c++) {
					movingAvgVal += int_array[data_position + i - c];
				}
				array_smooth[data_position + i] = movingAvgVal / avg_legth;
			}
		}
	}
	void launch_getMovingAvg(const int* int_array, const int& int_array_length, float* array_smooth, const int& avg_legth) {
		const int maxNumberOfThreads = max((int)thread::hardware_concurrency(), 4); //can return 0, for some plattforms
		const int numberOfThreads = 2;// maxNumberOfThreads;
		const int chunkSize = int_array_length / numberOfThreads;
		const int chunkSize_remainder = int_array_length % chunkSize;
		vector<future<void>> results;
		for (int i = 0; i < numberOfThreads - 1; ++i) {
			results.push_back(async(launch::async, getMovingAvg, int_array, int_array_length, array_smooth, i * chunkSize, chunkSize, avg_legth));
		}
		getMovingAvg(int_array, int_array_length, array_smooth, (numberOfThreads - 1) * chunkSize, chunkSize + chunkSize_remainder, avg_legth);
		for (int i = 0; i < numberOfThreads - 1; ++i) {
			results[i].wait();
		}	
	}

}
