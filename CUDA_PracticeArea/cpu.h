#pragma once

#include <algorithm>

using namespace std;

namespace CPU {

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
