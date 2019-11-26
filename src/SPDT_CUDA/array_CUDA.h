#pragma once

#ifdef __CUDACC__
#define CUDA_HOST __host__ 
#define CUDA_DEVICE __device__
#else
#define CUDA_HOST
#define CUDA_DEVICE
#endif

#include <math.h>
#include <vector>
#include <assert.h>
#include <memory.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define EPS 1e-9

// global variables
// record the information of all histograms
extern float* histogram;

int get_total_array(int histogram_id, int feature_id, int label);
float sum_array(int histogram_id, int feature_id, int label, float value);
void merge_array_pointers(float *histo1, float *histo2, int max_bin_size);
void merge_array(int histogram_id1, int feature_id1, int label1, int histogram_id2, int feature_id2, int label2);
void uniform_array(std::vector<float> &u, int histogram_id, int feature_id, int label);
extern CUDA_DEVICE void update_array(int histogram_id, int feature_id, int label, float value,
									 int num_of_features, int num_of_classes, int max_bin_size, float* histogram);

/*
 * For A[][M][N][Z]
 * A[i][j][k][e] = A[N*Z*M*i+Z*N*j+k*Z+e]
 */
inline int RLOC(int i, int j, int k, int e, int M, int N, int Z);

/*
 * For A[][M][N][Z]
 * A[i][j][k] = A[N*Z*M*i+Z*N*j+k*Z]
 */
inline int RLOC(int i, int j, int k, int M, int N, int Z);

/*
 * For A[][M][N][Z]
 * A[i][j] = A[N*Z*M*i+Z*N*j]
 */
inline int RLOC(int i, int j, int M, int N, int Z);

/*
 * For A[][M][N][Z]
 * A[i] = A[N*Z*M*i]
 */
inline int RLOC(int i, int M, int N, int Z);

extern CUDA_HOST CUDA_DEVICE float get_bin_size(float* histo);
extern CUDA_HOST CUDA_DEVICE float *get_histogram_array(int histogram_id, int feature_id, int label,
	float *histogram, int num_of_features, int num_of_classes, int max_bin_size);