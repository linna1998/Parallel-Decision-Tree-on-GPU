#pragma once
#include <math.h>
#include <vector>
#include <assert.h>
#include <memory.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// global variables
// record the information of all histograms
extern float* histogram;

void print_array(float* histo);
float *get_histogram_array(int histogram_id, int feature_id, int label);
float *get_histogram_array(float *histo, int histogram_id, int feature_id, int label);
int get_total_array(int histogram_id, int feature_id, int label);
float sum_array(int histogram_id, int feature_id, int label, float value);
void merge_array_pointers(float *histo1, float *histo2);
void merge_array(int histogram_id1, int feature_id1, int label1, int histogram_id2, int feature_id2, int label2);
void uniform_array(std::vector<float> &u, int histogram_id, int feature_id, int label, float* histo);
void update_array(int histogram_id, int feature_id, int label, float value);

/*
 * For A[][M][N][Z]
 * A[i][j][k][e] = A[N*Z*M*i+Z*N*j+k*Z+e]
 */
int RLOC(int i, int j, int k, int e, int M, int N, int Z);

/*
 * For A[][M][N][Z]
 * A[i][j][k] = A[N*Z*M*i+Z*N*j+k*Z]
 */
int RLOC(int i, int j, int k, int M, int N, int Z);

/*
 * For A[][M][N][Z]
 * A[i][j] = A[N*Z*M*i+Z*N*j]
 */
int RLOC(int i, int j, int M, int N, int Z);

/*
 * For A[][M][N][Z]
 * A[i] = A[N*Z*M*i]
 */
int RLOC(int i, int M, int N, int Z);