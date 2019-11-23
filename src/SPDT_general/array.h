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

#define EPS 1e-9

// global variables
// record the information of all histograms
static double* histogram;
static int num_of_features;
static int num_of_classes = 2;
static int max_bin_size = 16;

int get_total_array(int histogram_id, int feature_id, int label);
double sum_array(int histogram_id, int feature_id, int label, double value);
void merge_array(int histogram_id1, int feature_id1, int label1, int histogram_id2, int feature_id2, int label2);
void uniform_array(std::vector<double> &u, int histogram_id, int feature_id, int label);
void update_array(int histogram_id, int feature_id, int label, double value);

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