#pragma once
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include "array_CUDA.h"

using namespace std;

#define POS_LABEL 1
#define NEG_LABEL 0

class Dataset {
public:	
	int num_of_data;
	int num_pos_label;
	
    // 0/1 label, might apply something as bitmap in the future
    int *label_ptr;
    float *value_ptr;
    int *histogram_id_ptr;
	
	ifstream myfile;

	int already_read_data;

	Dataset() {num_pos_label=0;}
	Dataset(int _num_of_data):		
		num_of_data(_num_of_data) {
		already_read_data = 0;
		num_pos_label = 0;
		label_ptr = (int*)calloc(_num_of_data, sizeof(int));
    	value_ptr = (float*)calloc(_num_of_data * num_of_features, sizeof(float));
    	histogram_id_ptr = (int*)calloc(_num_of_data, sizeof(int));
	}

	~Dataset() {
		free(label_ptr);
		free(value_ptr);
		free(histogram_id_ptr);
	}

	void open_read_data(string name);
	void read_a_data(int index);
	bool streaming_read_data(int N);
	void close_read_data();
};