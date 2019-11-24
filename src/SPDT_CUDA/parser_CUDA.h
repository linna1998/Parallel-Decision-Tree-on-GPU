#pragma once
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

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

	Dataset();
	Dataset(int _num_of_data);
	~Dataset();

	void open_read_data(string name);
	void read_a_data(int index);
	bool streaming_read_data(int N);
	void close_read_data();
};