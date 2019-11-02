#include <fstream>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

class Data {
public:
	int label;
	map<int, double> values;

	void read_a_data(int num_of_features, ifstream* myfile);
};

class Dataset {
public:
	int num_of_classes;
	int num_of_data;
	int num_of_features;
	vector<Data> dataset;	
	ifstream myfile;

	Dataset() {}
	Dataset(int _num_of_classes, int _num_of_data, int _num_of_features):
		num_of_classes(_num_of_classes),
		num_of_data(_num_of_data),
		num_of_features(_num_of_features) {}

	void open_read_data(string name);

	void streaming_read_data(int N);

	void close_read_data();

	void print_dataset();

};