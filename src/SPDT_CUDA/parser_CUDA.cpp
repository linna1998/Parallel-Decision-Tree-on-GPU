#include "parser_CUDA.h"
#include "array_CUDA.h"
#include "tree_CUDA.h"

Dataset::Dataset() {
	num_pos_label=0;
}

Dataset::Dataset(int _num_of_data):		
	num_of_data(_num_of_data) {
	already_read_data = 0;
	num_pos_label = 0;
	label_ptr = (int*)calloc(_num_of_data, sizeof(int));
	value_ptr = (float*)calloc(_num_of_data * num_of_features, sizeof(float));
	histogram_id_ptr = (int*)calloc(_num_of_data, sizeof(int));
}

Dataset::~Dataset() {	
	free(label_ptr);
	free(value_ptr);
	free(histogram_id_ptr);	
}

void Dataset::read_a_data(int index) {		
	float tmpvalue;	
	
	bool isFirst = true;
	int feature;

	string str, prev, indexstr, valuestr;
	getline(myfile, str);	

	size_t npos;

	// cout << "str: " << str << endl;	

	while (str.size() > 0) {
		npos = str.find(' ');
		if (npos == string::npos) break;
		prev = str.substr(0, npos);		
		str = str.substr(npos + 1);
		if (prev.size() == 0 || prev[0] < '+') continue;				
		if (isFirst) {
			// cout << "prev: " << prev << endl;
			label_ptr[index] = stoi(prev) == -1 ? 0 : stoi(prev);					
			isFirst = false;
		} else {
			// cout << "prev: " << prev << endl;
			npos = prev.find(':');
			indexstr = prev.substr(0, npos);
			// cout << "indexstr: " << indexstr << endl;
			feature = stoi(indexstr);
			valuestr = prev.substr(npos + 1);
			// cout << "valuestr: " << valuestr << endl;
			tmpvalue = stod(valuestr);
			// cout << "index: " << index << "tmpvalue: " << tmpvalue << endl;
			value_ptr[index * num_of_features + feature - 1] = tmpvalue;			
		}
	}
}

void Dataset::open_read_data(string name) {
	myfile.open(name, fstream::in);
}

/* return whether there are still data left or not */
bool Dataset::streaming_read_data(int N) {		
	for (int i = 0; i < N; i++) {
		read_a_data(i);		
		if (label_ptr[i] == POS_LABEL) num_pos_label++;
		already_read_data++;
		if (already_read_data == num_of_data) {
			break;
		}
	}

	return (already_read_data < num_of_data);
}

void Dataset::close_read_data() {
	myfile.close();
}
