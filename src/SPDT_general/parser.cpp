#include "parser.h"
#include "tree.h"

double Data::get_value(int feature_id) {
	if (values.find(feature_id) == values.end()) {
		return 0;
	}
	return values[feature_id];
}

void Data::read_a_data(ifstream* myfile) {	
	int index;
	double tmpvalue;	
	
	bool isFirst = true;

	string str, prev, indexstr, valuestr;
	getline(*myfile, str);	

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
			label = stoi(prev);
			// cout << "label: " << label << endl;
			
			if (label != POS_LABEL && num_of_classes == 2) label = 0;
			if (num_of_classes > 2) {
				label = label - 1;
			}

			isFirst = false;
		} else {
			// cout << "prev: " << prev << endl;
			npos = prev.find(':');
			indexstr = prev.substr(0, npos);
			// cout << "indexstr: " << indexstr << endl;
			index = stoi(indexstr);
			valuestr = prev.substr(npos + 1);
			// cout << "valuestr: " << valuestr << endl;
			tmpvalue = stod(valuestr);
			// cout << "index: " << index << "tmpvalue: " << tmpvalue << endl;
			values[index - 1] = tmpvalue;
		}
	}
}

void Dataset::open_read_data(string name) {
	myfile.open(name, fstream::in);
}

/* return whether there are still data left or not */
bool Dataset::streaming_read_data(int N) {	
	dataset.clear();
	dataset.shrink_to_fit();
	dataset = vector<Data>(N);

	for (int i = 0; i < N; i++) {
		dataset[i].read_a_data(&myfile);		
		if (dataset[i].label == POS_LABEL) num_pos_label++;
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