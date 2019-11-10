#include "parser.h"

void Data::read_a_data(int num_of_features, ifstream* myfile) {	
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
			if (label == -1) label = 0;
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
			values[index] = tmpvalue;
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
		dataset[i].read_a_data(num_of_features, &myfile);		
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

void Dataset::print_dataset() {
	cout << "begin printing dataset" << endl;
	for (int i = 0; i < dataset.size(); i++) {
		cout << "data id: " << i << endl;
		cout << "label: " << dataset[i].label << endl;

		for (map<int, double>::iterator it = dataset[i].values.begin(); it != dataset[i].values.end(); it++) {
			cout << "index: " << it->first << " value: " << it->second << endl;
		}
	}

}

//int main() {
//	Dataset testDataset(3, 391, 20);
//	bool hasNext = true;
//
//	testDataset.open_read_data("./data/svmguide2.txt");
//
//	while (true) {
//		hasNext = testDataset.streaming_read_data(10);
//		testDataset.print_dataset();
//
//		if (!hasNext) break;
//	}		
//
//	testDataset.close_read_data();
//
//	return 0;
//}