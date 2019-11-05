#include "parser.h"

void Data::read_a_data(int num_of_features, ifstream* myfile) {
	int index;
	double tmpvalue;
	char c;
	*myfile >> label;
	for (int i = 0; i < num_of_features; i++) {
		*myfile >> index;
		*myfile >> c;
		*myfile >> tmpvalue;
		values[index] = tmpvalue;
	}
}

void Dataset::open_read_data(string name) {
	myfile.open(name, fstream::in);
}

/* return whether there are still data left or not */
bool Dataset::streaming_read_data(int N) {
	Data tmpdata;
	dataset.clear();

	for (int i = 0; i < N; i++) {
		tmpdata.read_a_data(num_of_features, &myfile);
		dataset.push_back(tmpdata);
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