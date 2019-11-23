#include "histogram.h"
#include "math.h"
#include "tree.h"

BinTriplet::BinTriplet() {	
	this->value = 0;
	this->freq = 0;	
}

BinTriplet::BinTriplet(double _value, int _freq) {
	this->value = _value;
	this->freq = _freq;
}


Histogram::Histogram(const int max_bin, BinTriplet* _bins) {	
	if (_bins == NULL) {
		printf("Initialize Histogram with null pointer\n");
		exit(1);
	}
	this->bins = _bins;
	// A trick here
	// Use the last empty place for in-place update function
	this->max_bin = max_bin - 1;
	this->bin_size = 0;	
}

Histogram::Histogram(const int max_bin) {	
	this->max_bin = max_bin - 1;
	this->bin_size = 0;	
}

int Histogram::get_total() {
	int t = 0;	
	int i = 0;	
	// #pragma omp parallel private(i)
	{
		// #pragma omp for schedule(dynamic) reduction(+:t)
		for (i = 0; i < bin_size; i++){
			t += bins[i].freq;
		}
	}
	return t;
}

// build vector of BinTriplet from histogram
void Histogram::ptr2vec(std::vector<BinTriplet>& res) {	
	int i = 0;
	res.clear();
	res.shrink_to_fit();

	for (i = 0; i < bin_size; i++) {		
		res.push_back(BinTriplet(bins[i].value, bins[i].freq));
	}		
}

// build histogram from vector into Histogram
void Histogram::vec2ptr(std::vector<BinTriplet>& vec) {	
	
	int i = 0;
	bin_size = vec.size();		

	for (i = 0; i < bin_size; i++) {
		bins[i].freq = vec[i].freq;
		bins[i].value = vec[i].value;
	}	
	vec.clear();
	vec.shrink_to_fit();	
}

void sortBin(std::vector<BinTriplet>& vec) {	
	sort(vec.begin(), vec.end(), [](const BinTriplet &a, const BinTriplet &b) {
		return a.value < b.value;
	});	
}

void printVector(std::vector<BinTriplet>& vec) {
	printf("size: %d\n", vec.size());

	for (int i = 0; i < vec.size(); i++) {
		printf("%d: (%f, %d) ", i, vec[i].value, vec[i].freq);
	}	
}

void mergeBin(std::vector<BinTriplet>& vec) {
	BinTriplet newbin;
	int index = 0;

	// find the min value of difference
	for (int i = 0; i < vec.size() - 1; i++) {
		if (vec[i + 1].value - vec[i].value
			< vec[index + 1].value - vec[index].value) {
			index = i;
		}
	}

	// merge vec[index], vecs[index + 1] into a new element
	newbin.freq = vec[index].freq + vec[index + 1].freq;
	newbin.value = (vec[index].value * vec[index].freq
		+ vec[index + 1].value * vec[index + 1].freq) /
		newbin.freq;

	// erase vec[index + 1]
	// change vec[index] with newbin
	vec.erase(vec.begin() + index + 1);
	vec[index].freq = newbin.freq;
	vec[index].value = newbin.value;

	mergeSame(vec);
}

// size of the bin: reduce by one
void Histogram::mergeBinArray() {
	BinTriplet newbin;
	int index = 0;

	// find the min value of difference
	for (int i = 0; i < bin_size - 1; i++) {
		if (bins[i + 1].value - bins[i].value
			< bins[index + 1].value - bins[index].value) {
			index = i;
		}
	}

	// merge bins[index], bins[index + 1] into a new element
	newbin.freq = bins[index].freq + bins[index + 1].freq;
	newbin.value = (bins[index].value * bins[index].freq
		+ bins[index + 1].value * bins[index + 1].freq) /
		newbin.freq;

	// change vec[index] with newbin
	bins[index].freq = newbin.freq;
	bins[index].value = newbin.value;	

	// erase vec[index + 1]
	for (int i = index + 1; i <= bin_size - 2; i++) {
		bins[i].freq = bins[i + 1].freq;
		bins[i].value = bins[i + 1].value;
 	}
	bin_size--;
	
	mergeSameArray();
}

void Histogram::update(double value) {	
	int index = 0;	
	// If there are values in the bin equals to the value here
	for (int i = 0; i < bin_size; i++) {
		if (abs(bins[i].value - value) < EPS) {			
			bins[i].freq++;					
			return;
		}
	}

	// put the next element into the correct place in bin_size
	// find the index to insert value
	// bins[index - 1].value < value
	// bins[index].value > value
	for (int i = 0; i < bin_size; i++) {
		if (bins[i].value > value) {
			index = i;
			break;
		}
	}

	// move the [index, bin_size - 1] an element further
	for (int i = bin_size; i >= index + 1; i--) {
		bins[i].value = bins[i - 1].value;
		bins[i].freq = bins[i - 1].freq;		
	}
	bin_size++;

	// put value into the place of bins[index]
	bins[index].value = value;
	bins[index].freq = 1;	

	if (bin_size <= max_bin) {
		return;
	}
	
	mergeBinArray();	
			
	return;
}

double Histogram::sum(double value) {
	check(__LINE__);
	std::vector<BinTriplet> vec;
	int index = 0;
	double mb = 0;
	double s = 0;
	ptr2vec(vec);

	if (bin_size == 1) {
		return vec[0].freq;
	}

	if (value < vec[0].value) {
		return 0;
	}

	if (value >= vec[vec.size() - 1].value) {
		for (int i = 0; i < vec.size(); i++) {
			s += vec[i].freq;			
		}
		return s;
	}

	for (index = 0; index + 1 < vec.size(); index++) {
		if (vec[index].value <= value && vec[index + 1].value > value) {
			break;
		}
	}		

	if (abs(vec[index + 1].value - vec[index].value) <= EPS) {
		printVector(vec);
		printf("index: %d\n", index);
		printf("value: %f\n", value);
		exit(1);
	}

	if (abs(vec[index + 1].value - vec[index].value) > EPS) {
		mb = (vec[index + 1].freq - vec[index].freq);
		mb = mb * (value - vec[index].value);
		mb = mb / (vec[index + 1].value - vec[index].value);
		mb = vec[index].freq + mb;
	} else {
		fprintf(stderr, "abs(vec[index + 1].value - vec[index].value) > EPS");
		exit(-1);
		mb = vec[index].freq;
	}
	
	if (abs(vec[index + 1].value - vec[index].value) > EPS) {
		s = (vec[index].freq + mb) / 2;
		s = s * (value - vec[index].value);
		s = s / (vec[index + 1].value - vec[index].value);
	} else {
		fprintf(stderr, "(vec[index + 1].value - vec[index].value) == 0");
		exit(-1);
		s = (vec[index].freq + mb) / 2;
	}


	for (int j = 0; j < index; j++) {
		s = s + vec[j].freq;
	}

	s = s + ((double)vec[index].freq) / 2;
	vec2ptr(vec);
	check(__LINE__);
	return s;
}

void mergeSame(std::vector<BinTriplet>& vec) {
	for (int i = 0; i + 1 < vec.size(); i++) {
		if (abs(vec[i].value - vec[i+1].value) < EPS) {
			vec[i].freq += vec[i+1].freq;
			vec.erase(vec.begin() + i + 1);
			i--;
		}
	}
}

void Histogram::mergeSameArray() {
	for (int i = 0; i + 1 < bin_size; i++) {
		if (abs(bins[i].value - bins[i + 1].value) < EPS) {
			bins[i].freq += bins[i + 1].freq;			
			// erase vec[i + 1]
			for (int j = i + 1; j <= bin_size - 2; j++) {
				bins[j].freq = bins[j + 1].freq;
				bins[j].value = bins[j + 1].value;
			}
			bin_size--;
			i--;
		}
	}
}

void Histogram::merge(Histogram &h) {
	check(__LINE__);
	int index = 0;
	std::vector<BinTriplet> vec;
	ptr2vec(vec);

	std::vector<BinTriplet> hvec;
	h.ptr2vec(hvec);

	vec.insert(vec.end(), hvec.begin(), hvec.end());
	
	sortBin(vec);

	// merge the same values in vec
	mergeSame(vec);

	while (vec.size() > max_bin) {
		mergeBin(vec);		
	}

	vec2ptr(vec);
	h.vec2ptr(hvec);	
	check(__LINE__);
	return;
}
void Histogram::print(){
	int i=0;
	printf("bin_size %d\n", bin_size);
	printf("[");
	while(i<bin_size){
		printf("(%.4f, %d) ", bins[i].value, bins[i].freq);		
		i++;
	}
	printf("]\n");

}

void Histogram::check(int lineno) {	
	#ifdef DEBUG
	int i = 0;	
	for (i = 0; i < bin_size; i++) {		
		if (bins[i].freq == 0) {
			printf("check lineno: %d\n", lineno);
			printf("%d: (%.4f, %d) ", i, bins[i].value, bins[i].freq);
			printf("Freq = 0\n");
			exit(1);
		} 	
		if (i + 1 < bin_size && abs(bins[i].value - bins[i + 1].value) < EPS) {		
			printf("check lineno: %d\n", lineno);
			printf("%d: (%.4f, %d) ", i, bins[i].value, bins[i].freq);
			printf("%d: (%.4f, %d) ", i + 1, bins[i + 1].value, bins[i + 1].freq);
			exit(1);
		}	
	}	
	#endif
}

void Histogram::uniform(std::vector<double> &u, int B) {
	check(__LINE__);
	double tmpsum = 0;
	double s = 0;
	int index = 0;
	double a = 0, b = 0, c = 0, d = 0, z = 0;	
	double uj = 0;
	std::vector<BinTriplet> vec;
	ptr2vec(vec);

	u.clear();

	if (vec.size() <= 1) {
		return;
	}
	
	for (int i = 0; i < vec.size(); i++) {
		tmpsum += vec[i].freq;
	}	

	for (int j = 0; j <= B - 2; j++) {
		s = tmpsum * (j + 1) / B;		
		
		for (index = 0; index + 1 < vec.size(); index++) {
			
			if (sum(vec[index].value) < s
				&& s < sum(vec[index + 1].value)) {
				break;
			}
		}

		d = s - sum(vec[index].value);		

		a = vec[index + 1].freq - vec[index].freq;
		b = 2 * vec[index].freq;
		c = -2 * d;		
		
		if (abs(a) > EPS && b * b - 4 * a * c >= 0) {
			z = -b + sqrt(b * b - 4 * a * c);
			z = z / (2 * a);
		} else if (abs(b) > EPS) {
			// b * z + c = 0
			z = -c / b;
		} else {
			z = 0;
		}
		if (z < 0) z = 0;
		if (z > 1) z = 1;
		
		uj = vec[index].value + z * (vec[index + 1].value - vec[index].value);		
		u.push_back(uj);				
	}
	vec2ptr(vec);	
	check(__LINE__);
	return;
}

void Histogram::clear() {
	bin_size = 0;
	bins = NULL;
}