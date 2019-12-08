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
	this->max_bin = max_bin;
	this->bin_size = 0;	
}

Histogram::Histogram(const int max_bin) {	
	this->max_bin = max_bin;
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

void Histogram::update(double value) {	
	check(__LINE__);		
	std::vector<BinTriplet> vec;
	ptr2vec(vec);	

	// If there are values in the bin equals to the value here
	for (int i = 0; i < vec.size(); i++) {
		if (abs(vec[i].value - value) < EPS) {			
			vec[i].freq++;
			vec2ptr(vec);
			check(__LINE__);					
			return;
		}
	}

	vec.push_back(BinTriplet(value, 1));	

	sortBin(vec);		

	if (vec.size() <= max_bin) {
		vec2ptr(vec);			
		check(__LINE__);		
		return;
	}
	
	mergeBin(vec);	
	
	vec2ptr(vec);	
	check(__LINE__);	
	return;
}

double Histogram::sum(double value) {
	check(__LINE__);
	int index = 0;
	double mb = 0;
	double s = 0;

	if (value < bins[0].value) {
		return 0;
	}

	if (value >= bins[bin_size - 1].value) {
		for (int i = 0; i < bin_size; i++) {
			s += bins[i].freq;			
		}
		return s;
	}
	
	if (bin_size == 1) {
		return bins[0].freq;
	}
	
	for (index = 0; index + 1 < bin_size; index++) {
		if (bins[index].value <= value && bins[index + 1].value > value) {
			break;
		}
	}		

	if (abs(bins[index + 1].value - bins[index].value) <= EPS) {
		// printVector(vec);
		printf("index: %d\n", index);
		printf("value: %f\n", value);
		exit(1);
	}

	if (abs(bins[index + 1].value - bins[index].value) > EPS) {
		mb = (bins[index + 1].freq - bins[index].freq);
		mb = mb * (value - bins[index].value);
		mb = mb / (bins[index + 1].value - bins[index].value);
		mb = bins[index].freq + mb;
	} else {
		fprintf(stderr, "abs(vec[index + 1].value - vec[index].value) > EPS");
		exit(-1);
		mb = bins[index].freq;
	}
	
	if (abs(bins[index + 1].value - bins[index].value) > EPS) {
		s = (bins[index].freq + mb) / 2;
		s = s * (value - bins[index].value);
		s = s / (bins[index + 1].value - bins[index].value);
	} else {
		fprintf(stderr, "(vec[index + 1].value - vec[index].value) == 0");
		exit(-1);
		s = (bins[index].freq + mb) / 2;
	}

	for (int j = 0; j < index; j++) {
		s = s + bins[j].freq;
	}

	s = s + ((double)bins[index].freq) / 2;
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

void Histogram::merge(Histogram &h, int B) {
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

	while (vec.size() > B) {
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
	u.clear();

	if (bin_size <= 1) {
		return;
	}
	
	for (int i = 0; i < bin_size; i++) {
		tmpsum += bins[i].freq;
	}	

	for (int j = 0; j <= B - 2; j++) {
		s = tmpsum * (j + 1) / B;		
		
		for (index = 0; index + 1 < bin_size; index++) {
			
			if (sum(bins[index].value) < s
				&& s < sum(bins[index + 1].value)) {
				break;
			}
		}

		d = s - sum(bins[index].value);		

		a = bins[index + 1].freq - bins[index].freq;
		b = 2 * bins[index].freq;
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
		
		uj = bins[index].value + z * (bins[index + 1].value - bins[index].value);		
		u.push_back(uj);				
	}
	check(__LINE__);
	return;
}

void Histogram::clear() {
	bin_size = 0;
	bins = NULL;
}