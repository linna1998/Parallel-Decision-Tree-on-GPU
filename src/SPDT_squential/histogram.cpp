#include "histogram.h"
#include "math.h"

BinTriplet::BinTriplet() {	
	this->value = 0;
	this->freq = 0;	
}

BinTriplet::BinTriplet(double _value, int _freq) {
	this->value = _value;
	this->freq = _freq;
}


Histogram::Histogram(const int max_bin, BinTriplet* _bins) {	
	this->bins = _bins;
	this->max_bin = max_bin;
	this->bin_size = 0;	
}

Histogram::Histogram(const int max_bin) {	
	this->max_bin = max_bin;
	this->bin_size = 0;	
}

// build vector of BinTriplet from histogram
void Histogram::ptr2vec(std::vector<BinTriplet>& res) {	
	int i = 0;
	res.clear();

	for (i = 0; i < bin_size; i++) {
		res.push_back(bins[i]);
	}	
}

// build histogram from vector into Histogram
void Histogram::vec2ptr(std::vector<BinTriplet>& vec) {	
	int i = 0;
	bin_size = vec.size();

	for (i = 0; i < bin_size; i++) {
		bins[i] = vec[i];
	}

	vec.clear();
	vec.shrink_to_fit();	
}

void Histogram::sortBin() {
	std::vector<BinTriplet> vec;
	ptr2vec(vec);
	// printf("sortBin: size=%d\n", vec.size());
	sort(vec.begin(), vec.end(), [](const BinTriplet &a, const BinTriplet &b) {
		return a.value < b.value;
	});
	vec2ptr(vec);
	// printf("sortBin: end\n");
}


void Histogram::mergeBin() {

	BinTriplet newbin;
	int index = 0;

	std::vector<BinTriplet> vec;
	ptr2vec(vec);
	// printf("mergeBin: begin size=%d\n", vec.size());

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
		+ vec[index + 1].value * (vec[index + 1].freq + vec[index + 1].freq)) /
		newbin.freq;

	// erase vec[index + 1]
	// change vec[index] with newbin
	vec.erase(vec.begin() + index + 1);
	vec[index] = newbin;
	vec2ptr(vec);
	// printf("mergeBin: end\n");

}

void Histogram::update(double value) {	
	std::vector<BinTriplet> vec;
	ptr2vec(vec);
	printf("update: begin size=%d\n", vec.size());

	// If there are values in the bin equals to the value here
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i].value == value) {			
			vec[i].freq++;
			vec2ptr(vec);
			printf("update: end\n");
			return;
		}
	}

	vec.push_back(BinTriplet(value, 1));

	sortBin();
	printf("size=%d, max_bin=%d", vec.size(), max_bin);
	if (vec.size() <= max_bin) {
		vec2ptr(vec);
		printf("update: end\n");
		return;
	}
	
	mergeBin();
	
	vec2ptr(vec);
	printf("update: end\n");
	return;
}

double Histogram::sum(double value) {	
	std::vector<BinTriplet> vec;
	int index = 0;
	double mb = 0;
	double s = 0;
	ptr2vec(vec);
	for (index = 0; index < vec.size() - 1; index++) {
		if (vec[index].value <= value && vec[index + 1].value > value) {
			break;
		}
	}

	mb = (vec[index + 1].freq - vec[index].freq);
	mb = mb * (value - vec[index].value);
	mb = mb / (vec[index + 1].value - vec[index].value);
	mb = vec[index].value + mb;

	s = (vec[index].freq + mb) / 2;
	s = s * (value - vec[index].value);
	s = s / (vec[index + 1].value - vec[index].value);

	for (int j = 0; j < index; j++) {
		s = s + vec[j].freq;
	}

	s = s + vec[index].freq;
	vec2ptr(vec);	
	return s;
}

void Histogram::merge(Histogram &h, int B) {
	printf("Histogram::merge begin\n");
	int index = 0;
	std::vector<BinTriplet> vec;
	ptr2vec(vec);

	printf("old size: %d\n", vec.size());

	std::vector<BinTriplet> hvec;
	h.ptr2vec(hvec);

	printf("insert size: %d\n", hvec.size());

	vec.insert(vec.end(), hvec.begin(), hvec.end());

	sortBin();

	while (vec.size() > B) {
		mergeBin();
	}
	printf("new size: %d\n", vec.size());

	vec2ptr(vec);
	h.vec2ptr(hvec);	
	
	printf("Histogram::merge end\n");
	return;
}

void Histogram::uniform(std::vector<double> &u, int B) {
	printf("Histogram::uniform begin. B=%d\n", B);
	double tmpsum = 0;
	double s = 0;
	int index = 0;
	double a = 0, b = 0, c = 0, d = 0, z = 0;	
	double uj = 0;
	std::vector<BinTriplet> vec;
	ptr2vec(vec);

	u.clear();

	if (vec.size() == 0) {
		printf("Histogram::uniform, vec.size() = 0, end\n");
		return;
	}

	for (int i = 0; i < vec.size(); i++) {
		tmpsum += vec[i].freq;
	}	

	for (int j = 0; j <= B - 2; j++) {
		s = tmpsum * j / B;		
		
		for (index = 0; index < (int)vec.size() - 1; index++) {

			if (sum(vec[index].value) < s
				&& s < sum(vec[index + 1].value)) {
				break;
			}
		}

		d = s - sum(vec[index].value);

		a = vec[index + 1].freq - vec[index].freq;
		b = 2 * vec[index].freq;
		c = -2 * d;
		z = -b + sqrt(b * b - 4 * a * c);
		z = z / (2 * a);

		uj = vec[index].value + z * (vec[index + 1].value - vec[index].value);
		u.push_back(uj);
	}
	vec2ptr(vec);
	printf("Histogram::uniform end\n");
	return;
}

void Histogram::clear() {
	bin_size = 0;
	bins = NULL;
}