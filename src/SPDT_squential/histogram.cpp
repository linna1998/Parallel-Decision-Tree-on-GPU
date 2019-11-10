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

void Histogram::sortBin(std::vector<BinTriplet>& vec) {	
	sort(vec.begin(), vec.end(), [](const BinTriplet &a, const BinTriplet &b) {
		return a.value < b.value;
	});	
}


void Histogram::mergeBin(std::vector<BinTriplet>& vec) {	

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
		+ vec[index + 1].value * (vec[index + 1].freq + vec[index + 1].freq)) /
		newbin.freq;

	// erase vec[index + 1]
	// change vec[index] with newbin
	vec.erase(vec.begin() + index + 1);
	vec[index] = newbin;	

}

void Histogram::update(double value) {	
	check();		
	std::vector<BinTriplet> vec;
	ptr2vec(vec);	

	// If there are values in the bin equals to the value here
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i].value == value) {			
			vec[i].freq++;
			vec2ptr(vec);
			check();					
			return;
		}
	}

	vec.push_back(BinTriplet(value, 1));	

	sortBin(vec);		

	if (vec.size() <= max_bin) {
		vec2ptr(vec);			
		check();		
		return;
	}
	
	mergeBin(vec);	
	
	vec2ptr(vec);	
	check();	
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

	if (vec[index + 1].value - vec[index].value != 0) {
		mb = (vec[index + 1].freq - vec[index].freq);
		mb = mb * (value - vec[index].value);
		mb = mb / (vec[index + 1].value - vec[index].value);
		mb = vec[index].freq + mb;
	} else {
		mb = vec[index].freq;
	}
	
	if (vec[index + 1].value - vec[index].value != 0) {
		s = (vec[index].freq + mb) / 2;
		s = s * (value - vec[index].value);
		s = s / (vec[index + 1].value - vec[index].value);
	} else {
		s = (vec[index].freq + mb) / 2;
	}


	for (int j = 0; j < index; j++) {
		s = s + vec[j].freq;
	}

	s = s + vec[index].freq;
	vec2ptr(vec);			
	return s;
}

void Histogram::merge(Histogram &h, int B) {
	check();
	print();
	int index = 0;
	std::vector<BinTriplet> vec;
	ptr2vec(vec);

	std::vector<BinTriplet> hvec;
	h.ptr2vec(hvec);

	vec.insert(vec.end(), hvec.begin(), hvec.end());
	
	sortBin(vec);

	while (vec.size() > B) {
		mergeBin(vec);
	}

	vec2ptr(vec);
	h.vec2ptr(hvec);	
	check();
	print();
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

void Histogram::check(){
	int i=0;	
	while(i<bin_size){		
		if (bins[i].freq == 0) {
			printf("%d: (%.4f, %d) ", i, bins[i].value, bins[i].freq);
			printf("Freq = 0\n");
			exit(1);
		} 
		i++;
	}	
}

void Histogram::uniform(std::vector<double> &u, int B) {
	printf("uniform: start. B: %d\n", B);	
	double tmpsum = 0;
	double s = 0;
	int index = 0;
	double a = 0, b = 0, c = 0, d = 0, z = 0;	
	double uj = 0;
	std::vector<BinTriplet> vec;
	ptr2vec(vec);

	u.clear();

	if (vec.size() == 0) {
		return;
	}
	
	for (int i = 0; i < vec.size(); i++) {
		tmpsum += vec[i].freq;
	}	

	for (int j = 0; j <= B - 2; j++) {
		s = tmpsum * (j + 1) / B;		
		
		for (index = 0; index < (int)vec.size() - 1; index++) {
			
			if (sum(vec[index].value) < s
				&& s < sum(vec[index + 1].value)) {
				break;
			}
		}

		d = s - sum(vec[index].value);
		printf("d: %f\n", d);

		a = vec[index + 1].freq - vec[index].freq;
		b = 2 * vec[index].freq;
		c = -2 * d;
		printf("a: %f, b %f, c %f\n", a, b, c);
		
		if (a != 0 && b * b - 4 * a * c >= 0) {
			z = -b + sqrt(b * b - 4 * a * c);
			z = z / (2 * a);
		} else if (b != 0) {
			// b * z + c = 0
			z = -c / b;
		} else {
			z = 0;
		}
		printf("z: %f\n", z);

		uj = vec[index].value + z * (vec[index + 1].value - vec[index].value);		
		u.push_back(uj);

		if (isnan(uj) < 0) {
			exit(1);
		}
	}
	vec2ptr(vec);	
	printf("uniform: end\n");	
	return;
}

void Histogram::clear() {
	bin_size = 0;
	bins = NULL;
}