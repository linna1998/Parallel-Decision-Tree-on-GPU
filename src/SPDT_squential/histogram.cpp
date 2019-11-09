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

Histogram::Histogram() {
	this->bins.clear();
	this->max_bin = 255;
}

Histogram::Histogram(int max_bin) {
	this->bins.clear();
	this->max_bin = max_bin;	
}

Histogram::Histogram(int max_bin, BinTriplet* _bins) {
	this->bins.clear();
	this->bins = _bins;
	this->max_bin = max_bin;	
}

void Histogram::sortBin() {
	sort(bins.begin(), bins.end(), [](const BinTriplet &a, const BinTriplet &b) {
		return a.value < b.value;
	});
}


void Histogram::mergeBin() {
	BinTriplet newbin;
	int index = 0;

	// find the min value of difference
	for (int i = 0; i < bins.size() - 1; i++) {
		if (bins[i + 1].value - bins[i].value
			< bins[index + 1].value - bins[index].value) {
			index = i;
		}
	}

	// merge bins[index], bins[index + 1] into a new element
	newbin.freq = bins[index].freq + bins[index + 1].freq;

	newbin.value = (bins[index].value * bins[index].freq
		+ bins[index + 1].value * (bins[index + 1].freq + bins[index + 1].freq)) /
		newbin.freq;

	// erase bins[index + 1]
	// change bins[index] with newbin
	bins.erase(bins.begin() + index + 1);
	bins[index] = newbin;
}

void Histogram::update(double value) {	
	// If there are values in the bin equals to the value here
	for (int i = 0; i < bins.size(); i++) {
		if (bins[i].value == value) {			
			bins[i].freq++;
			return;
		}
	}

	bins.push_back(BinTriplet(value, 1));

	sortBin();

	if (bins.size() <= max_bin) return;
	
	mergeBin();
	
	return;
}

double Histogram::sum(double value) {
	int index = 0;
	double mb = 0;
	double s = 0;
	for (index = 0; index < bins.size() - 1; index++) {
		if (bins[index].value <= value && bins[index + 1].value > value) {
			break;
		}
	}

	mb = (bins[index + 1].freq - bins[index].freq);
	mb = mb * (value - bins[index].value);
	mb = mb / (bins[index + 1].value - bins[index].value);
	mb = bins[index].value + mb;

	s = (bins[index].freq + mb) / 2;
	s = s * (value - bins[index].value);
	s = s / (bins[index + 1].value - bins[index].value);

	for (int j = 0; j < index; j++) {
		s = s + bins[j].freq;
	}

	s = s + bins[index].freq;

	return s;
}

void Histogram::merge(Histogram &h, int B) {
	int index = 0;

	bins.insert(bins.end(), h.bins.begin(), h.bins.end());

	sortBin();

	while (bins.size() > B) {
		mergeBin();
	}
	return;
}

void Histogram::uniform(std::vector<double> &u, int B) {
	double tmpsum = 0;
	double s = 0;
	int index = 0;
	double a = 0, b = 0, c = 0, d = 0, z = 0;	
	double uj = 0;
	u.clear();

	for (int i = 0; i <= B - 1; i++) {
		tmpsum += bins[i].freq;
	}

	for (int j = 0; j <= B - 2; j++) {
		s = tmpsum * j / B;
		
		for (index = 0; index < bins.size() - 1; index++) {
			if (sum(bins[index].value) < s
				&& s < sum(bins[index + 1].value)) {
				break;
			}
		}

		d = s - sum(bins[index].value);

		a = bins[index + 1].freq - bins[index].freq;
		b = 2 * bins[index].freq;
		c = -2 * d;
		z = -b + sqrt(b * b - 4 * a * c);
		z = z / (2 * a);

		uj = bins[index].value + z * (bins[index + 1].value - bins[index].value);
		u.push_back(uj);
	}
	return;
}

void Histogram::clear() {
	max_bin = 0;
	bins.clear();
}