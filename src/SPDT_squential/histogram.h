#pragma once
#include <algorithm>
#include <vector>

// scala version: 
// https://github.com/soundcloud/spdt/blob/master/compute/src/main/scala/com.soundcloud.spdt/Histogram.scala


class BinTriplet {
public:
    double value;
    int freq; // number of samples in this bin    
	BinTriplet();
	BinTriplet(double _value, int _freq);
};


class Histogram {
public:
	int max_bin;       
	// bins are always sorted, with value in increasing order
    BinTriplet* bins;
	int bin_size;
	
	Histogram();
    Histogram(int max_bin);
	Histogram(int max_bin, BinTriplet* _bins);

	void ptr2vec(std::vector<BinTriplet>& res);
	void vec2ptr(std::vector<BinTriplet>& vec);

	void sortBin();
	void mergeBin();
    void update(double value);
    double sum(double value);
    void merge(Histogram &h, int B);
	void uniform(std::vector<double> &u, int B);

	void clear();
};