#pragma once
#include <algorithm>
#include <memory.h>
#include <vector>
#include "parser.h"


// scala version: 
// https://github.com/soundcloud/spdt/blob/master/compute/src/main/scala/com.soundcloud.spdt/Histogram.scala

class SplitPoint{
public:
    // used to store the spliting information on a given histogram.
    int feature_id;
    double feature_value;
	double gain;
	double entropy;
    SplitPoint();
    SplitPoint(int feature_id, double feature_value);
    bool decision_rule(Data& data);
    inline SplitPoint& operator = (const SplitPoint& split){
        this->feature_id = split.feature_id;
        this->feature_value = split.feature_value;
		this->gain = split.gain;
		this->entropy = split.entropy;
        return *this;
    }
};

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
	Histogram(){
		this->max_bin = 0;
		this->bin_size = 0;	
		};
	Histogram(const int max_bin);
	Histogram(const int max_bin, BinTriplet* _bins);

	void ptr2vec(std::vector<BinTriplet>& res);
	void vec2ptr(std::vector<BinTriplet>& vec);	
	void mergeBinArray();
	void mergeSameArray();
    void update(double value);
    double sum(double value);
    void merge(Histogram &h);
	void uniform(std::vector<double> &u, int B);
	void print();
	void check(int lineno);
	void clear();
	int get_total();
	inline Histogram& operator = (const Histogram& h){
		this->max_bin = h.max_bin;
		this->bin_size = h.bin_size;
		this->bins = new BinTriplet[this->max_bin];
		memcpy(this->bins, h.bins, sizeof(BinTriplet) * max_bin);
		return *this;
	}
};

void printVector(std::vector<BinTriplet>& vec);
void mergeSame(std::vector<BinTriplet>& vec);
void sortBin(std::vector<BinTriplet>& vec);
void mergeBin(std::vector<BinTriplet>& vec);