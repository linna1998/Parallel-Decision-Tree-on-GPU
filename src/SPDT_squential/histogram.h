#include <vector>
#include "parser.h"

typedef struct{
    double bin_value;
    int bin_frequency; // number of samples in this bin
    int num_pos_frequency; // number of positive samples in this bin
    
} bin_triplet;

class Histogram{
public:

    int max_bin;
    Dataset& dataset;
    // bins shape: (number of features, number of bins)
    std::vector<std::vector<bin_triplet> > bins;
    Histogram(Dataset& dataset, int start_idx, int end_idx, int max_bin = 255);
    void merge();
    void sum();
    void update();

};