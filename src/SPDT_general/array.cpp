
#include "array.h"
double *get_histogram_array(int histogram_id, int feature_id, int label) {
    return histogram + 
        RLOC(histogram_id, feature_id, label, 0, 
        num_of_features, num_of_classes, (max_bin_size + 1) * 2 + 1);
}

double get_bin_freq(double *histo, int index) {
    return *(histo + index * 2 + 1);
}

double get_bin_value(double *histo, int index) {
    return *(histo + index * 2 + 2);
}

void set_bin_freq(double *histo, int index, double freq) {
    *(histo + index * 2 + 1) = freq;
}

void set_bin_value(double *histo, int index, double value) {
    *(histo + index * 2 + 2) = value;
}

int get_total_array(int histogram_id, int feature_id, int label) {
    int t = 0;
    double *histo = get_histogram_array(histogram_id, feature_id, label);
    int bin_size = *histo;
    for (int i = 0; i < bin_size; i++){
        t += get_bin_freq(histo, i);
    }
    return t;
}

double sum_array(int histogram_id, int feature_id, int label, double value) {	
	int index = 0;
	double mb = 0;
	double s = 0;

    double *histo = get_histogram_array(histogram_id, feature_id, label);
    int bin_size = *histo;

	if (bin_size == 1) {
		return get_bin_freq(histo, 0);
	}

    // value < the first value in histo
	if (value < get_bin_value(histo, 0)) {
		return 0;
	}

    // value >= the last value in histogram
	if (value >= get_bin_value(histo, bin_size - 1)) {
		for (int i = 0; i < bin_size; i++) {
			s += get_bin_freq(histo, i);			
		}
		return s;
	}

	for (index = 0; index + 1 < bin_size; index++) {
		if (get_bin_value(histo, index) <= value 
            && get_bin_value(histo, index + 1) > value) {
			break;
		}
	}		

	if (abs(get_bin_value(histo, index + 1) - get_bin_value(histo, index)) <= EPS) {
		// printVector(vec);
		printf("index: %d\n", index);
		printf("value: %f\n", value);
		exit(1);
	}

	if (abs(get_bin_value(histo, index + 1) - get_bin_value(histo, index)) > EPS) {
		mb = get_bin_freq(histo, index + 1) - get_bin_freq(histo, index);
		mb = mb * (value - get_bin_value(histo, index));
		mb = mb / (get_bin_value(histo, index + 1) - get_bin_value(histo, index));
		mb = get_bin_freq(histo, index) + mb;
	} else {
		fprintf(stderr, "abs(vec[index + 1].value - vec[index].value) > EPS");
		exit(-1);		
	}
	
	if (abs(get_bin_value(histo, index + 1) - get_bin_value(histo, index)) > EPS) {
		s = (get_bin_freq(histo, index) + mb) / 2;
		s = s * (value - get_bin_value(histo, index));
		s = s / (get_bin_value(histo, index + 1) - get_bin_value(histo, index));
	} else {
		fprintf(stderr, "(vec[index + 1].value - vec[index].value) == 0");
		exit(-1);		
	}

	for (int j = 0; j < index; j++) {
		s = s + get_bin_freq(histo, j);
	}

	s = s + ((double)get_bin_freq(histo, index)) / 2;	
	return s;
}

void merge_same_array(double *histo) {
    int bin_size = *histo;
    for (int i = 0; i + 1 < bin_size; i++) {        
		if (abs(get_bin_value(histo, i) - get_bin_value(histo, i + 1)) < EPS) {
			set_bin_freq(histo, i, get_bin_freq(histo, i) + get_bin_freq(histo, i + 1));			
			
            // erase vec[i + 1]
			for (int j = i + 1; j <= bin_size - 2; j++) {
                set_bin_freq(histo, j, get_bin_freq(histo, j + 1));
                set_bin_value(histo, j, get_bin_value(histo, j + 1));
			}
			bin_size--;
			i--;
		}
	}
    *histo = bin_size;
}

void merge_bin_array(double *histo) {    
	int index = 0;
    double new_freq = 0;
    double new_value = 0;

    int bin_size = *histo;

	// find the min value of difference
	for (int i = 0; i < bin_size - 1; i++) {
		if (get_bin_value(histo, i + 1) - get_bin_value(histo, i)
			< get_bin_value(histo, index + 1) - get_bin_value(histo, index)) {
			index = i;
		}
	}

	// merge bins[index], bins[index + 1] into a new element
	new_freq = get_bin_freq(histo, index) + get_bin_freq(histo, index + 1);
	new_value = (get_bin_value(histo, index) * get_bin_freq(histo, index)
		+ get_bin_value(histo, index + 1) * get_bin_freq(histo, index + 1)) /
		new_freq;

	// change vec[index] with newbin
	set_bin_freq(histo, index, new_freq);
    set_bin_value(histo, index, new_value);

	// erase vec[index + 1]
	for (int i = index + 1; i <= bin_size - 2; i++) {
        set_bin_freq(histo, i, get_bin_freq(histo, i + 1));
        set_bin_value(histo, i, get_bin_value(histo, i + 1));
 	}
	bin_size--;
    *histo = bin_size;

    merge_same_array(histo);
}

void merge_array(int histogram_id1, int feature_id1, int label1, int histogram_id2, int feature_id2, int label2) {
	double *histo1 = get_histogram_array(histogram_id1, feature_id1, label1);
    double *histo2 = get_histogram_array(histogram_id2, feature_id2, label2);

    double *histo_merge = new double[max_bin_size * 4 + 1];

    int index1 = 0, index2 = 0;
    int bin_size1 = *histo1;
    int bin_size2 = *histo2;
    int bin_size_merge = 0;
    
    while (index1 < bin_size1 && index2 < bin_size2) {
        if (get_bin_value(histo1, index1) < get_bin_value(histo2, index2)) {
            // put the index1 in histo1 to the next place of histo_merge
            set_bin_freq(histo_merge, bin_size_merge, get_bin_freq(histo1, index1));
            set_bin_value(histo_merge, bin_size_merge, get_bin_value(histo1, index1));
            index1++;
        } else {
            // put the index2 in histo2 to the next place of histo_merge
            set_bin_freq(histo_merge, bin_size_merge, get_bin_freq(histo2, index2));
            set_bin_value(histo_merge, bin_size_merge, get_bin_value(histo2, index2));            
            index2++;
        }
        bin_size_merge++;
    }
	*histo_merge = bin_size_merge;

	// merge the same values in vec
	merge_same_array(histo_merge);

	while (*histo_merge > max_bin_size) {
		merge_bin_array(histo_merge);		
	}

    // copy from histo_merge into histo1    
    *histo1 = bin_size_merge;
    for (int i = 0; i < bin_size_merge; i++) {
        set_bin_freq(histo1, i, get_bin_freq(histo_merge, i));
        set_bin_value(histo1, i, get_bin_value(histo_merge, i));
    }
	delete []histo_merge;
	return;
}

void uniform_array(std::vector<double> &u, int histogram_id, int feature_id, int label) {	
    double *histo = get_histogram_array(histogram_id, feature_id, label);
    int bin_size = get_bin_size(*histo);
    int B = bin_size;
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
		tmpsum += get_bin_freq(histo, i);
	}	

	for (int j = 0; j <= B - 2; j++) {
		s = tmpsum * (j + 1) / B;		
		
		for (index = 0; index + 1 < bin_size; index++) {
			
			if (sum_array(histogram_id, feature_id, label, get_bin_value(histo, index)) < s
				&& s < sum_array(histogram_id, feature_id, label, get_bin_value(histo, index + 1))) {
				break;
			}
		}

		d = s - sum_array(histogram_id, feature_id, label, get_bin_value(histo, index));

		a = get_bin_freq(histo, index + 1) - get_bin_freq(histo, index);
		b = 2 * get_bin_freq(histo, index);
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
		
		uj = get_bin_value(histo, index) + z * (get_bin_value(histo, index + 1) - get_bin_value(histo, index));		
		u.push_back(uj);				
	}
	
	return;
}