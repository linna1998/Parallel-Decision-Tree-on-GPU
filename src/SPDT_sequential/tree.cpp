#include "../SPDT_general/tree.h"
#include "panel.h"
#include <assert.h>
#include <queue>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <time.h>
#include "../SPDT_general/array.h"
double COMPRESS_TIME = 0;
double SPLIT_TIME = 0;


/*
 * This function return the best split point at a given leaf node.
 * Best split is store in `split`
*/
void DecisionTree::find_best_split(TreeNode *node, SplitPoint &split)
{
    clock_t start, end;
    start = clock();       

    std::vector<SplitPoint> results;
    for (int i = 0; i < this->datasetPointer->num_of_features; i++)
    {
        // merge different labels
        // put the result back into (node->histogram_id, i, 0)
        for (int k = 1; k < this->datasetPointer->num_of_classes; k++) {
            merge_array(node->histogram_id, i, 0, node->histogram_id, i, k);
        }

        std::vector<double> possible_splits;
        uniform_array(possible_splits, node->histogram_id, i, 0);

        dbg_assert(possible_splits.size() <= this->max_bin_size);
        for (auto& split_value: possible_splits)
        {
            SplitPoint t = SplitPoint(i, split_value);
            get_gain(node, t, i);
            results.push_back(t);
        }
    }
    std::vector<SplitPoint>::iterator best_split = std::max_element(results.begin(), results.end(),
                                                                    [](const SplitPoint &l, const SplitPoint &r) { return l.gain < r.gain; });

    split.feature_id = best_split->feature_id;
    split.feature_value = best_split->feature_value;
    split.gain = best_split->gain;
    end = clock();   
    SPLIT_TIME += ((double) (end - start)) / CLOCKS_PER_SEC; 
}

/*
 * This function compress the data into histograms.
 * Each unlabeled leaf would have a (num_feature, num_class) histograms
 * This function takes the assumption that each leaf is re-initialized (we use a batch mode)
*/
void DecisionTree::compress(vector<Data> &data)
{
    clock_t start, end;
    start = clock();  
    int feature_id = 0, class_id = 0;
    // Construct the histogram. and navigate each data to its leaf.
    TreeNode* cur;
    int c=0;
    for(auto& point : data){
        cur = navigate(point);
        if (cur->label > -1)
            continue;
        cur->data_size ++;
        for (int attr = 0; attr < this->datasetPointer->num_of_features; attr++)
        {                            
            update_array(cur->histogram_id, attr, point.label, point.get_value(attr));              
        }
    }

    end = clock();   
    COMPRESS_TIME += ((double) (end - start)) / CLOCKS_PER_SEC; 
}