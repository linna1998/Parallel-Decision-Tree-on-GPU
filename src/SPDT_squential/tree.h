#pragma once
#include <stdint.h>
#include "parser.h"
#include "histogram.h"

class TreeNode
{
public:
    bool isLeaf = false;
    // if isLeaf = True, label would be defined as the majority class in the subset
    int label;
    TreeNode* left_node;
    TreeNode* right_node;
    Histogram* histogram;
    // Each node contains a subset of data. In order to reduce the memory requirement,
    // dta_start_idx, dta_end_idx are used to store the start/end index of the data belonging to this node.
    int dta_start_idx; 
    int dta_end_idx;
    // used to store the spliting information on a given histogram.
    int optimal_split_feature_idx;
    double optimal_split_feature_value;
    double entropy;

    TreeNode();
    void set_label(int label);
    void set_histogram(Histogram* histogram);
    void set_dta_idx(int dta_start_idx, int dta_end_idx);
    void set_split_info(
        int optimal_split_feature_idx, 
        double optimal_split_feature_value, 
        double entropy);

};

class DecisionTree
{
private:
    TreeNode* root;
    int num_leaves;
    int max_num_leaves;
    int max_depth;
    int max_bin_size;
    int min_node_size;
    // dataset_index is a global index of data. e.g. [0,3,2,1]
    // this is used to partition the dataset. each node have a `start/end index` of this array
    vector<int> global_partition_idx; 

public:
    DecisionTree();
    DecisionTree(int max_num_leaves, int max_depth, int min_node_size);
    
    void train(Dataset& train_data, const int batch_size = 64);
    void train_on_batch(Dataset& train_data);
    void test(Dataset& test_data);
    void split(Dataset& data, int optimal_split_bin_idx);
    // this function adjust the `global_partition_idx`
    void find_best_split(TreeNode* node, int& split_feature_id, float& split_feature_value);
    void construct_histogram(Dataset& Dataset, int start_idx, int end_idx);
};