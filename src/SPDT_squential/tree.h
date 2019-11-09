#pragma once
#include <memory.h>
#include <stdint.h>
#include <string.h>
#include "parser.h"
#include "histogram.h"

typedef std::vector<Histogram> Histogram_FEATURE;
typedef std::vector<Histogram_FEATURE> Histogram_LEAF;
typedef std::vector<Histogram_LEAF> Histogram_ALL;
typedef BinTriplet Bin_t;
typedef Bin_t* Bin_ptr;

class SplitPoint{
public:
    // used to store the spliting information on a given histogram.
    int feature_id;
    double feature_value;
    double entropy;
    SplitPoint();
    SplitPoint(int feature_id, double feature_value, double entropy);
    bool decition_rule(Data& data);

    inline SplitPoint& operator = (SplitPoint& split){
        this->feature_id = split.feature_id;
        this->feature_value = split.feature_value;
        this->entropy = split.entropy;
        return *this;
    }
};

class TreeNode
{
public:
    bool is_leaf = false;    
    bool has_new_data = false;
    int label; // -1 means no label
    int depth;

    TreeNode* left_node;
    TreeNode* right_node;
    Histogram_LEAF* histogram_ptr;
    int histogram_id;   

    vector<Data*> data_ptr;
    SplitPoint split_ptr;

    TreeNode(int depth);
    void set_label();    
    void init();    
    void split(SplitPoint& best_split, vector<Data*>& left, vector<Data*>& right);
};

class DecisionTree
{
private:
    TreeNode* root = NULL;
    int num_leaves;
    int depth;
    int max_num_leaves = 64;
    int max_depth = 8;
    int max_bin_size = 255;
    int min_node_size = 32;
    int cur_depth = 0;
    Histogram_ALL histogram;
    // three dimensions for the global histogram.    
    int num_unlabled_leaves;

    Dataset* datasetPointer; 
    // histogram size (num_leaf, num_feature, num_class)
    Bin_ptr bin_ptr;

public:

    DecisionTree();
    DecisionTree(int max_num_leaves, int max_depth, int min_node_size);
        
    void train(Dataset& train_data, const int batch_size = 64);
    void train_on_batch(Dataset& train_data);
    void test(Dataset& test_data);
    // this function adjust the `global_partition_idx`
    void find_best_split(TreeNode* node, SplitPoint& split);
    void compress(vector<Data>& data, vector<TreeNode* >& unlabled_leaf);
    vector<TreeNode*> __get_unlabeled(TreeNode* node);
    void batch_initialize(TreeNode* node);
    void initialize(Dataset &train_data, const int batch_size);
    void init_histogram(vector<TreeNode* >& unlabled_leaf);
    TreeNode* navigate(Data& d);
    bool is_terminated(TreeNode* node);
};