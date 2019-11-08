#pragma once
#include <stdint.h>
#include "parser.h"
#include "histogram.h"

typedef std::vector<Histogram> Histogram_FEATURE;
typedef std::vector<Histogram_FEATURE> Histogram_LEAF;
typedef std::vector<std::shared_ptr<Histogram_LEAF> > Histogram_ALL;

class SplitPoint{
public:
    // used to store the spliting information on a given histogram.
    int feature_id;
    double feature_value;
    double entropy;
    SplitPoint();
    SplitPoint(int feature_id, double feature_value, double entropy);
    void copy_from(SplitPoint& split);
    bool decition_rule(Data& data);

};

class TreeNode
{
public:
    bool is_leaf = false;    
    bool has_new_data = false;
    int label; // -1 means no label
    TreeNode* left_node;
    TreeNode* right_node;

    std::shared_ptr<Histogram_LEAF> histogram_ptr;   
    vector<Data*> data_ptr;
    SplitPoint split_ptr;

    TreeNode();
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
    int max_num_leaves;
    int max_depth;
    int max_bin_size;
    int min_node_size;
    Dataset* datasetPointer; 
    // histogram size (num_leaf, num_feature, num_class)
    Histogram_ALL histogram;

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
    void initialize(TreeNode* node);
    TreeNode* navigate(Data& d);
    bool is_terminated(TreeNode* node);
};