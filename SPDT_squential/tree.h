#include <stdint.h>
#include <dataset.h>

class TreeNode
{
public:
    bool isLeaf = false;
    // if isLeaf = True, label would be defined as the majority class in the subset
    int label;
    TreeNode* left_node;
    TreeNode* right_node;
    // Each node contains a subset of data. In order to reduce the memory requirement,
    // dta_start_idx, dta_end_idx are used to store the start/end index of the data belonging to this node.
    uint32_t dta_start_idx; 
    uint32_t dta_end_idx;
    // used to store the spliting information on a given histogram.
    int optimal_split_bin_idx;
    double optimal_split_bin_value;
    double entropy_gain;

};

class DecisionTree
{
private:
    TreeNode* root;
    int num_leaves;
    int max_num_leaves;
    int max_depth;
    int min_node_size;

public:
    DecisionTree(int max_num_leaves = -1, int max_depth = -1, int min_node_size = 0);
    
    void train(Dataset& train_data, const int batch_size = 64);
    void train_on_batch(Dataset& train_data);
    void test(Dataset& test_data);
};