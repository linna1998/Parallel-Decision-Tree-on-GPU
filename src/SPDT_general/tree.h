#pragma once
#include <assert.h>
#include <memory.h>
#include <stdint.h>
#include <string.h>
#include "parser.h"
#include "histogram.h"
#include <omp.h>

typedef std::vector<Histogram> Histogram_FEATURE;
typedef std::vector<Histogram_FEATURE> Histogram_LEAF;
typedef std::vector<Histogram_LEAF> Histogram_ALL;
typedef BinTriplet Bin_t;
typedef Bin_t* Bin_ptr;
#define DEBUG
#ifdef DEBUG
/* When DEBUG is defined, these form aliases to useful functions */
#define dbg_printf(...) printf(__VA_ARGS__)
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printheap(...) print_heap(__VA_ARGS__)
#else
/* When DEBUG is not defined, no code gets generated for these */
/* The sizeof() hack is used to avoid "unused variable" warnings */
#define dbg_printf(...) (sizeof(__VA_ARGS__), -1)
#define dbg_requires(expr) (sizeof(expr), 1)
#define dbg_assert(expr) (sizeof(expr), 1)
#define dbg_ensures(expr) (sizeof(expr), 1)
#define dbg_printheap(...) ((void)sizeof(__VA_ARGS__))
#endif

class TreeNode
{
public:
    int id;
    bool is_leaf = false;    
    int label; // -1 means no label
    int depth;
    double entropy;
    TreeNode* left_node;
    TreeNode* right_node;
    Histogram_LEAF* histogram_ptr;
    int histogram_id;   
    int num_pos_label;

    vector<Data*> data_ptr;
    int data_size;
    SplitPoint split_ptr;

    TreeNode(int depth, int id);
    void set_label();    
    void init();    
    void split(SplitPoint& best_split, TreeNode*  left, TreeNode*  right);
    void printspaces();
    void print();
    void clear();
};

class DecisionTree
{
private:
    TreeNode* root;
    int num_leaves;
    int num_nodes;
    int depth;
    int max_num_leaves;
    int max_depth;
    int max_bin_size;
    int min_node_size;
    int cur_depth;
    double min_gain;
    Histogram_ALL histogram;
    // three dimensions for the global histogram.    
    int num_unlabled_leaves;

    Dataset* datasetPointer; 
    // histogram size (num_leaf, num_feature, num_class)
    Bin_ptr bin_ptr;

public:

    DecisionTree();
    ~DecisionTree();
    DecisionTree(int max_num_leaves, int max_depth, int min_node_size, int max_bin_size);
    
    void self_check();
    void train(Dataset& train_data, const int batch_size = 64);
    void train_on_batch(Dataset& train_data);
    double test(Dataset& test_data);
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