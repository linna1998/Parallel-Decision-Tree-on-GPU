#pragma once
#include <assert.h>
#include <memory.h>
#include <stdint.h>
#include <string.h>
#include "parser.h"
#include "histogram.h"
#include <omp.h>

// #define DEBUG
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
    bool has_new_data = false;
    int label; // -1 means no label
    int depth;
    double entropy;
    TreeNode* left_node;
    TreeNode* right_node;    
    int histogram_id;   
    int num_pos_label;

    vector<Data*> data_ptr;
    SplitPoint split_ptr;

    TreeNode(int depth, int id);
    void set_label();    
    void init();    
    void split(SplitPoint& best_split, TreeNode*  left, TreeNode*  right);
    void printspaces();
    void print();
    void clear();
};

// a global variable
// record the information of all histograms
double* histogram;
int num_of_features;
int num_of_classes;
int max_bin_size = -1;

class DecisionTree
{
private:
    TreeNode* root;
    int num_leaves;
    int num_nodes;
    int depth;
    int max_num_leaves;
    int max_depth;    
    int min_node_size;
    int cur_depth;
    double min_gain;    
    // three dimensions for the global histogram.    
    int num_unlabled_leaves;

    Dataset* datasetPointer; 
    // histogram size (num_leaf, num_feature, num_class)    

public:

    DecisionTree();
    ~DecisionTree();
    DecisionTree::DecisionTree(int max_num_leaves, int max_depth, int min_node_size);
    
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

/*
 * For A[][M][N][Z]
 * A[i][j][k][e] = A[N*Z*M*i+Z*N*j+k*Z+e]
 */
int RLOC(int i, int j, int k, int e, int M, int N, int Z){
    return N*Z*M*i+Z*N*j+k*Z+e;
}

/*
 * For A[][M][N][Z]
 * A[i][j][k] = A[N*Z*M*i+Z*N*j+k*Z]
 */
int RLOC(int i, int j, int k, int M, int N, int Z){
    return N*Z*M*i+Z*N*j+Z*k;
}

/*
 * For A[][M][N][Z]
 * A[i][j] = A[N*Z*M*i+Z*N*j]
 */
int RLOC(int i, int j, int M, int N, int Z){
    return N*Z*M*i+Z*N*j;
}

/*
 * For A[][M][N][Z]
 * A[i] = A[N*Z*M*i]
 */
int RLOC(int i, int M, int N, int Z){
    return N*Z*M*i;
}