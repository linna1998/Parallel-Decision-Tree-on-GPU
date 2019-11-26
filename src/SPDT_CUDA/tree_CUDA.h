#pragma once
#include <assert.h>
#include <memory.h>
#include <stdint.h>
#include <string.h>
#include "parser_CUDA.h"
#include "array_CUDA.h"
#include <queue>
#include <algorithm>
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

#define EPS 1e-9
#define magic 234

extern float COMPRESS_TIME;
extern float SPLIT_TIME;
extern float COMMUNICATION_TIME;

extern int num_of_features;
extern int num_of_classes;
extern int max_bin_size;
extern int max_num_leaves;

extern long long SIZE;

struct GlobalConstants {

    int num_of_features;
    int num_of_data;
    int max_bin_size;
    int num_of_classes;
    float* cuda_histogram_ptr;
    int* cuda_histogram_id_ptr;
    int* cuda_label_ptr;
    float* cuda_value_ptr;

};

class SplitPoint{
public:
    // used to store the spliting information on a given histogram.
    int feature_id;
    float feature_value;
	float gain;
	float entropy;
    Dataset* datasetPointer; 
    SplitPoint();
    SplitPoint(int feature_id, float feature_value, Dataset* datasetPointer);
    bool decision_rule(int data_index);
    inline SplitPoint& operator = (const SplitPoint& split){
        this->feature_id = split.feature_id;
        this->feature_value = split.feature_value;
		this->gain = split.gain;
		this->entropy = split.entropy;
        return *this;
    }
};

class TreeNode
{
public:
    int id;
    bool is_leaf = false;    
    int label; // -1 means no label
    int depth;
    float entropy;
    TreeNode* left_node;
    TreeNode* right_node;    
    int histogram_id;   
    int num_pos_label;
    Dataset* datasetPointer; 
    
    int data_size;
    SplitPoint split_ptr;

    TreeNode(int depth, int id, Dataset* datasetPointer);
    void set_label();    
    void init();    
    void split(SplitPoint& best_split, TreeNode* left, TreeNode* right);
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
    int min_node_size;
    int cur_depth;
    float min_gain;    
    // three dimensions for the global histogram.    
    int num_unlabled_leaves;

    Dataset* datasetPointer; 
    // histogram size (num_leaf, num_feature, num_class)    

    float *cuda_histogram_ptr;
    // 0/1 label, might apply something as bitmap in the future
    int *cuda_label_ptr;
    float *cuda_value_ptr;
    // record, every data sample is bind to which histogram id
    int *cuda_histogram_id_ptr;

public:

    DecisionTree();
    ~DecisionTree();
    DecisionTree(int max_num_leaves, int max_depth, int min_node_size){
        this->max_num_leaves = max_num_leaves;
        this->max_depth = max_depth;
        this->min_node_size = min_node_size;
        this->depth = 0;
        this->num_leaves = 0;    
        this->cur_depth = 0;
        this->root = NULL;    
        this->min_gain = 1e-3;
        this->num_nodes = 0;
    };
    
    void self_check();
    void train(Dataset& train_data, const int batch_size = 64);
    void train_on_batch(Dataset& train_data);
    float test(Dataset& test_data);
    // this function adjust the `global_partition_idx`
    void find_best_split(TreeNode* node, SplitPoint& split);
    void compress(vector<TreeNode *> &unlabeled_leaf);
    vector<TreeNode*> __get_unlabeled(TreeNode* node);
    void batch_initialize(TreeNode* node);
    void initialize(Dataset &train_data, const int batch_size);
    void init_histogram(vector<TreeNode* >& unlabled_leaf);
    TreeNode* navigate(int data_index);
    bool is_terminated(TreeNode* node);

    void initCUDA();
    void terminateCUDA();
};
