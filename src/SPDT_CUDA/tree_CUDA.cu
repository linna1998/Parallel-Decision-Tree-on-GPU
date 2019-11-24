#include "tree_CUDA.h"
#include "parser_CUDA.h"
#include "array_CUDA.h"
#include "../SPDT_general/timing.h"
#include "panel.h"
#include <assert.h>
#include <queue>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

float COMPRESS_TIME = 0.f;
float SPLIT_TIME = 0.f;
float COMMUNICATION_TIME = 0.f;
long long SIZE = 0 ;

int num_of_features = -1;
int num_of_classes = -1;
int max_bin_size = -1;
int max_num_leaves = -1;

__global__ void
navigate_samples_kernel() {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void
histogram_update_kernel(
    int data_size, 
    int num_of_threads,    
    float *cuda_histogram_ptr, 
    int *cuda_label_ptr,
    float *cuda_value_ptr,
    int *cuda_histogram_id_ptr,
    int num_of_features,
    int num_of_classes,
    int max_bin_size) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // blockIdx.x: the update data id
    // threadIdx.x: the update feature id

    if (blockIdx.x >= data_size || threadIdx.x >= num_of_threads)
       return;
    
    // re-write part of the DecisionTree::compress function      
    update_array(
        cuda_histogram_id_ptr[blockIdx.x], 
        threadIdx.x, 
        cuda_label_ptr[blockIdx.x], 
        cuda_value_ptr[blockIdx.x * num_of_features + threadIdx.x],
        cuda_histogram_ptr,
        num_of_features,
        num_of_classes,
        max_bin_size);
}

SplitPoint::SplitPoint()
{
    feature_id = -1;
    feature_value = 0;
    entropy = 0;
}

SplitPoint::SplitPoint(int feature_id, float feature_value, Dataset* datasetPointer)
{
    this->feature_id = feature_id;
    this->feature_value = feature_value;
    this->entropy = 0;
    this->datasetPointer = datasetPointer;
}
/*
 * Reture True if the data is larger or equal than the split value
 */
bool SplitPoint::decision_rule(int data_index)
{
    dbg_ensures(entropy >= -EPS);
    dbg_ensures(gain >= -EPS);
    dbg_ensures(feature_id >= 0);
    return datasetPointer->value_ptr[data_index * num_of_features + feature_id] >= feature_value;    
}

// constructor function
TreeNode::TreeNode(int depth, int id, Dataset* datasetPointer)
{
    this->id = id;
    this->depth = depth;
    is_leaf = false;
    label = -1;    
    histogram_id = -1;    
    left_node = NULL;
    right_node = NULL;
    entropy = -1.f;
    num_pos_label=0;
    data_size = 0;
    is_leaf = true;
    this->datasetPointer = datasetPointer;
}


void TreeNode::init()
{
    label = -1;
    histogram_id = -1;    
    left_node = NULL;
    right_node = NULL;
    is_leaf = true;
    return;
}

/*
 * Set label for the node as the majority class.
 */
void TreeNode::set_label()
{
    this->is_leaf = true;
    this->label = (this->num_pos_label >= (int)this->data_size / 2) ? POS_LABEL : NEG_LABEL;
}

/*
 * This function split the data according to the best split feature id and value.
 * The data would be appended to the `left` if the data value is smaller than the split value
 */
void TreeNode::split(SplitPoint &best_split, TreeNode* left, TreeNode* right)
{
    this->split_ptr = best_split;
    this->entropy = best_split.entropy;
    float split_value = best_split.feature_value;
    int num_pos_label_left=0;
    int num_pos_label_right=0;
    for (int i = 0; i < this->datasetPointer->num_of_data; i++) {
        if (this->datasetPointer->histogram_id_ptr[i] != this->histogram_id) {
            continue;
        }
        float p_value = this->datasetPointer->value_ptr[i * num_of_features + best_split.feature_id];
        if (best_split.decision_rule(i)) {
            this->datasetPointer->histogram_id_ptr[i] = right->histogram_id;
            num_pos_label_right = (this->datasetPointer->label_ptr[i] == POS_LABEL) ? num_pos_label_right + 1 : num_pos_label_right;
            right->data_size++;
        } else {
            this->datasetPointer->histogram_id_ptr[i] = left->histogram_id;
            num_pos_label_left = (this->datasetPointer->label_ptr[i] == POS_LABEL) ? num_pos_label_left + 1 : num_pos_label_left;
            left->data_size++;
        }
    }
   
    left->num_pos_label = num_pos_label_left;
    right->num_pos_label = num_pos_label_right;

    dbg_assert(left->num_pos_label >= 0);
    dbg_assert(right->num_pos_label >= 0);
    dbg_assert(left->num_pos_label + right->num_pos_label == this->num_pos_label);
}

void TreeNode::printspaces() {
    int i = 0;
    for (i = 0; i < depth * 2; i++) {
        printf(" ");
    }
}

void TreeNode::print() {
    printspaces();
    printf("TreeNode: \n");
    printspaces();
    printf("depth: %d\n", depth);
    printspaces();
    printf("label %d\n", label);
    printspaces();
    printf("is_leaf %d\n", is_leaf);
    printspaces();
    printf("hasLeft: %d\n", left_node != NULL);
    printspaces();
    printf("hasRight: %d\n", right_node != NULL);    
    if (left_node != NULL) {
        left_node->print();
    }

    if (right_node != NULL) {
        right_node->print();
    }
}

void TreeNode::clear(){
    if (left_node != NULL) left_node->clear();
    if (right_node != NULL) right_node->clear();    
}

DecisionTree::DecisionTree()
{
    this->max_num_leaves = 64;
    this->max_depth = -1;
    this->min_node_size = 1;
    this->depth = 0;
    this->num_leaves = 0;    
    this->cur_depth = 0;
    this->root = NULL;    
    this->min_gain = 1e-3;
    this->num_nodes = 0;

}

DecisionTree::~DecisionTree(){    
    root->clear();
}

void DecisionTree::initCUDA() {
    int data_size = this->datasetPointer->num_of_data;
    // Construct the histogram. and navigate each data to its leaf.  
    long long number = (long long) max_num_leaves * num_of_features * num_of_classes * ((max_bin_size + 1) * 2 + 1);        

    cudaMalloc((void **)&cuda_histogram_ptr, sizeof(float) * number);
    cudaMalloc((void **)&cuda_label_ptr, sizeof(int) * data_size);
    cudaMalloc((void **)&cuda_value_ptr, sizeof(float) * data_size * num_of_features);
    cudaMalloc((void **)&cuda_histogram_id_ptr, sizeof(int) * data_size);

    cudaMemcpy(cuda_histogram_ptr,
        histogram,
        sizeof(float) * number,
        cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_label_ptr,
        this->datasetPointer->label_ptr,
        sizeof(int) * data_size,
        cudaMemcpyHostToDevice);  
    cudaMemcpy(cuda_value_ptr,
        this->datasetPointer->value_ptr,
        sizeof(float) * data_size * num_of_features,
        cudaMemcpyHostToDevice);  
    cudaMemcpy(cuda_histogram_id_ptr,
        this->datasetPointer->histogram_id_ptr,
        sizeof(int) * data_size,
        cudaMemcpyHostToDevice); 
}

void DecisionTree::terminateCUDA() {
    cudaFree(cuda_histogram_ptr);
    cudaFree(cuda_label_ptr);
    cudaFree(cuda_value_ptr);
    cudaFree(cuda_histogram_id_ptr);
}

/* 
 * Return true if the node should be a leaf.
 * This is determined by the min-node-size, max-depth, max_num_leaves
*/
bool DecisionTree::is_terminated(TreeNode *node)
{
    if (min_node_size != -1 && node->data_size <= min_node_size)
    {
        printf("Node [%d] terminated: min_node_size=%d >= %d\n", node->id, min_node_size, node->data_size);
        return true;
    }

    if (max_depth != -1 && node->depth >= this->max_depth)
    {
        printf("Node [%d] terminated: max_depth\n", node->id);
        return true;
    }

    if (max_num_leaves != -1 && this->num_leaves >= this->max_num_leaves)
    {
        printf("Node [%d] terminated: max_num_leaves\n", node->id);
        return true;
    }

    if (!node->num_pos_label || node->num_pos_label == (int) node->data_size){
        dbg_assert(node->entropy < EPS);
        printf("Node [%d] terminated: all samples belong to same class\n",node->id);
        return true; 
    }
    printf("[%d] num_data=%d, num_pos=%d\n", node->id, node->data_size, node->num_pos_label);
    return false;
}

void DecisionTree::initialize(Dataset &train_data, const int batch_size){
    this->datasetPointer = &train_data;    
    root = new TreeNode(0, this->num_nodes++, datasetPointer);  
    root->data_size = train_data.num_of_data;
    if (histogram != NULL) {        
        delete[] histogram;
    }
    
    SIZE  = (long long) max_num_leaves * num_of_features * num_of_classes * ((max_bin_size + 1) * 2 + 1);    
    printf("Init Root Node [%.4f] MB\n", SIZE * sizeof(float) / 1024.f / 1024.f);
    
    histogram = new float[SIZE];
    memset(histogram, 0, SIZE * sizeof(float));          
    printf("Init success\n");
}

void DecisionTree::train(Dataset &train_data, const int batch_size)
{
    int hasNext = TRUE;
    initialize(train_data, batch_size);
	while (TRUE) {
		hasNext = train_data.streaming_read_data(batch_size);	
        printf("Train size (%d, %d, %d)\n", train_data.num_of_data, 
                num_of_features, num_of_classes);
                
        Timer t = Timer();
        t.reset();
        initCUDA();
        COMMUNICATION_TIME += t.elapsed();

        train_on_batch(train_data);        
		if (!hasNext) break;
	}		
    
    train_data.close_read_data();
    terminateCUDA(); 
    printf("COMPRESS TIME: %f\nSPLIT TIME: %f\nCOMMUNICATION TIME: %f\n", 
        COMPRESS_TIME, SPLIT_TIME, COMMUNICATION_TIME);   
    return;
}

float DecisionTree::test(Dataset &test_data) {    

    int i = 0;
    int correct_num = 0;
    test_data.streaming_read_data(test_data.num_of_data);

    for (i = 0; i < test_data.num_of_data; i++) {
        assert(navigate(i)->label != -1);        
        if (navigate(i)->label == test_data.label_ptr[i]) {
            correct_num++;
        }
    }    
    return (float)correct_num / (float)test_data.num_of_data;
}

/*
 * Calculate the entropy gain = H(Y) - H(Y|X)
 * H(Y|X) needs parameters p(X<a), p(Y=0|X<a), p(Y=0|X>=a)
 * Assuming binary classification problem
 */
void get_gain(TreeNode* node, SplitPoint& split, int feature_id){
    int total_sum = node->data_size;
    dbg_ensures(total_sum > 0);
    float sum_class_0 = get_total_array(node->histogram_id, feature_id, NEG_LABEL);
    float sum_class_1 = get_total_array(node->histogram_id, feature_id, POS_LABEL);
    dbg_assert((sum_class_1 - node->num_pos_label) < EPS);
    float left_sum_class_0 = sum_array(node->histogram_id, feature_id, NEG_LABEL, split.feature_value);
    float right_sum_class_0 = sum_class_0 - left_sum_class_0;
    float left_sum_class_1 = sum_array(node->histogram_id, feature_id, POS_LABEL, split.feature_value);
    float right_sum_class_1 = sum_class_1 - left_sum_class_1;
    float left_sum = left_sum_class_0 + left_sum_class_1;
    float right_sum = right_sum_class_0 + right_sum_class_1;

    float px = (left_sum_class_0 + left_sum_class_1) / (1.0 * total_sum); // p(x<a)
    float py_x0 = (left_sum <= EPS) ? 0.f : left_sum_class_0 / left_sum;                            // p(y=0|x < a)
    float py_x1 = (right_sum <= EPS) ? 0.f : right_sum_class_0 / right_sum;                          // p(y=0|x >= a)
    // printf("sum_class_1=%f, sum_class_0=%f, right_sum = %f, right_sum_class_0 = %f right_sum_class_1= %f\n", sum_class_1, sum_class_0, right_sum, right_sum_class_0, right_sum_class_1);
    // printf("py_x0 = %f, py_x1 = %f\n", py_x0, py_x1);
    dbg_ensures(py_x0 >= -EPS && py_x0 <= 1+EPS);
    dbg_ensures(py_x1 >= -EPS && py_x1 <= 1+EPS);
    dbg_ensures(px >= -EPS && px <= 1+EPS);
    float entropy_left = ((1-py_x0) < EPS || py_x0 < EPS) ? 0 : -py_x0 * log2((double)py_x0) - (1-py_x0)*log2((double)1-py_x0);
    float entropy_right = ((1-py_x1) < EPS || py_x1 < EPS) ? 0 : -py_x1 * log2((double)py_x1) - (1-py_x1)*log2((double)1-py_x1);
    float H_YX = px * entropy_left + (1-px) * entropy_right;
    float px_prior = sum_class_0 / (sum_class_0 + sum_class_1);
    dbg_ensures(px_prior > 0 && px_prior < 1);
    split.entropy = ((1-px_prior) < EPS || px_prior < EPS) ? 0 : -px_prior * log2((double)px_prior) - (1-px_prior) * log2((double)1-px_prior);
    split.gain = split.entropy - H_YX;
    // printf("%f = %f - %f\n", split.gain, split.entropy, H_YX);
    dbg_ensures(split.gain >= -EPS);
}

/*
 * This function return the best split point at a given leaf node.
 * Best split is store in `split`
*/
void DecisionTree::find_best_split(TreeNode *node, SplitPoint &split)
{              
    assert(node != NULL);

    std::vector<SplitPoint> results;

    for (int i = 0; i < num_of_features; i++)
    {
        // merge different labels
        // put the result back into (node->histogram_id, i, 0)
        for (int k = 1; k < num_of_classes; k++) {
            merge_array(node->histogram_id, i, 0, node->histogram_id, i, k);
        }

        std::vector<float> possible_splits;
        uniform_array(possible_splits, node->histogram_id, i, 0);        
        dbg_assert(possible_splits.size() <= max_bin_size);
        for (auto& split_value: possible_splits)
        {
            SplitPoint t = SplitPoint(i, split_value, datasetPointer);
            get_gain(node, t, i);
            results.push_back(t);
        }
    }
    std::vector<SplitPoint>::iterator best_split = std::max_element(results.begin(), results.end(),
                                                                    [](const SplitPoint &l, const SplitPoint &r) { return l.gain < r.gain; });

    split.feature_id = best_split->feature_id;
    split.feature_value = best_split->feature_value;
    split.gain = best_split->gain;
}

/*
 * This function compress the data into histograms.
 * Each unlabeled leaf would have a (num_feature, num_class) histograms
 * This function takes the assumption that each leaf is re-initialized (we use a batch mode)
*/
void DecisionTree::compress(vector<TreeNode *> &unlabeled_leaf) {
    int block_num = this->datasetPointer->num_of_data;
    int thread_per_block = num_of_features; 
    long long number = (long long) max_num_leaves * num_of_features * num_of_classes * ((max_bin_size + 1) * 2 + 1);        
    
    cudaMemcpy(cuda_histogram_ptr,
        histogram,
        sizeof(float) * number,
        cudaMemcpyHostToDevice);    
    cudaMemcpy(cuda_histogram_id_ptr,
        this->datasetPointer->histogram_id_ptr,
        sizeof(int) * this->datasetPointer->num_of_data,
        cudaMemcpyHostToDevice); 
                                
    // https://stackoverflow.com/questions/31598021/cuda-cudamemcpy-struct-of-arrays
    // reference for moving objects from host to device in CUDA

    histogram_update_kernel<<<block_num, thread_per_block>>>(
        block_num,
        num_of_features,                              
        cuda_histogram_ptr, 
        cuda_label_ptr,
        cuda_value_ptr,
        cuda_histogram_id_ptr,
        num_of_features,
        num_of_classes,
        max_bin_size);  
    cudaDeviceSynchronize();       
    
    cudaMemcpy(histogram,
        cuda_histogram_ptr,
        sizeof(float) * number,
        cudaMemcpyDeviceToHost);  
    
    float *histo = NULL;
    int bin_size = 0;

    for (int i = 0; i < num_of_features; i++) {
        for (int j = 0; j < num_of_classes; j++) {
            histo = get_histogram_array(0, i, j, histogram, num_of_features, num_of_classes, max_bin_size);
            bin_size = get_bin_size(histo);
            printf("[%d][%d]: bin_size %d\n", i, j, bin_size);
        }
    }    
    
}

/*
 * Serial version of training.
*/
void DecisionTree::train_on_batch(Dataset &train_data)
{       
    float pos_rate = ((float) train_data.num_pos_label) / train_data.num_of_data;
    dbg_assert(pos_rate > 0 && pos_rate < 1);
    root->num_pos_label = train_data.num_pos_label;
    root->entropy = - pos_rate * log2((double)pos_rate) - (1-pos_rate) * log2((double)(1-pos_rate));
    batch_initialize(root); // Reinitialize every leaf in T as unlabeled.
    vector<TreeNode *> unlabeled_leaf = __get_unlabeled(root);
    dbg_assert(unlabeled_leaf.size() <= max_num_leaves);
    while (!unlabeled_leaf.empty())
    {        
        // each while loop would add a new level node.
        this->cur_depth++;
        printf("depth [%d] finished\n", this->cur_depth);
        vector<TreeNode *> unlabeled_leaf_new; 
        if (unlabeled_leaf.size() > max_num_leaves) {
            for (int i = 0; i < unlabeled_leaf.size(); i++) {
                unlabeled_leaf[i]->set_label();
                this->num_leaves++;
            }
            break;
        }       
        init_histogram(unlabeled_leaf);
        Timer t1 = Timer();
        t1.reset();
        compress(unlabeled_leaf); 
        COMPRESS_TIME += t1.elapsed();         
        for (auto &cur_leaf : unlabeled_leaf)
        {            
            if (is_terminated(cur_leaf))
            {         
                cur_leaf->set_label();
                this->num_leaves++;             
            }
            else
            {                
                SplitPoint best_split;
                Timer t2 = Timer();
                t2.reset();
                find_best_split(cur_leaf, best_split);
                SPLIT_TIME += t2.elapsed();                
                dbg_ensures(best_split.gain >= -EPS);
                if (best_split.gain <= min_gain){
                    printf("Node terminated: gain=%.4f <= %.4f\n", min_node_size, best_split.gain, min_gain);
                    cur_leaf->set_label();
                    this->num_leaves++;               
                    continue;
                }
                cur_leaf->left_node = new TreeNode(this->cur_depth, this->num_nodes++, datasetPointer);
                cur_leaf->right_node = new TreeNode(this->cur_depth, this->num_nodes++, datasetPointer);
                cur_leaf->split(best_split, cur_leaf->left_node, cur_leaf->right_node);
                cur_leaf->is_leaf = false;
                cur_leaf->label = -1;
                unlabeled_leaf_new.push_back(cur_leaf->left_node);
                unlabeled_leaf_new.push_back(cur_leaf->right_node);
            }
        }
        unlabeled_leaf = unlabeled_leaf_new;
        unlabeled_leaf_new.clear(); 
    }
    self_check();    
}


void DecisionTree::self_check(){
    queue<TreeNode *> q;
    q.push(root);
    int count_leaf=0;
    int count_nodes=0;
    while (!q.empty())
    {
        auto tmp_ptr = q.front();
        q.pop();
        count_nodes++;
        if (tmp_ptr == NULL)
        {
            // should never reach here.
            fprintf(stderr, "ERROR: The tree contains node that have only one child\n");
            exit(-1);
        }
        else if ((tmp_ptr->left_node == NULL) && (tmp_ptr->right_node == NULL))
        {
            dbg_requires(tmp_ptr->is_leaf);
            dbg_requires(tmp_ptr->label == POS_LABEL || tmp_ptr->label == NEG_LABEL);
            count_leaf++;
        }
        else
        {
            dbg_requires(!tmp_ptr->is_leaf);
            dbg_requires(tmp_ptr->label == -1);
            q.push(tmp_ptr->left_node);
            q.push(tmp_ptr->right_node);
        }
    }
    dbg_assert(count_leaf == num_leaves);
    dbg_assert(count_nodes == num_nodes);
    printf("------------------------------------------------\n");
    printf("| Num_leaf: %d, num_nodes: %d, max_depth: %d | \n", num_leaves, num_nodes, cur_depth);
    printf("------------------------------------------------\n");

}


/* 
 * This function reture all the unlabeled leaf nodes in a breadth-first manner.
*/
vector<TreeNode *> DecisionTree::__get_unlabeled(TreeNode *node)
{
    queue<TreeNode *> q;
    q.push(node);
    vector<TreeNode *> ret;
    while (!q.empty())
    {
        auto tmp_ptr = q.front();
        q.pop();
        if (tmp_ptr == NULL)
        {
            // should never reach here.
            fprintf(stderr, "ERROR: The tree contains node that have only one child\n");
            exit(-1);
        }
        else if ((tmp_ptr->left_node == NULL) && (tmp_ptr->right_node == NULL) && tmp_ptr->label < 0)
        {
            ret.push_back(tmp_ptr);
        }
        else
        {
            q.push(tmp_ptr->left_node);
            q.push(tmp_ptr->right_node);
        }
    }
    return ret;
}

/*
 * initialize each leaf as unlabeled.
 */
void DecisionTree::batch_initialize(TreeNode *node)
{
    int feature_id = 0, class_id = 0;

    if (node == NULL)
    {
        // should never reach here.
        fprintf(stderr, "ERROR: The tree contains node that have only one child\n");
        exit(-1);
    }
    else if ((node->left_node == NULL) && (node->right_node == NULL))
    {
       node->init();
    }
    else
    {
        batch_initialize(node->left_node);
        batch_initialize(node->right_node);
    }
    return;
}


/*
 *
 */
TreeNode *DecisionTree::navigate(int data_index)
{
    TreeNode *ptr = this->root;
    while (!ptr->is_leaf)
    {
        dbg_assert(ptr->right_node != NULL && ptr->left_node != NULL);
        ptr = (ptr->split_ptr.decision_rule(data_index)) ? ptr->right_node : ptr->left_node;
    }
    return ptr;
}

/*
 * This function initialize the histogram for each unlabeled leaf node.
 * Also, potentially, it would free the previous histogram.
 */
void DecisionTree::init_histogram(vector<TreeNode *> &unlabeled_leaf)
{
    int c = 0;      
    assert(unlabeled_leaf.size() <= max_num_leaves);
    
    for (auto &p : unlabeled_leaf)
        p->histogram_id = c++;
}
