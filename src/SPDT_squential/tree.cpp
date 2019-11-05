#include "tree.h"
#include <queue>

// constructor function
TreeNode::TreeNode() {

}

void TreeNode::set_label(int label){
    this->label = label;
}

void TreeNode::set_histogram(Histogram* histogram){
    this->histogram = histogram;
}

void TreeNode::set_dta_idx(int dta_start_idx, int dta_end_idx){
    this->dta_start_idx = dta_start_idx;
    this->dta_end_idx = dta_end_idx;
}

void TreeNode::set_split_info(
    int optimal_split_feature_idx, 
    double optimal_split_feature_value, 
    double entropy) {
        this->optimal_split_feature_idx = optimal_split_feature_idx;
        this->optimal_split_feature_value = optimal_split_feature_value;
        this->entropy = entropy;
}

DecisionTree::DecisionTree() {
    this->max_num_leaves = -1;
    this->max_depth = -1;
    this->min_node_size = 0;
}

DecisionTree::DecisionTree(int max_num_leaves, int max_depth, int min_node_size) {
    this->max_num_leaves = max_num_leaves;
    this->max_depth = max_depth;
    this->min_node_size = min_node_size;
}

void DecisionTree::train(Dataset& train_data, const int batch_size) {
	TreeNode *root = new TreeNode();
    root->set_dta_idx(0, train_data.num_of_data-1);
    this->root = root;
    queue<TreeNode*> que;
    que.push(root);
	Histogram *histogram = new Histogram(this->max_bin_size);
    root->histogram = histogram;
}

void DecisionTree::train_on_batch(Dataset& train_data) {

}

void DecisionTree::test(Dataset& train_data) {

}

void DecisionTree::split(Dataset& data, int optimal_split_bin_idx) {

}

void DecisionTree::find_best_split(
    TreeNode* node, 
    int& split_feature_id, 
    float& split_feature_value) {

}

void DecisionTree::construct_histogram(
    Dataset& Dataset, 
    int start_idx, 
    int end_idx) {
        
}

int main() {
    return 0;
}