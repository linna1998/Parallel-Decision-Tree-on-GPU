#include "tree.h"
#include <queue>
#include "histogram.h"

void DecisionTree::set_label(int label){
    this->label = label;
}

void DecisionTree::set_histogram(Histogram& hist){
    this->histgram = hist;
}

void DecisionTree::set_dta_idx(int dta_start_idx, int dta_end_idx){
    this->dta_start_idx = dta_start_idx;
    this->dta_end_idx = dta_end_idx;
}

void DecisionTree::train(Dataset& train_data, const int batch_size){
    auto root = new TreeNode();
    root->set_dta_idx(0, train_data.num_of_data-1);
    this->root = root;
    queue<TreeNode*> que;
    que.push(root);
    auto Histogram = new Histogram(train_data, root->dta_start_idx, root->dta_end_idx, this->max_bin_size);

}