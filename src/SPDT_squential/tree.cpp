#include "tree.h"
#include "panel.h"
#include <assert.h>
#include <queue>
#include <stdio.h>
#include <algorithm>
#include <math.h>


/*
 * For A[][M][N][Z]
 * A[i][j][k][e] = A[N*Z*M*i+Z*N*j+k*Z+e]
 */
inline int RLOC(int i, int j, int k, int e, int& M, int &N, int& Z){
    return N*Z*M*i+Z*N*j+k*Z*k+e;
}

/*
 * For A[][M][N][Z]
 * A[i][j][k] = A[N*Z*M*i+Z*N*j+k*Z]
 */
inline int RLOC(int i, int j, int k, int& M, int &N, int& Z){
    return N*Z*M*i+Z*N*j+k*Z*k;
}

/*
 * For A[][M][N][Z]
 * A[i][j] = A[N*Z*M*i+Z*N*j]
 */
inline int RLOC(int i, int j, int& M, int &N, int& Z){
    return N*Z*M*i+Z*N*j;
}

/*
 * For A[][M][N][Z]
 * A[i] = A[N*Z*M*i]
 */
inline int RLOC(int i, int& M, int &N, int& Z){
    return N*Z*M*i;
}

SplitPoint::SplitPoint()
{
    feature_id = -1;
    feature_value = 0;
    entropy_left = 0;
    entropy_right = 0;
    entropy_before = 0;
}

SplitPoint::SplitPoint(int feature_id, double feature_value)
{
    this->feature_id = feature_id;
    this->feature_value = feature_value;
    this->entropy_left = 0;
    this->entropy_right = 0;
    this->entropy_before = 0;

}
/*
 * Reture True if the data is larger or equal than the split value
 */
bool SplitPoint::decision_rule(Data &data)
{
    return data.values[feature_id] >= feature_value;
}

// constructor function
TreeNode::TreeNode(int depth, int id)
{
    this->id = id;
    this->depth = depth;
    is_leaf = false;
    has_new_data = false;
    label = -1;
    // remove this if you want to keep the previous batch data.
    data_ptr.clear();
    histogram_id = -1;
    histogram_ptr = NULL;
    left_node = NULL;
    right_node = NULL;
    entropy = 0.f;
    num_pos_label=0;
}


void TreeNode::init()
{
    has_new_data = false;
    label = -1;
    histogram_id = -1;
    histogram_ptr = NULL;
    left_node = NULL;
    right_node = NULL;
    return;
}

/*
 * Set label for the node as the majority class.
 */
void TreeNode::set_label()
{
    this->is_leaf = true;
    this->label = (this->num_pos_label >= (int)this->data_ptr.size() / 2) ? POS_LABEL : NEG_LABEL;
}

/*
 * This function split the data according to the best split feature id and value.
 * The data would be appended to the `left` if the data value is smaller than the split value
 */
void TreeNode::split(SplitPoint &best_split, TreeNode* left, TreeNode* right)
{
    this->split_ptr = best_split;
    double split_value = best_split.feature_value;
    int num_pos_lebel_left=0;
    int num_pos_lebel_right=0;
    for (auto &p : this->data_ptr)
    {
        double p_value = p->values[best_split.feature_id];
        if (p_value >= split_value){
            right->data_ptr.push_back(p);
            num_pos_lebel_right = (p->label == POS_LABEL) ? num_pos_lebel_right+1 : num_pos_lebel_right;
        }
        else{
            left->data_ptr.push_back(p);
            num_pos_lebel_left = (p->label == POS_LABEL) ? num_pos_lebel_left+1 : num_pos_lebel_left;
        }
    }
    left->entropy = best_split.entropy_left;
    left->num_pos_label = num_pos_lebel_left;
    right->num_pos_label = num_pos_lebel_right;
    right->entropy = best_split.entropy_right;
    if (left->id == 14){
        printf("[%d] num_pos=%d/%d, entropy=%.4f\n", left->id, num_pos_lebel_left, left->data_ptr.size(), left->entropy);
    }
    if (right->id == 14){
        printf("[%d] num_pos=%d/%d, entropy=%.4f\n", right->id, num_pos_lebel_right, right->data_ptr.size(), right->entropy);
    }
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
    data_ptr.clear();
}

DecisionTree::DecisionTree()
{
    this->max_num_leaves = -1;
    this->max_depth = -1;
    this->min_node_size = 0;
    this->depth = 0;
    this->num_leaves = 0;
    this->cur_depth = 0;
    this->root = NULL;
    this->bin_ptr = NULL;
    num_nodes = 0;

}

DecisionTree::~DecisionTree(){
    delete[] bin_ptr;
    root->clear();
}

DecisionTree::DecisionTree(int max_num_leaves, int max_depth, int min_node_size)
{
    this->max_num_leaves = max_num_leaves;
    this->max_depth = max_depth;
    this->min_node_size = min_node_size;
    this->depth = 0;
    this->num_leaves = 0;
    this->cur_depth = 0;
    this->root = NULL;
    this->bin_ptr = NULL;
    num_nodes = 0;
}

/* 
 * Return true if the node should be a leaf.
 * This is determined by the min-node-size, max-depth, max_num_leaves
*/
bool DecisionTree::is_terminated(TreeNode *node)
{
    if (node->data_ptr.size() <= min_node_size)
    {
        dbg_printf("Node [%d] terminated: min_node_size=%d >= %d\n", min_node_size, node->data_ptr.size(), node->id);
        return true;
    }

    if (node->depth >= this->max_depth)
    {
        dbg_printf("Node [%d] terminated: max_depth\n", node->id);
        return true;
    }

    if (this->num_leaves >= this->max_num_leaves)
    {
        dbg_printf("Node [%d] terminated: max_num_leaves\n", node->id);
        return true;
    }

    if (!node->num_pos_label || node->num_pos_label == (int) node->data_ptr.size()){
        dbg_assert(node->entropy < EPS);
        dbg_printf("Node [%d] terminated: all samples belong to same class\n",node->id);
        return true; 
    }

    if (node->entropy < EPS){
        if (node->id == 14){
            (*node->histogram_ptr)[0][0].print();
            (*node->histogram_ptr)[0][1].print();
        }
        // dbg_assert(node->num_pos_label == 0 || node->num_pos_label == (int) node->data_ptr.size());
        dbg_printf("Node [%d] terminated: entropy=%.4f < %.8f and num_pos=%d/%d\n", node->id, node->entropy, EPS, node->num_pos_label, node->data_ptr.size());
        return true; 
    }

    return false;
}

void DecisionTree::initialize(Dataset &train_data, const int batch_size){
    this->datasetPointer = &train_data;
    dbg_printf("Init Root Node\n");
    root = new TreeNode(0, this->num_nodes++);  
    root->is_leaf = true;
    if (bin_ptr != NULL) {        
        delete[] bin_ptr;
    }
    long long number = max_num_leaves * datasetPointer->num_of_features * datasetPointer->num_of_classes * max_bin_size;    
    bin_ptr = new Bin_t[number];
    memset(bin_ptr, 0, number * sizeof(Bin_t));  
    histogram = Histogram_ALL(max_num_leaves, Histogram_LEAF(train_data.num_of_features, Histogram_FEATURE(train_data.num_of_classes, Histogram(max_bin_size))));
}

void DecisionTree::train(Dataset &train_data, const int batch_size)
{
    int hasNext = TRUE;
    initialize(train_data, batch_size);
	while (TRUE) {
		hasNext = train_data.streaming_read_data(batch_size);		
        // train_data.print_dataset();
        dbg_printf("Train size (%d, %d, %d)\n", train_data.num_of_data, 
                train_data.num_of_features, train_data.num_of_classes);
        train_on_batch(train_data);        
		if (!hasNext) break;
	}		

	train_data.close_read_data();
    root->print();
    return;
}

void DecisionTree::test(Dataset &train_data)
{
}

/*
 * Calculate the entropy gain = H(Y) - H(Y|X)
 * H(Y|X) needs parameters p(X<a), p(Y=0|X<a), p(Y=0|X>=a)
 * Assuming binary classification problem
 */
void get_gain(TreeNode* node, SplitPoint& split, int feature_id){
    int total_sum = node->data_ptr.size();
    dbg_ensures(total_sum > 0);
    double left_sum_class_0 = (*node->histogram_ptr)[feature_id][0].sum(split.feature_value);
    double right_sum_class_0 = (*node->histogram_ptr)[feature_id][0].get_total() - left_sum_class_0;
    double left_sum_class_1 = (*node->histogram_ptr)[feature_id][1].sum(split.feature_value);
    double right_sum_class_1 = (*node->histogram_ptr)[feature_id][1].get_total() - left_sum_class_1;
    double px = (left_sum_class_0 + left_sum_class_1) / (1.0 * total_sum); // p(x<a)
    double py_x0 = left_sum_class_0 / (left_sum_class_0 + left_sum_class_1); // p(y=0|x < a)
    double py_x1 = right_sum_class_0 / (right_sum_class_0 + right_sum_class_1); // p(y=0|x >= a)
    dbg_ensures(py_x0 >= 0 && py_x0 <= 1);
    dbg_ensures(py_x1 >= 0 && py_x1 <= 1);
    dbg_ensures(px >= 0 && px <= 1);
    split.entropy_left = ((1-py_x0) < EPS || py_x0 < EPS) ? 0 : -py_x0 * log2(py_x0) - (1-py_x0)*log2(1-py_x0);
    split.entropy_right = ((1-py_x1) < EPS || py_x1 < EPS) ? 0 : -py_x1 * log2(py_x1) - (1-py_x1)*log2(1-py_x1);
    double H_YX = px * split.entropy_left + (1-px) * split.entropy_right;
    split.gain = split.entropy_before - H_YX;
    dbg_printf("%.4f = %.4f - %.4f\n", split.gain, split.entropy_before, H_YX);
    // dbg_ensures(split.gain >= 0);
}

/*
 * This function return the best split point at a given leaf node.
 * Best split is store in `split`
*/
void DecisionTree::find_best_split(TreeNode *node, SplitPoint &split)
{
    std::vector<SplitPoint> results;
    for (int i = 0; i < this->datasetPointer->num_of_features; i++)
    {
        // merge different labels
        Histogram& hist = (*node->histogram_ptr)[i][0];
        Histogram merged_hist;
        merged_hist = hist;
        for (int k = 1; k < this->datasetPointer->num_of_classes; k++)
            merged_hist.merge((*node->histogram_ptr)[i][k], this->max_bin_size);
        
        if (i==12){
            printf("merge \n");
            merged_hist.print();
            printf("class 0 \n");
            (*node->histogram_ptr)[i][0].print();
            printf("class 1 \n");
            (*node->histogram_ptr)[i][1].print();
        }
        std::vector<double> possible_splits;
        merged_hist.uniform(possible_splits, merged_hist.bin_size);
        if (i==12){
            printf("possible_splits = [ ");
            for (auto& split_value: possible_splits)
                printf("(%.4f)", split_value);
            printf("]\n");                
        }
        for (auto& split_value: possible_splits)
        {
            SplitPoint t = SplitPoint(i, split_value);
            t.entropy_before = node->entropy;
            get_gain(node, t, i);
            results.push_back(t);
        }
    }
    std::vector<SplitPoint>::iterator best_split = std::max_element(results.begin(), results.end(),
                                                                    [](const SplitPoint &l, const SplitPoint &r) { return l.gain < r.gain; });

    split.feature_id = best_split->feature_id;
    split.feature_value = best_split->feature_value;
    split.entropy_right = best_split->entropy_right;
    split.entropy_left = best_split->entropy_left;
    split.entropy_before = best_split->entropy_before;
    split.gain = best_split->gain;
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
        else if ((tmp_ptr->left_node == NULL) && (tmp_ptr->right_node == NULL))
        {
            dbg_requires(tmp_ptr->is_leaf);
            dbg_requires(tmp_ptr->label < 0);
            if (tmp_ptr->is_leaf && tmp_ptr->label < 0)
            {
                ret.push_back(tmp_ptr);
            }
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
 * Serial version of training.
*/
void DecisionTree::train_on_batch(Dataset &train_data)
{
    
    for(auto& data: train_data.dataset)
        root->data_ptr.push_back(&data);
    double pos_rate = ((double) train_data.num_pos_label) / train_data.num_of_data;
    dbg_assert(pos_rate > 0 && pos_rate < 1);
    root->num_pos_label = train_data.num_pos_label;
    root->entropy = - pos_rate * log2(pos_rate) - (1-pos_rate) * log2((1-pos_rate));
    // Reinitialize every leaf in T as unlabeled.
    batch_initialize(root);
    vector<TreeNode *> unlabeled_leaf = __get_unlabeled(root);

    while (!unlabeled_leaf.empty())
    {
        // each while loop would add a new level node.
        this->cur_depth++;
        vector<TreeNode *> unlabeled_leaf_new;
        init_histogram(unlabeled_leaf);        
        compress(train_data.dataset, unlabeled_leaf);        
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
                find_best_split(cur_leaf, best_split);
                if (best_split.gain <= min_gain){
                    dbg_printf("Node terminated: gain=%.4f <= %.4f\n", min_node_size, best_split.gain, min_gain);
                    cur_leaf->set_label();
                    this->num_leaves++;               
                    continue;
                }
                dbg_printf("best split: id=%d, value=%.4f, gain=%.4f\n", best_split.feature_id, best_split.feature_value, best_split.gain);
                cur_leaf->left_node = new TreeNode(this->cur_depth, this->num_nodes++);
                cur_leaf->right_node = new TreeNode(this->cur_depth, this->num_nodes++);
                cur_leaf->split(best_split, cur_leaf->left_node, cur_leaf->right_node);
                unlabeled_leaf_new.push_back(cur_leaf->left_node);
                unlabeled_leaf_new.push_back(cur_leaf->right_node);
                this->num_leaves--;                
            }
        }
        unlabeled_leaf = unlabeled_leaf_new;
        unlabeled_leaf_new.clear();        
    }
}

/*
 * This function compress the data into histograms.
 * Each unlabeled leaf would have a (num_feature, num_class) histograms
 * This function takes the assumption that each leaf is re-initialized (we use a batch mode)
*/
void DecisionTree::compress(vector<Data> &data, vector<TreeNode *> &unlabled_leaf)
{
    int feature_id = 0, class_id = 0;
    // Construct the histogram. and navigate each data to its leaf.
    for (auto& node: unlabled_leaf){
        for (auto &d: node->data_ptr)
        {
            node->has_new_data = true;
            for (int attr = 0; attr < this->datasetPointer->num_of_features; attr++)
            {            
                if (d->values.find(attr) != d->values.end()) {
                    (*(node->histogram_ptr))[attr][d->label].update(d->values[attr]);
                }     
            }
        }
    }
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
 * This function initialize the histogram for each unlabeled leaf node.
 * Also, potentially, it would free the previous histogram.
 */
void DecisionTree::init_histogram(vector<TreeNode *> &unlabled_leaf)
{
    int c = 0;      
    for (auto &p : unlabled_leaf)
    {
        p->histogram_id = c++;
        for (int feature_id = 0; feature_id < datasetPointer->num_of_features; feature_id++)
            for (int class_id = 0; class_id < datasetPointer->num_of_classes; class_id++) {                
                histogram[p->histogram_id][feature_id][class_id].clear();
                histogram[p->histogram_id][feature_id][class_id].bins = &bin_ptr[RLOC(p->histogram_id, feature_id, class_id, datasetPointer->num_of_features, datasetPointer->num_of_classes, max_bin_size)];                
                histogram[p->histogram_id][feature_id][class_id].check();
            }                

        p->histogram_ptr = &histogram[p->histogram_id];   
    }
}

/*
 *
 */
TreeNode *DecisionTree::navigate(Data &d)
{
    TreeNode *ptr = this->root;
    while (!ptr->is_leaf)
    {
        ptr = (ptr->split_ptr.decision_rule(d)) ? ptr->right_node : ptr->left_node;
    }

    return ptr;
}