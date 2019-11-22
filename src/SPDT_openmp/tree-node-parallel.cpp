#include "tree.h"
#include "panel.h"
#include <assert.h>
#include <queue>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <time.h>

#define NUM_OF_THREADS 8
double COMPRESS_TIME = 0;
double SPLIT_TIME = 0;

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

/*
 * Reture True if the data is larger or equal than the split value
 */
bool SplitPoint::decision_rule(Data &data)
{
    dbg_ensures(entropy >= -EPS);
    dbg_ensures(gain >= -EPS);
    dbg_ensures(feature_id >= 0);
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
    entropy = -1.f;
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
    this->entropy = best_split.entropy;
    double split_value = best_split.feature_value;
    int num_pos_label_left=0;
    int num_pos_label_right=0;
    
    // shared vector in openmp parallel ?
    // #pragma omp parallel for schedule(dynamic) shared(right, left, num_pos_label_right, num_pos_label_left)
    for (int i = 0; i < this->data_ptr.size(); i++)
    {
        Data* p = this->data_ptr[i];
        double p_value = p->values[best_split.feature_id];
        if (best_split.decision_rule(*p)) {
            right->data_ptr.push_back(p);
            num_pos_label_right = (p->label == POS_LABEL) ? num_pos_label_right+1 : num_pos_label_right;
        }
        else {
            left->data_ptr.push_back(p);
            num_pos_label_left = (p->label == POS_LABEL) ? num_pos_label_left+1 : num_pos_label_left;
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
    data_ptr.clear();
}

DecisionTree::DecisionTree()
{
    this->max_num_leaves = 64;
    this->max_depth = -1;
    this->min_node_size = 1;
    this->depth = 0;
    this->num_leaves = 0;
    this->max_bin_size = 12;
    this->cur_depth = 0;
    this->root = NULL;
    this->bin_ptr = NULL;
    this->min_gain = 1e-3;
    this->num_nodes = 0;

}

DecisionTree::~DecisionTree(){
    delete[] bin_ptr;
    root->clear();
}

DecisionTree::DecisionTree(int max_num_leaves, int max_depth, int min_node_size, int max_bin_size)
{
    this->max_num_leaves = max_num_leaves;
    this->max_depth = max_depth;
    this->min_node_size = min_node_size;
    this->depth = 0;
    this->num_leaves = 0;
    this->max_bin_size = max_bin_size;
    this->cur_depth = 0;
    this->root = NULL;
    this->bin_ptr = NULL;
    this->min_gain = 1e-3;
    this->num_nodes = 0;
}

/* 
 * Return true if the node should be a leaf.
 * This is determined by the min-node-size, max-depth, max_num_leaves
*/
bool DecisionTree::is_terminated(TreeNode *node)
{
    if (min_node_size != -1 && node->data_ptr.size() <= min_node_size)
    {
        dbg_printf("Node [%d] terminated: min_node_size=%d >= %d\n", node->id, min_node_size, node->data_ptr.size());
        return true;
    }

    if (max_depth != -1 && node->depth >= this->max_depth)
    {
        dbg_printf("Node [%d] terminated: max_depth\n", node->id);
        return true;
    }

    if (max_num_leaves != -1 && this->num_leaves >= this->max_num_leaves)
    {
        dbg_printf("Node [%d] terminated: max_num_leaves\n", node->id);
        return true;
    }

    if (!node->num_pos_label || node->num_pos_label == (int) node->data_ptr.size()){
        dbg_assert(node->entropy < EPS);
        dbg_printf("Node [%d] terminated: all samples belong to same class\n",node->id);
        return true; 
    }
    dbg_printf("[%d] num_data=%d, num_pos=%d\n", node->id, node->data_ptr.size(), node->num_pos_label);
    return false;
}

void DecisionTree::initialize(Dataset &train_data, const int batch_size){
    this->datasetPointer = &train_data;
    root = new TreeNode(0, this->num_nodes++);  
    if (bin_ptr != NULL) {        
        delete[] bin_ptr;
    }
    long long number = (long long)max_num_leaves * datasetPointer->num_of_features * datasetPointer->num_of_classes * max_bin_size;    
    dbg_printf("Init Root Node [%.4f] MB\n", number * sizeof(Bin_t) / 1024.f);
    
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
        dbg_printf("Train size (%d, %d, %d)\n", train_data.num_of_data, 
                train_data.num_of_features, train_data.num_of_classes);
        train_on_batch(train_data);        
		if (!hasNext) break;
	}		

	train_data.close_read_data();    
    return;
}

double DecisionTree::test(Dataset &test_data) {    

    int i = 0;
    int correct_num = 0;
    test_data.streaming_read_data(test_data.num_of_data);

    // #pragma omp parallel for schedule(dynamic) reduction(+:correct_num)        
    for (i = 0; i < test_data.num_of_data; i++) {
        // printf("lable 1: %d, label 2: %d\n", 
        //     navigate(test_data.dataset[i])->label,
        //     test_data.dataset[i].label);
        // dbg_assert(navigate(test_data.dataset[i])->label != -1);
        if (navigate(test_data.dataset[i])->label == test_data.dataset[i].label) {
            correct_num++;
        }
    }    
    return (double)correct_num / (double)test_data.num_of_data;
}

/*
 * Calculate the entropy gain = H(Y) - H(Y|X)
 * H(Y|X) needs parameters p(X<a), p(Y=0|X<a), p(Y=0|X>=a)
 * Assuming binary classification problem
 */
void get_gain(TreeNode* node, SplitPoint& split, int feature_id){
    int total_sum = node->data_ptr.size();
    dbg_ensures(total_sum > 0);
    double sum_class_0 = (double)(*node->histogram_ptr)[feature_id][NEG_LABEL].get_total();
    double sum_class_1 = (double)(*node->histogram_ptr)[feature_id][POS_LABEL].get_total();
    dbg_assert((sum_class_1-node->num_pos_label) < EPS);
    double left_sum_class_0 = (*node->histogram_ptr)[feature_id][NEG_LABEL].sum(split.feature_value);
    double right_sum_class_0 = sum_class_0 - left_sum_class_0;
    double left_sum_class_1 = (*node->histogram_ptr)[feature_id][POS_LABEL].sum(split.feature_value);
    double right_sum_class_1 = sum_class_1 - left_sum_class_1;
    double left_sum = left_sum_class_0 + left_sum_class_1;
    double right_sum = right_sum_class_0 + right_sum_class_1;

    double px = (left_sum_class_0 + left_sum_class_1) / (1.0 * total_sum); // p(x<a)
    double py_x0 = (left_sum <= EPS) ? 0.f : left_sum_class_0 / left_sum;                            // p(y=0|x < a)
    double py_x1 = (right_sum <= EPS) ? 0.f : right_sum_class_0 / right_sum;                          // p(y=0|x >= a)
    // printf("sum_class_1=%f, sum_class_0=%f, right_sum = %f, right_sum_class_0 = %f right_sum_class_1= %f\n", sum_class_1, sum_class_0, right_sum, right_sum_class_0, right_sum_class_1);
    // printf("py_x0 = %f, py_x1 = %f\n", py_x0, py_x1);
    dbg_ensures(py_x0 >= -EPS && py_x0 <= 1+EPS);
    dbg_ensures(py_x1 >= -EPS && py_x1 <= 1+EPS);
    dbg_ensures(px >= -EPS && px <= 1+EPS);
    double entropy_left = ((1-py_x0) < EPS || py_x0 < EPS) ? 0 : -py_x0 * log2(py_x0) - (1-py_x0)*log2(1-py_x0);
    double entropy_right = ((1-py_x1) < EPS || py_x1 < EPS) ? 0 : -py_x1 * log2(py_x1) - (1-py_x1)*log2(1-py_x1);
    double H_YX = px * entropy_left + (1-px) * entropy_right;
    double px_prior = sum_class_0 / (sum_class_0 + sum_class_1);
    dbg_ensures(px_prior > 0 && px_prior < 1);
    split.entropy = ((1-px_prior) < EPS || px_prior < EPS) ? 0 : -px_prior * log2(px_prior) - (1-px_prior) * log2(1-px_prior);
    split.gain = split.entropy - H_YX;
    // printf("%f = %f - %f\n", split.gain, split.entropy, H_YX);
    dbg_ensures(split.gain >= -EPS);
}

void reduce(std::vector<SplitPoint> *v1, int begin, int end) {
    if (end - begin == 1) return;
    int pivot = (begin + end)/2;
    #pragma omp task
    reduce(v1, begin, pivot);
    #pragma omp task
    reduce(v1, pivot, end);
    #pragma omp taskwait
    v1[begin].insert(v1[begin].end(), v1[pivot].begin(), v1[pivot].end());    
}

/*
 * This function return the best split point at a given leaf node.
 * Best split is store in `split`
*/
void DecisionTree::find_best_split(TreeNode *node, SplitPoint &split)
{
    clock_t start, end;
    start = clock();    
    SplitPoint best_split = SplitPoint();
    int tot = 0; // used to count the number of results
    #pragma omp barrier
    #pragma omp parallel for schedule(static) num_threads(NUM_OF_THREADS)
    for (int i = 0; i < this->datasetPointer->num_of_features; i++)
    {
        int tid = omp_get_thread_num();
        // merge different labels
        Histogram& hist = (*node->histogram_ptr)[i][0];
        Histogram merged_hist;
        merged_hist = hist;
        for (int k = 1; k < this->datasetPointer->num_of_classes; k++)
            merged_hist.merge((*node->histogram_ptr)[i][k], this->max_bin_size);

        std::vector<double> possible_splits;
        merged_hist.uniform(possible_splits, merged_hist.bin_size);
        for (int j=0; j<possible_splits.size(); j++)
        {
            SplitPoint t = SplitPoint(i, possible_splits[j]);
            get_gain(node, t, i);
            if (t.gain > best_split.gain)
                best_split = t;
        }
    }
    split.feature_id = best_split.feature_id;
    split.feature_value = best_split.feature_value;
    split.gain = best_split.gain;
    end = clock();   
    SPLIT_TIME += ((double) (end - start)) / CLOCKS_PER_SEC; 
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
 * Serial version of training.
*/
void DecisionTree::train_on_batch(Dataset &train_data)
{
    
    for(auto& data: train_data.dataset)
        root->data_ptr.push_back(&data);
    clock_t start, end;
    double COMPRESS_TIME, SPLIT_TIME;
    COMPRESS_TIME = SPLIT_TIME = 0;

    double pos_rate = ((double) train_data.num_pos_label) / train_data.num_of_data;
    dbg_assert(pos_rate > 0 && pos_rate < 1);
    root->num_pos_label = train_data.num_pos_label;
    root->entropy = - pos_rate * log2(pos_rate) - (1-pos_rate) * log2((1-pos_rate));
    // Reinitialize every leaf in T as unlabeled.
    batch_initialize(root);
    vector<TreeNode *> unlabeled_leaf = __get_unlabeled(root);
    dbg_assert(unlabeled_leaf.size() <= max_num_leaves);
    while (!unlabeled_leaf.empty())
    {        
        // each while loop would add a new level node.
        this->cur_depth++;
        vector<TreeNode *> unlabeled_leaf_new; 
        if (unlabeled_leaf.size() > max_num_leaves) {
            for (int i = 0; i < unlabeled_leaf.size(); i++) {
                unlabeled_leaf[i]->set_label();
                this->num_leaves++;
            }
            break;
        }       
        init_histogram(unlabeled_leaf);        
        compress(train_data.dataset, unlabeled_leaf);  

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < unlabeled_leaf.size(); i++) {
            TreeNode* cur_leaf = unlabeled_leaf[i];            
            if (is_terminated(cur_leaf))
            {         
                cur_leaf->set_label();
                this->num_leaves++;             
            }
            else
            {                
                SplitPoint best_split;
                find_best_split(cur_leaf, best_split);
                dbg_ensures(best_split.gain >= -EPS);
                if (best_split.gain <= min_gain){
                    dbg_printf("Node terminated: gain=%.4f <= %.4f\n", min_node_size, best_split.gain, min_gain);
                    cur_leaf->set_label();
                    this->num_leaves++;               
                    continue;
                }
                cur_leaf->left_node = new TreeNode(this->cur_depth, this->num_nodes++);
                cur_leaf->right_node = new TreeNode(this->cur_depth, this->num_nodes++);
                cur_leaf->split(best_split, cur_leaf->left_node, cur_leaf->right_node);
                this->num_leaves = (cur_leaf->is_leaf) ? this->num_leaves - 1 : this->num_leaves;          
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

/*
 * This function compress the data into histograms.
 * Each unlabeled leaf would have a (num_feature, num_class) histograms
 * This function takes the assumption that each leaf is re-initialized (we use a batch mode)
*/
void DecisionTree::compress(vector<Data> &data, vector<TreeNode *> &unlabeled_leaf) {
    clock_t start, end;
    start = clock();
    int feature_id = 0, class_id = 0;
    // Construct the histogram. and navigate each data to its leaf.
    // #pragma omp parallel for schedule(dynamic)    
    for (int i = 0; i < unlabeled_leaf.size(); i++) {
        TreeNode* node = unlabeled_leaf[i];
        if (node->data_ptr.size() > 0) {
            node->has_new_data = true;
        }                
        for (int i = 0; i < node->data_ptr.size(); i++) {
            Data* d = node->data_ptr[i];                               
            for (int attr = 0; attr < this->datasetPointer->num_of_features; attr++) {            
                if (d->values.find(attr) != d->values.end()) {
                    (*(node->histogram_ptr))[attr][d->label].update(d->values[attr]);
                }     
            }
        }
    }
    end = clock();   
    COMPRESS_TIME += ((double) (end - start)) / CLOCKS_PER_SEC; 
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
void DecisionTree::self_check(){
    #ifdef DEBUG
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
    dbg_printf("------------------------------------------------\n");
    dbg_printf("| Num_leaf: %d, num_nodes: %d, max_depth: %d | \n", num_leaves, num_nodes, cur_depth);
    dbg_printf("------------------------------------------------\n");
    #endif

}
/*
 * This function initialize the histogram for each unlabeled leaf node.
 * Also, potentially, it would free the previous histogram.
 */
void DecisionTree::init_histogram(vector<TreeNode *> &unlabeled_leaf)
{
    int c = 0;      
    assert(unlabeled_leaf.size() <= max_num_leaves);
    
    for (auto &p : unlabeled_leaf) {
        p->histogram_id = c++;  
    }   
    // #pragma omp parallel for schedule(dynamic)   
    for (int i = 0; i < unlabeled_leaf.size(); i++) {
        TreeNode* p = unlabeled_leaf[i];      
        for (int feature_id = 0; feature_id < datasetPointer->num_of_features; feature_id++) {           
            for (int class_id = 0; class_id < datasetPointer->num_of_classes; class_id++) {                
                histogram[p->histogram_id][feature_id][class_id].clear();
                histogram[p->histogram_id][feature_id][class_id].bins = &bin_ptr[RLOC(p->histogram_id, feature_id, class_id, datasetPointer->num_of_features, datasetPointer->num_of_classes, max_bin_size)];                
                histogram[p->histogram_id][feature_id][class_id].check(__LINE__);
            }     
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
        dbg_assert(ptr->right_node != NULL && ptr->left_node != NULL);
        ptr = (ptr->split_ptr.decision_rule(d)) ? ptr->right_node : ptr->left_node;
    }
    return ptr;
}