#include "tree.h"
#include "panel.h"
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
    entropy = -1;
}

SplitPoint::SplitPoint(int feature_id, double feature_value, double entropy)
{
    feature_id = feature_id;
    feature_value = feature_value;
    entropy = entropy;
}
/*
 * Reture True if the data is larger or equal than the split value
 */
bool SplitPoint::decition_rule(Data &data)
{
    return data.values[feature_id] >= feature_value;
}

// constructor function
TreeNode::TreeNode(int depth)
{
    this->depth = depth;
    init();
}

void TreeNode::init()
{
    is_leaf = false;
    has_new_data = false;
    label = -1;
    // remove this if you want to keep the previous batch data.
    data_ptr.clear();
    histogram_id = -1;
    histogram_ptr = NULL;
    return;
}

/*
 * Set label for the node as the majority class.
 */
void TreeNode::set_label()
{
    this->is_leaf = true;
    int pos_count = 0;
    for (auto &p : this->data_ptr)
    {
        pos_count = (p->label == POS_LABEL) ? pos_count + 1 : pos_count;
    }
    this->label = (pos_count >= (int)this->data_ptr.size() / 2) ? POS_LABEL : NEG_LABEL;
}

/*
 * This function split the data according to the best split feature id and value.
 * The data would be appended to the `left` if the data value is smaller than the split value
 */
void TreeNode::split(SplitPoint &best_split, vector<Data *> &left, vector<Data *> &right)
{
    this->split_ptr = best_split;
    double split_value = best_split.feature_value;
    for (auto &p : this->data_ptr)
    {
        double p_value = p->values[best_split.feature_id];
        if (p_value >= split_value)
            right.push_back(p);
        else
            left.push_back(p);
    }
}

DecisionTree::DecisionTree()
{
    this->max_num_leaves = -1;
    this->max_depth = -1;
    this->min_node_size = 0;
    this->depth = 0;
    this->num_leaves = 0;
    this->cur_depth = 0;
}

DecisionTree::DecisionTree(int max_num_leaves, int max_depth, int min_node_size)
{
    this->max_num_leaves = max_num_leaves;
    this->max_depth = max_depth;
    this->min_node_size = min_node_size;
    this->depth = 0;
    this->num_leaves = 0;
    this->cur_depth = 0;
}

/* 
 * Return true if the node should be a leaf.
 * This is determined by the min-node-size, max-depth, max_num_leaves
*/
bool DecisionTree::is_terminated(TreeNode *node)
{
    if (node->data_ptr.size() <= min_node_size)
    {
        dbg_printf("Node terminated: min_node_size\n");
        return true;
    }

    if (node->depth >= this->max_depth)
    {
        dbg_printf("Node terminated: max_depth\n");
        return true;
    }

    if (this->num_leaves >= this->max_num_leaves)
    {
        dbg_printf("Node terminated: max_num_leaves\n");
        return true;
    }

    return false;
}

void DecisionTree::train(Dataset &train_data, const int batch_size)
{
    int hasNext = TRUE;
    if (root == NULL)
        root = new TreeNode(0);    

	while (TRUE) {
		hasNext = train_data.streaming_read_data(10);		
        train_data.print_dataset();
        train_on_batch(train_data);        
		if (!hasNext) break;
	}		

	train_data.close_read_data();
    return;
}

void DecisionTree::test(Dataset &train_data)
{
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
        Histogram &hist = (*node->histogram_ptr)[i][0];
        Histogram merged_hist = Histogram(this->max_bin_size, hist.bins);
        for (int k = 1; k < this->datasetPointer->num_of_classes; k++)
            merged_hist.merge((*node->histogram_ptr)[i][k], this->max_bin_size);

        std::vector<double> possible_splits;
        merged_hist.uniform(possible_splits, this->max_bin_size);
        // get the split value
        for (auto &split_value : possible_splits)
        {
            double gain = hist.sum(split_value);
            SplitPoint t = SplitPoint(i, split_value, gain);
            results.push_back(t);
        }
    }
    std::vector<SplitPoint>::iterator best_split = std::max_element(results.begin(), results.end(),
                                                                    [](const SplitPoint &l, const SplitPoint &r) { return l.entropy < r.entropy; });

    SplitPoint v = SplitPoint(best_split->feature_id, best_split->feature_value, best_split->entropy);
    split = v;
}

/* 
 * This function reture all the unlabeled leaf nodes in a breadth-first manner.
*/
vector<TreeNode *> DecisionTree::__get_unlabeled(TreeNode *node)
{
    queue<TreeNode *> q;
    q.push(node);
    vector<TreeNode *> ret;
    int c = 0;
    while (q.empty())
    {
        auto tmp_ptr = q.front();
        q.pop();
        if (node == NULL)
        {
            // should never reach here.
            fprintf(stderr, "ERROR: The tree contains node that have only one child\n");
            exit(-1);
        }
        else if ((node->left_node == NULL) && (node->left_node == NULL))
        {
            if (node->is_leaf && node->label < 0)
            {
                ret.push_back(node);
                node->histogram_id = c++;
            }
        }
        else
        {
            q.push(node->left_node);
            q.push(node->right_node);
        }
    }
    return ret;
}
/*
 * Serial version of training.
*/
void DecisionTree::train_on_batch(Dataset &train_data)
{
    // Reinitialize every leaf in T as unlabeled.
    initialize(root);
    vector<TreeNode *> unlabeled_leaf = __get_unlabeled(root);
    while (unlabeled_leaf.size() > 0)
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
                auto left_tree = new TreeNode(this->cur_depth);
                auto right_tree = new TreeNode(this->cur_depth);
                cur_leaf->split(best_split, left_tree->data_ptr, right_tree->data_ptr);
                unlabeled_leaf_new.push_back(left_tree);
                unlabeled_leaf_new.push_back(right_tree);
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
    for (auto &d : data)
    {
        auto node = DecisionTree::navigate(d);
        node->data_ptr.push_back(&d);
        node->has_new_data = true;
        for (int attr = 0; attr < this->datasetPointer->num_of_features; attr++)
        {
            (*(node->histogram_ptr))[attr][d.label].update(d.values[attr]);
        }
    }
}
/*
 * initialize each node. This function is called when a new batch comes.
 */
void DecisionTree::initialize(TreeNode *node)
{
    int feature_id = 0, class_id = 0;

    if (node == NULL)
    {
        // should never reach here.
        fprintf(stderr, "ERROR: The tree contains node that have only one child\n");
        exit(-1);
    }
    else if ((node->left_node == NULL) && (node->left_node == NULL))
    {
        node->init();
    }
    else
    {
        initialize(node->left_node);
        initialize(node->right_node);
    }
    return;
}

/*
 * This function initialize the histogram for each unlabeled leaf node.
 * Also, potentially, it would free the previous histogram.
 */
void DecisionTree::init_histogram(vector<TreeNode *> &unlabled_leaf)
{
    if (bin_ptr != NULL)
        delete[] bin_ptr;

    bin_ptr = new Bin_t[max_num_leaves * num_feature_num * num_class * max_bin_size];
    for (auto &p : unlabled_leaf)
    {
        for (int feature_id = 0; feature_id < datasetPointer->num_of_features; feature_id++)
        {
            for (int class_id = 0; class_id < datasetPointer->num_of_classes; class_id++)
            {
                //TODO: initialize the histogram.
                histogram[p->histogram_id][feature_id][class_id].clear();
                memset(&bin_ptr[RLOC(p->histogram_id, feature_id, class_id, num_feature_num, num_class, max_bin_size)], 0, max_bin_size * sizeof(Bin_t));
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
        ptr = (ptr->split_ptr.decition_rule(d)) ? ptr->right_node : ptr->left_node;
    }

    return ptr;
}