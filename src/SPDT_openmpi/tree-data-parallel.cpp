#include "../SPDT_general/tree.h"
#include "panel.h"
#include <assert.h>
#include <queue>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "../SPDT_general/timing.h"
#include "../SPDT_general/array.h"
#include "mpi.h"

double COMPRESS_TIME = 0.f;
double SPLIT_TIME = 0.f;
double COMMUNICATION_TIME = 0.f;
long long SIZE = 0;

int num_of_features = -1;
int num_of_classes = -1;
int max_bin_size = -1;
int max_num_leaves = -1;

#define MASTER 0

/*
 * This function compress the data into histograms.
 * Each unlabeled leaf would have a (num_feature, num_class) histograms
 * This function takes the assumption that each leaf is re-initialized (we use a batch mode)
*/

/*
 * This function split the data according to the best split feature id and value.
 * The data would be appended to the `left` if the data value is smaller than the split value
 */
void TreeNode::split(SplitPoint &best_split, TreeNode *left, TreeNode *right)
{
    this->split_ptr = best_split;
    this->entropy = best_split.entropy;
    float split_value = best_split.feature_value;
    int num_pos_lebel_left = 0;
    int num_pos_lebel_right = 0;
    for (auto &p : this->data_ptr)
    {
        float p_value = p->get_value(best_split.feature_id);
        if (best_split.decision_rule(*p))
        {
            right->data_ptr.push_back(p);
            num_pos_lebel_right = (p->label == POS_LABEL) ? num_pos_lebel_right + 1 : num_pos_lebel_right;
        }
        else
        {
            left->data_ptr.push_back(p);
            num_pos_lebel_left = (p->label == POS_LABEL) ? num_pos_lebel_left + 1 : num_pos_lebel_left;
        }
    }
    left->num_pos_label = num_pos_lebel_left;
    right->num_pos_label = num_pos_lebel_right;

    dbg_assert(left->num_pos_label >= 0);
    dbg_assert(right->num_pos_label >= 0);
    dbg_assert(left->num_pos_label + right->num_pos_label == this->num_pos_label);
}

/*
 * This function return the best split point at a given leaf node.
 * Best split is store in `split`
*/
void DecisionTree::find_best_split(TreeNode *node, SplitPoint &split)
{
    clock_t start, end;
    int taskid, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    start = clock();
    float *buf_merge = new float[2 * max_bin_size + 1];
    SplitPoint best_split = SplitPoint();
    for (int i = 0; i < num_of_features; i++)
    {
        // merge different labels
        // put the result back into (node->histogram_id, i, 0)
        float *histo_for_class_0 = get_histogram_array(node->histogram_id, i, 0);
        float *histo_for_class_1 = get_histogram_array(node->histogram_id, i, 1);
        memcpy(buf_merge, histo_for_class_0, sizeof(float) * (2 * max_bin_size + 1));
        std::vector<float> possible_splits;
        merge_array_pointers(buf_merge, histo_for_class_1);
        uniform_array(possible_splits, node->histogram_id, i, 0, buf_merge);
        dbg_assert(possible_splits.size() <= max_bin_size);
        for (auto &split_value : possible_splits)
        {
            SplitPoint t = SplitPoint(i, split_value);
            get_gain(node, t, i);
            if (best_split.gain < t.gain)
                best_split = t;
        }
    }
    split = best_split;
    end = clock();
    SPLIT_TIME += ((double)(end - start)) / CLOCKS_PER_SEC;
    delete[] buf_merge;
}

void DecisionTree::compress(vector<Data> &data)
{
    int taskid, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Request reqs;
    MPI_Status status;
    float *buffer;
    float *buffer2;
    int feature_id = 0, class_id = 0;
    // Construct the histogram. and navigate each data to its leaf.
    TreeNode *cur;
    int tasks_per_worker = (data.size() + numtasks - 1) / numtasks;
    for (int i = taskid * tasks_per_worker; i < (taskid + 1) * tasks_per_worker && i < data.size(); i++)
    {
        auto &point = data[i];
        cur = navigate(point);
        if (cur->label > -1)
            continue;

        cur->data_size++;
        for (int attr = 0; attr < num_of_features; attr++)
        {
            update_array(cur->histogram_id, attr, point.label, point.get_value(attr));
        }
    }

    Timer t = Timer();
    t.reset();
    if (taskid == MASTER)
    {
        buffer = new float[SIZE];
        buffer2 = new float[SIZE];
        int task_left = numtasks - 1;
        while (task_left > 0)
        {            
            t.reset();
            MPI_Recv(buffer, SIZE, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            COMMUNICATION_TIME += t.elapsed();
            memcpy(buffer2, buffer, SIZE * sizeof(float));
            for (int j = 0; j < num_unlabled_leaves; j++)
            { // merge the results in the master thread
                for (int k = 0; k < num_of_features; k++)
                {
                    for (int c = 0; c < num_of_classes; c++)
                    {
                        float *histo = get_histogram_array(j, k, c);
                    }
                }
            }
            task_left--;
            
        }
    }
    else
    {
        t.reset();
        MPI_Send(histogram, SIZE, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
        COMMUNICATION_TIME += t.elapsed();
    }
    t.reset();
    MPI_Bcast(histogram, SIZE, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    COMMUNICATION_TIME += t.elapsed();   

    if (taskid == MASTER) {
        delete[] buffer;
        delete[] buffer2;
    } 

}

SplitPoint::SplitPoint()
{
    feature_id = -1;
    feature_value = 0;
    entropy = 0;
}

SplitPoint::SplitPoint(int feature_id, float feature_value)
{
    this->feature_id = feature_id;
    this->feature_value = feature_value;
    this->entropy = 0;
}
/*
 * Reture True if the data is larger or equal than the split value
 */
bool SplitPoint::decision_rule(Data &data)
{
    dbg_ensures(entropy >= -EPS);
    dbg_ensures(gain >= -EPS);
    dbg_ensures(feature_id >= 0);
    return data.get_value(feature_id) >= feature_value;
}

// constructor function
TreeNode::TreeNode(int depth, int id)
{
    this->id = id;
    this->depth = depth;
    is_leaf = false;
    label = -1;
    // remove this if you want to keep the previous batch data.
    data_ptr.clear();
    histogram_id = -1;
    left_node = NULL;
    right_node = NULL;
    entropy = -1.f;
    num_pos_label = 0;
    data_size = 0;
    is_leaf = true;
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
    this->label = (this->num_pos_label >= (int)this->data_ptr.size() / 2) ? POS_LABEL : NEG_LABEL;
}

void TreeNode::printspaces()
{
    int i = 0;
    for (i = 0; i < depth * 2; i++)
    {
        printf(" ");
    }
}

void TreeNode::print()
{
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
    if (left_node != NULL)
    {
        left_node->print();
    }

    if (right_node != NULL)
    {
        right_node->print();
    }
}

void TreeNode::clear()
{
    if (left_node != NULL)
        left_node->clear();
    if (right_node != NULL)
        right_node->clear();
    data_ptr.clear();
}

DecisionTree::DecisionTree()
{
    this->max_depth = -1;
    this->min_node_size = 1;
    this->depth = 0;
    this->num_leaves = 0;
    this->cur_depth = 0;
    this->root = NULL;
    this->min_gain = 1e-3;
    this->num_nodes = 0;
    this->num_unlabled_leaves = 0;
}

DecisionTree::~DecisionTree()
{
    root->clear();
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

    if (max_num_leaves != -1 && this->num_leaves >= max_num_leaves)
    {
        dbg_printf("Node [%d] terminated: max_num_leaves\n", node->id);
        return true;
    }

    if (!node->num_pos_label || node->num_pos_label == (int)node->data_ptr.size())
    {
        dbg_assert(node->entropy < EPS);
        dbg_printf("Node [%d] terminated: all samples belong to same class\n", node->id);
        return true;
    }
    dbg_printf("[%d] num_data=%d, num_pos=%d\n", node->id, node->data_ptr.size(), node->num_pos_label);
    return false;
}

void DecisionTree::initialize(Dataset &train_data, const int batch_size)
{
    this->datasetPointer = &train_data;
    root = new TreeNode(0, this->num_nodes++);
    if (histogram != NULL)
    {
        delete[] histogram;
    }
    SIZE = (long long)max_num_leaves * num_of_features * num_of_classes * ((max_bin_size + 1) * 2 + 1);
    printf("Init Root Node [%.4f] MB\n", SIZE * sizeof(float) / 1024.f / 1024.f);

    histogram = new float[SIZE];
    memset(histogram, 0, SIZE * sizeof(float));
    printf("Init success\n");
}

void DecisionTree::train(Dataset &train_data, const int batch_size)
{
    int hasNext = TRUE;
    initialize(train_data, batch_size);
    while (TRUE)
    {
        hasNext = train_data.streaming_read_data(batch_size);
        dbg_printf("Train size (%d, %d, %d)\n", train_data.num_of_data,
                   num_of_features, num_of_classes);
        train_on_batch(train_data);
        if (!hasNext)
            break;
    }

    train_data.close_read_data();
    return;
}

double DecisionTree::test(Dataset &test_data)
{

    int i = 0;
    int correct_num = 0;
    test_data.streaming_read_data(test_data.num_of_data);

    for (i = 0; i < test_data.num_of_data; i++)
    {
        assert(navigate(test_data.dataset[i])->label != -1);
        if (navigate(test_data.dataset[i])->label == test_data.dataset[i].label)
        {
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
void get_gain(TreeNode *node, SplitPoint &split, int feature_id)
{
    int taskid, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    int total_sum = node->data_ptr.size();
    dbg_ensures(total_sum > 0);
    double sum_class_0 = get_total_array(node->histogram_id, feature_id, NEG_LABEL);
    double sum_class_1 = get_total_array(node->histogram_id, feature_id, POS_LABEL);
    dbg_assert((sum_class_1 - node->num_pos_label) < EPS);
    double left_sum_class_0 = sum_array(node->histogram_id, feature_id, NEG_LABEL, split.feature_value);
    double right_sum_class_0 = sum_class_0 - left_sum_class_0;
    double left_sum_class_1 = sum_array(node->histogram_id, feature_id, POS_LABEL, split.feature_value);
    double right_sum_class_1 = sum_class_1 - left_sum_class_1;
    double left_sum = left_sum_class_0 + left_sum_class_1;
    double right_sum = right_sum_class_0 + right_sum_class_1;
    double px = (left_sum_class_0 + left_sum_class_1) / (1.0 * total_sum);   // p(x<a)
    double py_x0 = (left_sum <= EPS) ? 0.f : left_sum_class_0 / left_sum;    // p(y=0|x < a)
    double py_x1 = (right_sum <= EPS) ? 0.f : right_sum_class_0 / right_sum; // p(y=0|x >= a)
    dbg_ensures(py_x0 >= -EPS && py_x0 <= 1 + EPS);
    dbg_ensures(py_x1 >= -EPS && py_x1 <= 1 + EPS);
    dbg_ensures(px >= -EPS && px <= 1 + EPS);
    double entropy_left = ((1 - py_x0) < EPS || py_x0 < EPS) ? 0 : -py_x0 * log2(py_x0) - (1 - py_x0) * log2(1 - py_x0);
    double entropy_right = ((1 - py_x1) < EPS || py_x1 < EPS) ? 0 : -py_x1 * log2(py_x1) - (1 - py_x1) * log2(1 - py_x1);
    double H_YX = px * entropy_left + (1 - px) * entropy_right;
    double px_prior = sum_class_0 / (sum_class_0 + sum_class_1);
    dbg_ensures(px_prior > 0 && px_prior < 1);
    split.entropy = ((1 - px_prior) < EPS || px_prior < EPS) ? 0 : -px_prior * log2(px_prior) - (1 - px_prior) * log2(1 - px_prior);
    split.gain = split.entropy - H_YX;
    // if (taskid == MASTER)
    //     printf("%.7f = %.7f - %.7f\n", split.gain, split.entropy, H_YX);
    dbg_ensures(split.gain >= -EPS);
}

void DecisionTree::self_check()
{
    queue<TreeNode *> q;
    q.push(root);
    int count_leaf = 0;
    int count_nodes = 0;
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

    num_unlabled_leaves = c;
    memset(histogram, 0, SIZE * sizeof(float));
}

/*
 * Serial version of training.
*/
void DecisionTree::train_on_batch(Dataset &train_data)
{
    int taskid, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    for (auto &data : train_data.dataset)
        root->data_ptr.push_back(&data);

    float pos_rate = ((float)train_data.num_pos_label) / train_data.num_of_data;
    dbg_assert(pos_rate > 0 && pos_rate < 1);
    root->num_pos_label = train_data.num_pos_label;
    root->entropy = -pos_rate * log2(pos_rate) - (1 - pos_rate) * log2((1 - pos_rate));
    batch_initialize(root); // Reinitialize every leaf in T as unlabeled.
    vector<TreeNode *> unlabeled_leaf = __get_unlabeled(root);
    dbg_assert(unlabeled_leaf.size() <= max_num_leaves);
    while (!unlabeled_leaf.empty())
    {
        // each while loop would add a new level node.
        this->cur_depth++;
        if (taskid == MASTER)
            printf("depth [%d] finished\n", this->cur_depth);
        vector<TreeNode *> unlabeled_leaf_new;
        if (unlabeled_leaf.size() > max_num_leaves)
        {
            for (int i = 0; i < unlabeled_leaf.size(); i++)
            {
                unlabeled_leaf[i]->set_label();
                this->num_leaves++;
            }
            break;
        }
        init_histogram(unlabeled_leaf);
        Timer t;
        t.reset();
        compress(train_data.dataset);
        COMPRESS_TIME += t.elapsed();
        for (auto &cur_leaf : unlabeled_leaf)
        {
            if (is_terminated(cur_leaf))
            {
                cur_leaf->set_label();
                this->num_leaves++;
            }
            else
            {
                SplitPoint best_split = SplitPoint();
                Timer t;
                t.reset();
                find_best_split(cur_leaf, best_split);
                SPLIT_TIME += t.elapsed();
                dbg_ensures(best_split.gain >= -EPS);
                if (best_split.gain <= min_gain)
                {
                    dbg_printf("Node terminated: gain=%.4f <= %.4f\n", best_split.gain, min_gain);
                    cur_leaf->set_label();
                    this->num_leaves++;
                    continue;
                }
                cur_leaf->left_node = new TreeNode(this->cur_depth, this->num_nodes++);
                cur_leaf->right_node = new TreeNode(this->cur_depth, this->num_nodes++);
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
