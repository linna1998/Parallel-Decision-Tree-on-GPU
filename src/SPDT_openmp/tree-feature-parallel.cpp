#include "../SPDT_general/tree.h"
#include "panel.h"
#include <assert.h>
#include <queue>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "../SPDT_general/array.h"
#include "../SPDT_general/timing.h"

void prefix_printf(const char* format, ...){
    va_list args;
    printf("FEATURE ");
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
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
    int num_pos_lebel_left=0;
    int num_pos_lebel_right=0;
    for (auto &p : this->data_ptr)
    {
        float p_value = p->get_value(best_split.feature_id);
        if (best_split.decision_rule(*p)){
            right->data_ptr.push_back(p);
            num_pos_lebel_right = (p->label == POS_LABEL) ? num_pos_lebel_right+1 : num_pos_lebel_right;
        }
        else{
            left->data_ptr.push_back(p);
            num_pos_lebel_left = (p->label == POS_LABEL) ? num_pos_lebel_left+1 : num_pos_lebel_left;
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
    float** buf_merge = (float**) malloc(NUM_OF_THREAD * sizeof(float*));
    for (int k=0; k<NUM_OF_THREAD; k++)
        buf_merge[k] = (float*) malloc(sizeof(float) * (2 * max_bin_size + 1));

    SplitPoint* results = new SplitPoint[NUM_OF_THREAD];
    for (int j = 0; j<NUM_OF_THREAD; j++)
        results[j] = SplitPoint();

    int tot = 0; // used to count the number of results
    #pragma barrier
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_of_features; i++)
    {
        int tid = omp_get_thread_num();
        // merge different labels
        float* histo_for_class_0 = get_histogram_array(node->histogram_id, i, 0);
        float* histo_for_class_1 = get_histogram_array(node->histogram_id, i, 1);
        memcpy(buf_merge[tid], histo_for_class_0, sizeof(float) * (2 * max_bin_size + 1));
        std::vector<float> possible_splits;
        merge_array_pointers(buf_merge[tid], histo_for_class_1);
        uniform_array(possible_splits, node->histogram_id, i, 0, buf_merge[tid]);
        for (int j=0; j<possible_splits.size(); j++)
        {
            SplitPoint t = SplitPoint(i, possible_splits[j]);
            get_gain(node, t, i);
            if (t.gain > results[tid].gain)
                results[tid] = t;
        }
    }

    SplitPoint best_split;
    best_split = results[0];
    for(int ii=0; ii < NUM_OF_THREAD; ii++){
        if (results[ii].gain > best_split.gain)
            best_split = results[ii];
    }

    split.feature_id = best_split.feature_id;
    split.feature_value = best_split.feature_value;
    split.gain = best_split.gain;
    delete[] results;
    for (int k=0; k<NUM_OF_THREAD; k++)
        free(buf_merge[k]);
    free(buf_merge);
}


/*
 * This function compress the data into histograms.
 * Each unlabeled leaf would have a (num_feature, num_class) histograms
 * This function takes the assumption that each leaf is re-initialized (we use a batch mode)
*/
void DecisionTree::compress(vector<Data> &data)
{
    int feature_id = 0, class_id = 0;
    // Construct the histogram. and navigate each data to its leaf.
    TreeNode* cur;
    int c=0;
    for(auto& point : data){
        cur = navigate(point);
        if (cur->label > -1)
            continue;
        cur->data_size ++;
        for (int attr = 0; attr < num_of_features; attr++)
            update_array(cur->histogram_id, attr, point.label, point.get_value(attr));              
    }
}


/*
 * Serial version of training.
*/
void DecisionTree::train_on_batch(Dataset &train_data)
{
    
    for(auto& data: train_data.dataset)
        root->data_ptr.push_back(&data);

    float pos_rate = ((float) train_data.num_pos_label) / train_data.num_of_data;
    dbg_assert(pos_rate > 0 && pos_rate < 1);
    root->num_pos_label = train_data.num_pos_label;
    root->entropy = - pos_rate * log2(pos_rate) - (1-pos_rate) * log2((1-pos_rate));
    batch_initialize(root); // Reinitialize every leaf in T as unlabeled.
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
        Timer t1 = Timer();
        t1.reset();
        compress(train_data.dataset); 
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
                SplitPoint best_split = SplitPoint();
                Timer t2 = Timer();
                t2.reset();
                find_best_split(cur_leaf, best_split);
                SPLIT_TIME += t2.elapsed();
                dbg_ensures(best_split.gain >= -EPS);
                if (best_split.gain <= min_gain){
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

