#include "../SPDT_general/tree.h"
#include "panel.h"
#include <assert.h>
#include <queue>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <time.h>
#include "../SPDT_general/array.h"
#include "../SPDT_general/timing.h"

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
    clock_t start, end;
    start = clock();       
    float* buf_merge = new float[2 * max_bin_size + 1];
    SplitPoint best_split = SplitPoint();
    for (int i = 0; i < num_of_features; i++)
    {
        // merge different labels
        // put the result back into (node->histogram_id, i, 0)
        float* histo_for_class_0 = get_histogram_array(node->histogram_id, i, 0);
        float* histo_for_class_1 = get_histogram_array(node->histogram_id, i, 1);
        memcpy(buf_merge, histo_for_class_0, sizeof(float) * (2 * max_bin_size + 1));
        std::vector<float> possible_splits;
        merge_array_pointers(buf_merge, histo_for_class_1);
        uniform_array(possible_splits, node->histogram_id, i, 0, buf_merge);
        dbg_assert(possible_splits.size() <= max_bin_size);
        for (auto& split_value: possible_splits)
        {
            SplitPoint t = SplitPoint(i, split_value);
            get_gain(node, t, i);
            if (best_split.gain < t.gain)
                best_split = t;
        }
    }
    split = best_split;
    end = clock();   
    // SPLIT_TIME += ((double) (end - start)) / CLOCKS_PER_SEC; 
    delete[] buf_merge;
}

/*
 * This function compress the data into histograms.
 * Each unlabeled leaf would have a (num_feature, num_class) histograms
 * This function takes the assumption that each leaf is re-initialized (we use a batch mode)
*/
void DecisionTree::compress(vector<Data> &data)
{
    clock_t start, end;
    start = clock();  
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

    end = clock();   
    COMPRESS_TIME += ((float) (end - start)) / CLOCKS_PER_SEC; 
}
