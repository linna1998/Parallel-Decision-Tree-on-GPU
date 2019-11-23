#include "../SPDT_general/tree.h"
#include "../SPDT_general/array.h"
#include "mpi.h"

#define MASTER 0

extern double* histogram;
/*
 * This function compress the data into histograms.
 * Each unlabeled leaf would have a (num_feature, num_class) histograms
 * This function takes the assumption that each leaf is re-initialized (we use a batch mode)
*/
void DecisionTree::compress(vector<Data> &data)
{
    clock_t start, end;
    start = clock();
    int taskid, numtasks;
    double* buffer;
    double* buffer2;

    long long msg_size = max_num_leaves * datasetPointer->num_of_features * datasetPointer->num_of_classes * max_bin_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);  
    int feature_id = 0, class_id = 0;
    // Construct the histogram. and navigate each data to its leaf.
    TreeNode* cur;
    int c=0;
    for(int i=taskid; i<data.size(); i+=numtasks){
        auto& point = data[i];
        cur = navigate(point);
        if (cur->label > -1)
            continue;
        cur->data_size ++;
        for (int attr = 0; attr < this->datasetPointer->num_of_features; attr++)
        {                            
            update_array(cur->histogram_id, attr, point.label, point.get_value(attr));                  
        }
    }

    if (taskid != MASTER){
        MPI_Send(histogram, number, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
    }
    else{
        buffer = new double[number];
        buffer2 = new double[number];
        MPI_Recv(buffer, number, MPI_DOUBLE, MPI_ANYSOURCE, 0, MPI_COMM_WORLD);
        memcpy(buffer2, buffer, sizeof(double) * number);
    }

    end = clock();   
    COMPRESS_TIME += ((double) (end - start)) / CLOCKS_PER_SEC; 
}