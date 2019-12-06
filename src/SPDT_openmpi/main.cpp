
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "mpi.h"
#include "../SPDT_general/tree.h"
#include "../SPDT_general/timing.h"


vector<string> names = {"a1a", "ijcnn1", "avazu-app", "rcv1", "covtype", "generated"};

vector<int> trainSize = {1605, 49990, 40428967, 20242, 581012, 40000};
vector<int> testSize = {30956, 91701, 4577464, 677399, -1, 10000};
vector<int> featureNum = {123, 22, 1000000, 47236, 54, 300};

string help_msg = "-l: max_num_leaf.\n-d: max_depth.\n-n: number of"\
                  "threads.\n-b: max_bin_size\n-l: max_num_leaf\n-e: min_node_size\n";

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int taskid, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);  
    int index = 0;
    double cpu_time_used_train;
    double cpu_time_used_test;
    int c;
    int num_of_thread = -1;    
    int min_node_size = -1;
    int max_depth = -1;

    while((c = getopt(argc, argv, "i:")) != -1 ){
        switch (c)
        {
        case 'i':
            index = (int)std::atoi(optarg);            
            break;        
        default:
            break;
        }
    }
   
    num_of_thread = (num_of_thread == -1)? 8 : num_of_thread;
    max_num_leaves = (max_num_leaves == -1) ? 64 : max_num_leaves;
    max_depth = (max_depth == -1) ? 9 : max_depth;
    min_node_size = (min_node_size == -1) ? 32 : min_node_size;
    // the global max_bin_size
    max_bin_size = (max_bin_size == -1) ? 256 : max_bin_size;
    num_of_features = featureNum[index];
    num_of_classes = 2;
    if (taskid == 0)
        printf("[%d] max_num_leaf=%d, max_depth=%d, min_node_size=%d, max_bin_size=%d\n", taskid,
                max_num_leaves, max_depth, min_node_size, max_bin_size);
            
    string trainName = "./data/" + names[index] + ".train.txt";
    cout << "dataset name: " << names[index] << endl;
    DecisionTree decisionTree(max_depth, min_node_size);
    Dataset trainDataset(trainSize[index]);
    trainDataset.open_read_data(trainName);
    Timer t = Timer();
    t.reset();
    decisionTree.train(trainDataset, trainSize[index]);
    cpu_time_used_train = t.elapsed();
  
    // test
    string testName = "./data/" + names[index] + ".test.txt";
    Dataset testDataset(testSize[index]);
    testDataset.open_read_data(testName);	
    if (taskid == 0){
        printf("[%d] Train Time: %f\n", taskid, cpu_time_used_train);
        printf("[%d] COMPRESS TIME: %f\n[%d] SPLIT TIME: %f\n", taskid, COMPRESS_TIME, SPLIT_TIME); 
        t.reset();
        printf("[%d] COMMUNICATION time: %f\n", taskid, COMMUNICATION_TIME);
        printf("[%d] Train Time: %f\n", taskid, cpu_time_used_train);
        printf("[%d] Training Correct Rate: %f\n", taskid, decisionTree.test(trainDataset)); 
        printf("[%d] Testing Correct Rate: %f\n", taskid, decisionTree.test(testDataset)); 
        cpu_time_used_test = t.elapsed();
        printf("[%d] Test Time: %f\n", taskid, cpu_time_used_test);
    }
    MPI_Finalize();
    return 0;  
}