
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "mpi.h"
#include "../SPDT_general/tree.h"
#include "../SPDT_general/timing.h"


vector<string> names = {"a1a", "ijcnn1", "avazu-app", "rcv1", "covtype", "generated",
                        "big_size_small_feature", "middle_size_small_feature", "small_size_small_feature", "tiny_size_small_feature",
                        "middle_size_middle_feature", "middle_size_big_feature", // testing feature parallel
                        };

vector<int> trainSize = {1605, 49990, 40428967, 20242, 581012, 40000,
                        990000, 99000, 9900, 990,
                        99000, 99000};
vector<int> testSize = {30956, 91701, 4577464, 677399, -1, 10000,
                        110000, 11000, 1100, 110,
                        11000, 11000};
vector<int> featureNum = {123, 22, 1000000, 47236, 54, 300,
                          50, 50, 50, 50,
                          200, 1000};
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
    int thread_num = 1;
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
    max_bin_size = (max_bin_size == -1) ? 64 : max_bin_size;
    num_of_features = featureNum[index];
    num_of_classes = 2;
    string trainName = "./data/" + names[index] + ".train.txt";
    prefix_printf("DATASET: %s\n", trainName.c_str());
    prefix_printf("SIZE: (%d, %d) \n", trainSize[index], num_of_features);
    prefix_printf("NUM_WORKERS: %d\n", numtasks);
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
    prefix_printf("COMPRESS_TIME: %f\n", COMPRESS_TIME); 
    prefix_printf("NET_COMPRESS_TIME: %f\n", COMPRESS_TIME - COMPRESS_COMMUNICATION_TIME); 
    prefix_printf("SPLIT_TIME: %f\n", SPLIT_TIME); 
    prefix_printf("NET_SPLIT_TIME: %f\n", SPLIT_TIME - SPLIT_COMMUNICATION_TIME); 
    prefix_printf("COMPRESS_COMMUNICATION_Time: %f\n", COMPRESS_COMMUNICATION_TIME);
    prefix_printf("SPLIT_COMMUNICATION_Time: %f\n", SPLIT_COMMUNICATION_TIME);
    prefix_printf("Train_Time: %f\n", cpu_time_used_train);
    prefix_printf("Training_Correct_Rate: %f\n", decisionTree.test(trainDataset));
    t.reset();
    prefix_printf("Testing_Correct_Rate: %f\n", decisionTree.test(testDataset)); 
    cpu_time_used_test = t.elapsed();
    prefix_printf("Test_Time: %f\n", cpu_time_used_test);
    MPI_Finalize();
    return 0;  
}