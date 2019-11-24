#include "tree_CUDA.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

vector<string> names = {"a1a", "ijcnn1", "avazu-app", "rcv1", "covtype", "generated"};

vector<int> trainSize = {1605, 49990, 40428967, 20242, 581012, 40000};
vector<int> testSize = {30956, 91701, 4577464, 677399, -1, 10000};
vector<int> featureNum = {123, 22, 1000000, 47236, 54, 300};


string help_msg = "-l: max_num_leaf.\n-d: max_depth.\n-n: number of"\
                  "threads.\n-b: max_bin_size\n-l: max_num_leaf\n-e: min_node_size\n";
                  
int main(int argc, char **argv) {

    int index = 5;
    clock_t start, end;
    double cpu_time_used;
    int c;
    int num_of_thread = -1;    
    int min_node_size = -1;
    int max_num_leaf = -1;
    int max_depth = -1;
    // while((c = getopt(argc, argv, "b:n:l:d:eh")) != -1){
    //     switch (c)
    //     {
    //     case 'n':
    //         num_of_thread = (int)std::atoi(optarg);
    //         cout << "using " << num_of_thread << " threads" << endl;
    //         break;

    //     case 'b':
    //         max_bin_size = (int)std::atoi(optarg);
    //         break;

    //     case 'l':
    //         max_num_leaf = (int)std::atoi(optarg);
    //         break;

    //     case 'd':
    //         max_depth = (int)std::atoi(optarg);
    //         break;

    //     case 'e':
    //         cout << "e" << endl;
    //         min_node_size = (int)std::atoi(optarg);
    //         cout << "e" << endl;

    //         break;
    //     case 'h':
    //         cout << help_msg << endl;
    //         exit(0);
    //     default:
    //         break;
    //     }
    // }
    num_of_thread = (num_of_thread == -1)? 8 : num_of_thread;
    max_num_leaf = (max_num_leaf == -1) ? 64 : max_num_leaf;
    max_depth = (max_depth == -1) ? 9 : max_depth;
    min_node_size = (min_node_size == -1) ? 32 : min_node_size;
    // the global max_bin_size
    max_bin_size = (max_bin_size == -1) ? 32 : max_bin_size;
    num_of_features = featureNum[index];
    num_of_classes = 2;

    printf("max_num_leaf=%d, max_depth=%d, min_node_size=%d, max_bin_size=%d\n", 
            max_num_leaf, max_depth, min_node_size, max_bin_size);
            
    string trainName = "./data/" + names[index] + ".train.txt";
    DecisionTree decisionTree(max_num_leaf, max_depth, min_node_size);
    Dataset trainDataset(trainSize[index]);
    trainDataset.open_read_data(trainName);
    start = clock();     
    decisionTree.train(trainDataset, trainSize[index]);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("train time: %f\n", cpu_time_used);
    
    // test
    string testName = "./data/" + names[index] + ".test.txt";
    Dataset testDataset(testSize[index]);
    testDataset.open_read_data(testName);	

    start = clock();   
    printf("correct rate: %f\n", decisionTree.test(testDataset));     
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("test time: %f\n", cpu_time_used);

    return 0;

}