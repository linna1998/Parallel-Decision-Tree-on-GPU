#include "tree.h"
#include <time.h>

vector<string> names = {"a1a", "ijcnn1"};

int trainSize[2] = {1605, 49990};
int testSize[2] = {30956, 91701};
int featureNum[2] = {123, 22};


int main() {

    int index = 1;
    clock_t start, end;
    double cpu_time_used;

    string trainName = "./data/" + names[index] + ".train.txt";
    string testName = "./data/" + names[index] + ".test.txt";

    DecisionTree decisionTree(64, 9, 32);

    Dataset trainDataset(2, trainSize[index], featureNum[index]);
    Dataset testDataset(2, testSize[index], featureNum[index]);
	

    trainDataset.open_read_data(trainName);
    testDataset.open_read_data(testName);	
    
    start = clock();     
    decisionTree.train(trainDataset, trainSize[index]);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("train time: %f\n", cpu_time_used);

    start = clock();   
    printf("correct rate: %f\n", decisionTree.test(testDataset));     
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("test time: %f\n", cpu_time_used);

    return 0;

}