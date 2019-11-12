#define DEBUG 1
#include "tree.h"

vector<string> names = {"a1a", "ijcnn1"};

int trainSize[2] = {1605, 49990};
int testSize[2] = {30956, 91701};
int featureNum[2] = {123, 22};


int main() {

    int index = 1;

    string trainName = "./data/" + names[index] + ".train.txt";
    string testName = "./data/" + names[index] + ".test.txt";

    DecisionTree decisionTree;

    Dataset trainDataset(2, trainSize[index], featureNum[index]);
    Dataset testDataset(2, testSize[index], featureNum[index]);
	

    trainDataset.open_read_data(trainName);
    testDataset.open_read_data(testName);	
    
    decisionTree.train(trainDataset, trainSize[index]);
    std::cout << "correct rate: " << decisionTree.test(testDataset) << std::endl;     

    return 0;

}