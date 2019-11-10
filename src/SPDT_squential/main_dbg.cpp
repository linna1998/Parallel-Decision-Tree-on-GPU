#define DEBUG 1
#include "tree.h"


int main() {

    // DecisionTree decisionTree(256, 8, 128);

    // Dataset trainDataset(3, 391, 20);

    // trainDataset.open_read_data("./data/svmguide2.txt");    
    
    // decisionTree.train(trainDataset, 391);   


    DecisionTree decisionTree(256, 8, 128);

    Dataset trainDataset(2, 1605, 123);
    Dataset testDataset(2, 30956, 123);
	

    trainDataset.open_read_data("./data/a1a.train.txt");
    testDataset.open_read_data("./data/a1a.test.txt");	
    
    decisionTree.train(trainDataset, 1605);
    decisionTree.test(testDataset);
    

    return 0;

}