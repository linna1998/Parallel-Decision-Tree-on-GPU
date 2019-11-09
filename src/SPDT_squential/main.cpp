#include "tree.h"

int main() {

    DecisionTree decisionTree(256, 8, 128);

    Dataset trainDataset(2, 1605, 123);
    Dataset testDataset(2, 30956, 123);	
    
    trainDataset.open_read_data("./data/a1a.train.txt");        
    decisionTree.train(trainDataset, 1605);

	testDataset.open_read_data("./data/a1a.test.txt");	
    decisionTree.test(testDataset);

    // DecisionTree decisionTree(256, 8, 128);
    // Dataset trainDataset(2, 3, 30);
    // trainDataset.open_read_data("./data/test.txt"); 
    // decisionTree.train(trainDataset);           

    return 0;

}