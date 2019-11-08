#include "tree.h"

int main() {

    DecisionTree decisionTree(256, 8, 128);

    Dataset trainDataset(2, 1605, 123);
    Dataset testDataset(2, 30956, 123);

	bool hasNext = true;

    trainDataset.open_read_data("./data/a1a.train.txt");
    decisionTree.train(trainDataset);

	testDataset.open_read_data("./data/a1a.test.txt");	
    decisionTree.test(testDataset);

    return 0;

}