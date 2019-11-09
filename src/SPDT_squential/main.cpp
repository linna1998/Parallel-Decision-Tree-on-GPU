#include "tree.h"

int main() {

    DecisionTree decisionTree(256, 8, 128);

    Dataset trainDataset(2, 1605, 123);
    Dataset testDataset(2, 30956, 123);

	bool hasNext = true;

    trainDataset.open_read_data("./data/a1a.train.txt");
    testDataset.open_read_data("./data/a1a.test.txt");	

    dbg_printf("Train size (%d, %d, %d)\n", trainDataset.num_of_data, 
                trainDataset.num_of_features, trainDataset.num_of_classes);
    dbg_printf("Test size (%d, %d, %d)\n", trainDataset.num_of_data, 
                trainDataset.num_of_features, trainDataset.num_of_classes);
    
    decisionTree.train_on_batch(trainDataset);


    decisionTree.test(testDataset);

    return 0;

}