#define DEBUG 1
#include "tree.h"


int main() {

    DecisionTree decisionTree(256, 8, 128);

    Dataset trainDataset(3, 391, 20);

    trainDataset.open_read_data("./data/svmguide2.txt");    
    
    decisionTree.train(trainDataset, 391);    

    return 0;

}