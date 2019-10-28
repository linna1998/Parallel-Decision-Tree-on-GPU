# Parallel Decision Tree on GPU

ke Ding

Anxiang Zhang

## URL
https://linna1998.github.io/Parallel-Decision-Tree-on-GPU/

## Summary
We are going to implement decision tree foucsed on large data databases with a basic naïve version, and optimize it into several CPU parallel and GPU parallel versions mainly using data parallel approach. Finally, we are going to evaluate the speedup and memory performance of all these versions.

## Background

### What is decision tree?

#### Different sequential implementation
Sort

Histogram

Stream: Reading data in a stream, and update?

### Different parallel approaches of decision tree?

#### CPU Parallel:
1. Data parallel
We are mainly foucsed on the data parallel implementation. It is used for large datasets.

2. Voting parallel
Voting parallel is a special kind of data parallel implementation. There is a paper "A Communication-Efficient Parallel Algorithm for Decision Tree" talking about the voting parallel decision tree.

3. Attribute parallel

## Challenge

### Dependenties of workload

### Memory access characteristics

### Limited memory of GPU

### Load balance

### Bandwidth problem

### How to achieve high speedup

## Resources

### Codes
1. XGBoost: https://github.com/dmlc/xgboost
2. LightGBM: https://github.com/microsoft/LightGBM
3. thunderGBM: https://github.com/Xtra-Computing/thundergbm/tree/master/src/thundergbm
4. https://lightgbm.readthedocs.io/en/latest/Features.html#optimization-in-parallel-learning

### Papers
1. thunderGBM: Exploiting GPUs for Efficient Gradient Boosting Decision Tree Training
2. XGBoost: Scalable GPU Accelerated Learning
3. Accelerating the XGBoost algorithm using GPU computing
4. Exploiting GPUs for Efficient Gradient Boosting Decision Tree Training
5. Communication-Efficient Parallel Algorithm for Decision Tree

## Goals and deliverables

1. Implement a basic sequential code for decision tree algorithm
2. Implement data parallel code on CPU for decision tree algorithm based on the sequential code
3. Implement basic parallel code on GPU based on the previous results
4. Improve our GPU parallel version with some new algorithms

### Hope to achieve:
1. More parallel versions on GPU

### Demo and poster
1. Evaluate sequential, data parallel CPU, basic GPU, advanced GPU versions
2. Compare the memory usage of different versions
3. Compare the feature and data samples, find the best suiting algorithm (?)
4. In what situations, CUDA version can achieve high speedup (?)

## Platform:
1. Platform: parallel system: GHC, latedays
2. Language: mainly C/C++, CUDA

## Schedule:
10.30 Finish proposal and website

11.6 Finish the sequential version. Find the dataset for evaluation.

11.13 Finish the CPU parallel version

11.20 Start basic GPU parallel version

11.27 Improved GPU parallel 

12.4 Improved GPU parallel

12.11 Evaluate and find the trade-offs

12.18 Finish report
