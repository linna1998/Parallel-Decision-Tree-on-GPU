# Parallel Decision Tree on GPU

ke Ding

Anxiang Zhang

## URL
https://linna1998.github.io/Parallel-Decision-Tree-on-GPU/

## Summary
We are going to implement decision tree with a basic naïve version, and optimize it into several CPU parallel and GPU parallel versions. Finally, we are going to evaluate the speedup and memory performance of all these versions.

## Background

### What is decision tree?

#### Different sequential implementation
Sort
Histogram

### Different parallel approaches of decision tree?

#### CPU Parallel:
1. Data parallel

2. Voting parallel

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


### Papers
1. thunderGBM: Exploiting GPUs for Efficient Gradient Boosting Decision Tree Training
2. XGBoost: Scalable GPU Accelerated Learning
3. Accelerating the XGBoost algorithm using GPU computing
4. Exploiting GPUs for Efficient Gradient Boosting Decision Tree Training

## Goals and deliverables

1. Implement a basic sequential code for decision tree algorithm
2. Implement parallel codes on CPU for decision tree algorithm based on the sequential code
3. Implement parallel codes on GPU based on the previous results

### Hope to achieve:
1. More parallel versions on GPU

### Demo and poster
1. evaluation of sequential, CPU, CUDA speedup
2. compare the memory
3. Compare the feature and data samples, find the best suiting algorithm
4. In what situations, CUDA version can achieve high speedup

## Platform:
1. Platform: parallel system: GHC, latedays
2. Language: mainly C/C++

## Schedule:
10.30 Finish proposal and website
11.6 Finish the sequential version
11.13 Finish the CPU parallel version
11.20 Start basic GPU parallel version
11.27 Improved GPU parallel - improve sort
12.4 Improved GPU parallel – change the sequential version (LightGBM)
12.11 Evaluate and find the trade-offs
12.18 Finish report
