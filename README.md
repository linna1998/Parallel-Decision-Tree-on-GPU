# Parallel Decision Tree on GPU

ke Ding

Anxiang Zhang

## URL
https://linna1998.github.io/Parallel-Decision-Tree-on-GPU/

## Summary
We are going to implement a communication-efficient parallel version of gradient boosting decision tree (gbdt). At first, we are going to implement a sequential version and use CPU to parallelize it. This model is served as a baseline. Then, GPU would be applied to implement a faster version. Finally, we are going to evaluate the speedup, memory performance and bottlenecks of all these versions. The main challenge resides in the strategy we used to parallelize. There are multiple ways to parallel the decision tree building process, including data-parallel, feature-parallel and etc. Some strategies are easy to implement but suffer from loadbalance issue. Some strategies have high communication cost. In our study, we focus on reducing the communication cost during the tree construction process. Also, memory constraint would be taken into consideration since GPU has limited memory to use. 

## Background

### What is decision tree?

Decision tree ensemble algorithms are increasingly adopted as a crucial solution to modern machine learning applications such as classification. The major computation cost of training decision tree ensemble comes from training a single decision tree, and the key challenge of decision tree building process is the high cost in finding the best split for each leaf, which requires scanning through all the training data in the current sub-tree.[3]



#### Different sequential implementation

### Different parallel approaches of decision tree?

#### CPU Parallel:
There are multiple ways to parallel the decision tree building process. More specifically, there are mainly three methods

1. Task Parallel
Classification decision tree construction algorithms have natural concurrency, as once a node is generated, all of its children in the classification tree can be generated concurrently. But this method suffers from *major load imbalance* issue and *high communication cost*. So we would not use this kind of parallel strategy.[1]

2. Feature Parallel (Vertical Parallel)
When splitting the node, each feature data would be processed independently to find the best split point. Then each process would communicate to get the optimal way of splitting the data. Afterwards, one processor would partition the data based on the splitting point and then broadcast the data to other processors. This result addresses the load balance issue, but suffer from high communication cost when the data set is large. Which means this staregy does not suit for GPU parallelism. [2]

3. Data Parallel (Horizontal Parallel)
Data parallel partitions the dataset so that each processor could handle only a portion of dataset. Each process builds the histogram of all the features and then merge the histogram by communication. So method reduces communication cost very much and also support streamming data. Therefore, this method is suitable for GPU due to the fact that GPU has limited share memory to use. [2]

#### GPU Parallel:
Many existing publication focus on building communication-efficient and scalabel distributed decision tree, while there are limited exploration in GPU accelerations. On the one hand, GPU’s strict memory constraint this method does not scale to large dataset. On the other hand, GPU used SIMD instructions within a warp and thus instruction divergence is a big problem in GPU programming. [3]

## Challenge
- How to parallellize so that the model could be capable of handling streaming data. (Data-parallel is a possible way)

- How to parallellize in GPU given that GPU has a limited memory size. (<= 64 KB while the dataset could be million)

- How to maintain a good workload balance.

- How to achieve low communication cost as data set becomes larger (still scale as data size becomes larger).

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
2. Implement data parallel code on CPU for decision tree algorithm based on the sequential code. 
3. Implement basic parallel code on GPU based on the previous results
4. Improve our GPU parallel version with some new algorithms

### Hope to achieve:
1. High scalability on GPU
2. More speedup on GPU.

### Demo and poster
1. Evaluate sequential, data parallel CPU, basic GPU, advanced GPU versions.
2. Compare the memory usage of different versions.
3. Compare the feature and data samples, find the limitations of the parallel scheme.
4. Give in what situations, CUDA version can achieve high speedup.

## Platform:
1. Platform: parallel system: GHC, latedays
2. Language: mainly C/C++, CUDA

## Schedule:
10.30 Finish proposal and website.

11.6 Finish the sequential version. Find the dataset for evaluation.

11.13 Finish the CPU data parallel version.

11.22 Project Milestone. Finish basic GPU parallel version.

11.27 Improve GPU parallel version.

12.4 Evaluate and find the trade-offs between different implementations.

12.10 Finish final report and poster.


## References

[1]: Srivastava, Anurag, et al. "Parallel formulations of decision-tree classification algorithms." High Performance Data Mining. Springer, Boston, MA, 1999. 237-261.

[2]: Ben-Haim, Yael, and Elad Tom-Tov. "A streaming parallel decision tree algorithm." Journal of Machine Learning Research 11.Feb (2010): 849-872.

[3]: Zhang, Huan, Si Si, and Cho-Jui Hsieh. "GPU-acceleration for Large-scale Tree Boosting." arXiv preprint arXiv:1706.08359 (2017).

[4]: Jin, Ruoming, and Gagan Agrawal. "Communication and memory efficient parallel decision tree construction." Proceedings of the 2003 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2003.