# Parallel Streaming Decision Tree - Final Report
<center> ke Ding(keding@andrew.cmu.edu) || Anxiang Zhang (anxiangz@andrew.cmu.edu) </center>
## Summary
We implemented the sequential version, the OpenMP version, the OpenMPI version and the CUDA version for decision tree with histogram and compared the performance of four implementations.

## Takeaways


## Backgrounds

### Decisiton Tree
Decision Tree is widely used in Machine learning and it is simple and intuitive. (抄一点proposal)

### Streaming Decision Tree

...

```pseudocode
Call TreeBuilding procedure:
Initialize T to be a single unlabeled node.
foreach batch_data do:
	Reinitialize every leaf in T as unlabeled.
  while there are unlabeled leaves in T:
    Navigate the batch_data to the leaves
    Construct the histogram h(v, i, c) by calling COMPRESS procedure.
    for all unlabeled leaves v in T do:
      if v.should_terminate() or there are no samples reaching v:
        Label v
      else:
        Call find_best_split procedures:
          for all features i do:
            Merge the h(v, i, 1)..h(v, i, c) and get h(v, i)
            Determine the candidate splits by calling UNIFORM.
            Estimate the information gain of each candidata by calling SUM.
          endfor
        Split v with the highest gain.
      endif
    endfor
  endwhile
```

```pseudocode
Call COMPRESS procedure:
	Initalized empty histogram h(v, i, j) for unlabeled leaf v, feature i, class j
	foreach data point (x, y) do:
		if the sample belongs to leaf v then:
			foreach feature i do:
				Call UPDATE procefure to update h(v, i, y)
			endfor
		endif
	endfor
```


### Key Challenges
...

## Approach
There are mainly two functions in the tree building process that are costly. `compress` function and `find_best_split` function. And both of them could be parallelized. Since now to do not need to Also, we introduce our baseline parallel strategy ---- Node Parallel. 

### Node Parallel

This parallel method is the most intuitive method because there is a natural independent loop in the psedocode,`for all unlabeled leaves v in T do`. However, this work partitioning strategy suffers from one problem, namely,  **imbalanced Workload** : The workload is determined by the number of childs of this parent node. However, this is unpredictable. Due to the essential drawback of static assignment of this parallel paradigm, the workload is unbalanced so this strategy is expected to show worst speedup and bad scalability.

### Feature Parallel

In the `find_best_split` procedure, the `for all features i do` loop could be parallelized. One note is that the best split information should be manurally syncronized. One method is to introduce a local varaible for each worker and then merge the results afterwards. Another method is to use lock-based syncronization to protect the shared varaible. After testing, we implemented the first version. Since each thread only needs to send it's best split feature to the master, the communication cost is O(P)

### Data Parallel

In the `compress` procedure, the `foreach data point (x, y) do` is parallelized. But similarly to the feature parallel version, there is a big challenge. Each data point would update `h(v, i, y)`, which means there is huge contension for the histogram method. In order to encounter this problem, we reorder the loop in a way that there is no race condition. 

```pseudocode
Call COMPRESS procedure:
	Initalized empty histogram h(v, i, j) for unlabeled leaf v, feature i, class j
	#pragma omp parallel for
	foreach unlabeled leaf v do:
		foreach data point (x, y) in v do:
			foreach feature i do:
				Call UPDATE procefure to update h(v, i, y)
			endfor
		endif
	endfor
```

This strategy is used by using shared-memory model and implemente by OpenMP. However, this algorithm would suffer from some trivial work imbalance problem as each leaf contains unequal number of data points. But this could be matigated by dynamic scheduling.



We also introduce a message-passing model data-parallel version here. This strategy is essentially streaming and could handle as much as possible. This parallel algorithm is borrowed from [A Streaming Parallel Decision Tree Algorithm](http://www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf ).

![Streaming Compress](./img/compress.png "Streaming COMPRESS Algorithm")

This algorithm is essentially streaming and need explicit synchronization. After each work completes their work, all the histogram data would be transferred to the MASTER worker and do the merging process. After mergin completes, the result would be braodcasted to other workers. (For more information about merging and some convergence proof, please refer to the paper.) Since each worker would send the histogram to the MASTER, the communication cost is $O( W ×L×c×d)$, where $W$ is the number of workers, $L$ is the number of unlabeled leaves in the current iteration, $c$ is the number of labels and $d$ is the number of features.  For a summary for this algorithm,

- At most$ N/W$ operations by each processor in the updating phase.
- $O( W ×L×c×d)$ communication cost.
- $O( W ×L×c×d)$ for merging.

### Data-Feature Parallel

We successfully combined the above two parallel strategy together as they should be independent with each other.

## Implementation
### OpenMP & OpenMPI implementations
All the parallel approaches are implemented by OpenMP and also we did a message-passing  Data-Feature parallel version on OpenMPI. 

#### CUDA implementation
Further more, we implemented a parallel version on CUDA. 
### Key Data Structure

#### How we store our dataset in vector version

#### How we store our dataset in several serialized arrays for GPU
Why we do this: pass dataset to GPU
The limitation: difficult for operations, hard to write codes and debug, memory limits on GPU

#### How we serialize histograms
Pointers, malloc and seek.

#### Split Node

#### Decision Tree

### Key operations

#### Read/write through the vector version dataset
Use vector and map. Save space.

#### Read/write through the serialized array version dataset


#### Several operations for histograms in vector version
Operations: update, sum, merge, uniform, compress

#### Several operations for histograms in array version
Operations: update, sum, merge, uniform, compress. HARD to implement in array version

### Algorithm's inputs and outputs
Input: train dataset, test dataset. Some parameters of the dataset (Num of data, feature num, class num)

Output: A decision tree. Predict on the test dataset, to verify the correctness of decision tree.

### Computationally expensive parts

#### Get gain function

#### Assign dataset to leaf nodes

#### Update the histogram

#### Navigate the samples

### The parallel in program
Several parallel approaches

Parallel on data / leaf nodes, when assign dataset to leaf nodes

Parallel on histograms / data, when update the histogram

Parallel on data, when navigate the samples

## Results

### Node Parallel & Data-Feature Parallel Speedup analysis 

...

### Data-Feature Parallel Scalability over Data Size

#### OpenMP

...

####OpenMPI

...

### Data-Feature Parallel Scalability over Feature Size

...

### CUDA Speedup Analysis

[Speedup table in Evaluation folder]

Problems:
1. OpenMPI, train_time > split_time
2. openmp feature parallel, data index = 1/5, abort

## References

## List of work

