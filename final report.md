# Parallel Streaming Decision Tree on GPU - Final Report
<center> ke Ding(keding@andrew.cmu.edu) || Anxiang Zhang (anxiangz@andrew.cmu.edu) </center>

## Summary
We implemented the sequential version, the OpenMP version, the OpenMPI version and the CUDA version for decision tree with histogram and compared the performance of four implementations.

## Background

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


## Approach

## Results

[Speedup table in Evaluation folder]

Problems:
1. OpenMPI, train_time > split_time
2. openmp feature parallel, data index = 1/5, abort

## References

## List of work

