# Parallel Streaming Decision Tree on GPU - Project Checkpoint
<center> ke Ding(keding@andrew.cmu.edu) || Anxiang Zhang (anxiangz@andrew.cmu.edu) </center>

## Detailed Schedule
11.8 Find more references, find the dataset for evaluation and finish the data set I/O. [Anxiang Zhang, Ke Ding]

11.12 Drafted the Histogram part of this project. [Ke Ding]

11.12 Drafted the Tree construction part of this project. [Anxiang Zhang]

11.15 Debug and finish the CPU sequential version. [Anxiang Zhang, Ke Ding]

11.17 Drafted the CPU parallel version with OpenMP. [Ke Ding]

(expected)11.22 Finish basic CPU parallel version(openMP, openMPI).

(expected)11.29 Finish the GPU parallel version.

(expected)12.4 Evaluate and find the trade-offs between different implementations.

(expected)12.10 Finish final report and poster.

## Work Completed
We have already completed the sequential version of our program. Due to the different implementation methods of decision tree, we find more references to figure out how to construct a decision tree using histograms. Then Ke programed the histogram related functions and Anxiang used those function to program the sequential version of the decision tree construction. We used three days to debug.

We used more than two weeks to finish the sequential version because we tried to use a parallel-friendly to implement the tree. So the following parallel code would not be mush complicated compared to the sequential version. 

## How we are doing
We think that currently, we could successfully complete all the goals in the proposal. So our goal is still the same. 1) High scalability on GPU when data becomes larger. 2) More speedup on GPU.

## Plan for Poster Session
We plan to show the speed-up graphs for our sequential version, CPU parallel version and GPU parallel version on several datasets during the poster session.

## Preliminary Results
We have finished the sequential version, and we have gathered the tree building time and testing time on some datasets. Moreover, we used gprof to evaluate the time of each functions in our code.

We have evaluated our CPU parallel version with OpenMP. However, the performance of our CPU parallel version is less than expected. We planned to fix this issue in the future.

## Concerns

The concerns now is the memory problem. Currently, our algorithm used too many memories. Maybe this could make the GPU implementation and openMPI implementation somewhat hard. If this happens, we could modify the data structure to minimize the memory requirement.