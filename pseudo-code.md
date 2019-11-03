```pseudocode
Algorithm 1 Growing decision tree, breadth-first approach 
Input: A training set {(x1, y1), ...}
T <- Root
queue<-Root
Initalize Histogram
while(!queue.empty())
  T_next = queue.pop()
  if T_next satisfy stopping criterion then
  	 Label T_next by majority class
  	 Label T_next as Leaf
 	else:
 		 compute_histogram(T_next)
 		 feature_name, split_idx = EvaluateSplits(T_next->Histogram)
 		 new_histogram = split_data(feature_name, split_idx, T_next->Histogram)
  
```





Split Data:

Dataset = {(x1, 1), {x2, 0}, {x3, 1} ... }

Dataset index = [0, 1, 2, ...]

Histogram = {(x1, 1, 1), {x2, 1, 0}, {x3, 1, 1}}

在x2 split

