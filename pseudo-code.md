Each worker has a full view of the current tree being built. The workers accumulate samples in the leaf nodes of the tree, building a separate histogram h for each class in each leaf,

```pseudocode
Initialize T
Call COMPRESS procedure:
	Initalized empty histogram h(v, i, j) for unlabeled leaf v, feature i, 		class j
	foreach data point (x, y) do:
		if the sample belongs to leaf v then:
			foreach feature i do:
				Call UPDATE procefure to update h(v, i, y)
			endfor
		endif
	endfor

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
        Merge the h(v, i, 1)..h(v, i, c) and get h(v, i)
        Choose the candidate splits by calling UNIFORM.
        Estimate each candidata by calling SUM.
        Split v with the highest gain.
        
      endif
    endfor
  endwhile
	
```

Question:

1) How to store the global histogram h(v, i, j) so that it could be easily transferred among processors and also each leaf could access the corresponding h(v, i, j) with O(1) time.

2) How to let the leaf know (or is there a need for a leaf to know since he could access the histogram ) which subset of the data belongs to this leaf. (Better not store the data like what we do in quadtree building because it requires linear scanning of data and extra storage.)



Complexity Analysis:

Every **iteration** consists of an updating phase performed simultaneously by all the processors and a merging phase performed by the master processor. In the update phase, every processor **makes one pass on the data batch assigned to it**. The only memory allocation is for the histograms being constructed. The number of bins in the histograms is constant; hence, operations on histograms take a constant amount of time. Every processor performs at most *N*/*W* histogram updates, where *N* is the size of the data batch and *W* is the number of processors. There are *W* × *L* × *c* × *d* histograms, where *L* is the number of leaves in the current iteration, *c* is the number of labels, and *d* is the number of attributes. Assuming that *W*, *L*, *c*, and *d* are all independent of *N*, it follows that the space complexity is *O*(1).  The histograms are communicated to the master processor, which merges them and applies the sum and uniform procedures. If the uniform procedure is applied with a constant parameter *B* ̃, then the time complexity of the merging phase is *O*(1). 