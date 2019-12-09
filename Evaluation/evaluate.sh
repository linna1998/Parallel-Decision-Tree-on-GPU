#!/bin/bash
# baseline
TARGETBIN="./decision-tree"
# openmp
TARGETBIN_NODE="./decision-tree-node-openmp"
TARGETBIN_FEATURE="./decision-tree-feature-openmp"
TARGETBIN_DATA_FEATURE="./decision-tree-feature-data-openmp"
TARGETBIN_DATA2="./decision-tree-data-openmp"

# openmp
TARGETBIN_DATA="./decision-tree-data"

# cuda
TARGETBIN_CUDA="./decision-tree-cuda"

echo -e "Base Sequential mode"
for i in 1 6 7
do  
$TARGETBIN -i $i
done  
echo "------------------"
echo "------------------"

# echo -e "Base OpenMP Node-Parallel Mode" 
# for i in 1 6 7
# do  
# $TARGETBIN_NODE -i $i -n 4
# done 
# echo "------------------"
# echo "------------------"

# echo "Dominated OpenMP Data-Parallel Mode"
# for i in 1 6 7
# do  
# $TARGETBIN_DATA2 -i $i -n 4
# done
# echo "------------------"
# echo "------------------"

# echo "Dominated OpenMP Feature-Parallel Mode"
# for i in 1 6 7 8 9 10 11   
# do  
# for t in 1 2 4 8
# do
# $TARGETBIN_FEATURE -i $i -n $t
# done  
# done  
# echo "------------------"
# echo "------------------"

echo "Improved OpenMP Data-Feature-Parallel Mode"
for i in 1 6 7
do  
$TARGETBIN_DATA2 -i $i -n 4
done
echo "------------------"
echo "------------------"

echo "Message Passing Data-Feature-Parallel Mode"
for i in 1 6 7
do  
mpirun -np 4 $TARGETBIN_DATA -i $i
done  
echo "------------------"
echo "------------------"

echo "CUDA Version"
for i in 1 6 7
do  
$TARGETBIN_CUDA -i $i
done  
echo "------------------"

exit