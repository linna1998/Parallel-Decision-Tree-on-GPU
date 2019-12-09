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
$TARGETBIN -i 
done  
echo "------------------"
echo "------------------"

# echo -e "Base OpenMP Node-Parallel Mode" 
# for t in 1 2 4 8
# do
# $TARGETBIN_NODE -i 10 -n $t
# done  
# echo "------------------"
# echo "------------------"

# echo "Dominated OpenMP Data-Parallel Mode"
# for i in 1 6 7 8 9 10 11   
# do  
# for t in 1 2 4 8
# do
# $TARGETBIN_DATA2 -i $i -n $t
# done  
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


# echo "Improved OpenMP Data-Feature-Parallel Mode"
# for i in 1 6 7 8 9 10 11   
# do  
# for t in 1 2 4 8
# do
# $TARGETBIN_DATA_FEATURE -i $i -n $t
# done  
# done  
# echo "------------------"
# echo "------------------"

echo "Message Passing Data-Feature-Parallel Mode"
for t in 1 2 4 8
do
mpirun -np $t $TARGETBIN_DATA -i 12
done
echo "------------------"
echo "------------------"

# echo "CUDA Version"
# for i in 1 6 7 8 9 10 11  
# do  
# $TARGETBIN_CUDA -i $i
# done  

exit