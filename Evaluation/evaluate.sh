#!/bin/bash
TARGETBIN="./decision-tree"
# TARGETBIN_FEATURE="./decision-tree-feature"
TARGETBIN_DATA_FEATURE="./decision-tree-feature-data-openmp"
TARGETBIN_DATA="./decision-tree-data"
TARGETBIN_NODE="./decision-tree-node"
# TARGETBIN_DATA2="./decision-tree-data-openmp"
TARGETBIN_CUDA="./decision-tree-cuda"

echo -e "Base Sequential mode"
for i in 0 1 6 7 8 9 10 11  
do  
$TARGETBIN -i $i
done  
echo "------------------"
echo "------------------"

echo -e "Base OpenMP Node-Parallel Mode" 
for t in 1 2 4 8 16
do
$TARGETBIN -i $10 -n t
done  
echo "------------------"
echo "------------------"


echo "Improved OpenMP Data-Feature-Parallel Mode"
for i in 0 1 6 7 8 9 10 11   
do  
for t in 1 2 4 8 16
do
$TARGETBIN_DATA_FEATURE -i $i -n t
done  
done  
echo "------------------"
echo "------------------"

echo "Message Passing Data-Feature-Parallel Mode"
for i in 0 1 6 7 8 9 10 11  
do  
for t in 1 2 4 8
do
mpirun -np $t $TARGETBIN_DATA -i $i
done
done  
echo "------------------"
echo "------------------"

# echo "CUDA Version"
# for i in 0 1 6 7 8 9 10 11  
# do  
# $TARGETBIN_CUDA -i $i
# done  

exit