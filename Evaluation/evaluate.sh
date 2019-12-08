#!/bin/bash
TARGETBIN="./decision-tree"
TARGETBIN_FEATURE="./decision-tree-feature"
TARGETBIN_DATA="./decision-tree-data"
TARGETBIN_DATA2="./decision-tree-data-openmp"
TARGETBIN_CUDA="./decision-tree-cuda"

echo -e "Sequential mode"
for i in 0 1 5 6 7 8 9 10   
do  
$TARGETBIN -i $i
done  
echo "------------------"
echo "------------------"

echo "Feature parallel"
for i in 0 1 5 6 7 8 9 10    
do  
$TARGETBIN_FEATURE -i $i
done  
echo "------------------"
echo "------------------"
echo "Data openmp"
for i in 0 1 5 6 7 8 9 10     
do  
$TARGETBIN_DATA2 -i $i
done  
echo "------------------"
echo "------------------"

echo "Data openmpi"
for i in 0 1 5 6 7 8 9 10   
do  
for t in 1 2 4 8
do
mpirun -np $t $TARGETBIN_DATA -i $i
done
done  
echo "------------------"
echo "------------------"

# echo "CUDA mode"
# for i in 0 1 5  
# do  
# $TARGETBIN_CUDA -i $i
# done  

exit