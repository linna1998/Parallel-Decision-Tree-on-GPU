#!/bin/bash
TARGETBIN="./decision-tree"
TARGETBIN_FEATURE="./decision-tree-feature"
TARGETBIN_DATA="./decision-tree-data"
TARGETBIN_DATA2="./decision-tree-data-openmp"
TARGETBIN_CUDA="./decision-tree-cuda"


echo -e "Sequential mode"
for i in {0, 1, 5}  
do  
$TARGETBIN -i $i
done  

echo "Feature parallel"
for i in {0, 1, 5}  
do  
$TARGETBIN_FEATURE -i $i
done  

echo "Data openmp"
for i in {0, 1, 5}  
do  
$TARGETBIN_DATA2 -i $i
done  

echo "Data openmpi"
for i in {0, 1, 5}  
do  
$TARGETBIN_DATA -i $i
done  

echo "CUDA mode"
for i in {0, 1, 5}  
do  
$TARGETBIN_CUDA -i $i
done  

exit