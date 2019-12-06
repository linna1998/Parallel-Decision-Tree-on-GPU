#!/bin/bash
TARGETBIN="./decision-tree"
TARGETBIN_FEATURE="./decision-tree-feature"
TARGETBIN_DATA="./decision-tree-data"
TARGETBIN_DATA2="./decision-tree-data-openmp"
TARGETBIN_CUDA="./decision-tree-cuda"

DATASET="{0,1,5}"
echo -e "Sequential mode"
for i in $DATASET  
do  
$TARGETBIN -i $i
done  

echo -e "Feature parallel"
for i in {0, 1, 5}  
do  
$TARGETBIN_FEATURE -i $i
done  

echo -e "Data openmp"
for i in {0, 1, 5}  
do  
$TARGETBIN_DATA2 -i $i
done  

echo -e "Data openmpi"
for i in {0, 1, 5}  
do  
$TARGETBIN_DATA -i $i
done  

echo -e "CUDA mode"
for i in {0, 1, 5}  
do  
$TARGETBIN_CUDA -i $i
done  

exit