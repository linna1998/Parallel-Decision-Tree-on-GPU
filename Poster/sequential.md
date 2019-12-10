```C
Function BuildTree(n,A) // n: samples (rows), A: attributes 
  If empty(A) or all n(L) are the same
    status = leaf
    class = most common class in n(L) 
  else
    status = internal
    a <- bestAttributeSplitPoint(n,A)
    LeftNode = BuildTree(n(a=1), A \ {a}) 
    RightNode = BuildTree(n(a=0), A \ {a})
  end 
end
```