# Finding Global Optima in Nonconvex Stochastic Semidefinite Optimization with Variance Reduction
MATLAB MEX implementation of SVRG-SBB algorithms

This code replicates the Eurodist dataset experiments from the following paper:

- ["Finding Global Optima in Nonconvex Stochastic Semidefinite Optimization with Variance Reduction".<br>Jinshan Zeng, Ke Ma, Yuan Yao. _AISTATS 2018_.](https://arxiv.org/abs/1802.06232)

Please cite this paper if you use this code in your published research project.

## Requirement
In MATLAB:
``` 
>>mex -v -g -largeArrayDims *.c
```
and run the script with the mex function, data file in the same directory.