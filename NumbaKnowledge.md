## Things I've learned about Numba

#### Python code must be written in specific ways to play nice with Numba
- loops with numpy arrays are the gold standard
- object-oriented operations seem to not be good
- functions that apply independant operations to all the elements of an array can be "vectorized", which provides massive speedup

#### Numba organization
- jitted code can call other jitted code
- nopython parameter (or njit) will essentially force the code to be fully optimized - it will return an error if it can't be optimized, instead of running slowly
- passing functions is iffy at besthttps://stackoverflow.com/questions/59573365/using-a-function-object-as-an-argument-for-numba-njit-function 
- A guide to using CUDA - possibly even more speedups, especially given GPU access - is here https://curiouscoding.nl/posts/numba-cuda-speedup/