""" 
The data from Nested Sampling (NS) run(s) is a set of N_s streams (each saved into a file).
Each of the streams contains N_p ordered points, that are the values of the Likelihood L.

One can think about the whole dataset as a matrix: 
  - rows: values of the Likelihood
  - columns: streams.

GOAL: combine the streams (i.e. the columns of the matrix) into a single ordered list of points, and run the NS analysis on it.

PROBLEM: For high N_s and N_p, it is both computationally and memory expensive to do the sorting.

SOLUTION: 

    An advantages of NS is that one can split the analysis into independent chunks, 
    as the portions of the phase space (ranges of the integration variable X) are independent of each other.

    - "L" has a known minimum and known maximum. For instance, L=exp(-S) goes from exp(-S_{max}) to 1.
    - Each stream can be split into sub-chunks of index "i", with values from L_low^{(i)} and L_high^{(i)}.
    - Each stream is in ascending (or descending, according to the convention) order, with a similar distribution. 
    
    Thus, we can proceed as follows. Imagining that each stream is stored in a separate file, we can:
    - Pre-compute the memory available (or the percentage of memory available that the user wants to use). 
      This defines the resolution, i.e. the number of sub-chunks N_c of the values from L_min to L_max.
    - For each i, from 0 to N_c-1, we find the indices of the lines in each stream that correspond to the values L_low^{(i)} and L_high^{(i)}.
      In other words, we build a matrix of indices, where each column corresponds to a stream and each row corresponds to a sub-chunk.
    - The analysis works as a loop over the i values, each corresponding to an interval for the integration variable "X". 
      For each of them, we extract the values of L from each stream, from L_low^{(i)} and L_high^{(i)}.
        - Again for each "i", we combine these values into a single ordered array, and run the analysis on it.
            
    This allows to parallelize the analysis and to keep control over the memory used by each chunk.

EXAMPLE:
    Suppose we have N_s=3 streams, each with N_p=10 points, and that every time we run the analysis we sample n_s=2 streams (e.g. bootstrap sample).
        Stream 0: [0.06, 1.24, 2.00, 3.22, 4.37, 5.56, 6.84, 8.03, 9.09, 9.89]
        Stream 1: [1.23, 2.17, 2.52, 3.30, 4.40, 5.31, 6.71, 7.96, 8.73, 9.98]
        Stream 2: [0.03, 1.19, 2.48, 3.37, 4.41, 5.84, 6.51, 7.85, 9.82, 9.95]

    Let's assume our memory is such that we can store simultaneously only M=6 numbers.
    Then, we consider N_c chunks of size S_c=M/n_s chunks, i.e. N_c=ceil(N_p/S)=5 chunks, and find the matrix of indices corresponding to the beginning of each chunk.
    In doing that, we use the 1st stream as a reference, and find the chunk indices in the other streams.
        Stream 0: [0, 2, 4, 6, 8] (the indices are trivial, as this is the reference stream)
        Stream 1: [0, 1, 4, 6, 8]
        Stream 2: [0, 2, 4, 6, 7]

    NOTE: due to the fact the the distribution of the values in each stream is just similar, 
    the indices in the other streams are not too far from the indices in the reference stream.
    This however means that the chunks indices and sizes are not exactly the same in each stream.
        REMARK: One could also consider to average the values corresponding to the naive indices for each stream,
        and then run the chunk indexing on each stream.

    The NS analysis can then be run on each chunk independently.
""" 

import numpy as np
import psutil # available memory


# ! To be implemented !

