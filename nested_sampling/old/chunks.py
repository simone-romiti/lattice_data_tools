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
    - Open all the files and read the minimum and maximum values of L in each stream. 
      We store them in a list and find the global minimum and maximum values of L across all streams.
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
        Stream 0: [0.06, 1.24, 2.00, 3.22, 4.37, 5.56, 6.84, 8.03, 9.09, 9.99]
        Stream 1: [1.23, 2.17, 2.52, 3.30, 4.40, 5.31, 6.71, 7.96, 8.73, 9.97]
        Stream 2: [0.03, 1.19, 2.48, 3.37, 4.41, 5.84, 6.51, 7.85, 9.82, 9.95]

    Let's assume our memory is such that we can store simultaneously only M=6 numbers.
    Then, we consider N_c chunks of size S_c=M/n_s chunks, i.e. N_c=ceil(N_p/S)=5 chunks, 
    and find the matrix of indices corresponding to the beginning of each chunk.
    We recognize that L_min=0.06 and L_max=9.99. the rows of the chunks matrix read:
        Stream 0: [0, 2, 4, 6, 8]
        Stream 1: [0, 1, 4, 6, 8]
        Stream 2: [0, 2, 4, 6, 7]

    NOTE: Due to the fact the the distribution of the values in each stream is just similar, 
    the indices in the other streams are not too far from the indices in the reference stream.
    This however means that the chunks indices and sizes are not exactly the same in each stream.
        REMARK: (idea for the future) One could also consider to average the values corresponding to the naive indices for each stream,
        and then run the chunk indexing on each stream.

    The NS analysis can then be run on each chunk independently.
""" 

import sys
import numpy as np
import pandas as pd
from typing import List
import os
import psutil # available memory

def get_avail_memory_bites():
    return psutil.virtual_memory().available # available memory in bytes
#---

def get_chunks_map(files, float_type, N_s: int, n_s: int,  p: float = 0.9):
    """
    Get the map of chunks indices for each stream, given the minimum and maximum values of L, 
    the number of streams N_s, the number of streams to sample n_s, and the percentage of memory p.
    
    :param files: List of file paths containing the streams.
    :param float_type: Type of the float (e.g. np.float32, np.float64).
    :param L_min: Minimum value of L.
    :param L_max: Maximum value of L.
    :param N_s: Total number of streams.
    :param n_s: Number of streams to sample at each step.
    :param p: Percentage of memory to use (default is 0.9).
    
    :return: Matrix of chunk indices. Each row "i" is the list of beginning position of the i-th chunk.
    """
    machine_memory = get_avail_memory_bites() # available memory in bites
    allotted_memory = p * machine_memory #  size of each chunk in bites
    chunk_size_bites = allotted_memory / n_s #  size of each chunk in bites
    float_size_bites = sys.getsizeof(float_type)
    chunk_size = int(chunk_size_bites / float_size_bites) # number of elements in each chunk
    print(f"Float type size: {float_size_bites} bytes")
    print(f"Available memory: {machine_memory} bytes")
    print(f"Chunk size: {chunk_size} elements ({chunk_size_bites:.2f} bytes)")
    #---
    with open(files[0], 'r', encoding='utf-8') as f:
        N_p = sum(1 for _ in f)
    #---
    L_min_arr, L_max_arr = [], []
    N_p_arr = []
    for i in range(N_s):
        stream_i = pd.read_csv(files[i], header=None)[0].to_numpy(dtype=float_type) 
        L_min_arr.append(np.min(stream_i)) # minimum value of L in the stream
        L_max_arr.append(np.max(stream_i))
        N_p_arr.append(len(stream_i))
    #---
    L_min = min(L_min_arr) # minimum value of L across all streams
    L_max = max(L_max_arr) # maximum value of L across all streams
    N_p_tot = sum(N_p_arr) # total number of points across all streams
    N_p = max(N_p_arr) # maximum number of points in the streams
    N_c = int(np.ceil(N_p / chunk_size)) # number of chunks
    L_values = np.linspace(L_min, L_max, N_c + 1) # values of L for each chunk
    #---
    chunks_matrix = np.zeros((N_s, N_c), dtype=int) # matrix of chunk indices
    for i in range(N_s):
        stream_i = pd.read_csv(files[i], header=None)[0].to_numpy(dtype=float_type)
        for j in range(N_c):
            idx_list = np.argwhere(stream_i <= L_values[j])
            idx = 0
            if idx_list.size != 0:
                idx = np.argwhere(stream_i <= L_values[j])[-1][0]
            #---
            chunks_matrix[i, j] = idx
    #-------
    res = {
        "allotted_memory": allotted_memory,
        "chunks_matrix": chunks_matrix,
        "N_s": N_s,
        "N_c": N_c,
        "chunk_size": chunk_size,
        "L_min": L_min,
        "L_max": L_max,
        "N_p_tot": N_p_tot,
    }
    return res 
#---


if __name__ == "__main__":
    import os
    # Parameters
    N_s = 10  # number of streams
    N_p = int(1e6)  # points per stream
    float_type = np.float64
    n_s = 2  # number of streams to sample at each step

    # Generate random streams and save to temporary files
    temp_dir = "./temp/"
    os.makedirs(temp_dir, exist_ok=True)  # Create temporary directory if it doesn't exist
    files = []
    rng = np.random.default_rng(seed=42)
    for i in range(N_s):
        # Generate sorted random values between 0 and 10
        data = np.sort(rng.uniform(0, 10, N_p).astype(float_type))
        file_path = os.path.join(temp_dir, f"stream_{i}.csv")
        pd.DataFrame(data).to_csv(file_path, header=False, index=False)
        files.append(file_path)
    #---
    # Running get_chunks_map with a fraction "p" of available memory
    chunks_map = get_chunks_map(files, float_type, N_s, n_s, p=0.001)
    allotted_memory = chunks_map["allotted_memory"]
    chunks_matrix = chunks_map["chunks_matrix"] # the idea is that you save this matrix to a file, and then read it in the analysis script
    print("Chunks matrix")
    print(chunks_matrix)
    chunk_size = chunks_map["chunk_size"]
    N_p_tot = chunks_map["N_p_tot"]
    N_bts = 5 # number of bootstraps
    for b in range(N_bts):
        i_streams = np.random.randint(low=0, high=N_s, size=n_s)
        N_c = chunks_map["N_c"]
        for c in range(N_c):
            combined_stream = np.zeros(shape=(0), dtype=float_type)
            for i in range(n_s):
                c1 = chunks_matrix[i_streams, c][i]
                c2 = -1
                if (c + 1 < N_c):
                    c2 = chunks_matrix[i_streams, c+1]
                #---
                stream_i = pd.read_csv(files[0], header=None, skiprows=c1, nrows=(c2-c1))[0].to_numpy(dtype=float_type) 
                combined_stream = np.append(combined_stream, stream_i)
            #---
            combined_stream = np.sort(combined_stream) # sorting the values in the given chunk of values
            # =====================================================
            # Here you would run the NS analysis on combined_stream
            # =====================================================
    

