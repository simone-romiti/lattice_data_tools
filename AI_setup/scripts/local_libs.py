import numpy as np
import struct

# reading the .bin files produced by Giseppe, Antonio, Francesca, etc.
def read_bin(file_path):
    with open(file_path, 'rb') as file:
        # Read the header (3 64-bit integers: N_g, dummy, T)
        header_format = '3q'
        header_size = struct.calcsize(header_format)
        header_data = file.read(header_size)
        if not header_data:
            return None
        integers = struct.unpack(header_format, header_data)
        
        double_data = file.read()
        num_doubles = len(double_data) // struct.calcsize('d')
        doubles = struct.unpack(f'{num_doubles}d', double_data)
        
        N_g = integers[0] # number of configurations
        T = integers[2] # time extent in lattice units
        N_avg = 1 # For these binaries, N_avg is implicitly 1
        
        corr_flat = np.array(doubles)
        # Handle potential trailing double by slicing to the expected size
        expected_size = N_g * (T // 2 + 1) * N_avg
        corr_flat = corr_flat[:expected_size]
        # Reshape to (N_g, T // 2 + 1, N_avg) and transpose to (N_avg, N_g, T // 2 + 1)
        corr = np.transpose(corr_flat.reshape(N_g, T // 2 + 1, N_avg), axes=(2,0,1))
        return {"integers": integers, "confs": corr}
#-------
