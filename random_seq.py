import sys
import os
import random
import numpy as np

num_vector = int(sys.argv[1])
min_number = int(sys.argv[2])
seed = int(sys.argv[3])
seq_number = 2**(num_vector)

random.seed(seed)
np.random.seed(seed)
def get_seq(num_vector):
    vector_index = [str(i) for i in range(num_vector)]
    vector_index.append("c")

    seq = []
    
    while(True):
        s = random.choice(vector_index)
        seq.append(s)
        if s == "c":
            break

    vector_index.append("e")
    while(True):
        s = random.choice(vector_index)
        if s == "e":
	        break
        else:
	        seq.append(s)

    if len(seq) >= min_number:
        print(",".join(seq))
        return 1
        
    else:
        return 0
    

count = 0
while(count < seq_number):
    r = get_seq(num_vector)
    if r:
        count += 1
