
from mpi4py import MPI
import numpy as np
import pickle
import sys

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)

def merge_two_lists(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result

def merge_sorted_lists(sorted_lists):
    result = []
    for lst in sorted_lists:
        result = merge_two_lists(result, lst)
    return result

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        with open('temp_data.pkl', 'rb') as f:
            data = pickle.load(f)
        chunks = np.array_split(data, size)
        chunks = [chunk.tolist() for chunk in chunks]
    else:
        chunks = None

    local_data = comm.scatter(chunks, root=0)

    comm.Barrier()
    start_time = MPI.Wtime()
    local_sorted = quicksort(local_data)
    gathered = comm.gather(local_sorted, root=0)
    end_time = MPI.Wtime()

    if rank == 0:
        result = merge_sorted_lists(gathered)
        with open('temp_result.pkl', 'wb') as f:
            pickle.dump((end_time - start_time, result), f)
