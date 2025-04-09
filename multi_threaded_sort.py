from mpi4py import MPI
import numpy as np
import pickle
import sys

def quicksort(arr):
    """Optimized quicksort implementation"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)

def merge_two_lists(a, b):
    """Optimized merge implementation"""
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

def merge_sorted_lists(lists):
    """Optimized multi-list merge"""
    if not lists:
        return []
    while len(lists) > 1:
        new_lists = []
        for i in range(0, len(lists), 2):
            if i + 1 < len(lists):
                new_lists.append(merge_two_lists(lists[i], lists[i + 1]))
            else:
                new_lists.append(lists[i])
        lists = new_lists
    return lists[0]

def run_parallel(data, num_processes):
    """Run quicksort using specified number of processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != num_processes:
        if rank == 0:
            print(f"Error: Requested {num_processes} processes but got {size}")
        return None, None

    # Load data from disk (only by rank 0)
    if rank == 0:
        # Use numpy's array_split for more efficient chunking
        chunks = np.array_split(data, size)
        # Convert to list only once per chunk
        chunks = [chunk.tolist() for chunk in chunks]
    else:
        chunks = None

    # Distribute data using scatter
    local_data = comm.scatter(chunks, root=0)

    comm.Barrier()
    start = MPI.Wtime()
    
    # Sort local data
    sorted_local = quicksort(local_data)
    
    # Gather results using gather
    gathered = comm.gather(sorted_local, root=0)

    end = MPI.Wtime()

    if rank == 0:
        # Merge results
        merged = merge_sorted_lists(gathered)
        total_time = end - start
        return total_time, merged
    return None, None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: mpirun -n <num_processes> python3 multi_threaded_sort.py <data_file> <num_processes>")
        sys.exit(1)

    data_file = sys.argv[1]
    num_processes = int(sys.argv[2])

    # Load data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    # Run parallel sort
    parallel_time, parallel_result = run_parallel(data, num_processes)

    if MPI.COMM_WORLD.Get_rank() == 0:
        # Save results
        with open('multi_thread_result.pkl', 'wb') as f:
            pickle.dump((parallel_time, parallel_result), f)

        print(f"Parallel Time ({num_processes} processes): {parallel_time:.4f} seconds") 