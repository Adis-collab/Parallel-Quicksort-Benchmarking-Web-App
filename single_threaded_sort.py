import time
import numpy as np

def quicksort(arr):
    """Optimized quicksort implementation"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)

def run_single_threaded(data):
    """Run quicksort using only one process"""
    start = time.time()
    result = quicksort(data.tolist())
    end = time.time()
    return end - start, result

if __name__ == "__main__":
    import pickle
    import sys

    if len(sys.argv) != 2:
        print("Usage: python single_threaded_sort.py <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]

    # Load data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    # Run single-threaded sort
    single_time, single_result = run_single_threaded(data)

    # Save results
    with open('single_thread_result.pkl', 'wb') as f:
        pickle.dump((single_time, single_result), f)

    print(f"Single-threaded Time: {single_time:.4f} seconds") 