# 🧠 Parallel Quicksort Benchmarking Web App

A Flask-based web application that benchmarks and compares the performance of **single-threaded** and **MPI-based parallel Quicksort** algorithms on a multi-core system.

---

## 🚀 Features

- User input for:
  - Number of elements to sort
  - Number of MPI processes to use
- Benchmarks:
  - Single-threaded Quicksort
  - Parallel Quicksort using `mpi4py` and `mpirun`
- Displays:
  - Execution time (single vs parallel)
  - Speedup
  - Efficiency
  - Result correctness

---

## 📦 Requirements

- Python 3.x
- Flask
- Flask-CORS
- NumPy
- mpi4py
- MPI runtime (e.g., MPICH or OpenMPI)

### Install dependencies

```bash
pip install flask flask-cors numpy mpi4py
sudo apt install mpich  # or openmpi-bin
