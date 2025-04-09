from flask import Flask, request, render_template, jsonify
import numpy as np
import subprocess
import pickle
import os
import time
import re

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sort', methods=['POST'])
def sort():
    try:
        num_elements = int(request.form['num_elements'])
        num_processes = int(request.form['num_processes'])

        # Generate data
        data = np.random.randint(0, 1000000, size=num_elements)

        # Save data for both runs
        with open('data.pkl', 'wb') as f:
            pickle.dump(data, f)

        # Run single-threaded sort
        try:
            result = subprocess.run(
                ['python3', 'single_threaded_sort.py', 'data.pkl'],
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                return jsonify({'error': f'Single-threaded execution failed: {result.stderr}'})

            # Load single-threaded results
            with open('single_thread_result.pkl', 'rb') as f:
                single_time, single_result = pickle.load(f)

        except subprocess.TimeoutExpired:
            return jsonify({'error': 'Single-threaded execution timed out'})
        except Exception as e:
            return jsonify({'error': f'Single-threaded execution failed: {str(e)}'})

        # Run parallel sort
        try:
            result = subprocess.run(
                ['mpirun', '-n', str(num_processes), 'python3', 'multi_threaded_sort.py', 'data.pkl', str(num_processes)],
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                return jsonify({'error': f'Parallel execution failed: {result.stderr}'})

            # Load parallel results
            with open('multi_thread_result.pkl', 'rb') as f:
                parallel_time, parallel_result = pickle.load(f)

        except subprocess.TimeoutExpired:
            return jsonify({'error': 'Parallel execution timed out'})
        except Exception as e:
            return jsonify({'error': f'Parallel execution failed: {str(e)}'})

        # Calculate speedup and efficiency
        speedup = single_time / parallel_time
        efficiency = (speedup / num_processes) * 100

        # Clean up temporary files
        for file in ['data.pkl', 'single_thread_result.pkl', 'multi_thread_result.pkl']:
            if os.path.exists(file):
                os.remove(file)

        return render_template('index.html', 
            num_elements=num_elements,
            num_processes=num_processes,
            single_time=f"{single_time:.4f}",
            parallel_time=f"{parallel_time:.4f}",
            speedup=f"{speedup:.2f}",
            efficiency=f"{efficiency:.2f}"
        )

    except ValueError:
        return jsonify({'error': 'Invalid input values'})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
