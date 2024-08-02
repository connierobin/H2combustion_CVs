import os
import numpy as np

# File to keep track of the run number
run_number_file = 'run_number.txt'

# Initialize the run number
if not os.path.exists(run_number_file):
    with open(run_number_file, 'w') as f:
        f.write('0')

# Read the current run number
with open(run_number_file, 'r') as f:
    run_number = int(f.read().strip())

# Create a base directory for all simulation runs
base_dir = 'simulation_runs'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)