import matplotlib.pyplot as plt
import numpy as np
import csv

# Load \(\phi\) and \(\psi\) angles
phi_angles = []
psi_angles = []
with open('phi_psi_angles.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        phi_angles.append(float(row[0]))
        psi_angles.append(float(row[1]))

basin_1 = lambda phi, psi: 1.5 < phi < 2.1 and -2.8 < psi < -1.6  # Define basin 1 bounds
basin_2 = lambda phi, psi: 0.2 < phi < 0.8 and -0.7 < psi < 0.0  # Define basin 2 bounds
basin_labels = []
for phi, psi in zip(phi_angles, psi_angles):
    if basin_1(phi, psi):
        basin_labels.append(1)  # Basin 1
    elif basin_2(phi, psi):
        basin_labels.append(2)  # Basin 2
    else:
        basin_labels.append(0)  # Outside main basins

# Generate Ramachandran Plot
# Convert lists to numpy arrays
phi_angles = np.array(phi_angles)
psi_angles = np.array(psi_angles)
# Extend phi and psi for wrapping
phi_wrapped = np.concatenate([phi_angles, phi_angles])
psi_wrapped = np.concatenate([psi_angles, psi_angles + 2 * np.pi])
H, xedges, yedges = np.histogram2d(phi_wrapped, psi_wrapped, bins=50, range=[[-np.pi, np.pi], [-np.pi, 2 * np.pi]])
plt.figure(figsize=(6, 5))
plt.hist2d(phi_wrapped, psi_wrapped, bins=50, cmap='Blues')
plt.colorbar(label="Density")
plt.xlabel("Phi (radians)")
plt.ylabel("Psi (radians)")
plt.title("Ramachandran Plot")
plt.show()

# Load free energy data
free_energy = np.loadtxt('free_energy.csv', delimiter=',')
# Load the histogram bin edges
phi_edges = np.loadtxt('phi_edges.csv', delimiter=',')
psi_edges = np.loadtxt('psi_edges.csv', delimiter=',')
# Extend phi edges and data for wrapping
H_extended = np.concatenate([H, H], axis=0)  # Duplicate the histogram along the phi-axis
phi_edges_extended = np.concatenate([xedges[:-1], xedges[:-1] + 2 * np.pi])  # Shift phi edges
psi_edges = yedges  # Psi edges remain unchanged
# Adjust the plotting range
phi_range = (-np.pi, 2 * np.pi)  # Extend phi to wrap around
psi_range = (-np.pi, np.pi)      # Keep psi within standard range

# Generate Free Energy Surface Plot
plt.figure(figsize=(6, 5))
plt.imshow(free_energy.T, extent=[phi_edges[0], phi_edges[-1], psi_edges[0], psi_edges[-1]],
           origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label="Free Energy (kT)")
plt.xlabel("Phi (radians)")
plt.ylabel("Psi (radians)")
plt.title("Free Energy Surface")
plt.show()

# Load energy over time data
steps, potential_energy, temperature = [], [], []
with open('energy_over_time.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        steps.append(int(row[0]))
        potential_energy.append(float(row[1]))
        temperature.append(float(row[2]))

# Plot Potential Energy Over Time
plt.figure(figsize=(6, 5))
start_idx = 10000
end_idx = 20000
print(f'Plotting potential energy from step {start_idx} to step {end_idx} with max value either {len(steps)} or {steps[-1]}')
# old slow way
#for i in range(len(steps)):
#    if basin_labels[i] == 1:
#        plt.axvspan(steps[i], steps[i + 1] if i + 1 < len(steps) else steps[i],
#                    color='yellow', alpha=0.1)
#    elif basin_labels[i] == 2:
#        plt.axvspan(steps[i], steps[i + 1] if i + 1 < len(steps) else steps[i],
#                    color='orange', alpha=0.1)
# new fast way
basin_colors = {1: 'yellow', 2: 'orange'}
groups = []  # To store ranges of steps in the same basin
current_basin = None
start_step = None
for i, label in enumerate(basin_labels):
    if label != current_basin:
        if current_basin is not None:
            groups.append((start_step, steps[i - 1], current_basin))  # Close the previous group
        current_basin = label
        start_step = steps[i]
if current_basin is not None:  # Close the last group
    groups.append((start_step, steps[-1], current_basin))
for start, end, basin in groups:
    if start >= steps[start_idx] and end <= steps[end_idx] and basin in basin_colors:
        plt.axvspan(max(start, steps[start_idx]), min(end, steps[end_idx]), color=basin_colors[basin], alpha=0.3, edgecolor=None, antialiased=False, lw=0)

plt.plot(steps[start_idx:end_idx], potential_energy[start_idx:end_idx], label=f'Potential Energy from Step {steps[start_idx]} to {steps[end_idx]} (kJ/mol)')
plt.xlabel("Simulation Steps")
plt.ylabel("Potential Energy (kJ/mol)")
plt.title("Potential Energy Over Time")
plt.legend()
plt.show()

