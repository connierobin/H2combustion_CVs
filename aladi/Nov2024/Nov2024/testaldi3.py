import sys
import csv
from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np
import matplotlib.pyplot as plt

print('finished importing')

# Load the PDB file
pdb = PDBFile('alanine-dipeptide.pdb')

# Create the system
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

# Assign positions
positions = pdb.positions
topology = pdb.topology
print('set up alanine dipeptide structure')

num_steps = 1000000
temperature = 300 * kelvin
friction = 1 / picosecond
step_size = 2 * femtoseconds
integrator = LangevinIntegrator(temperature, friction, step_size)

simulation = Simulation(topology, system, integrator)
simulation.context.setPositions(positions)

print('set up system and simulation')

# Step 2: Minimization and Equilibration
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)

# Add a reporter to show progress in the terminal
simulation.reporters.append(StateDataReporter(sys.stdout, 100, step=True, time=True,
                                              potentialEnergy=True, temperature=True,
                                              progress=True, remainingTime=True,
                                              speed=True, totalSteps=num_steps, separator="\t"))

# Also add reporters to save data for analysis
simulation.reporters.append(StateDataReporter('energies.log', 10, step=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(DCDReporter('trajectory.dcd', 10))  # Save trajectory

# Function to compute torsion angles
def compute_torsion(context, torsion_indices):
    """
    Compute torsion angles (dihedrals) for given indices in a simulation context.

    Parameters:
    - context: OpenMM Context object.
    - torsion_indices: List of 4-tuples, where each tuple specifies the atom indices for a torsion.

    Returns:
    - torsion_angles: List of torsion angles in radians.
    """
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True)

    torsion_angles = []
    for indices in torsion_indices:
        p1, p2, p3, p4 = [positions[i] for i in indices]
        
        # Vectors along the bonds
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # Normalize b2 for perpendicular component
        b2_norm = b2 / np.linalg.norm(b2)
        
        # Perpendicular vectors
        v1 = b1 - np.dot(b1, b2_norm) * b2_norm
        v2 = b3 - np.dot(b3, b2_norm) * b2_norm
        
        # Calculate angle using arctan2 for proper sign
        x = np.dot(v1, v2)
        y = np.dot(np.cross(b2_norm, v1), v2)
        angle = np.arctan2(y, x)
        torsion_angles.append(angle)
    
    return torsion_angles

phi_angles = []
psi_angles = []

# Atom indices for phi and psi dihedrals in alanine dipeptide
phi_indices = (4, 6, 8, 14)  # Backbone atoms defining phi angle
psi_indices = (6, 8, 14, 16)  # Backbone atoms defining psi angle

# Run the simulation (10,000 steps as an example)
for step in range(num_steps):
    simulation.step(1)  # Advance one step

    if step % 10 == 0:  # Calculate every 10 steps (matches DCDReporter frequency)
        # Get positions from the current state
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)

        # Compute phi and psi angles
        phi_angle = compute_torsion(simulation.context, [phi_indices])[0]
        psi_angle = compute_torsion(simulation.context, [psi_indices])[0]
        phi_angles.append(phi_angle)
        psi_angles.append(psi_angle)

print('done simulating, now record the data and plot it')

# Save phi and psi angles to a CSV file
with open('phi_psi_angles.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['phi', 'psi'])  # Header
    writer.writerows(zip(phi_angles, psi_angles))

# Step 4: Ramachandran Plot (phi vs psi angle distribution)
plt.figure(figsize=(6, 5))
plt.hist2d(phi_angles, psi_angles, bins=50, cmap='Blues')
plt.colorbar(label="Density")
plt.xlabel("Phi (radians)")
plt.ylabel("Psi (radians)")
plt.title("Ramachandran Plot")
plt.show()

# Step 5: Free Energy Surface Plot
# Bin the data and calculate free energy as -kT * ln(P)
H, xedges, yedges = np.histogram2d(phi_angles, psi_angles, bins=50)
prob_density = H / np.sum(H)  # Normalize to get probability density
free_energy = -np.log(prob_density + 1e-8)  # Add small value to avoid log(0)
free_energy -= np.min(free_energy)  # Shift min to zero
# Save free energy data to a CSV file
np.savetxt('free_energy.csv', free_energy, delimiter=',')
# Save the histogram bin edges
np.savetxt('phi_edges.csv', xedges, delimiter=',')
np.savetxt('psi_edges.csv', yedges, delimiter=',')

plt.figure(figsize=(6, 5))
plt.imshow(free_energy.T, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label="Free Energy (kT)")
plt.xlabel("Phi (radians)")
plt.ylabel("Psi (radians)")
plt.title("Free Energy Surface")
plt.show()

# Step 6: Energy Over Time Plot
# Load energies from log file
steps, potential_energy, temperature = [], [], []
with open('energies.log') as f:
    for line in f:
        if line.startswith("#") or line.startswith("Step"):
            continue
        #data = line.split()
        values = line.split(",")
        steps.append(int(values[0]))
        potential_energy.append(float(values[1]))
        temperature.append(float(values[2]))

# Save energy over time to a CSV file
with open('energy_over_time.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'potential_energy', 'temperature'])  # Header
    writer.writerows(zip(steps, potential_energy, temperature))

# Plot potential energy over time
plt.figure(figsize=(6, 5))
plt.plot(steps, potential_energy, label='Potential Energy (kJ/mol)')
plt.xlabel("Simulation Steps")
plt.ylabel("Potential Energy (kJ/mol)")
plt.title("Potential Energy Over Time")
plt.legend()
plt.show()

