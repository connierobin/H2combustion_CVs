#from openmmtools import testsystems
from openmm import *
from openmm.app import *
from openmm.unit import *
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from openmm.app import PDBFile, ForceField
from openmm import *
from openmm.unit import *


print('finished importing')

# Load the PDB file
pdb = PDBFile('alanine-dipeptide.pdb')

# Create the system
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

# Assign positions
positions = pdb.positions
topology = pdb.topology



#print('finished importing')

# Step 1: Set up the alanine dipeptide system and simulation
#testsystem = testsystems.AlanineDipeptideVacuum()
#system = testsystem.system
#topology = testsystem.topology
#positions = testsystem.positions

print('set up alanine dipeptide structure')

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
simulation.reporters.append(StateDataReporter(stdout, 100, step=True, time=True,
                                              potentialEnergy=True, temperature=True,
                                              progress=True, remainingTime=True,
                                              speed=True, totalSteps=10000, separator="\t"))

# Also add reporters to save data for analysis
simulation.reporters.append(StateDataReporter('energies.log', 10, step=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(DCDReporter('trajectory.dcd', 10))  # Save trajectory

# Run the simulation (10,000 steps as an example)
simulation.step(10000)

# Step 3: Load trajectory and calculate \phi and \psi angles
traj = md.load('trajectory.dcd', top=md.Topology.from_openmm(topology))
phi_indices, psi_indices = md.compute_phi(traj), md.compute_psi(traj)
phi_angles = md.compute_dihedrals(traj, phi_indices).flatten()
psi_angles = md.compute_dihedrals(traj, psi_indices).flatten()

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
        data = line.split()
        steps.append(int(data[0]))
        potential_energy.append(float(data[1]))
        temperature.append(float(data[2]))

# Plot potential energy over time
plt.figure(figsize=(6, 5))
plt.plot(steps, potential_energy, label='Potential Energy (kJ/mol)')
plt.xlabel("Simulation Steps")
plt.ylabel("Potential Energy (kJ/mol)")
plt.title("Potential Energy Over Time")
plt.legend()
plt.show()

