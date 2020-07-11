#!/bin/bash
# ------------------------------------------------------------------------------------------------------
# Coded by Joao Morado 18.06.2019
# Interface file used to run a SQM calculation
# ------------------------------------------------------------------------------------------------------
# Input variables
label=$1
n_atoms=$2
amber_input="sqm_${label}.in"
head_lines=$((${n_atoms} + 1))

# Output variables
amber_output="sqm_${label}.out"
potential_energy_output="epot_${label}"
forces_output="forces_${label}"


# Run SP calculation
sqm -O -i ${amber_input} -o ${amber_output}

# Extract Potential Energy
grep "QMMM: SCF Energy =" sqm_${label}.out | awk {'print $7'} > ${potential_energy_output}
grep -A${n_atoms} "QMMM: Forces on QM atoms from SCF calculation" sqm_${label}.out | head -n +${head_lines} | tail -n ${n_atoms} | awk {'print $4 "  " $5 "  " $6'} > ${forces_output}


