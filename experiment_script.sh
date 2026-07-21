#!/bin/bash
#SBATCH --account=XXXXXX   # Choose the project to be billed
#SBATCH --time=00:30:00             # Maximum duration of the job. Upper limit depends on partition.
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks. Upper limit depends on partition.
#SBATCH --cpus-per-task=20           # How many processors work on one task. Upper limit depends on number of CPUs per node.
#SBATCH --mem-per-cpu=2G            # Minimum memory required per usable allocated CPU.  Default units are megabytes.
#SBATCH --partition=small            # Which queue to use. Defines maximum time, memory, tasks, nodes and local storage for job

srun /projappl/XXXXXX/DESDEO/DESDEO/.venv/bin/python main.py