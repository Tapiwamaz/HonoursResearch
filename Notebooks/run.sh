#!/bin/bash
#SBATCH --job-name=centroids
#SBATCH --output=centroids.log
#SBATCH --error=centroids_err.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # Using more cores for biggpu            
#SBATCH --partition=stampede
#SBATCH --time=3-00:00:00      # 3 days max runtime 


python random.py


echo "Tapedza!!! Mwari Ngaakudzwe!"
