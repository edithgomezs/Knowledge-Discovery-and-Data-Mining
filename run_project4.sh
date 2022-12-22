#!/bin/bash
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --mem=0
RUNPATH=/home/ashrestha4/CS548
cd $RUNPATH
source environment_folder/bin/activate
python3 clean_data.py