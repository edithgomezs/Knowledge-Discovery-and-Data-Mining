#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=128gb
RUNPATH=/home/dmhofmann/CS548_Proj4
cd $RUNPATH
source environment_folder/bin/activate
python3 rf.py