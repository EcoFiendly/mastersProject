#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=8:mem=16gb
module load anaconda3/personal
echo "About to optimize model"
python3 $HOME/Code/pop/popModOptHPCarray.py
echo "Finished optimization"
# end of file