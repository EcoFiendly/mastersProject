#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=8:mem=32gb
module load anaconda3/personal
echo "About to optimize model"
python3 $HOME/Code/modelOptimization.py
echo "Finished optimization"
# end of file
