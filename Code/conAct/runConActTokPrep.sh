#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=8:mem=12gb
module load anaconda3/personal
echo "Tokenization"
python3 $HOME/Code/conAct/conActTokenPrepHPC.py
echo "Done"
# end of file