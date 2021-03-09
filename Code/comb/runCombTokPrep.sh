#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=8:mem=16gb
module load anaconda3/personal
echo "Tokenization"
python3 $HOME/Code/comb/combTokPrepHPC.py
echo "Done"
# end of file