#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=32:mem=62gb
module load anaconda3/personal
echo "About plot t-SNE"
python3 $HOME/Code/comb/tsne_plot.py
echo "Finished plotting"
# end of file
