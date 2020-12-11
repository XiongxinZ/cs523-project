#!/bin/bash -l

#$ -l h_rt=24:00:00
#$ -N create_data
#$ -j y
#$ -pe omp 2

source project_nnunet/bin/activate
module load python3/3.6.9
module load cuda/10.1
module load pytorch/1.3
cd ours

python create_data_structure.py
