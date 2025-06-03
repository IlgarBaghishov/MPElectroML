#!/bin/sh
#SBATCH -N 1
#SBATCH -n 72
#SBATCH -o ll_out
#SBATCH -p gh
#SBATCH -t 48:00:00
#SBATCH -A CHE21006

module unload xalt

python -u run_analysis.py
