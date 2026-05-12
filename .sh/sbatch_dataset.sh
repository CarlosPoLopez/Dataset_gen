#!/bin/sh
#SBATCH -J fhn_dataset
#SBATCH -o Logs_SLURM/salida_%A_%a.o
#SBATCH -e Logs_SLURM/error_%A_%a.e
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 8:00:00
#SBATCH --array=1-50%30

module purge
module load cesga/system miniconda3

ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Generando laberinto $ID..."
python main_ida.py $ID && python main_vuelta.py $ID