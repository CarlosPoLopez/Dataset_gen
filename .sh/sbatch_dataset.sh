#!/bin/sh
#SBATCH -J fhn_dataset
#SBATCH -o Logs_SLURM/salida_%A_%a.o
#SBATCH -e Logs_SLURM/error_%A_%a.e
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 8:00:00
#SBATCH --array=1-20

module purge
module load cesga/system miniconda3

ID=$(python -c "import time; print($SLURM_ARRAY_TASK_ID + int(time.time()) % 100000)")

echo "Generando laberinto $ID..."
python main_ida.py $ID && python main_vuelta.py $ID