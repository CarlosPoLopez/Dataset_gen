#!/bin/sh
#SBATCH -J fhn_completo
#SBATCH -o Logs_SLURM/salida_%A_%a.o    
#SBATCH -e Logs_SLURM/error_%A_%a.e  
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 04:00:00

module purge
module load cesga/system miniconda3

ID=$(python -c "import time; print(int(time.time()) % 100000)")
echo "--- ID de este laberinto: $ID ---"

echo "--- Iniciando Fase 1: IDA ---"
python main_ida.py $ID && \
echo "--- Fase 1 completada. Iniciando Fase 2: VUELTA ---" && \
python main_vuelta.py $ID

echo "--- Simulación completa ---"