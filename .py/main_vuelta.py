import numpy as np
import time
import os
from bucle import FCTS

import sys
job_id = sys.argv[1] if len(sys.argv) > 1 else os.getenv('SLURM_ARRAY_TASK_ID', '999')

N = 500
T = 200000

deltat = 0.002
deltax = 0.25      
epsilon, alpha, beta, Du, Dv = 10.0, 5.0, 0.1, 1, 5.0

""" u_min, u_max, v_min, v_max = -0.6505, 0.7526, -0.3752, 0.3263 """
u_min, u_max, v_min, v_max = -0.6505, 0.877, -0.3752, 0.197


F_matriz = np.load(f'Dataset_Matrices/matriz_F_laberinto_{job_id}.npy')
u_inicial = np.load(f'Dataset_Matrices/estado_u_ida_{job_id}.npy')
v_inicial = np.load(f'Dataset_Matrices/estado_v_ida_{job_id}.npy')


inicio = time.time()

u_final , v_final = FCTS(u_inicial, v_inicial, u_max, v_max, u_min, v_min, deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F_matriz, job_id)

fin = time.time()
print(f'Simulación terminada en {round((fin-inicio)/60, 2)}minutos')

#Guardar datos (zip .npy)
np.savez_compressed(f'Dataset_Matrices/muestra_{job_id}.npz',
                    laberinto=F_matriz,        
                    solucion=u_final)