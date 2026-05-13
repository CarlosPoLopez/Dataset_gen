import numpy as np
import time
import os
from bucle import dufort_frankel
from generar_laberinto import generar_laberinto
from generar_laberinto import generar_laberinto_3
from generar_laberinto import generar_laberinto_perfecto
from collections import deque
import sys

def laberinto_tiene_solucion(matriz, start, end):

    """Comprueba rápidamente si existe un camino de 0s (pasillos) 
    entre el punto de inicio y el punto final.
    """
    N_filas, N_cols = matriz.shape
    visitados = set()
    cola = deque([start])
    visitados.add(start)

    # Movimientos permitidos: arriba, abajo, izquierda, derecha
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while cola:
        x, y = cola.popleft()

        # Si llegamos a la zona final, ¡es válido!
        if (x, y) == end:
            return True

        for dx, dy in direcciones:
            nx, ny = x + dx, y + dy
            # Comprobamos que no nos salimos del mapa y que es pasillo (suponiendo que 0 es pasillo)
            if 0 <= nx < N_filas and 0 <= ny < N_cols:
                if matriz[nx, ny] == 0 and (nx, ny) not in visitados:
                    visitados.add((nx, ny))
                    cola.append((nx, ny))

    # Si la cola se vacía y no hemos llegado, no hay solución
    return False


job_id = sys.argv[1] if len(sys.argv) > 1 else os.getenv('SLURM_ARRAY_TASK_ID', '999')
np.random.seed(int(job_id))

N = 500
L = N + 1
T = 780000 

#Genera laberintos hasta que encuentre uno con salida
intentos = 0
while True:
    intentos += 1
    #matriz_laberinto = generar_laberinto_3()
    matriz_laberinto = generar_laberinto(L, l=65, n=0.67, grosor=4)
    
    # Comprobamos un punto en la zona de inicio (ej. 30,30) y otro en el final (ej. N-30, N-30)
    # Suponemos que en matriz_laberinto el valor 0 es pasillo y 1 es pared.
    if laberinto_tiene_solucion(matriz_laberinto, start=(30, 30), end=(N-30, N-30)):
        break # Salimos del bucle con un laberinto bueno
    else:
        # Si no hay solución, el bucle repite y genera otro distinto silenciosamente
        pass


deltat = 0.002 
deltax = 0.09
epsilon, alpha, beta, Du, Dv, F_pasillo, F_pared = 10.0, 8.0, -0.33, 0.05, 4.0, 0, -3.0
F_matriz = np.where(matriz_laberinto == 1, F_pared, F_pasillo)

u_min, u_max, v_min, v_max = -0.9109, 0.9581, -0.1551, 0.0785   


u_inicial = np.full((N+1, N+1), u_min)
v_inicial = np.full((N+1,N+1), v_min)


u_inicial[5:55, 5:55] = u_max
v_inicial[5:55, 5:55] = v_max


inicio = time.time()

u_final , v_final = dufort_frankel(u_inicial, v_inicial, u_max, v_max, u_min, v_min, deltat, deltax, N, T, epsilon, alpha, beta, Du, Dv, F_matriz, job_id)

fin = time.time()
print(f'Simulación terminada en {round((fin-inicio)/60, 2)}minutos')

#Guardar datos
np.save(f'Dataset_Matrices/estado_u_ida_{job_id}.npy', u_final)
np.save(f'Dataset_Matrices/estado_v_ida_{job_id}.npy', v_final)
np.save(f'Dataset_Matrices/matriz_F_laberinto_{job_id}.npy', F_matriz)