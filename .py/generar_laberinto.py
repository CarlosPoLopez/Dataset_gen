import numpy as np
import matplotlib.pyplot as plt


def _limpiar_generadores(A, N, margen=60):
    """
    Garantiza que las zonas de inicio y fin estén completamente despejadas.
    Se llama siempre al final de cada función generadora.
    El margen debe ser >= al tamaño del generador en main_ida/vuelta (55 por defecto).
    """
    A[5:5+margen, 5:5+margen]             = 0   # esquina inicio
    A[N-5-margen:N-5, N-5-margen:N-5]    = 0   # esquina fin
    return A


def generar_laberinto(L, l, n, grosor=4):
    N    = L - 1
    muro = 5
    A    = np.zeros((L, L))

    A[:, :muro]  = 1
    A[L-muro:,:] = 1
    A[:, L-muro:]= 1
    A[:muro, :]  = 1

    for i in range(0, L - l, l):
        for j in range(0, L - l, l):
            if np.random.rand() > n:
                A[i:i+l+grosor, j+l:j+l+grosor] = 1
            if np.random.rand() > n:
                A[i+l:i+l+grosor, j:j+l+grosor] = 1

    return _limpiar_generadores(A, N)


def generar_laberinto_3(L=501, canal=75, peso=-0.3):
    N    = L - 1
    dimx = L
    A    = np.zeros((dimx, dimx))

    numerosi = np.zeros((dimx, dimx))
    numerosd = np.zeros((dimx, dimx))
    numerosr = np.zeros((dimx, dimx))
    numerosb = np.zeros((dimx, dimx))

    for i in range(0, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            contador = 6
            while contador > 2:
                numerosi[i,j] = np.round(peso + np.random.rand())
                numerosd[i,j] = np.round(peso + np.random.rand())
                numerosr[i,j] = np.round(peso + np.random.rand())
                numerosb[i,j] = np.round(peso + np.random.rand())
                contador = numerosi[i,j] + numerosd[i,j] + numerosr[i,j] + numerosb[i,j]

    for i in range(0, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            if numerosi[i,j] == 1:
                A[i:i+4,          j:j+canal] = 1
            if numerosb[i,j] == 1:
                A[i:i+canal,      j:j+4]     = 1
            if numerosd[i,j] == 1:
                A[i+canal:i+canal+4, j:j+canal] = 1
            if numerosr[i,j] == 1:
                A[i:i+canal,      j+canal:j+canal+4] = 1

    A[:, :3]  = 1
    A[:, -3:] = 1
    A[:3, :]  = 1
    A[-3:, :] = 1

    return _limpiar_generadores(A, N)


if __name__ == '__main__':
    lab = generar_laberinto(L=501, l=65, n=0.67, grosor=4)
    plt.figure(figsize=(6,6))
    plt.pcolormesh(lab, cmap='binary')
    plt.axis('off')
    plt.tight_layout()
    plt.show()