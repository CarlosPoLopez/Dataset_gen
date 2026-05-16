import numpy as np
import matplotlib.pyplot as plt


# ── Funciones originales ────────────────────────────────────────────────────

def generar_laberinto(L, l, n, grosor=4):
    muro = 5
    A = np.zeros((L, L))
    A[:, :muro] = 1
    A[L-muro:, :] = 1
    A[:, L-muro:] = 1
    A[:muro, :] = 1
    for i in range(0, L - l, l):
        for j in range(0, L - l, l):
            n1 = np.random.rand()
            n2 = np.random.rand()
            if n1 > n:
                A[i:i+l+grosor, j+l:j+l+grosor] = 1
            if n2 > n:
                A[i+l:i+l+grosor, j:j+l+grosor] = 1
    return A

def generar_laberinto_1():
    dimx = 501
    canal = 55
    aa = np.zeros((dimx, dimx))
    for i in range(canal, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            aa[i-1:i+2, j:j+canal] = np.round(0.25 + np.random.rand())
    aa[:, :3] = 1
    aa[:, -3:] = 1
    aa[:3, :] = 1
    aa[-3:, :-canal] = 1
    return aa

def generar_laberinto_2():
    dimx = 501
    canal = 55
    extra = 5
    peso = -0.30
    aa = np.zeros((dimx, dimx))
    for i in range(0, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            if np.round(peso + np.random.rand()) == 1:
                aa[i:i+3, j:j+canal+extra] = 1
            if np.round(peso + np.random.rand()) == 1:
                aa[i:i+canal+extra, j:j+3] = 1
    aa[:, :3] = 1
    aa[:, -3:] = 1
    aa[:3, :] = 1
    aa[-3:, :-canal] = 1
    return aa

def generar_laberinto_3():
    dimx = 501
    canal = 65
    extra = 0
    peso = -0.2
    aa = np.zeros((dimx, dimx))
    numerosi = np.zeros((dimx, dimx))
    numerosd = np.zeros((dimx, dimx))
    numerosr = np.zeros((dimx, dimx))
    numerosb = np.zeros((dimx, dimx))
    for i in range(0, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            contador = 6
            while contador > 2:
                numerosi[i, j] = np.round(peso + np.random.rand())
                numerosd[i, j] = np.round(peso + np.random.rand())
                numerosr[i, j] = np.round(peso + np.random.rand())
                numerosb[i, j] = np.round(peso + np.random.rand())
                contador = numerosi[i,j]+numerosd[i,j]+numerosr[i,j]+numerosb[i,j]
    for i in range(0, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            if numerosi[i, j] == 1:
                aa[i:i+4, j:j+canal+extra] = 1
            if numerosb[i, j] == 1:
                aa[i:i+canal+extra, j:j+4] = 1
            if numerosd[i, j] == 1:
                aa[i+canal:i+canal+4, j:j+canal+extra] = 1
            if numerosr[i, j] == 1:
                aa[i:i+canal+extra, j+canal:j+canal+4] = 1
    aa[:, :3] = 1
    aa[:, -3:] = 1
    aa[:3, :] = 1
    aa[-3:, :] = 1
    return aa

def generar_laberinto_4():
    dimx = 501
    canal = 50
    extra = 1
    peso = 0.50
    aa = np.zeros((dimx, dimx))
    for i in range(0, dimx - 2*canal, 2*canal):
        for j in range(0, dimx - 2*canal, 2*canal):
            rand_val = np.round(peso + 2 * np.random.rand())
            if rand_val == 1:
                aa[i:i+3, j:j+2*canal+extra] = 1
                aa[i+canal:i+3+canal, j:j+2*canal+extra] = 1
            elif rand_val == 2:
                aa[i:i+2*canal+extra, j:j+3] = 1
                aa[i:i+2*canal+extra, j+canal:j+3+canal] = 1
    aa[:, :3] = 1
    aa[:, -3:] = 1
    aa[:3, :] = 1
    aa[-3:, :] = 1
    return aa


# ── Utilidad: calcular dimx exacto ──────────────────────────────────────────

def dimx_exacto(nc, canal, grosor_pared=3):
    """
    Devuelve el dimx exacto para que no haya offset ni bordes desiguales.

    Parámetros
    ----------
    nc           : número de celdas por lado
    canal        : tamaño de cada celda en píxeles
    grosor_pared : grosor de las paredes (por defecto 3)

    Ejemplo
    -------
    canal = 100
    gp    = 3
    dimx  = dimx_exacto(nc=9,  canal=canal, grosor_pared=gp)  # → 930
    dimx  = dimx_exacto(nc=15, canal=canal, grosor_pared=gp)  # → 1548
    lab   = generar_laberinto_perfecto(dimx=dimx, canal=canal, grosor_pared=gp)
    """
    return nc * (canal + grosor_pared) + grosor_pared


# ── Nueva función: laberinto perfecto sin islas ─────────────────────────────

def generar_laberinto_perfecto(dimx=501, canal=50, grosor_pared=3, semilla=None):
    """
    Genera un laberinto perfecto mediante backtracking DFS.

    Propiedades garantizadas:
      - Sin islas: toda pared tiene al menos un extremo tocando el perímetro
        o conectado a otra pared que lo hace (el conjunto de paredes es conexo).
      - Sin bucles: hay exactamente un camino entre cualquier par de celdas.
      - Completamente resoluble: todas las celdas son alcanzables.

    Parámetros
    ----------
    dimx        : tamaño del array cuadrado de salida (píxeles).
    canal       : tamaño interior de cada celda en píxeles.
    grosor_pared: grosor de las paredes entre celdas (y del perímetro).
    semilla     : semilla opcional para np.random (reproducibilidad).

    Devuelve
    --------
    aa : np.ndarray de forma (dimx, dimx), dtype int
         1 = pared, 0 = pasillo/celda abierta.

    Cómo funciona
    -------------
    1. Se parte de una matriz completamente llena de paredes (todo 1).
    2. Se organiza una rejilla de celdas separadas por paredes de 'grosor_pared'.
    3. El DFS iterativo visita cada celda exactamente una vez y "talla"
       el pasillo hacia su vecino: abre tanto la celda destino como la
       franja de pared que las separa.
    4. Al ser un árbol generador (spanning tree), el grafo de paredes
       resultante es conexo y no hay paredes flotantes.
    """
    if semilla is not None:
        np.random.seed(semilla)

    gp = grosor_pared  # alias corto

    # Número de celdas que caben en cada dimensión
    nc = (dimx - gp) // (canal + gp)

    # Ajustar canal para ocupar el máximo espacio posible dentro de dimx,
    # minimizando el offset residual (≤ 1 px por lado en la práctica).
    canal = (dimx - gp * (nc + 1)) // nc

    # Offset residual para centrar los píxeles que puedan sobrar
    dimx_real = nc * (canal + gp) + gp
    offset = (dimx - dimx_real) // 2

    # Posición en píxeles de la esquina superior-izquierda de la celda (i, j)
    def px(k):
        return offset + gp + k * (canal + gp)

    # ── Inicializar: todo paredes ──────────────────────────────────────────
    aa = np.ones((dimx, dimx), dtype=int)

    # ── DFS iterativo ──────────────────────────────────────────────────────
    visited = np.zeros((nc, nc), dtype=bool)

    # Abrir celda inicial (0, 0)
    r0, c0 = 0, 0
    aa[px(r0):px(r0)+canal, px(c0):px(c0)+canal] = 0
    visited[r0, c0] = True

    stack = [(r0, c0)]
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        r, c = stack[-1]

        # Vecinos no visitados
        vecinos = [
            (r + dr, c + dc)
            for dr, dc in direcciones
            if 0 <= r + dr < nc and 0 <= c + dc < nc
            and not visited[r + dr, c + dc]
        ]

        if vecinos:
            # Elegir vecino al azar
            nr, nc_ = vecinos[np.random.randint(len(vecinos))]

            # ── Abrir la pared entre (r,c) y (nr,nc_) ──────────────────
            if nr == r:          # misma fila → pared vertical entre columnas
                jmin = min(c, nc_)
                # franja de columnas entre las dos celdas
                aa[px(r):px(r)+canal, px(jmin)+canal:px(jmin)+canal+gp] = 0
            else:                # misma columna → pared horizontal entre filas
                imin = min(r, nr)
                aa[px(imin)+canal:px(imin)+canal+gp, px(c):px(c)+canal] = 0

            # ── Abrir celda destino ─────────────────────────────────────
            aa[px(nr):px(nr)+canal, px(nc_):px(nc_)+canal] = 0
            visited[nr, nc_] = True
            stack.append((nr, nc_))
        else:
            stack.pop()   # retroceder (backtrack)

    # ── Forzar perímetro sellado ───────────────────────────────────────────
    aa[:gp, :]  = 1
    aa[-gp:, :] = 1
    aa[:, :gp]  = 1
    aa[:, -gp:] = 1

    return aa


# ── Demo ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    #lab = generar_laberinto(L=501, l=65, n=0.67, grosor=4)
    #lab = generar_laberinto_2(L=501, canal=55, peso=-0.3, grosor = 3)
    canal = 60
    gp    = 5
    dimx  = dimx_exacto(nc=9, canal=canal, grosor_pared=gp)  
    print(dimx)
    lab_perfecto = generar_laberinto_perfecto(dimx=dimx, canal=canal, grosor_pared=gp, semilla=42)
    plt.pcolormesh(lab_perfecto, cmap='binary')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


