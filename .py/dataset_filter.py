import numpy as np
import os
import glob

# ═══════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════
U_MAX            = 0.877
TOLERANCIA       = 0.15
UMBRAL_AMARILLO  = U_MAX - TOLERANCIA   # 0.727

MIN_CELDAS       = 500

DIR         = 'Dataset_Matrices/'
DIR_PLOTS   = 'Dataset_Plots/'

# ═══════════════════════════════════════════════
#  FUNCIÓN DE VALIDACIÓN
# ═══════════════════════════════════════════════

def camino_es_valido(solucion, laberinto):
    """
    Devuelve (bool, int) → (válido, nº celdas activas).
    Criterio: las celdas activas (>= UMBRAL_AMARILLO) deben formar
    un componente conexo que toque ambas esquinas del laberinto.
    """
    N = solucion.shape[0] - 1
    activas = solucion >= UMBRAL_AMARILLO
    n_activas = int(np.sum(activas))

    if n_activas < MIN_CELDAS:
        return False, n_activas

    from collections import deque
    inicio   = (5, 5)
    fin_zona = (N - 85, N - 85)

    if not activas[inicio]:
        return False, n_activas

    visitados = np.zeros_like(activas, dtype=bool)
    visitados[inicio] = True
    cola = deque([inicio])
    alcanza_fin = False

    while cola:
        x, y = cola.popleft()
        if x >= fin_zona[0] and y >= fin_zona[1]:
            alcanza_fin = True
            break
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx <= N and 0 <= ny <= N:
                if activas[nx, ny] and not visitados[nx, ny]:
                    visitados[nx, ny] = True
                    cola.append((nx, ny))

    return alcanza_fin, n_activas


# ═══════════════════════════════════════════════
#  ARCHIVOS ASOCIADOS A UN job_id
# ═══════════════════════════════════════════════

def npy_de_muestra(job_id):
    """Devuelve todos los .npy de Dataset_Matrices asociados a job_id."""
    patrones = [
        f'estado_u_ida_{job_id}.npy',
        f'estado_v_ida_{job_id}.npy',
        f'estado_u_vuelta_{job_id}.npy',
        f'estado_v_vuelta_{job_id}.npy',
        f'matriz_F_laberinto_{job_id}.npy',
    ]
    return [os.path.join(DIR, nombre) for nombre in patrones]


def plots_fallidos(job_id):
    """Devuelve los .png de Dataset_Plots asociados a job_id (solo si fallan)."""
    nombres = [
        f'ida_{job_id}.png',
        f'vuelta_{job_id}.png',
    ]
    return [os.path.join(DIR_PLOTS, nombre) for nombre in nombres]


def borrar_fichero(path, borrar, lista_borrados):
    """Intenta borrar un fichero (si borrar=True) y registra el resultado."""
    if os.path.exists(path):
        if borrar:
            os.remove(path)
        lista_borrados.append(('OK' if borrar else 'PENDIENTE', path))
    else:
        lista_borrados.append(('NO EXISTE', path))


# ═══════════════════════════════════════════════
#  ESCANEO Y FILTRADO
# ═══════════════════════════════════════════════

def run(borrar=False):
    ficheros = sorted(glob.glob(f'{DIR}muestra_*.npz'))

    if not ficheros:
        print("No se encontraron archivos muestra_*.npz en", DIR)
        return

    modo = "BORRADO REAL" if borrar else "MODO SEGURO (sin borrar)"
    print(f"\n{'='*65}")
    print(f"  {modo}")
    print(f"  Matrices : {DIR}")
    print(f"  Plots    : {DIR_PLOTS}")
    print(f"{'='*65}\n")

    validos    = 0
    eliminados = 0

    for f in ficheros:
        job_id = os.path.basename(f).replace('muestra_', '').replace('.npz', '')
        datos     = np.load(f)
        solucion  = datos['solucion']
        laberinto = datos['laberinto']

        valido, n_celdas = camino_es_valido(solucion, laberinto)

        # ── Archivos .npy asociados (se borran SIEMPRE, fallen o no) ──
        npy_asociados = npy_de_muestra(job_id)

        if valido:
            print(f"  [{job_id:>4}]  OK      {n_celdas:>7} celdas activas")
            validos += 1

            # Borrar .npy aunque el .npz sea válido
            log_npy = []
            for path in npy_asociados:
                borrar_fichero(path, borrar, log_npy)
            if borrar:
                borrados = [e for e in log_npy if e[0] == 'OK']
                if borrados:
                    for _, p in borrados:
                        print(f"           ↳ .npy borrado : {os.path.basename(p)}")
            else:
                existentes = [e for e in log_npy if e[0] == 'PENDIENTE']
                if existentes:
                    for _, p in existentes:
                        print(f"           ↳ .npy a borrar: {os.path.basename(p)}")

        else:
            print(f"  [{job_id:>4}]  FALLO   {n_celdas:>7} celdas activas  {'→ BORRADO' if borrar else ''}")
            eliminados += 1

            # Borrar el .npz
            if borrar:
                os.remove(f)
            else:
                print(f"           ↳ .npz a borrar: {os.path.basename(f)}")

            # Borrar .npy asociados
            log_npy = []
            for path in npy_asociados:
                borrar_fichero(path, borrar, log_npy)

            # Borrar plots (ida y vuelta) solo en caso de fallo
            log_plots = []
            for path in plots_fallidos(job_id):
                borrar_fichero(path, borrar, log_plots)

            accion = 'borrado' if borrar else 'a borrar'
            for _, p in log_npy:
                print(f"           ↳ .npy {accion}  : {os.path.basename(p)}")
            for _, p in [(s, p) for s, p in log_plots if s != 'NO EXISTE']:
                print(f"           ↳ .png {accion}  : {os.path.basename(p)}")

    print(f"\n{'='*65}")
    print(f"  Analizados : {validos + eliminados}")
    print(f"  Válidos    : {validos}")
    print(f"  Fallidos   : {eliminados}{'  (borrados)' if borrar else '  (conservados en modo seguro)'}")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    borrar = '--borrar' in sys.argv
    run(borrar=borrar)