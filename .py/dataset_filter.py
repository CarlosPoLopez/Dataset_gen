import numpy as np
import os
import glob
from collections import deque

# ═══════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════
U_MAX            = 0.877
TOLERANCIA       = 0.15
UMBRAL_AMARILLO  = U_MAX - TOLERANCIA   # 0.727

MIN_CELDAS       = 500
MIN_AREA_AGUJERO = 200   # píxeles²; ignora artefactos pequeños del fondo

DIR         = 'Dataset_Matrices/'
DIR_PLOTS   = 'Dataset_Plots/'

# ═══════════════════════════════════════════════
#  FUNCIÓN 1 — CAMINO ROTO / NO CONECTADO
# ═══════════════════════════════════════════════

def camino_es_valido(solucion, laberinto):
    """
    Devuelve (bool, int) → (válido, nº celdas activas).
    Criterio: las celdas activas (>= UMBRAL_AMARILLO) deben formar
    un componente conexo que toque ambas esquinas del laberinto.
    """
    N = solucion.shape[0] - 1
    activas   = solucion >= UMBRAL_AMARILLO
    n_activas = int(np.sum(activas))

    if n_activas < MIN_CELDAS:
        return False, n_activas

    inicio   = (5, 5)
    fin_zona = (N - 85, N - 85)

    if not activas[inicio]:
        return False, n_activas

    visitados        = np.zeros_like(activas, dtype=bool)
    visitados[inicio] = True
    cola             = deque([inicio])
    alcanza_fin      = False

    while cola:
        x, y = cola.popleft()
        if x >= fin_zona[0] and y >= fin_zona[1]:
            alcanza_fin = True
            break
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx <= N and 0 <= ny <= N:
                if activas[nx, ny] and not visitados[nx, ny]:
                    visitados[nx, ny] = True
                    cola.append((nx, ny))

    return alcanza_fin, n_activas


# ═══════════════════════════════════════════════
#  FUNCIÓN 2 — DOBLE CAMINO (BUCLE/LOOP)
# ═══════════════════════════════════════════════

def camino_tiene_doble(solucion):
    """
    Devuelve (bool, int) → (tiene_doble, nº agujeros detectados).

    Principio topológico: un camino simple es una línea abierta
    (sin ciclos). Un doble camino forma un bucle cerrado, que al
    binarizar la imagen deja uno o más "agujeros" encerrados en el
    fondo oscuro. Se cuentan esas regiones de fondo aisladas.

    Pasos:
      1. Umbralizar → máscara binaria del camino.
      2. Invertir   → regiones de fondo.
      3. Etiquetar  → componentes conexos del fondo (4-conn).
      4. El exterior (mayor área) se descarta; el resto son agujeros.
      5. Si hay ≥ 1 agujero con área > MIN_AREA_AGUJERO → doble camino.
    """
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_closing, disk

    # 1. Binarizar
    binaria = (solucion >= UMBRAL_AMARILLO).astype(np.uint8)

    # 2. Cierre morfológico suave para cerrar micro-gaps sin alterar topología
    binaria = binary_closing(binaria, disk(3)).astype(np.uint8)

    # 3. Invertir y etiquetar fondo
    fondo   = 1 - binaria
    etiq    = label(fondo, connectivity=1)   # 4-conectividad
    props   = regionprops(etiq)

    if not props:
        return False, 0

    # 4. Descartar la región exterior (la de mayor área)
    area_max = max(r.area for r in props)
    agujeros = [r for r in props if r.area < area_max and r.area >= MIN_AREA_AGUJERO]

    return len(agujeros) > 0, len(agujeros)


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
    motivos    = {'roto': 0, 'doble': 0}

    for f in ficheros:
        job_id    = os.path.basename(f).replace('muestra_', '').replace('.npz', '')
        datos     = np.load(f)
        solucion  = datos['solucion']
        laberinto = datos['laberinto']

        # ── Criterio 1: camino roto / no conectado ──────────────────
        valido, n_celdas = camino_es_valido(solucion, laberinto)

        # ── Criterio 2: doble camino (solo si el 1 pasa) ────────────
        tiene_doble, n_agujeros = (False, 0)
        if valido:
            tiene_doble, n_agujeros = camino_tiene_doble(solucion)

        # ── Clasificación final ──────────────────────────────────────
        es_bueno = valido and not tiene_doble

        npy_asociados = npy_de_muestra(job_id)

        if es_bueno:
            print(f"  [{job_id:>6}]  OK          {n_celdas:>7} celdas")
            validos += 1

            # Los .npy intermedios ya no hacen falta si el .npz es bueno
            log_npy = []
            for path in npy_asociados:
                borrar_fichero(path, borrar, log_npy)
            accion = 'borrado' if borrar else 'a borrar'
            for estado, p in log_npy:
                if estado != 'NO EXISTE':
                    print(f"           ↳ .npy {accion}  : {os.path.basename(p)}")

        else:
            # Determinar motivo para el log
            if not valido:
                motivo_str = f"ROTO        {n_celdas:>7} celdas"
                motivos['roto'] += 1
            else:
                motivo_str = f"DOBLE CAMINO  {n_agujeros} bucle{'s' if n_agujeros > 1 else ''}"
                motivos['doble'] += 1

            eliminados += 1
            print(f"  [{job_id:>6}]  {motivo_str}{'  → BORRADO' if borrar else ''}")

            # Borrar el .npz
            if borrar:
                os.remove(f)
            else:
                print(f"           ↳ .npz a borrar: {os.path.basename(f)}")

            # Borrar .npy asociados
            log_npy = []
            for path in npy_asociados:
                borrar_fichero(path, borrar, log_npy)

            # Borrar plots (ida y vuelta)
            log_plots = []
            for path in plots_fallidos(job_id):
                borrar_fichero(path, borrar, log_plots)

            accion = 'borrado' if borrar else 'a borrar'
            for _, p in log_npy:
                print(f"           ↳ .npy {accion}  : {os.path.basename(p)}")
            for estado, p in log_plots:
                if estado != 'NO EXISTE':
                    print(f"           ↳ .png {accion}  : {os.path.basename(p)}")

    print(f"\n{'='*65}")
    print(f"  Analizados   : {validos + eliminados}")
    print(f"  Válidos      : {validos}")
    print(f"  Fallidos     : {eliminados}{'  (borrados)' if borrar else '  (conservados en modo seguro)'}")
    if eliminados:
        print(f"    · Camino roto   : {motivos['roto']}")
        print(f"    · Doble camino  : {motivos['doble']}")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    borrar = '--borrar' in sys.argv
    run(borrar=borrar)