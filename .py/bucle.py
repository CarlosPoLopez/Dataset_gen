import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from numba import njit, prange


@njit(parallel=True, cache=True)
def _paso_dufort_frankel(u, v, u_old, v_old, u_new, v_new,
                          F, Cu, Cv, deltat, epsilon, alpha, beta, N):
    for i in prange(1, N):
        for j in range(1, N):
            su = u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1]
            sv = v[i+1, j] + v[i-1, j] + v[i, j+1] + v[i, j-1]
            funu = epsilon * (u_old[i,j] - u_old[i,j]**3 - v_old[i,j] + F[i,j])
            funv = u_old[i,j] - alpha * v_old[i,j] + beta
            u_new[i,j] = (2.0*deltat*funu + u_old[i,j]*(1.0 - 2.0*Cu) + Cu*su) / (1.0 + 2.0*Cu)
            v_new[i,j] = (2.0*deltat*funv + v_old[i,j]*(1.0 - 2.0*Cv) + Cv*sv) / (1.0 + 2.0*Cv)
    return u_new, v_new


@njit(parallel=True, cache=True)
def _paso_fcts(u, v, u_new, v_new, F,
               deltat, inv_dx2, epsilon, alpha, beta, Du, Dv, N):
    for i in prange(1, N):
        for j in range(1, N):
            lap_u = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4.0*u[i,j]) * inv_dx2
            lap_v = (v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1] - 4.0*v[i,j]) * inv_dx2
            u_new[i,j] = u[i,j] + deltat * (epsilon*(u[i,j] - u[i,j]**3 - v[i,j] + F[i,j]) + Du*lap_u)
            v_new[i,j] = v[i,j] + deltat * (u[i,j] - alpha*v[i,j] + beta + Dv*lap_v)
    return u_new, v_new


def _guardar_plot(u, F, N, ruta):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(u, cmap='magma', vmin=-1.5, vmax=1.5, origin='lower')
    overlay = np.zeros((N+1, N+1, 4))
    overlay[F < 0] = [0.8, 0.8, 0.8, 0.8]
    ax.imshow(overlay, origin='lower')
    plt.savefig(ruta)
    plt.close(fig)


def dufort_frankel(u, v, u_max, v_max, u_min, v_min,
                   deltat, deltax, N, T,
                   epsilon, alpha, beta, Du, Dv, F,
                   job_id):

    u     = np.ascontiguousarray(u, dtype=np.float64)
    v     = np.ascontiguousarray(v, dtype=np.float64)
    F     = np.ascontiguousarray(F, dtype=np.float64)

    u_om1 = u.copy()
    v_om1 = v.copy()
    u_n   = u.copy()
    v_n   = v.copy()
    u_np1 = u.copy()
    v_np1 = v.copy()

    Cu = (2.0 * deltat * Du) / (deltax**2)
    Cv = (2.0 * deltat * Dv) / (deltax**2)

    _guardar_plot(u_n, F, N, f'Dataset_Plots/ida_{job_id}.png')

    for t in range(T):

        u_np1, v_np1 = _paso_dufort_frankel(
            u_n, v_n, u_om1, v_om1, u_np1, v_np1,
            F, Cu, Cv, deltat, epsilon, alpha, beta, N)

        for arr in (u_np1, v_np1):
            arr[0,  :] = arr[1,  :]
            arr[-1, :] = arr[-2, :]
            arr[:,  0] = arr[:,  1]
            arr[:, -1] = arr[:, -2]

        np.copyto(u_om1, u_n)
        np.copyto(v_om1, v_n)
        np.add(u_n, u_np1, out=u_n);  u_n *= 0.5
        np.add(v_n, v_np1, out=v_n);  v_n *= 0.5

    return u_n, v_n


def FCTS(u, v, u_max, v_max, u_min, v_min,
         deltat, deltax, N, T,
         epsilon, alpha, beta, Du, Dv, F,
         job_id):

    u     = np.ascontiguousarray(u, dtype=np.float64)
    v     = np.ascontiguousarray(v, dtype=np.float64)
    F     = np.ascontiguousarray(F, dtype=np.float64)
    u_n   = u.copy()
    v_n   = v.copy()
    u_np1 = u.copy()
    v_np1 = v.copy()

    inv_dx2 = 1.0 / (deltax**2)

    for t in range(T):

        u_np1, v_np1 = _paso_fcts(
            u_n, v_n, u_np1, v_np1,
            F, deltat, inv_dx2, epsilon, alpha, beta, Du, Dv, N)

        u_np1[5:55,     5:55]      = u_max
        u_np1[N-55:N-5, N-55:N-5] = u_max
        v_np1[5:55,     5:55]      = v_max
        v_np1[N-55:N-5, N-55:N-5] = v_max

        u_n, u_np1 = u_np1, u_n
        v_n, v_np1 = v_np1, v_n

    _guardar_plot(u_n, F, N, f'Dataset_Plots/vuelta_{job_id}.png')

    return u_n, v_n