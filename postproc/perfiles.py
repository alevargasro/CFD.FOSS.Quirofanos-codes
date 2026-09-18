#!/usr/bin/env python3
"""perfiles.py
===========
Seccion D del pipeline: lineas de perfil (functionObject 'sets', formato
'raw') -- video mp4 y capturas cada N segundos del perfil de velocidad (u
otro campo) a lo largo de cada linea. Depende de postproc_common.py y de
tener ffmpeg instalado.
"""
import argparse
import glob
import gzip
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
except ImportError:
    print("Falta matplotlib. Instala con: pip install matplotlib", file=sys.stderr)
    raise

from common import (
    find_time_dirs, open_maybe_gz, clean_axis, FIELD_LABELS, TIME_LABEL,
)


_XY_FILE_RE = re.compile(r'^(?P<rest>.+)\.xy$')
# nombres de campo conocidos, mas largos primero (p_rgh antes que p) para que
# el sufijo se identifique sin ambiguedad
_KNOWN_XY_FIELDS = sorted(["p_rgh", "U", "T", "Age", "ACH", "p"], key=len, reverse=True)
# campos vectoriales conocidos en este pipeline (ocupan 3 columnas de datos
# en vez de 1) -- todo lo demas (T, p, p_rgh, Age, ACH) es escalar
_VECTOR_XY_FIELDS = {"U"}


def _split_setname_and_fields(stem):
    """OpenFOAM (setFormat raw) escribe TODOS los campos pedidos en UN SOLO
    archivo por linea, con el nombre de cada campo pegado al del set en
    orden alfabetico: 'lineX_Age_T_p_rgh_U.xy' trae Age, T, p_rgh y U del
    set 'lineX', no 4 archivos separados. Para separar 'lineX' del resto,
    se van despegando nombres de campo CONOCIDOS desde el final del nombre,
    uno a la vez (probando 'p_rgh' antes que 'p' para no cortar mal),
    hasta que no quede ninguno mas por quitar -- lo que sobra al final es
    el nombre del set. Devuelve (setname, [campos en el orden real del
    archivo, izquierda a derecha]) o (None, []) si no se reconocio nada."""
    remaining = stem
    fields_found = []
    changed = True
    while changed:
        changed = False
        for f in _KNOWN_XY_FIELDS:
            suffix = "_" + f
            if remaining.endswith(suffix):
                fields_found.insert(0, f)
                remaining = remaining[: -len(suffix)]
                changed = True
                break
    if not fields_found or not remaining:
        return None, []
    return remaining, fields_found


def discover_profile_sets(base_dir):
    """Recorre postProcessing/velocityProfiles/<tiempo>/*.xy (formato 'raw'
    del functionObject 'sets') y arma {(setname, field): {t: (dist, valores)}}.

    Formato real de cada archivo (verificado contra datos reales, no
    supuesto): UN archivo por linea con TODOS los campos juntos, sin
    encabezado. Cada fila es 'coord campo1 campo2 ...' donde 'coord' es
    la posicion a lo largo de la linea (un solo numero -- para una linea
    recta en X/Y/Z, OpenFOAM ya da solo esa coordenada, no x,y,z
    completos), y el resto son los valores de cada campo en ese punto en
    el mismo orden en que aparecen en el nombre del archivo (escalares
    ocupan 1 columna, U ocupa 3)."""
    time_dirs = find_time_dirs(base_dir)
    data = {}
    for _, folder in time_dirs:
        try:
            names = os.listdir(folder)
        except (FileNotFoundError, NotADirectoryError):
            continue
        for fname in names:
            stem = fname[:-3] if fname.endswith(".xy.gz") else fname
            m = _XY_FILE_RE.match(stem)
            if not m:
                continue
            setname, fields_in_file = _split_setname_and_fields(m.group("rest"))
            if setname is None:
                continue

            # columna de inicio de cada campo (col 0 es la coordenada)
            col_start = {}
            col = 1
            for f in fields_in_file:
                col_start[f] = col
                col += 3 if f in _VECTOR_XY_FIELDS else 1

            path = os.path.join(folder, fname)
            coords, per_field_rows = [], {f: [] for f in fields_in_file}
            try:
                with open_maybe_gz(path, "r") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) < col:
                            continue  # fila incompleta, se salta en vez de tronar
                        vals = [float(p) for p in parts]
                        coords.append(vals[0])
                        for f in fields_in_file:
                            c0 = col_start[f]
                            if f in _VECTOR_XY_FIELDS:
                                per_field_rows[f].append(vals[c0:c0 + 3])
                            else:
                                per_field_rows[f].append(vals[c0])
            except (FileNotFoundError, ValueError):
                continue
            if not coords:
                continue

            t = round(float(os.path.basename(folder)), 6)
            coords_arr = np.array(coords)
            for f in fields_in_file:
                data.setdefault((setname, f), {})[t] = (coords_arr, np.array(per_field_rows[f]))
    return data



def _profile_scalar(vals, component="mag"):
    vals = np.asarray(vals)
    if vals.ndim == 1:
        return vals
    if component == "mag":
        return np.linalg.norm(vals, axis=1)
    idx = {"x": 0, "y": 1, "z": 2}.get(component, 0)
    return vals[:, idx]



def run_profile_video_and_snapshots(base_dir, outdir, field="U", component="mag",
                                     snapshot_every=10.0, fps=10):
    """Para cada linea de perfil detectada: un video mp4 de la evolucion
    completa en el tiempo, y una imagen estatica cada 'snapshot_every'
    segundos simulados. Rango de color/eje Y fijo en todo el video (calculado
    de antemano sobre todos los tiempos), para que no salte de cuadro a
    cuadro como una animacion mal hecha."""
    sets = discover_profile_sets(base_dir)
    keys = [k for k in sets if k[1] == field]
    if not keys:
        print(f"[aviso] no se encontraron perfiles de '{field}' en {base_dir} "
              f"(revisa que 'fields' en el functionObject de perfiles incluya "
              f"'{field}')", file=sys.stderr)
        return

    dest = os.path.join(outdir, "perfiles")
    os.makedirs(dest, exist_ok=True)
    ylabel = FIELD_LABELS.get(field, field) + (f" ({component})" if component else "")

    for setname, _ in keys:
        series_by_t = sets[(setname, field)]
        times_sorted = sorted(series_by_t.keys())
        if not times_sorted:
            continue

        # rango fijo de X e Y sobre TODO el tiempo, para el video
        all_d, all_y = [], []
        for t in times_sorted:
            d, v = series_by_t[t]
            y = _profile_scalar(v, component)
            all_d.append(d)
            all_y.append(y)
        d_min = min(d.min() for d in all_d if len(d))
        d_max = max(d.max() for d in all_d if len(d))
        y_min = min(np.nanmin(y) for y in all_y if len(y))
        y_max = max(np.nanmax(y) for y in all_y if len(y))
        pad = (y_max - y_min) * 0.08 or max(abs(y_max), 1e-6) * 0.1

        set_dest = os.path.join(dest, setname)
        frames_dir = os.path.join(set_dest, "frames_video")
        os.makedirs(frames_dir, exist_ok=True)

        for k, t in enumerate(times_sorted):
            d, v = series_by_t[t]
            y = _profile_scalar(v, component)
            order = np.argsort(d)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(d[order], y[order], linewidth=1.8, color="tab:blue")
            ax.set_xlim(d_min, d_max)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xlabel("Distance along the line [m]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{setname} - {field} ({component}) | t = {t:.3g} s")
            ax.grid(True, alpha=0.3)
            clean_axis(ax)
            fig.tight_layout()
            fig.savefig(os.path.join(frames_dir, f"frame_{k:05d}.png"), dpi=130)
            plt.close(fig)

        video_path = os.path.join(set_dest, f"{setname}_{field}_{component}.mp4")
        cmd = ["ffmpeg", "-y", "-framerate", str(fps),
               "-i", os.path.join(frames_dir, "frame_%05d.png"),
               "-c:v", "libx264", "-crf", "20", "-preset", "fast",
               "-pix_fmt", "yuv420p", video_path]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode != 0:
            print(f"[aviso] ffmpeg fallo generando {video_path}: "
                  f"{r.stderr.decode(errors='ignore')[-400:]}", file=sys.stderr)
        else:
            print(f"Listo -> {video_path}")

        # imagenes estaticas cada 'snapshot_every' segundos (tiempo mas cercano disponible)
        snap_dir = os.path.join(set_dest, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        t_arr = np.array(times_sorted)
        t_final = t_arr[-1]
        targets = np.arange(0.0, t_final + snapshot_every, snapshot_every)
        picked = sorted(set(t_arr[np.argmin(np.abs(t_arr - tg))] for tg in targets))
        for t in picked:
            d, v = series_by_t[t]
            y = _profile_scalar(v, component)
            order = np.argsort(d)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(d[order], y[order], linewidth=1.8, color="tab:blue")
            ax.set_xlim(d_min, d_max)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xlabel("Distance along the line [m]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{setname} - {field} ({component}) | t = {t:.3g} s")
            ax.grid(True, alpha=0.3)
            clean_axis(ax)
            fig.tight_layout()
            fig.savefig(os.path.join(snap_dir, f"t_{t:07.2f}s.png"), dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
        print(f"Listo -> {snap_dir} ({len(picked)} instantes, cada ~{snapshot_every:.0f}s)")


