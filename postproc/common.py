#!/usr/bin/env python3
"""common.py
=========
Utilidades compartidas por ach.py, probes.py, topo.py y perfiles.py: lectura
de archivos de OpenFOAM (probes, functionObjects tipo .dat), traduccion de
etiquetas al ingles para las graficas, y helpers de matplotlib. No se corre
solo -- lo importan los otros 4 modulos y dashboard_plots.py.
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

FIELD_LABELS = {
    "U": "Velocity [m/s]",
    "p": "Pressure (p) [Pa]",
    "p_rgh": "Pressure (p_rgh) [m2/s2]",
    "T": "Temperature [K]",
    "Age": "Air age [s]",
    "ACH": "Air changes [h-1]",
}


SOURCE_LABELS = {
    "difusores": "Diffuser / Inlet / Outlet Probes",
    "esquinas": "Corner Probes",
    "paciente": "Patient Probes",
}


TIME_LABEL = "Time [s]"

# Todo lo que aparece EN las graficas (leyendas, titulos, nombres de
# archivo) debe salir en ingles, aunque el nombre real de la zona/sonda
# en los dicts de OpenFOAM se quede en espanol (eso es correcto e
# intencional, no se toca). Este diccionario traduce por palabra/token,
# asi que nombres compuestos como 'esquina_TL_piso' o 'banda_baja' se
# arman traduciendo cada pedazo reconocido y dejando intacto lo que no
# esta en el mapa (como los codigos TL/TR/BL/BRa/BRb).

_ES_EN_WORD_MAP = {
    "banda": "band", "bandas": "bands",
    "baja": "lower", "media": "middle", "alta": "upper",
    "esquina": "corner", "esquinas": "corners",
    "piso": "floor", "techo": "ceiling",
    "camilla": "table", "zonacamilla": "patient zone",
    "difusores": "diffusers", "difusor": "diffuser",
    "paciente": "patient", "pacientes": "patients",
    "perfiles": "profiles", "perfil": "profile",
    "punto": "point", "puntos": "points",
    "sonda": "probe", "sondas": "probes",
    "topo": "zone",
}



_BANDA_RE = re.compile(r'^banda_(baja|media|alta)$', re.IGNORECASE)
_ESQUINA_RE = re.compile(r'^esquina_([A-Za-z]+)_(piso|techo)$', re.IGNORECASE)



def translate_label(name):
    """'banda_baja' -> 'lower band'; 'esquina_TL_piso' -> 'TL corner (floor)';
    'punto 3' -> 'point 3'. Los dos patrones compuestos conocidos (banda_X,
    esquina_X_Y) se reordenan a como sonarian naturales en ingles; el resto
    se traduce palabra por palabra. Si ningun token tiene traduccion (como
    'outletS_1', que no es español, es un codigo propio), se devuelve TAL
    CUAL -- no hay que reformatear un identificador que no tenia nada que
    traducir."""
    if name is None:
        return name
    s = name.strip()

    m = _BANDA_RE.match(s)
    if m:
        return f"{_ES_EN_WORD_MAP[m.group(1).lower()]} band"
    m = _ESQUINA_RE.match(s)
    if m:
        code, nivel = m.group(1), m.group(2).lower()
        return f"{code} corner ({_ES_EN_WORD_MAP[nivel]})"

    low_full = s.lower()
    if low_full in _ES_EN_WORD_MAP:
        return _ES_EN_WORD_MAP[low_full]

    tokens = re.split(r'([ _])', s)
    any_translated = any(tok.lower() in _ES_EN_WORD_MAP
                          for tok in tokens if tok not in (" ", "_"))
    if not any_translated:
        return s  # nada que traducir -- no tocar el identificador original

    out = []
    for tok in tokens:
        if tok in (" ", "_"):
            out.append(" ")
            continue
        out.append(_ES_EN_WORD_MAP.get(tok.lower(), tok))
    return "".join(out).strip()



def line_style(n_points, base_width=1.0):
    """Con miles de instantes, un marcador por punto convierte la curva en una
    banda solida y se pierde el detalle. Los marcadores solo aportan cuando hay
    pocos puntos; por encima de ~120 se quitan, y entre medias se dibuja uno de
    cada N. El grosor tambien baja con la densidad de datos."""
    if n_points <= 60:
        return {"linewidth": base_width, "marker": "o", "markersize": 3.0}
    if n_points <= 400:
        return {"linewidth": base_width * 0.85, "marker": "o", "markersize": 2.0,
                "markevery": max(1, n_points // 60)}
    return {"linewidth": base_width * 0.7}



def clean_axis(ax):
    try:
        ax.ticklabel_format(useOffset=False, style="plain", axis="both")
    except AttributeError:
        pass
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))



_PROBE_HEADER_RE = re.compile(r'#\s*Probe\s+(\d+)\s*\(([^)]*)\)')
_VEC_RE = re.compile(r'\(([^)]*)\)')



def find_time_dirs(base_dir):
    try:
        names = sorted(os.listdir(base_dir))
    except (FileNotFoundError, NotADirectoryError):
        return []
    out = []
    for name in names:
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            t = float(name)
        except ValueError:
            continue
        out.append((t, full))
    out.sort(key=lambda x: x[0])
    return out



def parse_probe_locations(lines):
    probes = []
    for line in lines:
        if not line.startswith("#"):
            break
        m = _PROBE_HEADER_RE.match(line.strip())
        if m:
            idx = int(m.group(1))
            coords = tuple(float(x) for x in m.group(2).split())
            probes.append((idx, coords))
    return probes



def parse_probe_data_line(line):
    parts = line.split(None, 1)
    t = float(parts[0])
    rest = parts[1].strip() if len(parts) > 1 else ""
    if "(" in rest:
        vecs = _VEC_RE.findall(rest)
        values = [tuple(float(x) for x in v.split()) for v in vecs]
    else:
        values = [float(x) for x in rest.split()]
    return t, values



def open_maybe_gz(path, mode="r"):
    """Abre 'path' tal cual, o 'path.gz' si el primero no existe -- para que
    no importe si el caso tiene writeCompression on u off en su controlDict.
    Con gzip, el modo texto ('r') ya decodifica automaticamente."""
    if os.path.exists(path):
        return open(path, mode)
    gz_path = path + ".gz"
    if os.path.exists(gz_path):
        text_mode = "t" in mode or mode == "r"
        return gzip.open(gz_path, mode + ("t" if text_mode and "b" not in mode else ""))
    raise FileNotFoundError(path)



def probe_field_path_exists(folder, field_name):
    return (os.path.exists(os.path.join(folder, field_name))
            or os.path.exists(os.path.join(folder, field_name + ".gz")))



def read_probe_series(base_dir, field_name):
    time_dirs = find_time_dirs(base_dir)
    if not time_dirs:
        raise FileNotFoundError(f"No time subfolders found in {base_dir}")

    probes = None
    probes_src = None
    all_times, all_data = [], []
    for _, folder in time_dirs:
        fpath = os.path.join(folder, field_name)
        if not probe_field_path_exists(folder, field_name):
            continue
        with open_maybe_gz(fpath, "r") as f:
            lines = f.readlines()
        this_probes = parse_probe_locations(lines)
        if probes is None:
            probes = this_probes
            probes_src = folder
        elif this_probes and this_probes != probes:
            print(f"[AVISO IMPORTANTE] el conjunto de sondas cambio entre segmentos de "
                  f"la simulacion:\n"
                  f"    {probes_src}: {len(probes)} sondas\n"
                  f"    {folder}: {len(this_probes)} sondas\n"
                  f"  Esto pasa si se edito el diccionario de sondas entre reinicios. "
                  f"Los indices NO son comparables entre segmentos y las etiquetas de "
                  f"grupo pueden quedar mal asignadas. Revisa antes de usar estas "
                  f"graficas en un reporte.", file=sys.stderr)
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t, values = parse_probe_data_line(line)
            all_times.append(t)
            all_data.append(values)

    if probes is None:
        raise FileNotFoundError(f"'{field_name}' not found in any time subfolder of {base_dir}")

    order = np.argsort(all_times, kind="stable")
    dedup = {}
    n_dup = 0
    for i in order:
        key = round(all_times[i], 10)
        if key in dedup:
            n_dup += 1
        dedup[key] = all_data[i]
    if n_dup:
        print(f"[info] {base_dir}: {n_dup} instantes repetidos entre segmentos de reinicio; "
              f"se conservo el valor del segmento mas reciente.")
    final_times = sorted(dedup.keys())
    final_data = [dedup[t] for t in final_times]
    return probes, final_times, final_data



def extract_component(value, component):
    if isinstance(value, tuple):
        if component in ("mag", "magnitude", None):
            return float(np.linalg.norm(value))
        idx = {"x": 0, "y": 1, "z": 2}.get(component)
        return value[idx] if idx is not None else float(np.linalg.norm(value))
    return value



def find_dat_file(folder):
    candidates = sorted(glob.glob(os.path.join(folder, "*.dat"))
                         + glob.glob(os.path.join(folder, "*.dat.gz")))
    return candidates[0] if candidates else None



def _to_number(tok):
    try:
        return float(tok)
    except ValueError:
        return tok



def parse_func_data_line(line):
    if "(" in line:
        tokens, buf = [], ""
        for ch in line:
            if ch == "(":
                if buf.strip():
                    tokens.extend(buf.split())
                buf = ch
            elif ch == ")":
                buf += ch
                tokens.append(buf)
                buf = ""
            else:
                buf += ch
        if buf.strip():
            tokens.extend(buf.split())
        values = []
        for tok in tokens:
            if tok.startswith("("):
                values.append(tuple(float(x) for x in tok.strip("()").split()))
            else:
                values.append(_to_number(tok))
        return values
    return [_to_number(x) for x in line.split()]



def parse_func_header(lines):
    header = None
    for line in lines:
        if line.startswith("#"):
            header = line.lstrip("#").strip()
        else:
            break
    return header.split() if header else None



def read_function_series(base_dir):
    time_dirs = find_time_dirs(base_dir)
    if not time_dirs:
        raise FileNotFoundError(f"No time subfolders found in {base_dir}")
    column_names, all_rows = None, []
    for _, folder in time_dirs:
        fpath = find_dat_file(folder)
        if fpath is None:
            continue
        with open_maybe_gz(fpath, "r") as f:
            lines = [l.rstrip("\n") for l in f]
        if column_names is None:
            column_names = parse_func_header(lines)
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            all_rows.append(parse_func_data_line(line))
    if not all_rows:
        raise FileNotFoundError(f"No data found in any time subfolder of {base_dir}")
    times = [row[0] for row in all_rows]
    order = np.argsort(times, kind="stable")
    dedup = {}
    for i in order:
        dedup[round(times[i], 10)] = all_rows[i]
    final_times = sorted(dedup.keys())
    final_rows = [dedup[t] for t in final_times]
    if column_names is None:
        column_names = ["Time"] + [f"col{i}" for i in range(1, len(final_rows[0]))]
    return column_names, final_times, final_rows


