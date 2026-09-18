#!/usr/bin/env python3
"""probes.py
=========
Seccion B del pipeline: difusores/inlet/outlet (promedio de 9 sondas por
altura), esquinas (Age por esquina) y paciente (todas las variables por
sonda + dashboards de 6 paneles). Deteccion automatica de grupos por los
comentarios de los archivos probes_difusores/esquinas/paciente. Depende de
postproc_common.py.
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
    find_time_dirs, read_probe_series, extract_component, open_maybe_gz,
    probe_field_path_exists, line_style, clean_axis, translate_label,
    FIELD_LABELS, SOURCE_LABELS, TIME_LABEL,
)


_LABEL_HEIGHT_RE = re.compile(r'^([A-Za-z0-9_\-]+)\s*(?:\([^)]*\))?\s*@\s*([\d.]+)\s*cm')
_LABEL_WORD_RE = re.compile(r'^([A-Za-z0-9_\-]+)')
_OBJECT_NAME_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*$')
_STANDALONE_COMMENT_RE = re.compile(r'^//\s*(.*)$')
_POINT_LINE_RE = re.compile(
    r'^\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)\s*(?://\s*(.*))?$'
)

_DEF_FILE_HINTS = {
    "difusores": ["probes_difusores", "probesDifusores", "sondasDifusores"],
    "esquinas": ["probes_esquinas", "probesEsquinas", "sondasEsquinas"],
    "paciente": ["probes_paciente", "probesPaciente", "sondasPaciente"],
}



def parse_label(raw_label):
    """A partir del comentario ('diff_1 (...) @ 10cm', 'piso (z=0.1)...',
    'punto 1'), extrae (grupo, altura_cm). altura_cm es None si no aplica."""
    if not raw_label:
        return "group", None
    text = raw_label.strip()
    m = _LABEL_HEIGHT_RE.match(text)
    if m:
        return m.group(1), float(m.group(2))
    m2 = _LABEL_WORD_RE.match(text)
    if m2:
        return m2.group(1), None
    return text, None



def parse_probe_definition_file(path):
    """Lee un archivo tipo probes_difusores/esquinas/paciente (con los
    comentarios que identifican cada grupo) y devuelve:
        object_name : nombre del functionObject (define la carpeta en
                       postProcessing/), ej. 'inletOutletProbes'
        definitions : lista ordenada (mismo orden que probeLocations, que
                       es el mismo orden/indice que usa OpenFOAM al escribir
                       postProcessing/<obj>/<t>/<field>) de dicts:
                       {coords, raw_label, group, height}
    """
    with open(path, "r") as f:
        raw_lines = f.readlines()

    object_name = None
    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        m = _OBJECT_NAME_RE.match(s)
        if m:
            object_name = m.group(1)
        break

    definitions = []
    current_comment = None
    in_locations = False
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if not in_locations:
            if s.startswith("probeLocations"):
                in_locations = True
            continue
        if s == ");":
            break
        if s == "(":
            continue

        mc = _STANDALONE_COMMENT_RE.match(s)
        if mc:
            current_comment = mc.group(1).strip()
            continue

        mp = _POINT_LINE_RE.match(s)
        if mp:
            coords = (float(mp.group(1)), float(mp.group(2)), float(mp.group(3)))
            inline_comment = mp.group(4)
            raw_label = inline_comment.strip() if inline_comment else current_comment
            group, height = parse_label(raw_label)
            definitions.append({
                "coords": coords,
                "raw_label": raw_label,
                "group": group,
                "height": height,
            })

    if not definitions:
        raise ValueError(f"No se pudo parsear ninguna sonda de {path}")
    return object_name, definitions



def discover_definition_file(import_dir, kind, override=None):
    if override:
        if os.path.isfile(override):
            return override
        raise FileNotFoundError(f"No existe el archivo indicado: {override}")

    hints = _DEF_FILE_HINTS[kind]
    for h in hints:
        p = os.path.join(import_dir, "system", h)
        if os.path.isfile(p):
            return p
    for root, dirs, files in os.walk(import_dir):
        if "postProcessing" in root.split(os.sep):
            dirs[:] = []
            continue
        for h in hints:
            if h in files:
                return os.path.join(root, h)
    raise FileNotFoundError(
        f"No se encontro el archivo de definicion de sondas para '{kind}'. "
        f"Se busco en {os.path.join(import_dir, 'system')} y de forma recursiva "
        f"bajo {import_dir}. Usa --{kind}-def para indicarlo manualmente."
    )



def validate_alignment(probes, definitions, tol=1e-3):
    if len(probes) != len(definitions):
        print(f"[aviso] el numero de sondas en postProcessing ({len(probes)}) no coincide "
              f"con el archivo de definicion ({len(definitions)}); revisa que ambos esten "
              f"sincronizados (mismo caso, misma corrida).", file=sys.stderr)
        return
    for idx, coords in probes:
        if idx >= len(definitions):
            continue
        dc = definitions[idx]["coords"]
        if max(abs(coords[k] - dc[k]) for k in range(3)) > tol:
            print(f"[aviso] la sonda {idx} no coincide en coordenadas entre postProcessing "
                  f"{coords} y el archivo de definicion {dc}. Los nombres de grupo podrian "
                  f"estar mal asignados.", file=sys.stderr)
            return



def build_group_map(definitions):
    m = {}
    order_groups = []
    for i, d in enumerate(definitions):
        key = (d["group"], d["height"])
        if key not in m:
            m[key] = []
            if d["group"] not in order_groups:
                order_groups.append(d["group"])
        m[key].append(i)
    return m, order_groups



_KNOWN_FIELD_PRIORITY = ["U", "p", "p_rgh", "T", "Age", "ACH"]

# ---------------------------------------------------------------------------
# Magnitudes derivadas (heredadas del pipeline anterior)
# ---------------------------------------------------------------------------

P_REF = 101325.0    # Pa, presion de referencia para p_rgh
R_AIR = 287.058     # J/(kg K), constante del aire seco


def density_from_T_p(T_arr, p_arr, p_ref=P_REF, R=R_AIR):
    """rho = (p + p_ref) / (R T), gas ideal. p es p_rgh (o p) de OpenFOAM."""
    T = np.asarray(T_arr, dtype=float)
    p = np.asarray(p_arr, dtype=float) + p_ref
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = p / (R * T)
    return rho


def ach_from_age(age_arr):
    """ACH local = 3600 / Age. Age<=0 (arranque) queda como NaN, no como inf."""
    age = np.asarray(age_arr, dtype=float)
    out = np.full_like(age, np.nan)
    valid = age > 0
    out[valid] = 3600.0 / age[valid]
    return out


def detect_normal_axis(definitions, group):
    """Deduce el eje normal de un grupo de difusores/inlet/outlet a partir de
    la geometria: las 3 alturas (10/20/30 cm) estan separadas justamente a lo
    largo de la normal del patch, asi que el eje donde mas varia el centroide
    entre alturas ES la normal. Evita declarar a mano 'Uz para el inlet, Ux
    para outlet 1-2, Uy para 3-4' como hacia el codigo anterior.

    Devuelve 'x' / 'y' / 'z', o None si el grupo no tiene varias alturas."""
    by_height = {}
    for d in definitions:
        if d["group"] != group or d["height"] is None:
            continue
        by_height.setdefault(d["height"], []).append(d["coords"])
    if len(by_height) < 2:
        return None
    centroids = [np.mean(np.array(v), axis=0) for _, v in sorted(by_height.items())]
    spread = np.ptp(np.array(centroids), axis=0)
    return ["x", "y", "z"][int(np.argmax(spread))]


def detect_available_fields(base_dir):
    """Escanea TODOS los directorios de tiempo, no solo el primero. Esto
    importa cuando se agregaron campos a un functionObject a mitad de la
    corrida (ej. se anadio 'Age' a cornerProbes en un reinicio): el campo
    solo existe en los segmentos posteriores."""
    time_dirs = find_time_dirs(base_dir)
    if not time_dirs:
        return []
    seen = {}
    for _, folder in time_dirs:
        try:
            names = [f[:-3] if f.endswith(".gz") else f
                     for f in os.listdir(folder)
                     if os.path.isfile(os.path.join(folder, f)) and not f.startswith(".")]
        except (FileNotFoundError, NotADirectoryError):
            continue
        for n in names:
            seen.setdefault(n, []).append(folder)

    partial = {n: folders for n, folders in seen.items()
               if len(folders) < len(time_dirs)}
    for n, folders in sorted(partial.items()):
        print(f"[info] el campo '{n}' solo esta en {len(folders)} de {len(time_dirs)} "
              f"segmentos de {base_dir}; su serie de tiempo sera parcial "
              f"(probablemente se agregó al functionObject en un reinicio).")

    found = [f for f in _KNOWN_FIELD_PRIORITY if f in seen]
    extra = sorted(f for f in seen if f not in _KNOWN_FIELD_PRIORITY)
    return found + extra



def plot_field_single_source(base_dir, source_label, field, outdir, component="mag",
                              label_map=None, subfolder="timeseries"):
    probes, times, data = read_probe_series(base_dir, field)
    is_vector = len(data) > 0 and isinstance(data[0][0], tuple)
    comp = component if is_vector else None
    ylabel = FIELD_LABELS.get(field, field) + (f" ({comp})" if comp else "")

    n_probes = len(probes)
    ncol = 1 if n_probes <= 10 else (2 if n_probes <= 24 else 3)
    fig, ax = plt.subplots(figsize=(9 + 2.2 * ncol, 6))
    for idx, coords in probes:
        series = [extract_component(data[i][idx], comp) for i in range(len(times))]
        if label_map and idx in label_map:
            label = label_map[idx]
        else:
            label = f"{idx} ({coords[0]:.3f},{coords[1]:.3f},{coords[2]:.3f})"
        ax.plot(times, series, label=label, **line_style(len(times)))
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{SOURCE_LABELS.get(source_label, source_label)} - {field}")
    ax.grid(True, alpha=0.3)
    clean_axis(ax)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              ncol=ncol, borderaxespad=0.0)
    fig.tight_layout()

    dest = os.path.join(outdir, subfolder)
    os.makedirs(dest, exist_ok=True)
    out_path = os.path.join(dest, f"{source_label}_{field}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Listo -> {out_path}")


def load_probe_group(base_dir, definitions=None):
    """Carga TODOS los campos disponibles de un grupo de sondas y ademas
    calcula las magnitudes derivadas que traia el pipeline anterior:

        ACH  = 3600 / Age          (por sonda, no solo el global)
        rho  = (p + p_ref)/(R T)   (gas ideal)

    Devuelve {nombre: {"probes":..., "times":..., "data":...}}. Cada campo
    conserva su propio vector de tiempos, porque tras un reinicio un campo
    puede existir solo en algunos segmentos. Las derivadas se calculan sobre
    los instantes comunes a sus ingredientes."""
    fields = detect_available_fields(base_dir)
    if not fields:
        return {}

    store = {}
    for f in fields:
        try:
            probes, times, data = read_probe_series(base_dir, f)
        except (FileNotFoundError, ValueError) as e:
            print(f"[aviso] no se pudo leer '{f}' en {base_dir}: {e}", file=sys.stderr)
            continue
        if definitions is not None:
            validate_alignment(probes, definitions)
        store[f] = {"probes": probes, "times": times, "data": data}

    # ACH local por sonda, a partir de Age
    if "Age" in store and "ACH" not in store:
        src = store["Age"]
        n = len(src["probes"])
        ach_rows = []
        for row in src["data"]:
            ach_rows.append(list(ach_from_age([row[i] for i in range(n)])))
        store["ACH"] = {"probes": src["probes"], "times": src["times"], "data": ach_rows}

    # Densidad por gas ideal, a partir de T y p (o p_rgh)
    p_key = "p_rgh" if "p_rgh" in store else ("p" if "p" in store else None)
    if "T" in store and p_key:
        tT, tp = store["T"], store[p_key]
        common = sorted(set(round(t, 9) for t in tT["times"])
                        & set(round(t, 9) for t in tp["times"]))
        if common:
            iT = {round(t, 9): k for k, t in enumerate(tT["times"])}
            ip = {round(t, 9): k for k, t in enumerate(tp["times"])}
            n = min(len(tT["probes"]), len(tp["probes"]))
            rho_rows = []
            for t in common:
                Trow = [tT["data"][iT[t]][i] for i in range(n)]
                prow = [tp["data"][ip[t]][i] for i in range(n)]
                rho_rows.append(list(density_from_T_p(Trow, prow)))
            store["rho"] = {"probes": tT["probes"][:n], "times": common, "data": rho_rows}

    return store



def series_for(entry, idx, comp=None):
    """Serie temporal de una sonda. comp: 'mag'/'x'/'y'/'z' para vectores."""
    data = entry["data"]
    if not data:
        return np.array([])
    is_vec = isinstance(data[0][idx], tuple)
    c = comp if is_vec else None
    return np.array([extract_component(row[idx], c) for row in data], dtype=float)



def quantity_plan(store, normal_axis=None):
    """Lista ordenada de (clave, componente, etiqueta_archivo, etiqueta_eje).
    Si el grupo tiene una normal detectada, agrega la componente normal de U
    ademas de la magnitude -- el flujo util a traves de un difusor es el
    normal, no el modulo."""
    plan = []
    if "U" in store:
        plan.append(("U", "mag", "MagU", "Velocity |U| [m/s]"))
        if normal_axis:
            plan.append(("U", normal_axis, f"U{normal_axis}",
                         f"Normal velocity U{normal_axis} [m/s]"))
    for key in ["p_rgh", "p", "T", "Age", "ACH", "rho"]:
        if key in store:
            plan.append((key, None, key, FIELD_LABELS.get(key, key)))
    return plan


FIELD_LABELS["rho"] = "Density [kg/m3]"



def plot_difusores_averages(base_dir, definitions, outdir):
    """Una grafica por grupo (inlet, diff_*, outlet_* -- lo que exista, todo
    detectado automaticamente) y por magnitud, con una linea por altura, cada
    linea = promedio de las 9 sondas de esa altura."""
    dest = os.path.join(outdir, "difusores")
    os.makedirs(dest, exist_ok=True)

    store = load_probe_group(base_dir, definitions)
    if not store:
        if not os.path.isdir(base_dir):
            print(f"[aviso] no existe la carpeta {base_dir}; revisa si el functionObject "
                  f"correspondiente corrio en esta simulacion o si falto bajarla del "
                  f"cluster.", file=sys.stderr)
        else:
            print(f"[aviso] no se encontraron campos en {base_dir}", file=sys.stderr)
        return

    group_map, order_groups = build_group_map(definitions)
    csv_rows = []

    for group in order_groups:
        normal = detect_normal_axis(definitions, group)
        heights = sorted({h for (g, h) in group_map if g == group},
                         key=lambda x: (x is None, x))
        if not heights:
            continue
        for key, comp, fname, ylabel in quantity_plan(store, normal):
            entry = store[key]
            times = entry["times"]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(heights), 1)))
            n_per_height = None
            for h, color in zip(heights, colors):
                idxs = [i for i in group_map[(group, h)] if i < len(entry["probes"])]
                if not idxs:
                    continue
                n_per_height = len(idxs)
                stack = np.vstack([series_for(entry, i, comp) for i in idxs])
                avg = np.nanmean(stack, axis=0)
                ax.plot(times, avg, color=color,
                        label=f"{h:.0f} cm" if h is not None else group,
                        **line_style(len(times), base_width=1.5))
                if len(avg):
                    csv_rows.append({"group": group, "height_cm": h,
                                     "quantity": fname, "final_value": avg[-1],
                                     "mean_last_50s": float(np.nanmean(
                                         avg[np.array(times) >= max(times) - 50.0]))})
            ax.set_xlabel(TIME_LABEL)
            ax.set_ylabel(f"Average {ylabel}")
            ax.set_title(f"{group} - {fname} "
                         f"(average of {n_per_height} probes per height level)")
            ax.grid(True, alpha=0.3)
            clean_axis(ax)
            ax.legend()
            fig.tight_layout()
            out_path = os.path.join(dest, f"{group}_{fname}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Listo -> {out_path}")

    if csv_rows:
        csv_path = os.path.join(dest, "difusores_resumen.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Listo -> {csv_path}")



def plot_esquinas_age(base_dir, definitions, outdir):
    """Age de cada esquina vs t (sin promediar). Si el caso trae mas campos,
    tambien salen, porque el mismo grafico sirve para comparar esquinas."""
    dest = os.path.join(outdir, "esquinas")
    os.makedirs(dest, exist_ok=True)

    store = load_probe_group(base_dir, definitions)
    if not store:
        if not os.path.isdir(base_dir):
            print(f"[aviso] no existe la carpeta {base_dir}; revisa si el functionObject "
                  f"correspondiente corrio en esta simulacion o si falto bajarla del "
                  f"cluster.", file=sys.stderr)
        else:
            print(f"[aviso] no se encontraron campos en {base_dir}", file=sys.stderr)
        return
    if "Age" not in store:
        print(f"[aviso] no se encontro el campo 'Age' en {base_dir} "
              f"(disponibles: {sorted(store)}). Agrega 'Age' a la lista 'fields' de "
              f"cornerProbes en el caso de OpenFOAM para tener esta grafica.",
              file=sys.stderr)

    label_map = {i: f"{translate_label(d['group'])} ({d['coords'][0]:.2f}, {d['coords'][1]:.2f})"
                 for i, d in enumerate(definitions)}

    for key, comp, fname, ylabel in quantity_plan(store):
        entry = store[key]
        n = len(entry["probes"])
        ncol = 1 if n <= 10 else (2 if n <= 24 else 3)
        fig, ax = plt.subplots(figsize=(9 + 2.2 * ncol, 6))
        for i in range(n):
            ax.plot(entry["times"], series_for(entry, i, comp),
                    label=label_map.get(i, f"probe {i}"),
                    **line_style(len(entry["times"])))
        ax.set_xlabel(TIME_LABEL)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{SOURCE_LABELS['esquinas']} - {fname}")
        ax.grid(True, alpha=0.3)
        clean_axis(ax)
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  ncol=ncol, borderaxespad=0.0)
        fig.tight_layout()
        out_path = os.path.join(dest, f"esquinas_{fname}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Listo -> {out_path}")



def plot_paciente_all(base_dir, definitions, outdir):
    """Todas las variables vs t, sonda por sonda (sin promediar), mas un
    dashboard de 6 paneles por sonda (idea heredada del pipeline anterior:
    ver todo el entorno de un punto de un vistazo)."""
    dest = os.path.join(outdir, "paciente")
    os.makedirs(dest, exist_ok=True)

    store = load_probe_group(base_dir, definitions)
    if not store:
        if not os.path.isdir(base_dir):
            print(f"[aviso] no existe la carpeta {base_dir}; revisa si el functionObject "
                  f"correspondiente corrio en esta simulacion o si falto bajarla del "
                  f"cluster.", file=sys.stderr)
        else:
            print(f"[aviso] no se encontraron campos en {base_dir}", file=sys.stderr)
        return

    label_map = {i: translate_label(d["raw_label"] or f"punto {i + 1}") for i, d in enumerate(definitions)}

    # (a) una grafica por magnitud, todas las sondas superpuestas
    for key, comp, fname, ylabel in quantity_plan(store):
        entry = store[key]
        n = len(entry["probes"])
        fig, ax = plt.subplots(figsize=(11, 6))
        for i in range(n):
            ax.plot(entry["times"], series_for(entry, i, comp),
                    label=label_map.get(i, f"probe {i}"),
                    **line_style(len(entry["times"])))
        ax.set_xlabel(TIME_LABEL)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{SOURCE_LABELS['paciente']} - {fname}")
        ax.grid(True, alpha=0.3)
        clean_axis(ax)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0)
        fig.tight_layout()
        out_path = os.path.join(dest, f"paciente_{fname}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Listo -> {out_path}")

    # (b) dashboard de 6 paneles por sonda
    plot_probe_dashboards(store, label_map, os.path.join(dest, "dashboards"),
                          SOURCE_LABELS["paciente"])



_DASH_PANELS = [
    ("velocity", "Velocity [m/s]", "Velocity field"),
    ("T",   "Temperature [K]",  "Temperature"),
    ("p",   "Pressure [Pa]",    "Pressure"),
    ("rho", "Density [kg/m3]",  "Density (ideal gas)"),
    ("Age", "Air age [s]",      "Local mean age of air"),
    ("ACH", "ACH [1/h]",        "Local air changes per hour"),
]



def plot_probe_dashboards(store, label_map, dest, source_title):
    """6 paneles por sonda: velocidad (|U| + componentes), T, p, rho, Age, ACH.
    Los paneles sin datos quedan marcados, no vacios sin explicacion."""
    if "U" not in store and "T" not in store:
        return
    os.makedirs(dest, exist_ok=True)
    ref = store.get("U") or next(iter(store.values()))
    n_probes = len(ref["probes"])
    p_key = "p_rgh" if "p_rgh" in store else ("p" if "p" in store else None)

    for i in range(n_probes):
        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        fig.suptitle(f"{source_title} | {label_map.get(i, f'probe {i}')}",
                     fontsize=13, fontweight="bold")
        for ax, (kind, ylabel, title) in zip(axes.flat, _DASH_PANELS):
            drew = False
            if kind == "velocity" and "U" in store:
                e = store["U"]
                ax.plot(e["times"], series_for(e, i, "mag"), color="black",
                        linewidth=1.2, label="|U|")
                for c, col in zip(["x", "y", "z"], ["tab:blue", "tab:orange", "tab:green"]):
                    ax.plot(e["times"], series_for(e, i, c), color=col,
                            linestyle="--", alpha=0.75, linewidth=0.8, label=f"U{c}")
                ax.legend(fontsize=8)
                drew = True
            else:
                key = p_key if kind == "p" else kind
                if key and key in store:
                    e = store[key]
                    ax.plot(e["times"], series_for(e, i), linewidth=1.0)
                    drew = True
            if not drew:
                ax.text(0.5, 0.5, "sin datos", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            if drew:
                clean_axis(ax)
        for ax in axes[2]:
            ax.set_xlabel(TIME_LABEL, fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(dest, f"probe_{i + 1:02d}_dashboard.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"Listo -> {dest} ({n_probes} dashboards)")


