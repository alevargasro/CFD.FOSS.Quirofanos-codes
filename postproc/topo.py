#!/usr/bin/env python3
"""topo.py
=======
Sección C del pipeline: zonas Topo_* (volAverage por cellZone -- bandas,
esquinas piso/techo, zona del paciente), agrupadas por categoría con subplots
individuales y calidad de publicación científica.
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
    read_function_series, line_style, clean_axis, translate_label,
    FIELD_LABELS, TIME_LABEL,
)

# Información de categorías y nombres de archivos limpios
CAT_INFO = {
    "banda": {"title": "Band Zones", "suffix": "band"},
    "esquina": {"title": "Corner Zones", "suffix": "corners"},
    "esquina_piso": {"title": "Floor Corner Zones", "suffix": "corners_floor"},
    "esquina_techo": {"title": "Ceiling Corner Zones", "suffix": "corners_ceiling"},
    "paciente": {"title": "Patient Zone", "suffix": "patient"},
}


def apply_publication_style():
    """Aplica configuraciones estéticas de Matplotlib para artículos científicos."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern", "serif"],
        "font.size": 9.5,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 11.5,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "figure.autolayout": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def discover_topo_dirs(import_dir):
    """Carpetas postProcessing/Topo_<zona> -- una por cada cellZone."""
    pp = os.path.join(import_dir, "postProcessing")
    if not os.path.isdir(pp):
        return []
    return sorted(d for d in os.listdir(pp)
                  if d.startswith("Topo_") and os.path.isdir(os.path.join(pp, d)))


def classify_topo_name(zone_name):
    """Clasifica la zona topo.
    
    Separa las esquinas en piso y techo si la palabra clave está presente.
    """
    low = zone_name.lower()
    if "banda" in low:
        return "banda"
    if "esquina" in low or "corner" in low:
        if any(k in low for k in ["piso", "floor", "inferior", "bottom"]):
            return "esquina_piso"
        if any(k in low for k in ["techo", "ceiling", "roof", "superior", "top"]):
            return "esquina_techo"
        return "esquina"
    return "paciente"


_VOLAVG_WRAP_RE = re.compile(r'^\w+\((\w+)\)$')


def clean_topo_field_name(raw_field):
    """'volAverage(Age)' -> 'Age'."""
    m = _VOLAVG_WRAP_RE.match(raw_field)
    return m.group(1) if m else raw_field


def run_topo_section(import_dir, outdir):
    apply_publication_style()
    topo_dirs = discover_topo_dirs(import_dir)
    if not topo_dirs:
        print(f"[aviso] no se encontraron carpetas Topo_* en "
              f"{os.path.join(import_dir, 'postProcessing')}", file=sys.stderr)
        return

    dest = os.path.join(outdir, "topo")
    os.makedirs(dest, exist_ok=True)
    pp = os.path.join(import_dir, "postProcessing")

    # (categoria, campo) -> lista de (nombre_zona, times, valores)
    by_cat_field = {}
    summary_rows = []
    for tdir in topo_dirs:
        zone_name = tdir[len("Topo_"):]
        cat = classify_topo_name(zone_name)
        try:
            col_names, times, rows = read_function_series(os.path.join(pp, tdir))
        except (FileNotFoundError, ValueError) as e:
            print(f"[aviso] no se pudo leer {tdir}: {e}", file=sys.stderr)
            continue
        numeric_cols = [i for i in range(1, len(col_names))
                        if len(rows) and isinstance(rows[0][i], float)]
        for i in numeric_cols:
            raw_field = col_names[i] if i < len(col_names) else f"col{i}"
            field = clean_topo_field_name(raw_field)
            series = [row[i] for row in rows]
            by_cat_field.setdefault((cat, field), []).append((zone_name, times, series))
            t_final = times[-1]
            last50 = [s for t, s in zip(times, series) if t >= t_final - 50.0]
            summary_rows.append({
                "zona": zone_name, "categoria": cat, "campo": field,
                "valor_final": series[-1],
                "promedio_ultimos_50s": float(np.mean(last50)) if last50 else series[-1],
            })

    # Paleta de colores profesional
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]

    for (cat, field), entries in sorted(by_cat_field.items()):
        n_probes = len(entries)
        cat_meta = CAT_INFO.get(cat, {
            "title": translate_label(cat).title(),
            "suffix": cat.lower().replace(" ", "_")
        })

        # Configuración de grilla de subplots según número de zonas
        if n_probes == 1:
            nrows, ncols = 1, 1
            figsize = (7.0, 4.5)
        elif n_probes == 2:
            nrows, ncols = 2, 1
            figsize = (7.5, 5.5)
        elif n_probes == 3:
            nrows, ncols = 3, 1
            figsize = (7.5, 7.0)
        elif n_probes == 4:
            nrows, ncols = 2, 2
            figsize = (9.0, 6.0)
        elif n_probes in (5, 6):
            nrows, ncols = 2, 3
            figsize = (11.0, 6.0)
        else:
            ncols = 3
            nrows = int(np.ceil(n_probes / ncols))
            figsize = (11.0, 2.5 * nrows)

        fig, axes = plt.subplots(
            nrows, ncols, 
            figsize=figsize, 
            sharex=(ncols == 1 or nrows > 1),
            squeeze=False
        )
        axes_flat = axes.flatten()

        field_unit = FIELD_LABELS.get(field, field)

        for idx, (zone_name, times, series) in enumerate(entries):
            ax = axes_flat[idx]
            color = colors[idx % len(colors)]
            markevery = max(1, len(times) // 25)
            zone_label_en = translate_label(zone_name)

            ax.plot(
                times, series, 
                color=color, linestyle="-", marker="o", 
                markevery=markevery, label=zone_label_en
            )

            ax.set_title(zone_label_en, fontsize=9.5, fontweight="bold", pad=4)
            ax.set_ylabel(field_unit, fontsize=9)
            ax.grid(True, alpha=0.25, linestyle="--")
            clean_axis(ax)

            # Ajuste ajustado de eje Y para eliminar espacio desperdiciado
            y_min, y_max = np.min(series), np.max(series)
            if y_min == y_max:
                y_margin = 0.1 * (abs(y_min) if y_min != 0 else 1.0)
            else:
                y_margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            # Añadir etiqueta del eje X en la fila inferior
            row_idx = idx // ncols
            if row_idx == nrows - 1 or idx + ncols >= n_probes:
                ax.set_xlabel(TIME_LABEL, fontsize=9)

        # Ocultar ejes vacíos si no se llena la matriz
        for j in range(n_probes, nrows * ncols):
            fig.delaxes(axes_flat[j])

        fig.suptitle(f"Topo zones - {cat_meta['title']} - {field}", fontsize=11, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(dest, f"topo_{cat_meta['suffix']}_{field}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Listo -> {out_path}")

    if summary_rows:
        csv_path = os.path.join(dest, "topo_resumen.csv")
        df = pd.DataFrame(summary_rows)
        df.insert(1, "zone_en", df["zona"].map(translate_label))
        df.insert(3, "category_en", df["categoria"].map(translate_label))
        df.to_csv(csv_path, index=False)
        print(f"Listo -> {csv_path}")