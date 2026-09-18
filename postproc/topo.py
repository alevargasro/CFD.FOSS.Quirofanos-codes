#!/usr/bin/env python3
"""topo.py
=======
Seccion C del pipeline: zonas Topo_* (volAverage por cellZone -- bandas,
esquinas, zona del paciente), agrupadas por categoria con leyendas y titulos
traducidos al ingles. Depende de postproc_common.py.
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


def discover_topo_dirs(import_dir):
    """Carpetas postProcessing/Topo_<zona> -- una por cada cellZone que armo
    topoSetDict, con su volAverage de Age/ACH (o lo que se le haya pedido)."""
    pp = os.path.join(import_dir, "postProcessing")
    if not os.path.isdir(pp):
        return []
    return sorted(d for d in os.listdir(pp)
                  if d.startswith("Topo_") and os.path.isdir(os.path.join(pp, d)))



def classify_topo_name(zone_name):
    """banda_alta -> 'banda'; esquina_TL_piso -> 'esquina'; el resto
    (zonaCamilla, o cualquier zona nueva) -> 'paciente'."""
    low = zone_name.lower()
    if "banda" in low:
        return "banda"
    if "esquina" in low:
        return "esquina"
    return "paciente"



_VOLAVG_WRAP_RE = re.compile(r'^\w+\((\w+)\)$')


def clean_topo_field_name(raw_field):
    """'volAverage(Age)' -> 'Age' (para buscar en FIELD_LABELS y nombrar
    archivos limpios); si no matchea el patron, devuelve tal cual."""
    m = _VOLAVG_WRAP_RE.match(raw_field)
    return m.group(1) if m else raw_field



def run_topo_section(import_dir, outdir):
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

    for (cat, field), entries in sorted(by_cat_field.items()):
        n_probes = len(entries)
        ncol = 1 if n_probes <= 10 else (2 if n_probes <= 24 else 3)
        fig, ax = plt.subplots(figsize=(9 + 2.2 * ncol, 6))
        for zone_name, times, series in entries:
            ax.plot(times, series, label=translate_label(zone_name), **line_style(len(times)))
        ax.set_xlabel(TIME_LABEL)
        ax.set_ylabel(FIELD_LABELS.get(field, field))
        cat_en = translate_label(cat)
        ax.set_title(f"Topo zones - {cat_en} - {field} (spatial average per zone)")
        ax.grid(True, alpha=0.3)
        clean_axis(ax)
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  ncol=ncol, borderaxespad=0.0)
        fig.tight_layout()
        out_path = os.path.join(dest, f"topo_{cat_en.replace(' ', '_')}_{field}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Listo -> {out_path}")

    if summary_rows:
        csv_path = os.path.join(dest, "topo_resumen.csv")
        df = pd.DataFrame(summary_rows)
        df.insert(1, "zone_en", df["zona"].map(translate_label))
        df.insert(3, "category_en", df["categoria"].map(translate_label))
        df.to_csv(csv_path, index=False)
        print(f"Listo -> {csv_path}")


