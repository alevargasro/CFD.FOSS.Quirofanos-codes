#!/usr/bin/env python3
"""ach.py
======
Seccion A del pipeline: las 4 graficas clasicas de ACH (salida + promedio
volumetrico) mas el ACH por caudal (Q/V x 3600). Depende de postproc_common.py.
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

from common import (read_function_series, clean_axis, line_style, TIME_LABEL)


def get_age_series(base_dir):
    col_names, times, rows = read_function_series(base_dir)
    age_col = None
    for i in range(1, len(col_names)):
        if isinstance(rows[0][i], float):
            age_col = i
            break
    if age_col is None:
        raise ValueError(f"No numeric Age column found in {base_dir}")
    df = pd.DataFrame({"Time": times, "Age": [row[age_col] for row in rows]})
    df = df[(df["Time"] > 0) & (df["Age"] > 0)].reset_index(drop=True)
    df["ACH"] = 3600.0 / df["Age"]
    return df



def get_residencia(df, col_ach="ACH"):
    final_ach = df[col_ach].iloc[-1]
    df_temp = df.copy()
    df_temp["Var_%"] = (abs(df_temp[col_ach] - final_ach) / final_ach) * 100.0

    out_of_bounds = df_temp[df_temp["Var_%"] > 2.0]
    if out_of_bounds.empty:
        stable_row = df_temp.iloc[0]
    else:
        last_out_idx = out_of_bounds.index[-1]
        last_out_pos = df_temp.index.get_loc(last_out_idx)
        stable_row = (df_temp.iloc[last_out_pos + 1]
                      if last_out_pos < len(df_temp) - 1 else df_temp.iloc[-1])
    return stable_row["Time"], stable_row[col_ach], final_ach



def plot_ach_full_and_zoom(df, label, marker, color, dest_dir, file_prefix,
                            formula="3600 / Age", extra_col="Age",
                            extra_label="Final Age", extra_unit="s"):
    os.makedirs(dest_dir, exist_ok=True)
    t_res, ach_res, final_ach = get_residencia(df, "ACH")
    t_final = df["Time"].iloc[-1]
    has_extra = extra_col in df.columns
    final_age = df[extra_col].iloc[-1] if has_extra else float("nan")

    plt.figure(figsize=(10, 6))
    st = line_style(len(df), base_width=1.4)
    st.pop("marker", None); st.pop("markersize", None); st.pop("markevery", None)
    if len(df) <= 400:
        st.update(marker=marker, markersize=3, markevery=max(1, len(df) // 60))
    plt.plot(df["Time"], df["ACH"], color=color, label=f"ACH {label} ({formula})", **st)
    plt.axhline(y=20, color="tab:red", linestyle="--", linewidth=1.5,
                label="ASHRAE 170 minimum (20 ACH)")
    plt.axvline(x=t_res, color="tab:green", linestyle=":", linewidth=2,
                label=f"$t_{{residence}} = {t_res:.1f}$ s")
    plt.plot(t_res, ach_res, "go", markersize=8)

    text = (f"Final ACH: {final_ach:.2f} $h^{{-1}}$\n"
            + (f"{extra_label}: {final_age:.3g} {extra_unit}\n" if has_extra else "")
            + f"$t_{{residence}}$: {t_res:.1f} s")
    plt.annotate(text, xy=(t_res, ach_res),
                 xycoords="data", textcoords="axes fraction", xytext=(0.42, 0.16),
                 arrowprops=dict(facecolor="tab:green", shrink=0.05, width=1.5, headwidth=6),
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.85),
                 fontweight="bold")

    plt.title(f"Air Changes per Hour (ACH) - {label} (Full time)",
              fontsize=13, fontweight="bold")
    plt.xlabel("Simulation Time [s]", fontsize=11)
    plt.ylabel("ACH [$h^{-1}$]", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(dest_dir, f"{file_prefix}_FullTime.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Listo -> {p1}")

    t_start_50 = max(0, t_final - 50.0)
    df_50 = df[df["Time"] >= t_start_50]

    plt.figure(figsize=(10, 6))
    st50 = line_style(len(df_50), base_width=1.6)
    st50.pop("marker", None); st50.pop("markersize", None); st50.pop("markevery", None)
    if len(df_50) <= 400:
        st50.update(marker=marker, markersize=3.5, markevery=max(1, len(df_50) // 50))
    plt.plot(df_50["Time"], df_50["ACH"], color=color, label=f"ACH {label}", **st50)
    plt.axhline(y=20, color="tab:red", linestyle="--", linewidth=1.5,
                label="ASHRAE 170 minimum (20 ACH)")
    if t_res >= t_start_50:
        plt.axvline(x=t_res, color="tab:green", linestyle=":", linewidth=2,
                    label=f"$t_{{residence}} = {t_res:.1f}$ s")
        plt.plot(t_res, ach_res, "go", markersize=8)

    text_50 = (f"Average (last 50s): {df_50['ACH'].mean():.2f} $h^{{-1}}$\n"
               f"Final ACH: {final_ach:.2f} $h^{{-1}}$"
               + (f"\n{extra_label}: {final_age:.3g} {extra_unit}" if has_extra else ""))
    plt.annotate(text_50, xy=(df_50["Time"].iloc[-1], final_ach),
                 xycoords="data", textcoords="axes fraction", xytext=(0.04, 0.08),
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="aliceblue", alpha=0.9),
                 fontweight="bold")

    plt.title(f"Air Changes per Hour (ACH) - {label} "
              f"(Last 50s: t={t_start_50:.1f}s to {t_final:.1f}s)", fontsize=13, fontweight="bold")
    plt.xlabel("Simulation Time [s]", fontsize=11)
    plt.ylabel("ACH [$h^{-1}$]", fontsize=11)
    plt.xlim(t_start_50, t_final)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(dest_dir, f"{file_prefix}_Zoom_Last50s.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Listo -> {p2}")

    return p1, p2


# ---------------------------------------------------------------------------
# ACH por caudal:  Q = |phi| / rho   ->   ACH = (Q / V) * 3600
# (metodo heredado del pipeline anterior; independiente del campo Age)
# ---------------------------------------------------------------------------


_VOLUME_RE = re.compile(r"Total volume\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")



def detect_room_volume(import_dir, override=None):
    """Volumen del quirofano en m3. Si no se pasa --volume, lo busca en los
    logs de checkMesh del caso ('Total volume = X'), asi no hay que acordarse
    del numero por cada quirofano. Devuelve (volumen, procedencia)."""
    if override is not None:
        return float(override), "indicado con --volume"

    candidates = []
    for root, dirs, files in os.walk(import_dir):
        if "postProcessing" in root.split(os.sep):
            dirs[:] = []
            continue
        for f in files:
            if f.startswith("log.") and "checkMesh" in f:
                candidates.append(os.path.join(root, f))
    for path in sorted(candidates):
        try:
            with open(path, "r", errors="ignore") as fh:
                m = _VOLUME_RE.search(fh.read())
        except OSError:
            continue
        if m:
            try:
                vol = float(m.group(1))
            except ValueError:
                print(f"[aviso] no se pudo interpretar el volumen en {path}: "
                      f"'{m.group(1)}'", file=sys.stderr)
                continue
            if vol > 0:
                return vol, f"leido de {path}"
            print(f"[aviso] volumen no positivo en {path}: {vol}", file=sys.stderr)
    return None, None



def _first_numeric_column(col_names, rows):
    for i in range(1, len(rows[0])):
        if isinstance(rows[0][i], float):
            return i
    return None



def get_flow_ach_series(flujo_dir, densidad_dir, volume):
    """Combina el flujo masico en la salida (sum(phi), kg/s) con la densidad
    media del recinto para obtener el caudal volumetrico y de ahi el ACH."""
    cn_phi, t_phi, rows_phi = read_function_series(flujo_dir)
    cn_rho, t_rho, rows_rho = read_function_series(densidad_dir)

    i_phi = _first_numeric_column(cn_phi, rows_phi)
    i_rho = _first_numeric_column(cn_rho, rows_rho)
    if i_phi is None or i_rho is None:
        raise ValueError("no se encontro una columna numerica de phi o rho")

    df_phi = pd.DataFrame({"Time": t_phi, "phi": [r[i_phi] for r in rows_phi]})
    df_rho = pd.DataFrame({"Time": t_rho, "rho": [r[i_rho] for r in rows_rho]})
    df = pd.merge_asof(df_phi.sort_values("Time"), df_rho.sort_values("Time"),
                       on="Time", direction="nearest")

    df = df[(df["Time"] > 0) & (df["rho"] > 0)].reset_index(drop=True)
    if df.empty:
        raise ValueError("no quedaron instantes validos tras filtrar")
    df["Q_m3s"] = df["phi"].abs() / df["rho"]
    df["ACH"] = (df["Q_m3s"] / volume) * 3600.0
    return df



def run_flow_ach_section(flujo_dir, densidad_dir, volume, vol_source, outdir):
    dest = os.path.join(outdir, "ACH")
    os.makedirs(dest, exist_ok=True)

    for d, name in ((flujo_dir, "flujoOutlet"), (densidad_dir, "densidadMedia")):
        if not os.path.isdir(d):
            print(f"[aviso] no existe {d}; se omite el ACH por caudal "
                  f"(hace falta el functionObject '{name}').", file=sys.stderr)
            return

    if volume is None:
        print("[aviso] no se pudo determinar el volumen del quirofano y se omite "
              "el ACH por caudal. Pasa --volume <m3>, o deja un log.checkMesh del "
              "caso dentro de --import-dir para que lo lea solo.", file=sys.stderr)
        return

    print(f"[info] volumen del quirofano: {volume:.4g} m3 ({vol_source})")
    try:
        df = get_flow_ach_series(flujo_dir, densidad_dir, volume)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"[aviso] no se pudo calcular el ACH por caudal: {e}", file=sys.stderr)
        return

    plot_ach_full_and_zoom(df, "Flow rate", "^", "tab:orange", dest,
                           "5_6_ACH_FlowRate",
                           formula=f"Q/V x 3600, V={volume:.4g} m3",
                           extra_col="Q_m3s", extra_label="Final Q",
                           extra_unit="m3/s")
    csv_path = os.path.join(dest, "ACH_FlowRate_Report.csv")
    df.to_csv(csv_path, index=False)
    print(f"Listo -> {csv_path}")



def run_ach_section(salida_dir, age_volavg_dir, outdir):
    dest = os.path.join(outdir, "ACH")
    os.makedirs(dest, exist_ok=True)

    try:
        df_out = get_age_series(salida_dir)
        plot_ach_full_and_zoom(df_out, "Outlet", "o", "tab:blue", dest, "1_2_ACH_Outlet")
        df_out.to_csv(os.path.join(dest, "ACH_Outlet_Report.csv"), index=False)
        print(f"Listo -> {os.path.join(dest, 'ACH_Outlet_Report.csv')}")
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"[aviso] no se pudo generar el ACH de salida desde {salida_dir}: {e}\n"
              f"        (graficas 1 y 2 omitidas)", file=sys.stderr)

    if not os.path.isdir(age_volavg_dir):
        print(f"[aviso] no existe {age_volavg_dir}; se omiten las graficas 3 y 4 "
              f"(ACH volumetrico general).\n"
              f"        Esa corrida no incluyo el functionObject de promedio volumetrico "
              f"de Age. Puedes generarlo a posteriori con 'postProcess -func ...' si los "
              f"campos Age quedaron escritos en los directorios de tiempo, o apuntar a "
              f"otra carpeta con --age-volavg-dir.", file=sys.stderr)
        return

    try:
        df_vol = get_age_series(age_volavg_dir)
        plot_ach_full_and_zoom(df_vol, "Room Volume Average", "s", "tab:purple",
                                dest, "3_4_ACH_Volume")
        df_vol.to_csv(os.path.join(dest, "ACH_Volume_Report.csv"), index=False)
        print(f"Listo -> {os.path.join(dest, 'ACH_Volume_Report.csv')}")
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"[aviso] no se pudo generar el ACH volumetrico desde {age_volavg_dir}: {e}\n"
              f"        (graficas 3 y 4 omitidas)", file=sys.stderr)


# ---------------------------------------------------------------------------
# B. Sondas nuevas: identificacion automatica de grupos (difusores/esquinas/
#    paciente) a partir de los comentarios del archivo de definicion.
# ---------------------------------------------------------------------------

