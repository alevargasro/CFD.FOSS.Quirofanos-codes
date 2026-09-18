#!/usr/bin/env python3
"""
dashboard_plots.py
===================
Genera lo siguiente:

  A. Las 4 graficas de ACH.py (criticas, sin cambios de fondo):
       1. ACH a la salida (promedioSalida) -- todos los tiempos
       2. ACH a la salida -- zoom ultimos 50s
       3. ACH volumetrico general (AgeVolumeAverage) -- todos los tiempos
       4. ACH volumetrico general -- zoom ultimos 50s
     (cada una con linea ASHRAE 170 minimo, tiempo de residencia con
     criterio <=2% de variacion, y reporte CSV)

  B. Sondas nuevas (difusores/inlet/outlet, esquinas, paciente).
     El script IDENTIFICA AUTOMATICAMENTE cuantos difusores, esquinas
     y puntos de paciente hay -- lee los comentarios del archivo de
     definicion de sondas (el que tiene "// diff_1 @ 10cm", etc.) y
     arma los grupos a partir de eso, sin necesidad de tocar el codigo
     al cambiar de quirofano (Q1, Q4, el que sea).

       1. Difusores/inlet/outlet: promedio de las 9 sondas de cada
          altura (10/20/30 cm) vs t, una grafica por grupo y por
          variable (U, p, T...), con 3 lineas (una por altura).
       2. Esquinas: Age vs t, todas las esquinas en una sola grafica
          (sin promediar).
       3. Paciente: todas las variables disponibles vs t, sonda por
          sonda (sin promediar).

Titulos y ejes en ingles, maximo 3 decimales, sin notacion de offset.

USO
---
    python dashboard_plots.py --import-dir /mnt/datos/OpenFOAM_Data/imports/Q1_2026-08-04
    python dashboard_plots.py --import-dir /mnt/datos/OpenFOAM_Data/imports/Q4_2026-08-21

Por defecto:
  - Los archivos de definicion de sondas (con los comentarios "// diff_1 @ 10cm")
    se buscan en <import-dir>/system/probes_difusores, probes_esquinas,
    probes_paciente. Si estan en otro lado, usa --difusores-def / --esquinas-def
    / --paciente-def para indicar la ruta exacta.
  - Las carpetas de postProcessing correspondientes se derivan del NOMBRE DEL
    OBJETO declarado en cada archivo de definicion (ej. "inletOutletProbes",
    "cornerProbes", "patientProbes"), asi que tampoco hay que tocar nada si
    renombras las cosas de forma consistente. Si algo no calza, usa
    --difusores-dir / --esquinas-dir / --paciente-dir.
  - Los resultados se guardan en <results-base>/<nombre-de-la-carpeta-Q#_fecha>,
    o sea al mismo nivel que "imports" y con el mismo nombre de carpeta que
    trajiste (Q1_2026-08-04, Q4_2026-08-21, etc.)

Requisitos: matplotlib, numpy, pandas
    pip install matplotlib numpy pandas
"""
import argparse
import glob
import gzip
import os
import re
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


def check_convergence(df, col_ach="ACH", tol_pct=1.0):
    """El criterio del 2% compara contra el ULTIMO valor, asi que en una
    corrida truncada siempre declara convergencia: los puntos finales estan
    por construccion cerca de si mismos. Esto mira si la serie TODAVIA se
    mueve, comparando la media del ultimo 10% del tiempo contra la del 10%
    anterior. Devuelve (convergio, cambio_pct)."""
    t = df["Time"].to_numpy(dtype=float)
    y = df[col_ach].to_numpy(dtype=float)
    if len(t) < 10:
        return True, 0.0
    span = t[-1] - t[0]
    if span <= 0:
        return True, 0.0
    last = y[t >= t[-1] - 0.10 * span]
    prev = y[(t >= t[-1] - 0.20 * span) & (t < t[-1] - 0.10 * span)]
    if len(last) == 0 or len(prev) == 0:
        return True, 0.0
    m_last, m_prev = float(np.nanmean(last)), float(np.nanmean(prev))
    if m_last == 0:
        return True, 0.0
    change = abs(m_last - m_prev) / abs(m_last) * 100.0
    return change <= tol_pct, change


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

    converged, change_pct = check_convergence(df)
    title_suffix = "" if converged else "  [NO CONVERGIDO]"
    if not converged:
        print(f"[AVISO] '{label}': la serie de ACH aun cambia {change_pct:.2f}% entre "
              f"el ultimo 10% del tiempo y el 10% anterior. El valor final "
              f"({final_ach:.2f} ACH) NO es asintotico -- es donde se corto la "
              f"simulacion. El t_residencia del criterio del 2% no es fiable aqui: "
              f"ese criterio compara contra el ultimo punto, asi que en una corrida "
              f"truncada siempre declara convergencia. Alarga la corrida antes de "
              f"citar este numero en un reporte.", file=sys.stderr)
        plt.figtext(0.5, 0.01, f"Serie aun en transitorio: {change_pct:.2f}% de cambio "
                    f"en el ultimo 10% del tiempo. Valor final no asintotico.",
                    ha="center", fontsize=9, color="tab:red", style="italic")

    plt.title(f"Air Changes per Hour (ACH) - {label} (Full time){title_suffix}",
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

    label_map = {i: f"{d['group']} ({d['coords'][0]:.2f}, {d['coords'][1]:.2f})"
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

    label_map = {i: (d["raw_label"] or f"punto {i + 1}") for i, d in enumerate(definitions)}

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import-dir", required=True,
                     help="Carpeta del caso (ej. imports/Q4_2026-08-21), debe contener "
                          "'postProcessing/' y, por defecto, 'system/probes_*' con los "
                          "comentarios que identifican cada grupo de sondas.")
    ap.add_argument("--salida-dir", default=None, help="Override: postProcessing/promedioSalida")
    ap.add_argument("--age-volavg-dir", default=None, help="Override: postProcessing/AgeVolumeAverage")
    ap.add_argument("--flujo-dir", default=None, help="Override: postProcessing/flujoOutlet")
    ap.add_argument("--densidad-dir", default=None, help="Override: postProcessing/densidadMedia")
    ap.add_argument("--volume", type=float, default=None,
                     help="Volumen del quirofano en m3, para el ACH por caudal "
                          "(Q/V*3600). Si no se indica, se busca 'Total volume' en "
                          "los log.checkMesh que haya bajo --import-dir.")

    ap.add_argument("--difusores-def", default=None,
                     help="Ruta al archivo de definicion de sondas de difusores/inlet/outlet "
                          "(default: <import-dir>/system/probes_difusores)")
    ap.add_argument("--esquinas-def", default=None,
                     help="Ruta al archivo de definicion de sondas de esquinas "
                          "(default: <import-dir>/system/probes_esquinas)")
    ap.add_argument("--paciente-def", default=None,
                     help="Ruta al archivo de definicion de sondas de paciente "
                          "(default: <import-dir>/system/probes_paciente)")

    ap.add_argument("--difusores-dir", default=None,
                     help="Override de la carpeta postProcessing/<obj> para difusores")
    ap.add_argument("--esquinas-dir", default=None,
                     help="Override de la carpeta postProcessing/<obj> para esquinas")
    ap.add_argument("--paciente-dir", default=None,
                     help="Override de la carpeta postProcessing/<obj> para paciente")

    ap.add_argument("--outdir", default=None)
    ap.add_argument("--results-base", default="/mnt/datos/OpenFOAM_Data/results")
    args = ap.parse_args()

    pp = os.path.join(args.import_dir, "postProcessing")
    args.salida_dir = args.salida_dir or os.path.join(pp, "promedioSalida")
    args.age_volavg_dir = args.age_volavg_dir or os.path.join(pp, "AgeVolumeAverage")
    args.flujo_dir = args.flujo_dir or os.path.join(pp, "flujoOutlet")
    args.densidad_dir = args.densidad_dir or os.path.join(pp, "densidadMedia")

    if args.outdir:
        outdir = args.outdir
    else:
        case_name = os.path.basename(os.path.normpath(args.import_dir))
        outdir = os.path.join(args.results_base, case_name)
        print(f"[info] --outdir no indicado, usando: {outdir}")
    os.makedirs(outdir, exist_ok=True)

    print("\n=== A. Graficas de ACH (Salida + Promedio Volumetrico) ===")
    run_ach_section(args.salida_dir, args.age_volavg_dir, outdir)

    print("\n=== A.2 ACH por caudal (Q/V x 3600) ===")
    volume, vol_source = detect_room_volume(args.import_dir, args.volume)
    run_flow_ach_section(args.flujo_dir, args.densidad_dir, volume, vol_source, outdir)

    print("\n=== B.1 Difusores/Inlet/Outlet: promedio por altura vs t ===")
    def_path = discover_definition_file(args.import_dir, "difusores", args.difusores_def)
    obj_name, definitions = parse_probe_definition_file(def_path)
    difusores_dir = args.difusores_dir or os.path.join(pp, obj_name)
    print(f"[info] difusores: {def_path} -> {len(definitions)} sondas, "
          f"carpeta postProcessing: {difusores_dir}")
    plot_difusores_averages(difusores_dir, definitions, outdir)

    print("\n=== B.2 Esquinas: Age vs t ===")
    def_path = discover_definition_file(args.import_dir, "esquinas", args.esquinas_def)
    obj_name, definitions = parse_probe_definition_file(def_path)
    esquinas_dir = args.esquinas_dir or os.path.join(pp, obj_name)
    print(f"[info] esquinas: {def_path} -> {len(definitions)} sondas, "
          f"carpeta postProcessing: {esquinas_dir}")
    plot_esquinas_age(esquinas_dir, definitions, outdir)

    print("\n=== B.3 Paciente: todas las variables vs t (por sonda) ===")
    def_path = discover_definition_file(args.import_dir, "paciente", args.paciente_def)
    obj_name, definitions = parse_probe_definition_file(def_path)
    paciente_dir = args.paciente_dir or os.path.join(pp, obj_name)
    print(f"[info] paciente: {def_path} -> {len(definitions)} sondas, "
          f"carpeta postProcessing: {paciente_dir}")
    plot_paciente_all(paciente_dir, definitions, outdir)

    print("\nListo. Todo generado.")


if __name__ == "__main__":
    main()
