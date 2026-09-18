#!/usr/bin/env python3
"""
dashboard_plots.py
===================
Orquestador: llama a los 4 modulos de la carpeta 'postproc/' (ach.py,
probes.py, topo.py, perfiles.py) en orden, y escribe todo en resultsSbase.

No tiene logica propia de lectura/graficado -- eso vive en cada modulo por
separado, precisamente para que un cambio futuro en una seccion no arriesgue
romper las demas (cada modulo es un archivo independiente).

  A. ACH: salida + promedio volumetrico + ACH por caudal          -> ach.py
  B. Sondas: difusores/inlet/outlet, esquinas, paciente           -> probes.py
  C. Zonas Topo_* (bandas, esquinas, zona del paciente)           -> topo.py
  D. Perfiles de linea: video + capturas cada N segundos          -> perfiles.py

Titulos y ejes de TODAS las graficas en ingles (incluso lo que en los dicts
de OpenFOAM esta en espanol, como 'banda_baja' o 'piso'/'techo' -- eso se
traduce automaticamente solo para lo que se muestra en pantalla, nunca para
la busqueda real de archivos/carpetas).

USO
---
    python dashboard_plots.py --import-dir /mnt/datos/OpenFOAM_Data/imports/Q1_2026-08-04

Por defecto:
  - Los archivos de definicion de sondas se buscan en
    <import-dir>/system/probes_difusores, probes_esquinas, probes_paciente.
  - Las carpetas de postProcessing se derivan del nombre del OBJETO
    declarado en cada archivo de definicion.
  - Los resultados se guardan en <results-base>/<nombre-de-import-dir>.

Requisitos: matplotlib, numpy, pandas, ffmpeg (para D)
    pip install matplotlib numpy pandas
"""
import argparse
import os
import sys

# Los 4 modulos viven en 'postproc/', al lado de este archivo. Se agrega
# esa ruta al path SIN importar desde donde se corra el script (no depende
# del directorio de trabajo actual, solo de donde este dashboard_plots.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "postproc"))

# OJO: los 4 modulos se importan uno por uno, DENTRO de cada bloque try de
# main() (no aqui arriba) -- si estuvieran importados a nivel de modulo y
# UNO SOLO tuviera un error de sintaxis o no existiera, todo el script
# moriria antes de correr nada, justo lo que se queria evitar al separarlos.


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

    ap.add_argument("--perfiles-dir", default=None,
                     help="Override: postProcessing/velocityProfiles")
    ap.add_argument("--perfiles-field", default="U",
                     help="Campo a animar en los perfiles de linea (default: U)")
    ap.add_argument("--perfiles-component", default="mag",
                     help="Componente del campo de perfiles: mag/x/y/z (default: mag)")
    ap.add_argument("--perfiles-snapshot-every", type=float, default=10.0,
                     help="Cada cuantos segundos SIMULADOS sacar una imagen estatica "
                          "del perfil (default: 10)")
    ap.add_argument("--perfiles-fps", type=int, default=10,
                     help="Cuadros por segundo del video del perfil (default: 10)")
    ap.add_argument("--skip-topo", action="store_true",
                     help="No generar las graficas de las zonas Topo_*")
    ap.add_argument("--skip-perfiles", action="store_true",
                     help="No generar el video/capturas de los perfiles de linea")
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
    try:
        from ach import run_ach_section
        run_ach_section(args.salida_dir, args.age_volavg_dir, outdir)
    except Exception as e:
        print(f"[aviso] la seccion A (ACH) fallo sin tumbar el resto: {e}", file=sys.stderr)

    print("\n=== A.2 ACH por caudal (Q/V x 3600) ===")
    try:
        from ach import run_flow_ach_section, detect_room_volume
        volume, vol_source = detect_room_volume(args.import_dir, args.volume)
        run_flow_ach_section(args.flujo_dir, args.densidad_dir, volume, vol_source, outdir)
    except Exception as e:
        print(f"[aviso] la seccion A.2 (ACH por caudal) fallo sin tumbar el resto: {e}",
              file=sys.stderr)

    print("\n=== B.1 Difusores/Inlet/Outlet: promedio por altura vs t ===")
    try:
        from probes import discover_definition_file, parse_probe_definition_file, plot_difusores_averages
        def_path = discover_definition_file(args.import_dir, "difusores", args.difusores_def)
        obj_name, definitions = parse_probe_definition_file(def_path)
        difusores_dir = args.difusores_dir or os.path.join(pp, obj_name)
        print(f"[info] difusores: {def_path} -> {len(definitions)} sondas, "
              f"carpeta postProcessing: {difusores_dir}")
        plot_difusores_averages(difusores_dir, definitions, outdir)
    except Exception as e:
        print(f"[aviso] la seccion B.1 (difusores) fallo sin tumbar el resto: {e}",
              file=sys.stderr)

    print("\n=== B.2 Esquinas: Age vs t ===")
    try:
        from probes import discover_definition_file, parse_probe_definition_file, plot_esquinas_age
        def_path = discover_definition_file(args.import_dir, "esquinas", args.esquinas_def)
        obj_name, definitions = parse_probe_definition_file(def_path)
        esquinas_dir = args.esquinas_dir or os.path.join(pp, obj_name)
        print(f"[info] esquinas: {def_path} -> {len(definitions)} sondas, "
              f"carpeta postProcessing: {esquinas_dir}")
        plot_esquinas_age(esquinas_dir, definitions, outdir)
    except Exception as e:
        print(f"[aviso] la seccion B.2 (esquinas) fallo sin tumbar el resto: {e}",
              file=sys.stderr)

    print("\n=== B.3 Paciente: todas las variables vs t (por sonda) ===")
    try:
        from probes import discover_definition_file, parse_probe_definition_file, plot_paciente_all
        def_path = discover_definition_file(args.import_dir, "paciente", args.paciente_def)
        obj_name, definitions = parse_probe_definition_file(def_path)
        paciente_dir = args.paciente_dir or os.path.join(pp, obj_name)
        print(f"[info] paciente: {def_path} -> {len(definitions)} sondas, "
              f"carpeta postProcessing: {paciente_dir}")
        plot_paciente_all(paciente_dir, definitions, outdir)
    except Exception as e:
        print(f"[aviso] la seccion B.3 (paciente) fallo sin tumbar el resto: {e}",
              file=sys.stderr)

    print("\n=== C. Zonas Topo_* (promedio espacial por zona vs t) ===")
    if args.skip_topo:
        print("[info] --skip-topo indicado, se omite esta seccion")
    else:
        try:
            from topo import run_topo_section
            run_topo_section(args.import_dir, outdir)
        except Exception as e:
            print(f"[aviso] la seccion de zonas Topo fallo sin tumbar el resto: {e}",
                  file=sys.stderr)

    print("\n=== D. Perfiles de linea (video + capturas cada "
          f"{args.perfiles_snapshot_every:.0f}s) ===")
    if args.skip_perfiles:
        print("[info] --skip-perfiles indicado, se omite esta seccion")
    else:
        perfiles_dir = args.perfiles_dir or os.path.join(pp, "velocityProfiles")
        if not os.path.isdir(perfiles_dir):
            print(f"[aviso] no existe {perfiles_dir}; se omite la seccion de perfiles "
                  f"(revisa --perfiles-dir si el functionObject de perfiles tiene otro "
                  f"nombre).", file=sys.stderr)
        else:
            try:
                from perfiles import run_profile_video_and_snapshots
                run_profile_video_and_snapshots(
                    perfiles_dir, outdir,
                    field=args.perfiles_field, component=args.perfiles_component,
                    snapshot_every=args.perfiles_snapshot_every, fps=args.perfiles_fps)
            except Exception as e:
                print(f"[aviso] la seccion de perfiles fallo sin tumbar el resto: {e}",
                      file=sys.stderr)

    print("\nListo. Todo generado.")


if __name__ == "__main__":
    main()
