#!/usr/bin/env python3
"""
run_pipeline.py
===============
Script maestro de orquestación para ejecutar secuencialmente todo el flujo de
procesamiento de datos, generación de gráficos y visualizadores 3D en OpenFOAM.
"""

import argparse
import os
import sys
import subprocess
import time

# ---------------------------------------------------------------------------
# Definición de las etapas del pipeline en orden estricto de ejecución
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    {
        "id": "prepare",
        "script": "prepareVTPOFFilesV01.py",
        "description": "Preparación y conversión de archivos VTP/OFF",
        "args": []
    },
    {
        "id": "unir",
        "script": "unir_datos.py",
        "description": "Unificación y concatenación de archivos de datos",
        "args": []
    },
    {
        "id": "process",
        "script": "Process_data.py",
        "description": "Procesamiento numérico principal de variables CFD",
        "args": []
    },
    {
        "id": "ach",
        "script": "ACH.py",
        "description": "Cálculo de Renovaciones de Aire por Hora (ACH)",
        "args": []
    },
    {
        "id": "graficos_gen",
        "script": "Graficos.py",
        "description": "Generación de gráficos globales y resúmenes",
        "args": []
    },
    {
        "id": "graficos_ind",
        "script": "Graficos_Individuales.py",
        "description": "Generación de gráficos individuales por sonda/zona",
        "args": []
    },
    {
        "id": "vis_probes",
        "script": "visualize_probes.py",
        "description": "Visualizador 3D HTML de Sondas, Puntos y topoSets",
        "args": []
    },
    {
        "id": "vis_planes",
        "script": "visualize_planes.py",
        "description": "Visualizador 3D HTML de Planos de Corte (cuttingPlanes)",
        "args": []
    },
]


# ---------------------------------------------------------------------------
# Colores y Formato en Consola
# ---------------------------------------------------------------------------
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    print(f"{Colors.HEADER}{Colors.BOLD}" + "=" * 75)
    print(" 🚀 EJECUTOR MAESTRO DE PIPELINE CFD / OPENFOAM")
    print("=" * 75 + f"{Colors.ENDC}\n")


def run_script(step, continue_on_error=False):
    script_name = step["script"]
    description = step["description"]

    print(f"{Colors.OKCYAN}{Colors.BOLD}▶ Ejecutando [{script_name}]{Colors.ENDC}: {description}")

    if not os.path.exists(script_name):
        print(f"  {Colors.WARNING}⚠️  Advertencia: El archivo '{script_name}' no existe en el directorio actual. Omitiendo.{Colors.ENDC}\n")
        return False

    cmd = [sys.executable, script_name] + step["args"]
    start_time = time.time()

    try:
        # Ejecución interactiva transmitiendo salida en tiempo real
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"  {Colors.OKGREEN}✔ Completado con éxito en {elapsed:.2f}s{Colors.ENDC}\n")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"  {Colors.FAIL}❌ Error al ejecutar {script_name} (Código de salida: {e.returncode}) tras {elapsed:.2f}s{Colors.ENDC}")
        if not continue_on_error:
            print(f"  {Colors.FAIL}Abortando el pipeline. Usa --continue-on-error para ignorar fallos.{Colors.ENDC}\n")
            sys.exit(1)
        else:
            print(f"  {Colors.WARNING}Continuando con la siguiente etapa según la opción --continue-on-error...{Colors.ENDC}\n")
            return False


# ---------------------------------------------------------------------------
# CLI y Lógica Principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta en orden secuencial todos los scripts del caso CFD.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Permite continuar con el siguiente script si alguno falla."
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Lista de scripts a omitir por nombre (ej: --skip ACH.py Graficos.py)"
    )
    parser.add_argument(
        "--only", nargs="*", default=[],
        help="Ejecutar únicamente los scripts indicados por nombre."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Muestra el orden de los scripts registrados y sale."
    )

    args = parser.parse_args()

    print_banner()

    if args.list:
        print(f"{Colors.BOLD}Orden de ejecución programado:{Colors.ENDC}")
        for idx, step in enumerate(PIPELINE_STEPS, 1):
            print(f"  {idx}. {step['script']:<25} - {step['description']}")
        print()
        return

    # Filtrado de etapas
    steps_to_run = []
    for step in PIPELINE_STEPS:
        name = step["script"]
        if args.only and name not in args.only:
            continue
        if name in args.skip:
            print(f"{Colors.WARNING}⏭  Omitiendo voluntariamente: {name}{Colors.ENDC}")
            continue
        steps_to_run.append(step)

    if not steps_to_run:
        print(f"{Colors.WARNING}No hay scripts seleccionados para ejecutar.{Colors.ENDC}")
        return

    total_start = time.time()
    successful = 0
    failed = 0

    print(f"Comenzando ejecución de {len(steps_to_run)} etapas...\n" + "-" * 75)

    for step in steps_to_run:
        ok = run_script(step, continue_on_error=args.continue_on_error)
        if ok:
            successful += 1
        else:
            failed += 1

    total_elapsed = time.time() - total_start

    print("=" * 75)
    print(f"{Colors.BOLD}RESUMEN DEL PIPELINE:{Colors.ENDC}")
    print(f"  • Tiempo Total: {total_elapsed:.2f} segundos")
    print(f"  • Éxitos: {Colors.OKGREEN}{successful}{Colors.ENDC}")
    if failed > 0:
        print(f"  • Fallos/Omitidos: {Colors.FAIL}{failed}{Colors.ENDC}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()