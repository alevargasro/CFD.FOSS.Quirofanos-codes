#!/usr/bin/env python3
"""
preparar_vtp_sincronizado.py
============================
Sincroniza temporalmente planos VTP para ParaView respetando la línea
de tiempo global de la simulación.
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path

def fix_float_to_double(filepath):
    """Corrige Float32 a Float64 para compatibilidad con ParaView 5.9+."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if 'type="Float32"' in content:
            content = content.replace('type="Float32"', 'type="Float64"')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    except Exception:
        pass

def procesar_cortes(directorio_cortes, carpeta_salida="orderedVTPFiles", corregir_double=True):
    base_path = Path(directorio_cortes).resolve()
    out_path = base_path / carpeta_salida
    
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Escanear carpetas de tiempo y ordenarlas explícitamente por valor flotante
    tiempos_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name != carpeta_salida:
            try:
                t_val = float(item.name)
                tiempos_dirs.append((t_val, item))
            except ValueError:
                continue

    if not tiempos_dirs:
        print(f"❌ No se encontraron carpetas de tiempo numéricas en {base_path}")
        return

    # Orden numérico real (evita problemas de ordenamiento de texto)
    tiempos_dirs.sort(key=lambda x: x[0])
    tiempos_ordenados = [t[0] for t in tiempos_dirs]

    print(f"⌛ Rango de tiempos detectado: {tiempos_ordenados[0]} a {tiempos_ordenados[-1]} ({len(tiempos_ordenados)} pasos de tiempo)")

    # 2. Registrar todos los planos únicos
    planos = set()
    for _, folder in tiempos_dirs:
        for f in folder.iterdir():
            if f.is_file() and f.name.lower().endswith('.vtp'):
                planos.add(f.stem)

    planos = sorted(list(planos))
    print(f"📐 Planos detectados ({len(planos)}): {', '.join(planos)}\n")

    # 3. Procesar cada plano con sincronización a la línea de tiempo global
    for plano in planos:
        print(f"Procesando plano: {plano}...")
        archivos_series = []
        
        for idx, (t_val, folder) in enumerate(tiempos_dirs):
            archivo_origen = folder / f"{plano}.vtp"
            
            # Si no está en minúsculas/mayúsculas exactas, buscar ignorando mayúsculas
            if not archivo_origen.exists():
                coincidentes = [f for f in folder.iterdir() if f.stem.lower() == plano.lower() and f.suffix.lower() == '.vtp']
                if coincidentes:
                    archivo_origen = coincidentes[0]
                else:
                    # El plano no existe en este tiempo (ej. plano de abajo antes de 38.8)
                    continue

            # El índice (idx) corresponde al tiempo GLOBAL de la simulación
            nombre_destino = f"{plano}_{idx:05d}.vtp"
            path_destino = out_path / nombre_destino

            shutil.copy2(archivo_origen, path_destino)

            if corregir_double:
                fix_float_to_double(path_destino)

            # Registrar en el archivo de serie de ParaView
            archivos_series.append({
                "name": nombre_destino,
                "time": t_val
            })

        # Generar archivo .vtp.series para ParaView
        series_file = out_path / f"{plano}.vtp.series"
        series_data = {
            "file-series-version": "1.0",
            "files": archivos_series
        }
        with open(series_file, 'w', encoding='utf-8') as f:
            json.dump(series_data, f, indent=2)

        print(f"  └─ {len(archivos_series)} frames sincronizados -> {series_file.name}")

    print(f"\n✅ Proceso completado. Resultados en: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sincroniza cortes VTP para ParaView")
    parser.add_argument("-d", "--directory", required=True, help="Ruta a la carpeta cortesVTK")
    parser.add_argument("--no-correct", action="store_true", help="Desactivar corrección Float32->Float64")
    args = parser.parse_args()

    procesar_cortes(args.directory, corregir_double=not args.no_correct)