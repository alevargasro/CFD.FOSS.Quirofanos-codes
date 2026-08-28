#!/usr/bin/env python3
import os
import sys

def es_numero(valor):
    """Verifica si el nombre de una carpeta es numérico (un tiempo válido)."""
    try:
        float(valor)
        return True
    except ValueError:
        return False

# 1. Definir la carpeta raíz de postprocesamiento
base_dir = "postProcessing"

if not os.path.exists(base_dir):
    print(f"❌ Error: No se encontró la carpeta '{base_dir}'. Ejecuta este script en la raíz del caso.")
    sys.exit(1)

# Carpetas de imágenes o cortes que NO deben ser procesadas como tablas/datos
CARPETAS_A_IGNORAR = {"cortesVTK", "orderedVTPFiles", "__pycache__"}

print("==================================================")
print("🚀 Iniciando CONCATENACIÓN TOTAL AUTOMÁTICA (Quirófano 3)...")
print("==================================================")

# Detectar automáticamente TODAS las funciones configuradas en controlDict
todas_las_zonas = [
    d for d in os.listdir(base_dir) 
    if os.path.isdir(os.path.join(base_dir, d)) and d not in CARPETAS_A_IGNORAR and not d.startswith('.')
]

if not todas_las_zonas:
    print("❌ No se encontraron carpetas de postprocesamiento para concatenar.")
    sys.exit(1)

print(f"🔍 Se detectaron {len(todas_las_zonas)} zonas/funciones de OpenFOAM:")
for z in sorted(todas_las_zonas):
    print(f"   • {z}")
print("-" * 50)

# --- PROCESO ÚNICO Y DINÁMICO DE CONCATENACIÓN ---
for zona in sorted(todas_las_zonas):
    ruta_zona = os.path.join(base_dir, zona)
        
    # Detectar dinámicamente TODAS las carpetas de tiempo numéricas
    carpetas_tiempo = sorted(
        [d for d in os.listdir(ruta_zona) if os.path.isdir(os.path.join(ruta_zona, d)) and es_numero(d)],
        key=lambda x: float(x)
    )

    if not carpetas_tiempo:
        continue

    print(f"\n📂 Procesando Zona: '{zona}' ({len(carpetas_tiempo)} pasos de tiempo)...")

    # Identificar todos los archivos únicos dentro de los pasos de tiempo (sin importar la extensión)
    archivos_unicos = set()
    for t_dir in carpetas_tiempo:
        ruta_tiempo = os.path.join(ruta_zona, t_dir)
        for f in os.listdir(ruta_tiempo):
            ruta_completa = os.path.join(ruta_tiempo, f)
            if (os.path.isfile(ruta_completa) and 
                not f.startswith('.') and 
                not f.endswith('.py') and 
                not f.endswith('_concatenado') and 
                not f.endswith('_concatenado.dat')):
                archivos_unicos.add(f)

    for archivo_nombre in sorted(archivos_unicos):
        headers = []
        datos_combinados = {}
        
        # Respetar extensión .dat si el archivo original la tiene
        if archivo_nombre.endswith('.dat'):
            nombre_base = archivo_nombre[:-4]
            archivo_salida = os.path.join(ruta_zona, f"{nombre_base}_concatenado.dat")
        else:
            archivo_salida = os.path.join(ruta_zona, f"{archivo_nombre}_concatenado")
        
        for t_dir in carpetas_tiempo:
            ruta_archivo = os.path.join(ruta_zona, t_dir, archivo_nombre)
            if os.path.exists(ruta_archivo):
                with open(ruta_archivo, 'r') as f:
                    for line in f:
                        line_clean = line.strip()
                        if not line_clean: 
                            continue
                        if line_clean.startswith("#"):
                            if not datos_combinados and line_clean not in headers:
                                headers.append(line_clean)
                            continue
                        partes = line_clean.split()
                        if partes:
                            try:
                                datos_combinados[float(partes[0])] = line_clean
                            except ValueError:
                                continue
        
        if datos_combinados:
            with open(archivo_salida, 'w') as f_out:
                if headers:
                    f_out.write("\n".join(headers) + "\n")
                for t in sorted(datos_combinados.keys()):
                    f_out.write(datos_combinados[t] + "\n")
            print(f"   ✅ Unificado: {zona}/{os.path.basename(archivo_salida)} ({len(datos_combinados)} filas)")

print("\n==================================================")
print("🎉 ¡Proceso terminado! Todos los datos del Quirófano 3 fueron unificados.")
print("==================================================")