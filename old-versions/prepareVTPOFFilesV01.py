#!/usr/bin/env python3
import os
import sys
import shutil
from argparse import ArgumentParser

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def find_cortes_vtk_dir(start_dir):
    """
    Busca estrictamente la carpeta 'postProcessing/cortesVTK/', 
    ignorando carpetas como 'postProcessingV00' o 'postProcessing(ESTENO)'.
    """
    cwd = os.path.abspath(start_dir)

    # 1. Si ya estamos parados dentro de 'cortesVTK'
    if os.path.basename(cwd) == "cortesVTK":
        return cwd

    # 2. Verificación directa de rutas prioritarias exactas
    direct_paths = [
        os.path.join(cwd, "postProcessing", "cortesVTK"),
        os.path.join(cwd, "cortesVTK")
    ]

    for path in direct_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return os.path.abspath(path)

    # 3. Búsqueda explícita descartando variaciones de postProcessing
    for root, dirs, _ in os.walk(cwd):
        # Excluir explícitamente carpetas alternativas para que os.walk no entre en ellas
        dirs[:] = [d for d in dirs if not (d.startswith("postProcessingV") or "ESTENO" in d or d.endswith("V00"))]
        
        if os.path.basename(root) == "cortesVTK":
            path_parts = os.path.normpath(root).split(os.sep)
            if "postProcessing" in path_parts:
                return os.path.abspath(root)

    return cwd

def main():
    parser = ArgumentParser(
        description='''
        Ordena archivos VTP de OpenFOAM estrictamente dentro de postProcessing/cortesVTK/orderedVTPFiles
        ''')
    parser.add_argument("-f", "--file", dest='vtpDataFile', default=None, 
                        type=str, help="Nombre del archivo VTP (ej: plano_XZ). Si se omite, procesa todos.")
    parser.add_argument("-d", "--directory", dest='targetDir', default="./", 
                        type=str, help="Directorio raíz del caso o carpeta objetivo")
    parser.add_argument("-c", "--correct", dest='correctForParaview', action='store_true',
                        help="Corrige la precisión 'float' a 'double' para evitar errores en ParaView")

    args = parser.parse_args()
    
    # Apuntar exclusivamente a postProcessing/cortesVTK
    vtp_dir = find_cortes_vtk_dir(args.targetDir)
    storeDir = os.path.join(vtp_dir, "orderedVTPFiles")
    
    print("==================================================")
    print(f"📂 Carpeta de cortes seleccionada: {os.path.abspath(vtp_dir)}")
    print(f"📂 Destino de archivos ordenados: {os.path.abspath(storeDir)}")
    print("==================================================")

    if not os.path.exists(vtp_dir):
        print(f"❌ No se encontró la carpeta 'cortesVTK' dentro de 'postProcessing' en:\n   {os.path.abspath(args.targetDir)}")
        sys.exit(1)

    if not os.path.exists(storeDir):
        os.makedirs(storeDir)

    # 1. Obtener carpetas de tiempo dentro de postProcessing/cortesVTK
    subdirs = [d for d in os.listdir(vtp_dir) if os.path.isdir(os.path.join(vtp_dir, d)) and d != "orderedVTPFiles"]
    time_dirs = []
    
    for d in subdirs:
        if is_float(d):
            sd_path = os.path.join(vtp_dir, d)
            try:
                # Validar que la carpeta de tiempo contenga archivos .vtp
                if any(f.endswith('.vtp') for f in os.listdir(sd_path)):
                    time_dirs.append((float(d), d))
            except Exception:
                pass
            
    if not time_dirs:
        print(f"❌ No se encontraron carpetas numéricas de tiempo con archivos .vtp en:\n   {vtp_dir}")
        sys.exit(1)
        
    # Ordenar por valor numérico real del tiempo
    time_dirs.sort(key=lambda x: x[0])
    print(f"⏱️ Se detectaron {len(time_dirs)} pasos de tiempo (desde t={time_dirs[0][1]} hasta t={time_dirs[-1][1]})")

    # 2. Identificar los archivos .vtp a procesar
    vtp_files_to_process = []
    if args.vtpDataFile:
        base_name = args.vtpDataFile.replace(".vtp", "")
        vtp_files_to_process.append(f"{base_name}.vtp")
    else:
        detected_vtp = set()
        for _, original_dir in time_dirs:
            subdir_path = os.path.join(vtp_dir, original_dir)
            for file in os.listdir(subdir_path):
                if file.endswith(".vtp"):
                    detected_vtp.add(file)
        vtp_files_to_process = sorted(list(detected_vtp))
        print(f"🔍 Planos detectados automáticamente: {vtp_files_to_process}")

    if not vtp_files_to_process:
        print("❌ No se encontraron archivos VTP para procesar.")
        sys.exit(1)

    # 3. Copiar secuencialmente a postProcessing/cortesVTK/orderedVTPFiles/
    for vtp_filename in vtp_files_to_process:
        plane_name = vtp_filename.replace(".vtp", "")
        print(f"\n🔄 Procesando plano: '{plane_name}'")
        
        file_counter = 0
        for _, original_dir in time_dirs:
            file_origin = os.path.join(vtp_dir, original_dir, vtp_filename)
            
            if os.path.exists(file_origin):
                new_filename = f"{plane_name}_{file_counter:04d}.vtp"
                file_dest = os.path.join(storeDir, new_filename)
                
                shutil.copy2(file_origin, file_dest)
                file_counter += 1
                
                if args.correctForParaview:
                    try:
                        with open(file_dest, 'rb') as f_bin:
                            data = f_bin.read()
                        data = data.replace(b'type="float"', b'type="double"')
                        data = data.replace(b'type=\'float\'', b'type=\'double\'')
                        with open(file_dest, 'wb') as f_bin:
                            f_bin.write(data)
                    except Exception as e:
                        print(f"  ⚠️ Error ajustando precisión en {new_filename}: {e}")
                        
        print(f"  ✅ ¡Listo! Se generaron {file_counter} archivos de '{plane_name}' en 'orderedVTPFiles/'")

    print(f"\n🎉 ¡Proceso completado! Archivos guardados correctamente en:\n   {os.path.abspath(storeDir)}")

if __name__ == "__main__":
    main()