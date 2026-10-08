#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys

def parse_vector_column(series):
    """ Convierte columnas en formato vectorial de OpenFOAM '(ux uy uz)' a floats limpios """
    return (series.astype(str)
            .str.replace('(', '', regex=False)
            .str.replace(')', '', regex=False)
            .str.strip()
            .astype(float)
            .values)

def load_and_process_probes(folder_name):
    """ Función auxiliar para procesar todas las variables registradas por un grupo de sondas """
    base_path = os.path.join("postProcessing", folder_name)
    if not os.path.exists(base_path):
        return None

    # Buscar archivos de Presión (p_rgh o p)
    ruta_p = os.path.join(base_path, "p_rgh_concatenado")
    if not os.path.exists(ruta_p):
        ruta_p = os.path.join(base_path, "p_concatenado")

    # Buscar archivos de Velocidad
    ruta_U = os.path.join(base_path, "U_concatenado")
    
    # Rutas opcionales
    ruta_T   = os.path.join(base_path, "T_concatenado")
    ruta_Age = os.path.join(base_path, "Age_concatenado")
    ruta_ACH = os.path.join(base_path, "ACH_concatenado")

    # Verificación de archivos esenciales (Presión y Velocidad)
    if not os.path.exists(ruta_p) or not os.path.exists(ruta_U):
        # Si no hay datos de sondas válidos, retornar None
        return None

    print(f"📖 Leyendo y procesando grupo de sondas: '{folder_name}'...")

    # Lectura de archivos con Pandas
    data_p = pd.read_csv(ruta_p, sep=r'\s+', comment='#', header=None)
    data_U = pd.read_csv(ruta_U, sep=r'\s+', comment='#', header=None)
    
    # Lectura opcional/segura para variables adicionales
    data_T   = pd.read_csv(ruta_T, sep=r'\s+', comment='#', header=None) if os.path.exists(ruta_T) else None
    data_Age = pd.read_csv(ruta_Age, sep=r'\s+', comment='#', header=None) if os.path.exists(ruta_Age) else None
    data_ACH = pd.read_csv(ruta_ACH, sep=r'\s+', comment='#', header=None) if os.path.exists(ruta_ACH) else None

    # Tiempos de simulación
    t = data_p[0].values
    t_adquisicion = data_U[0].values
    t_T   = data_T[0].values if data_T is not None else t
    t_Age = data_Age[0].values if data_Age is not None else t
    t_ACH = data_ACH[0].values if data_ACH is not None else t

    # Extraer diccionarios escalares por sensor
    N_columnas_p = data_p.shape[1]
    probes_p   = {f'p{i}': data_p[i].values for i in range(1, N_columnas_p)}
    probes_T   = {f'T{i}': data_T[i].values for i in range(1, data_T.shape[1])} if data_T is not None else {}
    probes_Age = {f'Age{i}': data_Age[i].values for i in range(1, data_Age.shape[1])} if data_Age is not None else {}
    probes_ACH = {f'ACH{i}': data_ACH[i].values for i in range(1, data_ACH.shape[1])} if data_ACH is not None else {}

    # Descomponer componentes vectoriales de Velocidad (Ux, Uy, Uz) y Magnitud (|U|)
    N_columnas_U = data_U.shape[1]
    N_sensores = (N_columnas_U - 1) // 3

    probes_Ux, probes_Uy, probes_Uz, probes_MagU = {}, {}, {}, {}
    for n in range(1, N_sensores + 1):
        idx_x, idx_y, idx_z = 1 + 3*(n-1), 2 + 3*(n-1), 3 + 3*(n-1)
        ux = parse_vector_column(data_U[idx_x])
        uy = parse_vector_column(data_U[idx_y])
        uz = parse_vector_column(data_U[idx_z])
        
        probes_Ux[f'Ux{n}'] = ux
        probes_Uy[f'Uy{n}'] = uy
        probes_Uz[f'Uz{n}'] = uz
        probes_MagU[f'MagU{n}'] = np.sqrt(ux**2 + uy**2 + uz**2)

    # Perfiles espaciales interpolados en el tiempo (cada 0.5 s)
    intervalo = 0.5
    t_max = t_adquisicion.max()
    tiempos_ideales = np.arange(intervalo, t_max + intervalo, intervalo)
    tiempos_ideales = tiempos_ideales[tiempos_ideales <= t_max] 
    indices_perfil = [np.argmin(np.abs(t_adquisicion - t_id)) for t_id in tiempos_ideales]
    tiempos_objetivo = t_adquisicion[indices_perfil] 

    probes_PerfilUx, probes_PerfilUy = {}, {}
    for n in range(1, N_sensores + 1):
        idx_x, idx_y = 1 + 3*(n-1), 2 + 3*(n-1)
        probes_PerfilUx[f'Ux{n}'] = parse_vector_column(data_U.iloc[indices_perfil, idx_x])
        probes_PerfilUy[f'Uy{n}'] = parse_vector_column(data_U.iloc[indices_perfil, idx_y])

    # Construir diccionario compacto
    data_dict = {
        't': t, 't_adquisicion': t_adquisicion, 't_T': t_T, 't_Age': t_Age, 't_ACH': t_ACH,
        'probes_p': probes_p, 'probes_T': probes_T, 'probes_Age': probes_Age, 'probes_ACH': probes_ACH,
        'probes_Ux': probes_Ux, 'probes_Uy': probes_Uy, 'probes_Uz': probes_Uz, 'probes_MagU': probes_MagU,
        'N_columnas_p': N_columnas_p, 'N_sensores': N_sensores,
        'probes_PerfilUx': probes_PerfilUx, 'probes_PerfilUy': probes_PerfilUy,
        'tiempos_objetivo': tiempos_objetivo
    }

    tuple_out = (t, t_adquisicion, t_T, t_Age, t_ACH, probes_p, probes_T, probes_Age, probes_ACH,
                 probes_Ux, probes_Uy, probes_Uz, probes_MagU, N_columnas_p, N_sensores,
                 probes_PerfilUx, probes_PerfilUy, tiempos_objetivo)

    return data_dict, tuple_out


# =========================================================================
# 1. CARGAR AUTOMÁTICAMENTE TODOS LOS GRUPOS DE SONDAS DETECTADOS
# =========================================================================
sondas_dict = {}
base_post = "postProcessing"

if os.path.exists(base_post):
    # Buscar dinámicamente cualquier carpeta que empiece por 'sondas'
    carpetas_sondas = [d for d in os.listdir(base_post) if d.startswith("sondas") and os.path.isdir(os.path.join(base_post, d))]
    
    for folder_sonda in sorted(carpetas_sondas):
        res = load_and_process_probes(folder_sonda)
        if res:
            sondas_dict[folder_sonda], _ = res

print(f"✅ Se cargaron exitosamente {len(sondas_dict)} grupos de sondas en memoria: {list(sondas_dict.keys())}\n")


# =========================================================================
# 2. CARGAR Y CALCULAR DATOS MACROSCÓPICOS DEL QUIRÓFANO 3
# =========================================================================
print("📥 CALCULANDO DATOS GLOBALES Y MACROSCÓPICOS (Q3)...")
V_quirofano = 78.69  # Volumen total del quirófano en m³
base_path_macro = 'postProcessing'

path_phi     = os.path.join(base_path_macro, 'flujoOutlet', 'surfaceFieldValue_concatenado.dat')
path_rho     = os.path.join(base_path_macro, 'densidadMedia', 'volFieldValue_concatenado.dat')
path_age     = os.path.join(base_path_macro, 'calculoACH', 'volFieldValue_concatenado.dat')
path_salida  = os.path.join(base_path_macro, 'promedioSalida', 'surfaceFieldValue_concatenado.dat')
path_htc     = os.path.join(base_path_macro, 'htc', 'wallFieldValue_concatenado.dat')

data_ach_macro    = None
data_age_macro    = None
data_salida_macro = None
data_htc_macro    = None

# A) Cálculo del ACH Macroscópico: Q = |phi| / rho  ==>  ACH = (Q / V) * 3600
if os.path.exists(path_phi) and os.path.exists(path_rho):
    try:
        phi_df = pd.read_csv(path_phi, comment='#', sep=r'\s+', header=None, names=['Time', 'phi_sum'])
        rho_df = pd.read_csv(path_rho, comment='#', sep=r'\s+', header=None, names=['Time', 'rho_avg'])
        data_ach_macro = pd.merge(phi_df, rho_df, on='Time')
        data_ach_macro = data_ach_macro[data_ach_macro['Time'] > 0].copy()
        data_ach_macro['Q_m3s'] = abs(data_ach_macro['phi_sum']) / data_ach_macro['rho_avg']
        data_ach_macro['ACH'] = (data_ach_macro['Q_m3s'] / V_quirofano) * 3600.0
    except Exception as e:
        print(f"  ⚠️ Error procesando ACH macroscópico: {e}")

# B) Edad Promedio / Cálculo Global
if os.path.exists(path_age):
    try:
        data_age_macro = pd.read_csv(path_age, comment='#', sep=r'\s+', header=None)
        data_age_macro.columns = ['Time', 'Age_avg'] + list(range(2, data_age_macro.shape[1]))
        data_age_macro = data_age_macro[data_age_macro['Time'] > 0].copy()
    except Exception as e:
        print(f"  ⚠️ Error procesando edad volumétrica: {e}")

# C) Variables Promedio a la Salida (Outlet)
if os.path.exists(path_salida):
    try:
        temp_df = pd.read_csv(path_salida, comment='#', sep=r'\s+', header=None)
        if temp_df.shape[1] >= 3:
            temp_df.columns = ['Time', 'Age_out', 'T_out'] + list(range(3, temp_df.shape[1]))
            data_salida_macro = temp_df[['Time', 'Age_out', 'T_out']].copy()
        elif temp_df.shape[1] == 2:
            temp_df.columns = ['Time', 'Age_out']
            data_salida_macro = temp_df.copy()
    except Exception as e:
        print(f"  ⚠️ Nota al leer promedioSalida: {e}")

# D) Coeficiente de Transferencia de Calor (HTC)
if os.path.exists(path_htc):
    try:
        data_htc_macro = pd.read_csv(path_htc, comment='#', sep=r'\s+', header=None)
        data_htc_macro.columns = ['Time', 'HTC_avg'] + list(range(2, data_htc_macro.shape[1]))
    except Exception as e:
        print(f"  ⚠️ Nota al leer HTC: {e}")

print("🎉 Todos los datos macroscópicos y locales fueron procesados exitosamente.")
print("=========================================================================\n")
