#!/usr/bin/env python3
"""
General Plotting Engine for Operating Room Q3 (Without Dummies)
Processes spatially-averaged probe groups: Inlet, Outlets (1-4), Corners, and Patient.
Outputs high-resolution PNG plots with all labels in English.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

carpeta_principal = os.path.dirname(os.path.abspath(__file__))

try:
    from Process_data import *
except ImportError:
    print("❌ Error: 'Process_data.py' not found in the script directory.")
    sys.exit(1)


# =========================================================================
# HELPER FUNCTIONS & CALCULATIONS
# =========================================================================
def calculate_density(T_arr, p_rgh_arr, p_ref=101325.0, R=287.058):
    """Calculates fluid density rho [kg/m3] using Ideal Gas Equation"""
    T_k = np.array(T_arr)
    p_total = np.array(p_rgh_arr) + p_ref
    return p_total / (R * T_k)

def average_dict_sensors(sensor_dict, indices, prefix):
    """Computes the spatial mean across specified sensor indices over time"""
    arrays = [np.array(sensor_dict[f'{prefix}{i}']) for i in indices if f'{prefix}{i}' in sensor_dict]
    if not arrays:
        return None
    return np.mean(arrays, axis=0)


# =========================================================================
# 1. INLET & OUTLETS PLANE-AVERAGED PLOTS (9 PROBES PER DISTANCE)
# =========================================================================
def plot_grid_patch_averages(patch_name, probe_data, normal_comp='Uz'):
    """
    Generates time-series plots for 9-probe array patches at 10cm, 20cm, and 30cm offsets.
    Metrics: |U|, Normal Velocity, ACH, Age, T, p_rgh, Rho
    """
    output_dir = os.path.join(carpeta_principal, f"General_Plots_{patch_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    t_adq = probe_data['t_adquisicion']
    t_T = probe_data['t_T']
    t_Age = probe_data['t_Age']
    t_ACH = probe_data['t_ACH']
    t_p = probe_data['t']

    # Sensor groups for 10cm (1-9), 20cm (10-18), 30cm (19-27)
    groups = {
        '10 cm': list(range(1, 10)),
        '20 cm': list(range(10, 19)),
        '30 cm': list(range(19, 28))
    }
    colors = {'10 cm': 'tab:blue', '20 cm': 'tab:orange', '30 cm': 'tab:green'}

    # Field configurations: (dict_key_prefix, metric_title, y_label, time_array, filename)
    fields_to_plot = [
        ('MagU', 'Velocity Magnitude |U|', '|U| [m/s]', t_adq, 'MagU_average.png'),
        (normal_comp, f'Normal Velocity ({normal_comp})', f'{normal_comp} [m/s]', t_adq, f'{normal_comp}_average.png'),
        ('ACH', 'Air Changes per Hour (ACH)', 'ACH [1/h]', t_ACH, 'ACH_average.png'),
        ('Age', 'Mean Age of Air', 'Air Age [s]', t_Age, 'Age_average.png'),
        ('T', 'Temperature', 'Temperature [K]', t_T, 'Temperature_average.png'),
        ('p', 'Modified Pressure p_rgh', 'p_rgh [Pa]', t_p, 'Pressure_average.png')
    ]

    for prefix, title, ylabel, t_vec, fname in fields_to_plot:
        data_dict = probe_data[f'probes_{prefix}']
        if not data_dict:
            continue
        
        plt.figure(figsize=(9, 5.5))
        for label, indices in groups.items():
            mean_val = average_dict_sensors(data_dict, indices, prefix)
            if mean_val is not None:
                plt.plot(t_vec, mean_val, label=f'Offset {label}', color=colors[label], linewidth=2)
        
        plt.title(f'{patch_name} - Spatially Averaged {title}', fontsize=12, fontweight='bold')
        plt.xlabel('Time [s]', fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    # Calculate and plot Density (rho)
    if probe_data['probes_T'] and probe_data['probes_p']:
        plt.figure(figsize=(9, 5.5))
        for label, indices in groups.items():
            mean_T = average_dict_sensors(probe_data['probes_T'], indices, 'T')
            mean_p = average_dict_sensors(probe_data['probes_p'], indices, 'p')
            if mean_T is not None and mean_p is not None:
                rho = calculate_density(mean_T, mean_p)
                plt.plot(t_T, rho, label=f'Offset {label}', color=colors[label], linewidth=2)
        
        plt.title(f'{patch_name} - Spatially Averaged Density ($\\rho$)', fontsize=12, fontweight='bold')
        plt.xlabel('Time [s]', fontsize=11)
        plt.ylabel('Density $\\rho$ [kg/m³]', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, 'Density_average.png'), dpi=300, bbox_inches='tight')
        plt.close()


# =========================================================================
# 2. OUTLET COMPARISON PLOTS (OUTLET 1 vs 2 vs 3 vs 4)
# =========================================================================
def plot_outlet_comparisons():
    """Compares the averaged behavior across all 4 outlets at 10cm, 20cm, and 30cm"""
    output_dir = os.path.join(carpeta_principal, "General_Plots_Outlets_Comparison")
    os.makedirs(output_dir, exist_ok=True)

    outlets = ['sondasOutlet1', 'sondasOutlet2', 'sondasOutlet3', 'sondasOutlet4']
    out_labels = ['Outlet 1 (Left)', 'Outlet 2 (Right)', 'Outlet 3 (Front)', 'Outlet 4 (Back)']
    offsets = {'10 cm': list(range(1, 10)), '20 cm': list(range(10, 19)), '30 cm': list(range(19, 28))}

    for dist_label, indices in offsets.items():
        # Plot ACH Comparison across Outlets
        plt.figure(figsize=(9.5, 5.5))
        for out_key, out_name in zip(outlets, out_labels):
            if out_key in sondas_dict:
                p_data = sondas_dict[out_key]
                mean_ach = average_dict_sensors(p_data['probes_ACH'], indices, 'ACH')
                if mean_ach is not None:
                    plt.plot(p_data['t_ACH'], mean_ach, label=out_name, linewidth=2)
        
        plt.title(f'Outlets ACH Comparison at {dist_label} Offset', fontsize=12, fontweight='bold')
        plt.xlabel('Time [s]', fontsize=11)
        plt.ylabel('ACH [1/h]', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, f'Comparison_ACH_{dist_label.replace(" ", "")}.png'), dpi=300, bbox_inches='tight')
        plt.close()


# =========================================================================
# 3. CORNERS PLOTS (LOWER VS UPPER CORNERS)
# =========================================================================
def plot_corners_averages(probe_data):
    """Plots averaged metrics for Lower Corners (Z=0.15m) vs Upper Corners (Z=2.55m)"""
    output_dir = os.path.join(carpeta_principal, "General_Plots_sondasEsquinas")
    os.makedirs(output_dir, exist_ok=True)

    groups = {
        'Lower Corners (Z = 0.15 m)': [1, 2, 3, 4],
        'Upper Corners (Z = 2.55 m)': [5, 6, 7, 8]
    }
    colors = {'Lower Corners (Z = 0.15 m)': 'tab:blue', 'Upper Corners (Z = 2.55 m)': 'tab:red'}

    fields = [
        ('MagU', 'Velocity Magnitude |U|', '|U| [m/s]', probe_data['t_adquisicion'], 'MagU_average.png'),
        ('ACH', 'Air Changes per Hour (ACH)', 'ACH [1/h]', probe_data['t_ACH'], 'ACH_average.png'),
        ('Age', 'Mean Age of Air', 'Air Age [s]', probe_data['t_Age'], 'Age_average.png'),
        ('T', 'Temperature', 'Temperature [K]', probe_data['t_T'], 'Temperature_average.png'),
        ('p', 'Modified Pressure p_rgh', 'p_rgh [Pa]', probe_data['t'], 'Pressure_average.png')
    ]

    for prefix, title, ylabel, t_vec, fname in fields:
        data_dict = probe_data[f'probes_{prefix}']
        if not data_dict:
            continue

        plt.figure(figsize=(9, 5.5))
        for label, indices in groups.items():
            mean_val = average_dict_sensors(data_dict, indices, prefix)
            if mean_val is not None:
                plt.plot(t_vec, mean_val, label=label, color=colors[label], linewidth=2)

        plt.title(f'Corners Zone - {title}', fontsize=12, fontweight='bold')
        plt.xlabel('Time [s]', fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()


# =========================================================================
# 4. PATIENT PLOTS (TOP VS HEAD VS FEET)
# =========================================================================
def plot_patient_averages(probe_data):
    """Plots averaged metrics for Patient probes: Top (1-5), Head (6-7), and Feet (8-9)"""
    output_dir = os.path.join(carpeta_principal, "General_Plots_sondasPaciente")
    os.makedirs(output_dir, exist_ok=True)

    groups = {
        'Above Patient (Z = 1.22 m)': [1, 2, 3, 4, 5],
        'Head Zone (X = 2.46 m)': [6, 7],
        'Feet Zone (X = 4.59 m)': [8, 9]
    }
    colors = {
        'Above Patient (Z = 1.22 m)': 'tab:green',
        'Head Zone (X = 2.46 m)': 'tab:purple',
        'Feet Zone (X = 4.59 m)': 'tab:orange'
    }

    fields = [
        ('MagU', 'Velocity Magnitude |U|', '|U| [m/s]', probe_data['t_adquisicion'], 'MagU_average.png'),
        ('ACH', 'Air Changes per Hour (ACH)', 'ACH [1/h]', probe_data['t_ACH'], 'ACH_average.png'),
        ('Age', 'Mean Age of Air', 'Air Age [s]', probe_data['t_Age'], 'Age_average.png'),
        ('T', 'Temperature', 'Temperature [K]', probe_data['t_T'], 'Temperature_average.png'),
        ('p', 'Modified Pressure p_rgh', 'p_rgh [Pa]', probe_data['t'], 'Pressure_average.png')
    ]

    for prefix, title, ylabel, t_vec, fname in fields:
        data_dict = probe_data[f'probes_{prefix}']
        if not data_dict:
            continue

        plt.figure(figsize=(9, 5.5))
        for label, indices in groups.items():
            mean_val = average_dict_sensors(data_dict, indices, prefix)
            if mean_val is not None:
                plt.plot(t_vec, mean_val, label=label, color=colors[label], linewidth=2)

        plt.title(f'Patient Thermal & Velocity Environment - {title}', fontsize=12, fontweight='bold')
        plt.xlabel('Time [s]', fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()


# =========================================================================
# MAIN EXECUTION ROUTINE
# =========================================================================
if __name__ == "__main__":
    print("\n🚀 Starting Q3 General Post-Processing Execution...")

    if 'sondas_dict' in globals() and sondas_dict:
        # 1. Process Inlet
        if 'sondasInlet' in sondas_dict:
            print("  [+] Generating General Plots for Inlet...")
            plot_grid_patch_averages('sondasInlet', sondas_dict['sondasInlet'], normal_comp='Uz')

        # 2. Process Outlets 1 to 4
        outlet_normals = {
            'sondasOutlet1': 'Ux',
            'sondasOutlet2': 'Ux',
            'sondasOutlet3': 'Uy',
            'sondasOutlet4': 'Uy'
        }
        for out_name, n_comp in outlet_normals.items():
            if out_name in sondas_dict:
                print(f"  [+] Generating General Plots for {out_name}...")
                plot_grid_patch_averages(out_name, sondas_dict[out_name], normal_comp=n_comp)

        # 3. Generate Comparative Plots Across Outlets
        print("  [+] Generating Cross-Outlet Comparison Plots...")
        plot_outlet_comparisons()

        # 4. Process Corners
        if 'sondasEsquinas' in sondas_dict:
            print("  [+] Generating General Plots for Corners...")
            plot_corners_averages(sondas_dict['sondasEsquinas'])

        # 5. Process Patient
        if 'sondasPaciente' in sondas_dict:
            print("  [+] Generating General Plots for Patient...")
            plot_patient_averages(sondas_dict['sondasPaciente'])

    print("\n✅ GENERAL PLOTS GENERATION COMPLETED SUCCESSFULLY!")
