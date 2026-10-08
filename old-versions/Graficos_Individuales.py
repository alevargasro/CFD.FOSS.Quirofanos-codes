#!/usr/bin/env python3
"""
Individual Probe Dashboard Generator for Operating Room Q3 (Without Dummies)
Generates high-resolution 6-panel dashboards for individual sensors.
Includes physical fields: Velocity (|U|, Ux, Uy, Uz), Temperature, Pressure, 
Air Age, Local ACH, and Fluid Density (rho).
All labels and output files are generated in English.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

carpeta_principal = os.path.dirname(os.path.abspath(__file__))

try:
    from Process_data import *
except ImportError:
    print("❌ Error: 'Process_data.py' not found in the current directory.")
    sys.exit(1)


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================
def calculate_density(T_arr, p_rgh_arr, p_ref=101325.0, R=287.058):
    """Calculates density rho [kg/m3] using Ideal Gas Law"""
    T_k = np.array(T_arr)
    p_total = np.array(p_rgh_arr) + p_ref
    return p_total / (R * T_k)

def get_probe_description(group_name, probe_idx):
    """Returns a specific spatial description for key probes in Q3"""
    if "Inlet" in group_name or "Outlet" in group_name:
        if probe_idx == 5:
            return "(Center Sensor @ 10 cm Offset)"
        elif probe_idx == 14:
            return "(Center Sensor @ 20 cm Offset)"
        elif probe_idx == 23:
            return "(Center Sensor @ 30 cm Offset)"
        elif probe_idx <= 9:
            return f"(Grid Position #{probe_idx} @ 10 cm Offset)"
        elif probe_idx <= 18:
            return f"(Grid Position #{probe_idx-9} @ 20 cm Offset)"
        else:
            return f"(Grid Position #{probe_idx-18} @ 30 cm Offset)"

    elif "Esquinas" in group_name:
        corners_map = {
            1: "Lower Left-Front (Z=0.15m)",
            2: "Lower Right-Front (Z=0.15m)",
            3: "Lower Left-Back (Z=0.15m)",
            4: "Lower Right-Back (Z=0.15m)",
            5: "Upper Left-Front (Z=2.55m)",
            6: "Upper Right-Front (Z=2.55m)",
            7: "Upper Left-Back (Z=2.55m)",
            8: "Upper Right-Back (Z=2.55m)"
        }
        return f"({corners_map.get(probe_idx, 'Corner Zone')})"

    elif "Paciente" in group_name:
        patient_map = {
            1: "Above Patient - Head Region (Z=1.22m)",
            2: "Above Patient - Mid-Upper (Z=1.22m)",
            3: "Above Patient - Center (Z=1.22m)",
            4: "Above Patient - Mid-Lower (Z=1.22m)",
            5: "Above Patient - Feet Region (Z=1.22m)",
            6: "Head Side - Lower Level (Z=0.75m)",
            7: "Head Side - Upper Level (Z=1.05m)",
            8: "Feet Side - Lower Level (Z=0.75m)",
            9: "Feet Side - Upper Level (Z=1.05m)"
        }
        return f"({patient_map.get(probe_idx, 'Patient Zone')})"

    return f"(Probe #{probe_idx})"


# =========================================================================
# 6-PANEL INDIVIDUAL DASHBOARD GENERATOR
# =========================================================================
def generate_individual_dashboards(group_name, probe_data):
    """
    Generates a 6-panel comprehensive dashboard for each probe in the group:
    1. Velocity Magnitude & Components
    2. Temperature (T)
    3. Pressure (p_rgh)
    4. Density (rho)
    5. Mean Age of Air
    6. Local Air Changes per Hour (ACH)
    """
    N_sens = probe_data['N_sensores']
    if N_sens == 0:
        return

    output_dir = os.path.join(carpeta_principal, f"Individual_Dashboards_{group_name}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🚀 Generating Individual Dashboards for: {group_name} ({N_sens} probes)")

    t_p = probe_data['t']
    t_adq = probe_data['t_adquisicion']
    t_T = probe_data['t_T']
    t_Age = probe_data['t_Age']
    t_ACH = probe_data['t_ACH']

    p_dict = probe_data['probes_p']
    T_dict = probe_data['probes_T']
    Age_dict = probe_data['probes_Age']
    ACH_dict = probe_data['probes_ACH']
    Ux_dict = probe_data['probes_Ux']
    Uy_dict = probe_data['probes_Uy']
    Uz_dict = probe_data['probes_Uz']
    MagU_dict = probe_data['probes_MagU']

    for i in range(1, N_sens + 1):
        key_p, key_ux, key_uy, key_uz, key_mag = f'p{i}', f'Ux{i}', f'Uy{i}', f'Uz{i}', f'MagU{i}'
        key_T, key_Age, key_ACH = f'T{i}', f'Age{i}', f'ACH{i}'

        spatial_desc = get_probe_description(group_name, i)

        fig, axes = plt.subplots(3, 2, figsize=(14, 11))
        fig.suptitle(f'Dashboard - {group_name} | Probe {i} {spatial_desc}', fontsize=14, fontweight='bold', y=0.98)

        # Panel 1: Velocity Magnitude & Components
        ax1 = axes[0, 0]
        if MagU_dict and key_mag in MagU_dict:
            ax1.plot(t_adq, MagU_dict[key_mag], label='|U| Total', color='black', linewidth=2.0)
        if Ux_dict and key_ux in Ux_dict:
            ax1.plot(t_adq, Ux_dict[key_ux], label='Ux', color='tab:blue', linestyle='--', alpha=0.7)
        if Uy_dict and key_uy in Uy_dict:
            ax1.plot(t_adq, Uy_dict[key_uy], label='Uy', color='tab:orange', linestyle='--', alpha=0.7)
        if Uz_dict and key_uz in Uz_dict:
            ax1.plot(t_adq, Uz_dict[key_uz], label='Uz', color='tab:green', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Velocity [m/s]', fontsize=10)
        ax1.set_title('Velocity Field', fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right', fontsize='small')

        # Panel 2: Temperature
        ax2 = axes[0, 1]
        if T_dict and key_T in T_dict:
            ax2.plot(t_T, T_dict[key_T], color='tab:red', linewidth=1.5)
        ax2.set_ylabel('Temperature [K]', fontsize=10)
        ax2.set_title('Temperature ($T$)', fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)

        # Panel 3: Modified Pressure p_rgh
        ax3 = axes[1, 0]
        if p_dict and key_p in p_dict:
            ax3.plot(t_p, p_dict[key_p], color='tab:purple', linewidth=1.5)
        ax3.set_ylabel('p_rgh [Pa]', fontsize=10)
        ax3.set_title('Modified Pressure ($p_{rgh}$)', fontsize=11, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.5)

        # Panel 4: Fluid Density (rho)
        ax4 = axes[1, 1]
        if T_dict and key_T in T_dict and p_dict and key_p in p_dict:
            rho_vals = calculate_density(T_dict[key_T], p_dict[key_p])
            ax4.plot(t_T, rho_vals, color='tab:brown', linewidth=1.5)
        ax4.set_ylabel('Density [kg/m³]', fontsize=10)
        ax4.set_title('Fluid Density ($\\rho$)', fontsize=11, fontweight='bold')
        ax4.grid(True, linestyle='--', alpha=0.5)

        # Panel 5: Local Air Age
        ax5 = axes[2, 0]
        if Age_dict and key_Age in Age_dict:
            ax5.plot(t_Age, Age_dict[key_Age], color='tab:green', linewidth=1.5)
        ax5.set_xlabel('Time [s]', fontsize=10)
        ax5.set_ylabel('Air Age [s]', fontsize=10)
        ax5.set_title('Local Mean Age of Air', fontsize=11, fontweight='bold')
        ax5.grid(True, linestyle='--', alpha=0.5)

        # Panel 6: Local ACH
        ax6 = axes[2, 1]
        if ACH_dict and key_ACH in ACH_dict:
            ax6.plot(t_ACH, ACH_dict[key_ACH], color='tab:blue', linewidth=1.5)
        ax6.set_xlabel('Time [s]', fontsize=10)
        ax6.set_ylabel('ACH [1/h]', fontsize=10)
        ax6.set_title('Local Air Changes per Hour (ACH)', fontsize=11, fontweight='bold')
        ax6.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        
        # Save figure with descriptive name
        filename = f'probe_{i}_center_dashboard.png' if i in [5, 14, 23] and ("Inlet" in group_name or "Outlet" in group_name) else f'probe_{i}_dashboard.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

        if i % 9 == 0 or i == N_sens:
            print(f"   [+] Exported: {i}/{N_sens} probe dashboards.")


# =========================================================================
# MAIN EXECUTION ROUTINE
# =========================================================================
if __name__ == "__main__":
    print("\n🚀 Starting Q3 Individual Probe Dashboards Generation...")

    if 'sondas_dict' in globals() and sondas_dict:
        for nombre_sonda, probe_data in sondas_dict.items():
            generate_individual_dashboards(nombre_sonda, probe_data)

    print("\n🎉 INDIVIDUAL DASHBOARDS GENERATION COMPLETED SUCCESSFULLY!")
