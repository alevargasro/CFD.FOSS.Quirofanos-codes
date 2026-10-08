#!/usr/bin/env python3
"""
Unified ACH & Air Residence Time Processing Engine for Operating Room Q3
Generates 4 distinct plots:
1. Global Volumetric ACH (Full Time)
2. Global Volumetric ACH (Last 50s Zoom)
3. Multi-Outlet ACH Comparison (Full Time)
4. Multi-Outlet ACH Comparison (Last 50s Zoom)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

base_directory = os.path.dirname(os.path.abspath(__file__))

print("==================================================")
print("🧪 UNIFIED ACH & RESIDENCE TIME CALCULATION (Q3)")
print("==================================================")

# ---------------------------------------------------------------------------
# 1. FILE DISCOVERY (INDIVIDUAL OUTLETS & GLOBAL VOLUME)
# ---------------------------------------------------------------------------
def find_file(subfolder_keywords, file_names):
    """Searches for postProcessing files matching given keywords"""
    post_dir = os.path.join(base_directory, 'postProcessing')
    if not os.path.exists(post_dir):
        return None
    for root, dirs, files in os.walk(post_dir):
        if any(kw.lower() in root.lower() for kw in subfolder_keywords):
            for fn in file_names:
                if fn in files:
                    return os.path.join(root, fn)
    return None

# Locate Volume Average File
file_vol = find_file(['volfieldvalue', 'agevolumeaverage'], ['volFieldValue.dat', 'volFieldValue_concatenado.dat', 'surfaceFieldValue.dat'])

# Locate Outlet Files (Total and Individual 1-4)
file_outlet_total = find_file(['promediosalida', 'outlettotal', 'outlets'], ['surfaceFieldValue_concatenado.dat', 'surfaceFieldValue.dat'])

individual_outlets = {}
for i in range(1, 5):
    f_path = find_file([f'outlet{i}', f'salida{i}'], ['surfaceFieldValue_concatenado.dat', 'surfaceFieldValue.dat'])
    if f_path:
        individual_outlets[f'Outlet_{i}'] = f_path

if file_vol:
    print(f"📂 Global Volume File: {file_vol}")
else:
    print("⚠️ Warning: Global Volume File not found. Volumetric ACH will be skipped.")

if individual_outlets:
    print(f"📂 Individual Outlets Found: {list(individual_outlets.keys())}")
elif file_outlet_total:
    print(f"📂 Combined Outlet File: {file_outlet_total}")
else:
    print("❌ Error: No outlet data found in postProcessing.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. RESIDENCE TIME / STABILIZATION CALCULATION (2% VARIATION THRESHOLD)
# ---------------------------------------------------------------------------
def get_residence_time(df, col_ach):
    """Calculates residence time t_res when ACH stays within 2% of asymptotic final value"""
    if df.empty or col_ach not in df.columns:
        return 0.0, 0.0, 0.0
    final_ach = df[col_ach].iloc[-1]
    df_temp = df.copy()
    df_temp['Var_%'] = (abs(df_temp[col_ach] - final_ach) / (final_ach + 1e-9)) * 100.0
    
    out_of_bounds = df_temp[df_temp['Var_%'] > 2.0]
    if out_of_bounds.empty:
        stable_row = df_temp.iloc[0]
    else:
        last_out_idx = out_of_bounds.index[-1]
        last_out_pos = df_temp.index.get_loc(last_out_idx)
        stable_row = df_temp.iloc[last_out_pos + 1] if last_out_pos < len(df_temp) - 1 else df_temp.iloc[-1]
        
    return stable_row['Time'], stable_row[col_ach], final_ach

# ---------------------------------------------------------------------------
# 3. DATA PROCESSING & MERGING
# ---------------------------------------------------------------------------
output_folder = os.path.join(base_directory, "Postprocessing_ACH")
os.makedirs(output_folder, exist_ok=True)

# Load Global Volume Data
df_combined = None
if file_vol:
    df_vol = pd.read_csv(file_vol, comment='#', sep=r'\s+', header=None, usecols=[0, 1], names=['Time', 'Age_Volume'])
    df_vol = df_vol[(df_vol['Time'] > 0) & (df_vol['Age_Volume'] > 0)].copy()
    df_vol['ACH_Volume'] = 3600.0 / df_vol['Age_Volume']
    df_combined = df_vol.sort_values('Time').copy()

# Load Individual Outlets Data
df_outlets = pd.DataFrame()
if individual_outlets:
    for name, path in individual_outlets.items():
        df_temp = pd.read_csv(path, comment='#', sep=r'\s+', header=None, usecols=[0, 1], names=['Time', f'Age_{name}'])
        df_temp = df_temp[(df_temp['Time'] > 0) & (df_temp[f'Age_{name}'] > 0)].copy()
        df_temp[f'ACH_{name}'] = 3600.0 / df_temp[f'Age_{name}']
        
        if df_outlets.empty:
            df_outlets = df_temp
        else:
            df_outlets = pd.merge_asof(df_outlets.sort_values('Time'), df_temp.sort_values('Time'), on='Time')

elif file_outlet_total:
    df_temp = pd.read_csv(file_outlet_total, comment='#', sep=r'\s+', header=None, usecols=[0, 1], names=['Time', 'Age_OutletTotal'])
    df_temp = df_temp[(df_temp['Time'] > 0) & (df_temp['Age_OutletTotal'] > 0)].copy()
    df_temp['ACH_OutletTotal'] = 3600.0 / df_temp['Age_OutletTotal']
    df_outlets = df_temp

if df_combined is None:
    df_combined = df_outlets
else:
    df_combined = pd.merge_asof(df_combined, df_outlets, on='Time')

# ---------------------------------------------------------------------------
# 4. PLOTTING ROUTINES
# ---------------------------------------------------------------------------

# === GRAPH 1: GLOBAL VOLUMETRIC ACH (FULL TIME) ===
if file_vol:
    t_res_vol, ach_res_vol, final_ach_vol = get_residence_time(df_vol, 'ACH_Volume')
    t_final_vol = df_vol['Time'].iloc[-1]
    final_age_vol = df_vol['Age_Volume'].iloc[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(df_vol['Time'], df_vol['ACH_Volume'], marker='s', markersize=3, color='tab:purple', linewidth=1.5, label=r'Volumetric ACH ($3600 / \tau_{mean}$)')
    plt.axhline(y=20, color='tab:red', linestyle='--', linewidth=1.5, label='ASHRAE Standard 170 Minimum (20 ACH)')
    plt.axvline(x=t_res_vol, color='tab:green', linestyle=':', linewidth=2, label=f'$t_{{residence}} = {t_res_vol:.1f}$ s')
    plt.plot(t_res_vol, ach_res_vol, 'go', markersize=8)

    text_vol = f"Final Volumetric ACH: {final_ach_vol:.2f} h⁻¹\nMean Air Age: {final_age_vol:.1f} s\n$t_{{residence}}$: {t_res_vol:.1f} s"
    plt.annotate(text_vol, xy=(t_res_vol, ach_res_vol), xytext=(t_final_vol * 0.45, max(df_vol['ACH_Volume']) * 0.75),
                 arrowprops=dict(facecolor='tab:green', shrink=0.05, width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.85), fontweight='bold')

    plt.title('Global Volumetric Air Changes per Hour (ACH) - Full Time', fontsize=13, fontweight='bold')
    plt.xlabel('Simulation Time [s]', fontsize=11)
    plt.ylabel('ACH [h⁻¹]', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1_ACH_Global_Volume_FullTime.png'), dpi=300)
    plt.close()

    # === GRAPH 2: GLOBAL VOLUMETRIC ACH (LAST 50s ZOOM) ===
    t_start_vol_50 = max(0, t_final_vol - 50.0)
    df_vol_50 = df_vol[df_vol['Time'] >= t_start_vol_50]

    plt.figure(figsize=(10, 6))
    plt.plot(df_vol_50['Time'], df_vol_50['ACH_Volume'], marker='s', markersize=4, color='tab:purple', linewidth=1.8, label='Volumetric ACH')
    plt.axhline(y=20, color='tab:red', linestyle='--', linewidth=1.5, label='ASHRAE 170 Minimum (20 ACH)')

    if t_res_vol >= t_start_vol_50:
        plt.axvline(x=t_res_vol, color='tab:green', linestyle=':', linewidth=2, label=f'$t_{{residence}} = {t_res_vol:.1f}$ s')
        plt.plot(t_res_vol, ach_res_vol, 'go', markersize=8)

    text_vol_50 = f"Mean (Last 50s): {df_vol_50['ACH_Volume'].mean():.2f} h⁻¹\nFinal ACH: {final_ach_vol:.2f} h⁻¹"
    plt.annotate(text_vol_50, xy=(df_vol_50['Time'].iloc[-1], final_ach_vol), 
                 xytext=(t_start_vol_50 + 2, df_vol_50['ACH_Volume'].min() + (df_vol_50['ACH_Volume'].max() - df_vol_50['ACH_Volume'].min())*0.1),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.9), fontweight='bold')

    plt.title('Global Volumetric Air Changes per Hour (ACH) - Last 50 s', fontsize=13, fontweight='bold')
    plt.xlabel('Simulation Time [s]', fontsize=11)
    plt.ylabel('ACH [h⁻¹]', fontsize=11)
    plt.xlim(t_start_vol_50, t_final_vol)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '2_ACH_Global_Volume_Zoom_Last50s.png'), dpi=300)
    plt.close()

# === GRAPH 3: MULTI-OUTLET ACH COMPARISON (FULL TIME) ===
if not df_outlets.empty:
    plt.figure(figsize=(10, 6))
    ach_cols = [c for c in df_outlets.columns if c.startswith('ACH_')]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan']
    
    for idx, col in enumerate(ach_cols):
        label_name = col.replace('ACH_', '').replace('_', ' ')
        plt.plot(df_outlets['Time'], df_outlets[col], linewidth=1.8, color=colors[idx % len(colors)], label=label_name)
    
    plt.axhline(y=20, color='black', linestyle='--', linewidth=1.2, label='ASHRAE 170 Threshold (20 ACH)')
    plt.title('Air Extraction Performance Comparison Across Outlets - Full Time', fontsize=13, fontweight='bold')
    plt.xlabel('Simulation Time [s]', fontsize=11)
    plt.ylabel('ACH [h⁻¹]', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '3_ACH_Outlets_Comparison_FullTime.png'), dpi=300)
    plt.close()

    # === GRAPH 4: MULTI-OUTLET ACH COMPARISON (LAST 50s ZOOM) ===
    t_final_outlets = df_outlets['Time'].iloc[-1]
    t_start_out_50 = max(0, t_final_outlets - 50.0)
    df_outlets_50 = df_outlets[df_outlets['Time'] >= t_start_out_50]

    plt.figure(figsize=(10, 6))
    for idx, col in enumerate(ach_cols):
        label_name = col.replace('ACH_', '').replace('_', ' ')
        plt.plot(df_outlets_50['Time'], df_outlets_50[col], linewidth=1.8, color=colors[idx % len(colors)], label=label_name)

    plt.axhline(y=20, color='black', linestyle='--', linewidth=1.2, label='ASHRAE 170 Threshold (20 ACH)')
    plt.title('Air Extraction Performance Comparison Across Outlets - Last 50 s', fontsize=13, fontweight='bold')
    plt.xlabel('Simulation Time [s]', fontsize=11)
    plt.ylabel('ACH [h⁻¹]', fontsize=11)
    plt.xlim(t_start_out_50, t_final_outlets)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '4_ACH_Outlets_Comparison_Zoom_Last50s.png'), dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# 5. CONSOLIDATED CSV REPORT
# ---------------------------------------------------------------------------
csv_path = os.path.join(output_folder, 'ACH_Consolidated_Report.csv')
df_combined.to_csv(csv_path, index=False)

print("\n🎉 ACH Post-Processing completed successfully!")
print(f"📁 Results exported to: {output_folder}")
print(f"📄 CSV Report: {os.path.basename(csv_path)}")
