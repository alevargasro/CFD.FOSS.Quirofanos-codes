#!/usr/bin/env python3
"""
visualize_probes.py
====================
Visualizador genérico de sondas (probes) y zonas topoSet de OpenFOAM sobre 
la geometría STL de un caso, usando Plotly para generar un HTML 3D interactivo.

Escanea automáticamente:
 - Geometrías STL en 'constant/triSurface' (o rutas relativas)
 - Archivos de sondas dentro de 'Specific_functions' o 'system/functions'
 - Cajas de refinamiento/monitoreo en 'system/topoSetDict'
"""

import argparse
import glob
import os
import re
import sys
import numpy as np

try:
    from stl import mesh as stlmesh
except ImportError:
    stlmesh = None

try:
    import plotly.graph_objects as go
except ImportError:
    print("❌ Falta la librería Plotly. Instálala ejecutando: pip install plotly", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuración predeterminada de rutas y mapas de color
# ---------------------------------------------------------------------------
DEFAULT_STLS = [
    "constant/triSurface/walls.stl",
    "constant/triSurface/inlet.stl",
    "constant/triSurface/outlet.stl",
    "constant/triSurface/pacienteManta.stl",
]

PROBE_SEARCH_DIRS = [
    "system/Specific_functions",
    "Specific_functions",
    "system/functions",
    "functions",
    "system/specificFunctions",
    "specificFunctions",

]

TOPOSET_SEARCH_PATHS = [
    "system/topoSetDict",
    "topoSetDict",
    "../system/topoSetDict",
]

PATCH_COLOR_MAP = {
    "inlet": "#00d2ff",      # Cyan / Azul Claro
    "outlet": "#ff6b6b",     # Naranja / Rojo
    "paciente": "#e056fd",   # Magenta
    "manta": "#be2edd",      # Púrpura
    "wall": "#b2bec3",       # Gris Metalizado
    "muro": "#b2bec3",
    "piso": "#636e72",
    "techo": "#dfe6e9",
}

DEFAULT_STL_COLORS = [
    "#74b9ff", "#ff7675", "#a29bfe", "#55efc4", "#ffeaa7", "#fd79a8", "#81ecec"
]

_PROBE_COLORS = [
    "#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f1c40f", "#e67e22",
    "#1abc9c", "#d35400", "#8e44ad", "#27ae60", "#2980b9", "#f39c12"
]


# ---------------------------------------------------------------------------
# Búsqueda y Resolución Dinámica de Rutas
# ---------------------------------------------------------------------------
def resolve_file_path(filename):
    """Busca un archivo en las rutas relativas más comunes del proyecto."""
    if os.path.exists(filename):
        return filename

    basename = os.path.basename(filename)
    search_dirs = [
        "constant/triSurface",
        "triSurface",
        "system/Specific_functions",
        "Specific_functions",
        "system/specificFunctions",
        "specificFunctions",
        "system/functions",
        "system",
        ".",
        "..",
        "../constant/triSurface",
        "../system",
    ]

    for d in search_dirs:
        candidate = os.path.join(d, basename)
        if os.path.exists(candidate):
            return candidate

    return None


def discover_all_stls(input_stls=None):
    """Escanea la carpeta triSurface si no se especifican STLs válidos."""
    discovered = []

    if input_stls:
        for stl in input_stls:
            resolved = resolve_file_path(stl)
            if resolved and resolved not in discovered:
                discovered.append(resolved)

    # Si no se encontró ninguno de la lista por defecto, auto-escanear constant/triSurface
    if not discovered:
        search_folder = resolve_file_path("constant/triSurface") or resolve_file_path("triSurface")
        if search_folder and os.path.isdir(search_folder):
            for f in sorted(os.listdir(search_folder)):
                if f.endswith(".stl"):
                    discovered.append(os.path.join(search_folder, f))

    return discovered


def discover_all_probes(input_paths=None):
    """Escanea los directorios de 'Specific_functions' buscando archivos de sondas."""
    discovered = []

    for p_dir in PROBE_SEARCH_DIRS:
        resolved_dir = resolve_file_path(p_dir)
        if resolved_dir and os.path.isdir(resolved_dir):
            for root, _, files in os.walk(resolved_dir):
                for f in files:
                    if not f.startswith(".") and not f.endswith("~") and not f.endswith(".py") and not f.endswith(".html"):
                        full_path = os.path.join(root, f)
                        if full_path not in discovered:
                            discovered.append(full_path)

    if input_paths:
        for item in input_paths:
            resolved = resolve_file_path(item)
            if resolved:
                if os.path.isdir(resolved):
                    for root, _, files in os.walk(resolved):
                        for f in files:
                            if not f.startswith(".") and not f.endswith("~"):
                                fp = os.path.join(root, f)
                                if fp not in discovered:
                                    discovered.append(fp)
                elif os.path.isfile(resolved):
                    if resolved not in discovered:
                        discovered.append(resolved)

    return sorted(discovered)


# ---------------------------------------------------------------------------
# Parser de topoSetDict para extraer cajas (boxToCell)
# ---------------------------------------------------------------------------
def parse_toposet_boxes(path):
    """Extrae nombres y coordenadas min/max de las cajas en topoSetDict."""
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path, "r", errors="ignore") as f:
        content = f.read()

    # Regex para capturar bloques de topoSet con box (min) (max)
    action_blocks = re.findall(
        r'\{\s*name\s+([a-zA-Z0-9_]+);[\s\S]*?box\s*\(\s*([^)]+)\s*\)\s*\(\s*([^)]+)\s*\);',
        content
    )

    for name, min_str, max_str in action_blocks:
        try:
            p_min = [float(v) for v in min_str.replace(',', ' ').split()]
            p_max = [float(v) for v in max_str.replace(',', ' ').split()]
            if len(p_min) == 3 and len(p_max) == 3:
                boxes.append({
                    "name": name,
                    "min": p_min,
                    "max": p_max
                })
        except ValueError:
            continue

    return boxes


def add_toposet_box_to_fig(fig, box_data, color="#f1c40f"):
    """Dibuja una caja topoSet como volumen semitransparente con esquinas y bordes."""
    name = box_data["name"]
    x0, y0, z0 = box_data["min"]
    x1, y1, z1 = box_data["max"]

    # 8 Vértices de la caja 3D
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], # Base inferior
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]  # Tapa superior
    ])

    # Caras 3D
    i_idx = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
    j_idx = [1, 2, 5, 6, 1, 5, 2, 7, 3, 7, 2, 6]
    k_idx = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]

    group_id = f"group_box_{name}"

    fig.add_trace(
        go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=i_idx, j=j_idx, k=k_idx,
            color=color,
            opacity=0.20,
            name=f"Caja: {name}",
            legendgroup=group_id,
            showlegend=True,
            hoverinfo="name"
        )
    )

    # Aristas
    line_x = [v[0,0], v[1,0], v[2,0], v[3,0], v[0,0], None,
              v[4,0], v[5,0], v[6,0], v[7,0], v[4,0], None,
              v[0,0], v[4,0], None, v[1,0], v[5,0], None,
              v[2,0], v[6,0], None, v[3,0], v[7,0]]
    
    line_y = [v[0,1], v[1,1], v[2,1], v[3,1], v[0,1], None,
              v[4,1], v[5,1], v[6,1], v[7,1], v[4,1], None,
              v[0,1], v[4,1], None, v[1,1], v[5,1], None,
              v[2,1], v[6,1], None, v[3,1], v[7,1]]

    line_z = [v[0,2], v[1,2], v[2,2], v[3,2], v[0,2], None,
              v[4,2], v[5,2], v[6,2], v[7,2], v[4,2], None,
              v[0,2], v[4,2], None, v[1,2], v[5,2], None,
              v[2,2], v[6,2], None, v[3,2], v[7,2]]

    hover_esquinas = [
        f"<b>Caja topoSet:</b> {name}<br><b>X:</b> {pt[0]:.3f} m<br><b>Y:</b> {pt[1]:.3f} m<br><b>Z:</b> {pt[2]:.3f} m"
        for pt in v
    ]

    fig.add_trace(
        go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode="lines",
            line=dict(color="orange", width=4),
            legendgroup=group_id,
            showlegend=False,
            hoverinfo="none"
        )
    )

    # Esquinas
    fig.add_trace(
        go.Scatter3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            mode="markers",
            marker=dict(size=5, color="yellow", symbol="square", line=dict(color="black", width=1)),
            text=hover_esquinas,
            hoverinfo="text",
            legendgroup=group_id,
            showlegend=False
        )
    )


# ---------------------------------------------------------------------------
# Carga de STL con fallback
# ---------------------------------------------------------------------------
def _parse_stl_fallback(path):
    vertex_re = re.compile(r'vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)')
    verts = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = vertex_re.search(line)
            if m:
                verts.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])

    if len(verts) >= 3 and len(verts) % 3 == 0:
        verts = np.array(verts, dtype=np.float32)
        return verts.reshape(-1, 3, 3)

    return None


def read_stl_decimated(path, max_tris=25000, seed=0):
    tris = None

    if stlmesh is None:
        tris = _parse_stl_fallback(path)
    else:
        try:
            m = stlmesh.Mesh.from_file(path)
            if len(m.vectors) > 0:
                tris = m.vectors
        except Exception:
            tris = _parse_stl_fallback(path)

    if tris is None or len(tris) == 0:
        raise ValueError(f"No se pudieron extraer triángulos de {path}")

    n_tris = len(tris)
    if n_tris > max_tris:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_tris, size=max_tris, replace=False)
        tris = tris[idx]

    verts = tris.reshape(-1, 3)
    n_verts = len(verts)
    i_idx = np.arange(0, n_verts, 3)
    j_idx = i_idx + 1
    k_idx = i_idx + 2

    return verts[:, 0], verts[:, 1], verts[:, 2], i_idx, j_idx, k_idx


# ---------------------------------------------------------------------------
# Lectura de sondas
# ---------------------------------------------------------------------------
_POINT_RE = re.compile(r'\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)')


def parse_probes(path):
    groups = {}
    default_label = os.path.basename(path)
    current_label = default_label

    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("//"):
                text = line.lstrip("/ ").strip()
                text = re.split(r'\s+@\s+', text)[0]
                text = re.split(r'\s+-\s+', text)[0]
                current_label = text.strip() or default_label
                continue
            m = _POINT_RE.search(line)
            if m:
                x, y, z = (float(v) for v in m.groups())
                groups.setdefault(current_label, []).append((x, y, z))

    return groups


# ---------------------------------------------------------------------------
# Construcción de la Escena 3D
# ---------------------------------------------------------------------------
def get_patch_color(filename, index):
    name_lower = os.path.basename(filename).lower()
    for key, color in PATCH_COLOR_MAP.items():
        if key in name_lower:
            return color
    return DEFAULT_STL_COLORS[index % len(DEFAULT_STL_COLORS)]


def build_figure(stl_paths, probe_input_paths, max_tris=25000, stl_opacity=0.45):
    fig = go.Figure()

    # 1. Cargar geometrías STL
    resolved_stls = discover_all_stls(stl_paths)
    stls_cargados = 0

    for i, resolved_path in enumerate(resolved_stls):
        try:
            x, y, z, ii, jj, kk = read_stl_decimated(resolved_path, max_tris=max_tris)
            patch_name = os.path.basename(resolved_path)
            color = get_patch_color(resolved_path, i)

            print(f"[✅ STL Cargado] {resolved_path} ({len(ii)} triángulos)")
            stls_cargados += 1

            fig.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z, i=ii, j=jj, k=kk,
                    color=color,
                    opacity=stl_opacity,
                    name=patch_name,
                    showlegend=True,
                    hoverinfo="name",
                    flatshading=True,
                    lighting=dict(ambient=0.7, diffuse=0.8, fresnel=0.2, specular=0.3),
                    lightposition=dict(x=100, y=200, z=150)
                )
            )
        except Exception as e:
            print(f"[❌ Error STL] No se pudo leer {resolved_path}: {e}", file=sys.stderr)

    if stls_cargados == 0:
        print("\n[⚠️ ALERTA]: No se cargó ningún STL. Revisa la carpeta 'constant/triSurface/'.\n")

    # 2. Buscar y Renderizar Cajas de topoSetDict
    for topopath in TOPOSET_SEARCH_PATHS:
        resolved_topo = resolve_file_path(topopath)
        if resolved_topo:
            boxes = parse_toposet_boxes(resolved_topo)
            for box in boxes:
                print(f"[📦 topoSet Cargado] -> {box['name']}: {box['min']} a {box['max']}")
                add_toposet_box_to_fig(fig, box)
            break

    # 3. Cargar TODAS las sondas
    all_probe_files = discover_all_probes(probe_input_paths)
    color_i = 0
    probes_encontrados = 0

    print(f"\n🔍 Buscando sondas... Se encontraron {len(all_probe_files)} archivos:")

    for ppath in all_probe_files:
        try:
            groups = parse_probes(ppath)
            if not groups:
                continue

            base_filename = os.path.basename(ppath)
            print(f" [📍 Sonda Cargada] -> {ppath}")
            probes_encontrados += 1

            for group_label, pts in groups.items():
                pts = np.array(pts)
                
                if group_label == base_filename:
                    display_name = f"{base_filename} ({len(pts)} pts)"
                else:
                    display_name = f"{base_filename} / {group_label} ({len(pts)} pts)"

                hover_text = [
                    f"<b>Archivo:</b> {base_filename}<br>"
                    f"<b>Sonda/Grupo:</b> {group_label}<br>"
                    f"<b>X:</b> {p[0]:.4f} m<br><b>Y:</b> {p[1]:.4f} m<br><b>Z:</b> {p[2]:.4f} m"
                    for p in pts
                ]

                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=_PROBE_COLORS[color_i % len(_PROBE_COLORS)],
                            symbol="diamond",
                            line=dict(color="black", width=1)
                        ),
                        name=display_name,
                        text=hover_text,
                        hoverinfo="text",
                    )
                )
                color_i += 1
        except Exception as e:
            print(f"[⚠️ Error Sondas] No se pudo leer {ppath}: {e}", file=sys.stderr)

    if probes_encontrados == 0:
        print("[⚠️ Aviso] No se encontraron archivos de sondas válidos en 'Specific_functions'.")

    # Configuración global del visualizador Plotly
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            bgcolor="rgb(245, 245, 250)"
        ),
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        margin=dict(l=0, r=0, t=40, b=0),
        title="Visualización 3D Quirófano: Geometría, Sondas y Cajas topoSet",
    )
    return fig


# ---------------------------------------------------------------------------
# CLI / Ejecución principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stl", nargs="*", default=DEFAULT_STLS,
        help="Rutas a archivos STL (default: geometrías del quirófano)"
    )
    ap.add_argument(
        "--probes", nargs="*", default=None,
        help="Rutas adicionales o específicas a archivos/carpetas de sondas"
    )
    ap.add_argument(
        "--output", default="probes_view.html",
        help="Archivo HTML de salida (default: probes_view.html)"
    )
    ap.add_argument(
        "--max-tris", type=int, default=25000,
        help="Máximo de triángulos por STL antes de decimar (default: 25000)"
    )
    ap.add_argument(
        "--stl-opacity", type=float, default=0.45,
        help="Opacidad de la geometría STL, 0-1 (default: 0.45)"
    )
    args = ap.parse_args()

    print("🚀 Procesando visualización 3D de Quirófano, Sondas y topoSet...")
    fig = build_figure(
        args.stl, args.probes,
        max_tris=args.max_tris,
        stl_opacity=args.stl_opacity
    )

    fig.write_html(args.output, include_plotlyjs=True)
    print(f"\n🎉 ¡Proceso finalizado! Revisa el archivo HTML -> {args.output}")


if __name__ == "__main__":
    main()