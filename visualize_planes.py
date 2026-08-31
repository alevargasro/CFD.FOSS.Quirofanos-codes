#!/usr/bin/env python3
"""
visualize_planes.py
====================
Visualizador genérico de planos de corte (cuttingPlane) definidos en
functionObjects "surfaces" de OpenFOAM (ej. "cortesVTK"), superpuestos
sobre la geometría STL de un caso, usando Plotly para un HTML 3D interactivo.

Ajusta y recorta EXACTAMENTE los planos al interior del quirófano (AABB / Mesh Clipping)
para evitar que sobresalgan del techo, suelo o paredes.
"""

import argparse
import os
import re
import sys
import numpy as np

try:
    from stl import mesh as stlmesh
except ImportError:
    stlmesh = None

try:
    import trimesh
except ImportError:
    trimesh = None

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

PLANE_SEARCH_DIRS = [
    "system/General_functions",
    "General_functions",
    "system/functions",
    "functions",
    "system",
    "system/specificFunctions",
    "specificFunctions"
]

DEFAULT_DOMAIN_FILE = "constant/triSurface/walls.stl"

PATCH_COLOR_MAP = {
    "inlet": "#00d2ff",      # Cyan
    "outlet": "#ff6b6b",     # Naranja / Rojo
    "paciente": "#e056fd",   # Magenta
    "manta": "#be2edd",      # Púrpura
    "wall": "#b2bec3",       # Gris Metalizado
    "muro": "#b2bec3",
    "piso": "#636e72",
    "techo": "#dfe6e9",
}

DEFAULT_STL_COLORS = [
    "#74b9ff", "#ff7675", "#a29bfe", "#55efc4", "#ffeaa7", "#fd79a8"
]

_PLANE_COLORS = [
    "#ff4757", "#2ed573", "#1e90ff", "#ffa502", "#9b59b6", "#eccc68",
    "#70a1ff", "#ff6b81", "#7bed9f", "#e84118", "#00a8ff", "#9c88ff"
]


# ---------------------------------------------------------------------------
# Búsqueda y Resolución Dinámica de Rutas
# ---------------------------------------------------------------------------
def resolve_file_path(filename):
    if not filename:
        return None
    if os.path.exists(filename):
        return filename

    basename = os.path.basename(filename)
    search_dirs = [
        "constant/triSurface",
        "triSurface",
        "system/General_functions",
        "General_functions",
        "system/functions",
        "system",
        "system/specificFunctions",
        "specificFunctions",
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
    """Busca archivos STL disponibles en el directorio del proyecto."""
    discovered = []

    if input_stls:
        for stl in input_stls:
            resolved = resolve_file_path(stl)
            if resolved and resolved not in discovered:
                discovered.append(resolved)

    if not discovered:
        search_folder = resolve_file_path("constant/triSurface") or resolve_file_path("triSurface")
        if search_folder and os.path.isdir(search_folder):
            for f in sorted(os.listdir(search_folder)):
                if f.endswith(".stl"):
                    discovered.append(os.path.join(search_folder, f))

    return discovered


def discover_all_plane_files(input_paths=None):
    """Busca archivos de configuración OpenFOAM que contengan planos de corte."""
    discovered = []

    for p_dir in PLANE_SEARCH_DIRS:
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
# Carga de STL con parser robusto
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

    return verts[:, 0], verts[:, 1], verts[:, 2], i_idx, j_idx, k_idx, verts


# ---------------------------------------------------------------------------
# Parser flexible de bloques cuttingPlane
# ---------------------------------------------------------------------------
_POINT_RE = re.compile(r'point\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)')
_NORMAL_RE = re.compile(r'normal\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)')


def parse_cutting_planes(path):
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    # Eliminar comentarios de C/C++
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    planes = []
    block_re = re.compile(r'([A-Za-z0-9_]+)\s*\{([^}]*type\s+cuttingPlane;[^}]*)\}', re.DOTALL)
    matches = list(block_re.finditer(text))

    if not matches:
        subblock_re = re.compile(r'([A-Za-z0-9_]+)\s*\{([^{}]*point[^{}]*normal[^{}]*)\}', re.DOTALL)
        matches = list(subblock_re.finditer(text))

    for m in matches:
        name = m.group(1)
        body = m.group(2)

        pt_match = _POINT_RE.search(body)
        norm_match = _NORMAL_RE.search(body)

        if pt_match and norm_match:
            point = np.array([float(pt_match.group(1)), float(pt_match.group(2)), float(pt_match.group(3))])
            normal = np.array([float(norm_match.group(1)), float(norm_match.group(2)), float(norm_match.group(3))])

            n_norm = np.linalg.norm(normal)
            if n_norm > 0:
                normal = normal / n_norm

            planes.append({"name": name, "point": point, "normal": normal, "file": os.path.basename(path)})

    return planes


# ---------------------------------------------------------------------------
# Recorte Geométrico Exacto de Plano contra Bounding Box (AABB)
# ---------------------------------------------------------------------------
def clip_plane_to_aabb(point, normal, bbox_min, bbox_max):
    """Recorta un plano 3D (point, normal) al Bounding Box [bbox_min, bbox_max]."""
    normal = np.array(normal, dtype=float)
    n_norm = np.linalg.norm(normal)
    if n_norm == 0 or bbox_min is None or bbox_max is None:
        return None, None

    normal = normal / n_norm
    point = np.array(point, dtype=float)

    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max

    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ])

    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]

    intersections = []
    eps = 1e-7

    for idx1, idx2 in edges:
        p1 = corners[idx1]
        p2 = corners[idx2]
        d1 = np.dot(normal, p1 - point)
        d2 = np.dot(normal, p2 - point)

        if abs(d1) < eps:
            intersections.append(p1)
        elif abs(d2) < eps:
            pass
        elif (d1 * d2) < 0:
            t = d1 / (d1 - d2)
            p_int = p1 + t * (p2 - p1)
            intersections.append(p_int)

    if len(intersections) < 3:
        return None, None

    unique_pts = []
    for pt in intersections:
        if not any(np.allclose(pt, u, atol=1e-5) for u in unique_pts):
            unique_pts.append(pt)

    if len(unique_pts) < 3:
        return None, None

    pts = np.array(unique_pts)
    center = np.mean(pts, axis=0)

    # Base ortonormal local en el plano
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Ordenamiento angular en 2D
    rel = pts - center
    u_coords = rel @ u
    v_coords = rel @ v

    angles = np.arctan2(v_coords, u_coords)
    sort_idx = np.argsort(angles)
    pts_sorted = pts[sort_idx]

    # Triangulación en abanico
    faces = []
    n_pts = len(pts_sorted)
    for i in range(1, n_pts - 1):
        faces.append([0, i, i + 1])

    return pts_sorted, np.array(faces)


def load_domain_mesh(path):
    if trimesh is None:
        return None
    try:
        m = trimesh.load(path)
        if hasattr(m, 'merge_vertices'):
            m.merge_vertices()
        return m
    except Exception:
        return None


def clip_plane_to_domain(domain_mesh, point, normal):
    if domain_mesh is None:
        return None, None, None

    try:
        from trimesh.creation import triangulate_polygon
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        return None, None, None

    normal = normal / np.linalg.norm(normal)
    try:
        section = domain_mesh.section(plane_origin=point, plane_normal=normal)
    except Exception:
        return None, None, None

    if section is None:
        return None, None, None

    outline_segments = section.discrete
    if not outline_segments:
        for tol in (0.005, 0.02, 0.05):
            try:
                section.fill_gaps(distance=tol)
            except Exception:
                pass
            outline_segments = section.discrete
            if outline_segments:
                break

    if not outline_segments:
        return None, None, None

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    loops_2d = []
    for seg in outline_segments:
        rel = np.asarray(seg) - point
        s = rel @ u
        t = rel @ v
        loops_2d.append(np.column_stack([s, t]))

    try:
        if len(loops_2d) == 1:
            poly = ShapelyPolygon(loops_2d[0])
        else:
            areas = [ShapelyPolygon(loop).area for loop in loops_2d]
            ext_idx = int(np.argmax(areas))
            exterior = loops_2d[ext_idx]
            holes = [loop for i, loop in enumerate(loops_2d) if i != ext_idx]
            poly = ShapelyPolygon(exterior, holes=holes)

        verts2d, faces = triangulate_polygon(poly, engine="earcut")
    except Exception:
        return None, None, outline_segments

    if faces is None or len(faces) == 0:
        return None, None, outline_segments

    verts3d = point + np.outer(verts2d[:, 0], u) + np.outer(verts2d[:, 1], v)
    return verts3d, faces, outline_segments


# ---------------------------------------------------------------------------
# Construcción de la Escena 3D
# ---------------------------------------------------------------------------
def get_patch_color(filename, index):
    name_lower = os.path.basename(filename).lower()
    for key, color in PATCH_COLOR_MAP.items():
        if key in name_lower:
            return color
    return DEFAULT_STL_COLORS[index % len(DEFAULT_STL_COLORS)]


def build_figure(stl_paths, plane_input_paths, max_tris=25000, stl_opacity=0.30,
                 plane_opacity=0.45, domain_path=None):
    fig = go.Figure()
    all_pts = []

    # 1. Cargar geometrías STL
    resolved_stls = discover_all_stls(stl_paths)
    stls_cargados = 0

    for i, resolved_path in enumerate(resolved_stls):
        try:
            x, y, z, ii, jj, kk, pts = read_stl_decimated(resolved_path, max_tris=max_tris)
            all_pts.append(pts)
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

    # Límites globales del quirófano (Bounding Box AABB)
    bbox_min = bbox_max = None
    if all_pts:
        pts_concat = np.concatenate(all_pts, axis=0)
        bbox_min = pts_concat.min(axis=0)
        bbox_max = pts_concat.max(axis=0)

    # 2. Cargar Dominio Mesh si trimesh está disponible
    domain_mesh = None
    resolved_domain = resolve_file_path(domain_path or DEFAULT_DOMAIN_FILE)
    if resolved_domain:
        domain_mesh = load_domain_mesh(resolved_domain)
        if domain_mesh is not None:
            print(f"[✂️ Dominio STL Clicado] {resolved_domain}")
        elif trimesh is None:
            print("[ℹ️ Info] 'trimesh' no está instalado. Usando recorte AABB por defecto.")

    # 3. Cargar y recortar Planos
    all_plane_files = discover_all_plane_files(plane_input_paths)
    color_i = 0
    planos_encontrados = 0

    print(f"\n🔍 Buscando planos... Se encontraron {len(all_plane_files)} archivos de funciones:")

    for ppath in all_plane_files:
        try:
            planes = parse_cutting_planes(ppath)
            if not planes:
                continue

            print(f" [📐 Planos Cargados] -> {ppath} ({len(planes)} planos detectados)")
            planos_encontrados += 1

            for pl in planes:
                color = _PLANE_COLORS[color_i % len(_PLANE_COLORS)]
                verts3d = faces = None

                # Intentar Recorte por Sección Malla
                if domain_mesh is not None:
                    verts3d, faces, _ = clip_plane_to_domain(
                        domain_mesh, pl["point"], pl["normal"]
                    )

                # Fallback: Recorte Exacto AABB del Quirófano
                if verts3d is None or faces is None or len(verts3d) < 3:
                    verts3d, faces = clip_plane_to_aabb(
                        pl["point"], pl["normal"], bbox_min, bbox_max
                    )

                display_title = f"{pl['file']} / {pl['name']}"
                group_id = f"plane_{pl['name']}_{color_i}"

                if verts3d is not None and faces is not None and len(faces) > 0:
                    # Malla del Plano Recortado
                    fig.add_trace(
                        go.Mesh3d(
                            x=verts3d[:, 0], y=verts3d[:, 1], z=verts3d[:, 2],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            color=color,
                            opacity=plane_opacity,
                            name=display_title,
                            legendgroup=group_id,
                            showlegend=True,
                            hovertext=(
                                f"<b>Plano:</b> {pl['name']}<br>"
                                f"<b>Archivo:</b> {pl['file']}<br>"
                                f"<b>Punto:</b> ({pl['point'][0]:.3f}, {pl['point'][1]:.3f}, {pl['point'][2]:.3f})<br>"
                                f"<b>Normal:</b> ({pl['normal'][0]:.3f}, {pl['normal'][1]:.3f}, {pl['normal'][2]:.3f})"
                            ),
                            hoverinfo="text",
                        )
                    )

                    # Borde contorno cerrado vinculado a la misma leyenda
                    x_line = list(verts3d[:, 0]) + [verts3d[0, 0]]
                    y_line = list(verts3d[:, 1]) + [verts3d[0, 1]]
                    z_line = list(verts3d[:, 2]) + [verts3d[0, 2]]

                    fig.add_trace(
                        go.Scatter3d(
                            x=x_line, y=y_line, z=z_line,
                            mode="lines",
                            line=dict(color="black", width=3),
                            legendgroup=group_id,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

                color_i += 1

        except Exception as e:
            print(f"[⚠️ Error Planos] No se pudo leer {ppath}: {e}", file=sys.stderr)

    if planos_encontrados == 0:
        print("[⚠️ Aviso] No se encontraron archivos con definiciones de planos válidas en 'General_functions'.")

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
        title="Visualización 3D Quirófano: Planos de Corte Recortados al Dominio",
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
        help="Rutas a archivos STL de geometría (default: geometrías del quirófano)"
    )
    ap.add_argument(
        "--planes", nargs="*", default=None,
        help="Rutas específicas a archivos/carpetas de planos (default: busca en General_functions)"
    )
    ap.add_argument(
        "--output", default="planes_view.html",
        help="Archivo HTML de salida (default: planes_view.html)"
    )
    ap.add_argument(
        "--max-tris", type=int, default=25000,
        help="Máximo de triángulos por STL antes de decimar (default: 25000)"
    )
    ap.add_argument(
        "--stl-opacity", type=float, default=0.30,
        help="Opacidad de la geometría STL, 0-1 (default: 0.30)"
    )
    ap.add_argument(
        "--plane-opacity", type=float, default=0.45,
        help="Opacidad de los planos de corte, 0-1 (default: 0.45)"
    )
    ap.add_argument(
        "--domain", default=DEFAULT_DOMAIN_FILE,
        help="STL que define el volumen cerrado para recorte (default: walls.stl)"
    )
    args = ap.parse_args()

    print("🚀 Procesando visualización 3D de Planos de Corte y Quirófano...")
    fig = build_figure(
        args.stl, args.planes,
        max_tris=args.max_tris,
        stl_opacity=args.stl_opacity,
        plane_opacity=args.plane_opacity,
        domain_path=args.domain
    )

    fig.write_html(args.output, include_plotlyjs=True)
    print(f"\n🎉 ¡Proceso finalizado! Revisa el archivo HTML -> {args.output}")


if __name__ == "__main__":
    main()