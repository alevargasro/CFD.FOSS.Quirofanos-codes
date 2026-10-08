#!/usr/bin/env python3
"""
visualize_quirofano.py
=======================
Visualizador UNIFICADO: combina en un solo script lo que antes eran
visualize_probes.py + visualize_planes.py + visualize_topo.py.

Corrido SIN NINGUN ARGUMENTO, parado en la raiz del caso (misma carpeta
que contiene 0/, constant/, system/), hace todo solo:

  - Lee TODOS los STL que encuentre en constant/triSurface/
  - Si ninguno de esos STL parece ser un maniqui/paciente/mobiliario
    (paciente, mesa, lampara, cirujano, ayudante), avisa que es un
    "quirofano vacio" (sin dummies) -- no es un error, solo informa.
  - Recorre system/ completo (incluyendo system/specificFunctions/ y
    system/topoSetDict si existen) y clasifica AUTOMATICAMENTE cada
    archivo segun su contenido:
        * tiene "probeLocations"      -> sondas (probes)
        * tiene "cuttingPlane"        -> planos de corte
        * tiene "boxToCell"/topoSetDict -> zonas de topoSet
    No hace falta decirle nombres de archivo ni rutas exactas.
  - Genera UN SOLO HTML interactivo con BOTONES arriba para elegir que
    ver: Todos / Probes / Planos / Topo (la geometria STL siempre se ve,
    de fondo, en cualquier modo).
  - Los planos de corte SIEMPRE se dibujan rellenos con color (nunca solo
    el contorno), recortados a la geometria real de las paredes cuando es
    posible, y si eso falla (STL no cerrado, etc.) caen a un rectangulo
    relleno dimensionado para no salirse del dominio real de la sala.

USO
---
    # Recomendado -- parado en la raiz del caso, sin argumentos:
    cd /mnt/datos/OpenFOAM_Data/cases/Q1_Personal
    python3 visualize_quirofano.py

    # Con rutas manuales (si tu caso no sigue la convencion estandar):
    python3 visualize_quirofano.py \\
        --stl-dir constant/triSurface \\
        --functions-dir system/specificFunctions \\
        --topo-file system/topoSetDict \\
        --output mi_vista.html

Requisitos: numpy-stl, plotly, trimesh, shapely, mapbox_earcut, networkx
    pip install numpy-stl plotly trimesh shapely mapbox_earcut networkx --break-system-packages
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
    import trimesh
except ImportError:
    trimesh = None

try:
    import plotly.graph_objects as go
except ImportError:
    print("Falta plotly. Instala con: pip install plotly", file=sys.stderr)
    raise


DUMMY_KEYWORDS = ["paciente", "manta", "mesa", "lampara", "cirujano", "ayudante"]


def classify_zone_name(name):
    """Subcategoria de una zona topo, a partir de su nombre (banda_baja,
    esquina_3_alta, zonaCamilla...). Sirve para poder aislar 'solo bandas'
    o 'solo esquinas' en vez de un unico interruptor con las 16 juntas."""
    low = name.lower()
    if "banda" in low:
        return "banda"
    if "esquina" in low or "corner" in low:
        return "esquina"
    return "paciente"


# ===========================================================================
# 1. Lectura de STL
# ===========================================================================
def discover_stls(stl_dir):
    if not os.path.isdir(stl_dir):
        return []
    paths = sorted(glob.glob(os.path.join(stl_dir, "*.stl")) +
                    glob.glob(os.path.join(stl_dir, "*.STL")))
    # evitar duplicados si el filesystem no distingue mayusculas
    seen, out = set(), []
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def read_stl_decimated(path, max_tris=15000, seed=0):
    """Reduce el conteo de triangulos de un STL para que Plotly lo renderice
    rapido, PERO preservando la conectividad real de la malla (simplificacion
    por quadric decimation via trimesh), no un recorte al azar.

    Un muestreo aleatorio de triangulos dejaba la superficie con aspecto de
    'confeti': con una malla de cientos de miles de triangulos recortada a
    ~15000, la probabilidad de que dos triangulos vecinos sobrevivan JUNTOS
    es minuscula (con walls.stl de Q1: 311122 -> 15000 triangulos, esa
    probabilidad es de solo 0.23%), asi que casi ningun triangulo conservado
    tenia un vecino con el que formar una superficie continua. La
    decimation real fusiona la geometria en vez de tirar piezas sueltas, y
    el resultado se ve solido incluso muy simplificado.

    Si trimesh/fast-simplification no estan disponibles, o la decimation
    falla por algun motivo (malla no-manifold, etc.), cae de vuelta al
    metodo anterior (muestreo aleatorio con numpy-stl) en vez de romper."""
    if trimesh is not None:
        try:
            mesh = trimesh.load(path, force="mesh")
            full_verts = mesh.vertices  # bbox de referencia: SIEMPRE la malla completa,
                                         # sin decimar (igual que el metodo anterior,
                                         # necesario para que el respaldo de planos
                                         # calce con el tamano real de la sala)
            if len(mesh.faces) > max_tris:
                mesh = mesh.simplify_quadric_decimation(face_count=max_tris)
            verts = mesh.vertices
            faces = mesh.faces
            return (verts[:, 0], verts[:, 1], verts[:, 2],
                    faces[:, 0], faces[:, 1], faces[:, 2], full_verts)
        except Exception as e:
            print(f"[aviso] decimation real fallo en {path} ({e}); "
                  f"se usa muestreo aleatorio de respaldo.", file=sys.stderr)

    if stlmesh is None:
        raise RuntimeError("Ni trimesh ni numpy-stl estan disponibles para leer STL")
    m = stlmesh.Mesh.from_file(path)
    n_tris = len(m.vectors)
    if n_tris > max_tris:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_tris, size=max_tris, replace=False)
        tris = m.vectors[idx]
    else:
        tris = m.vectors
    verts = tris.reshape(-1, 3)
    n_verts = len(verts)
    i_idx = np.arange(0, n_verts, 3)
    j_idx = i_idx + 1
    k_idx = i_idx + 2
    return verts[:, 0], verts[:, 1], verts[:, 2], i_idx, j_idx, k_idx, m.points.reshape(-1, 3)


def is_dummy_name(filename):
    lower = filename.lower()
    return any(kw in lower for kw in DUMMY_KEYWORDS)


# ===========================================================================
# 2. Clasificador + parsers de archivos de "functions" / topoSetDict
# ===========================================================================
def classify_dict_file(path):
    """Lee un archivo de texto y adivina que tipo de contenido OpenFOAM
    tiene, basandose en palabras clave. Devuelve 'probes', 'planes',
    'topo', o None si no coincide con nada reconocido (o no es texto)."""
    try:
        with open(path, "r", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None
    if len(content) > 2_000_000:  # evita leer STL/archivos binarios grandes por error
        return None
    if "probeLocations" in content:
        return "probes"
    if "cuttingPlane" in content:
        return "planes"
    if "boxToCell" in content or "topoSetDict" in content:
        return "topo"
    if re.search(r'\btype\s+uniform\s*;', content) and "nPoints" in content and "axis" in content:
        return "lines"
    return None


def discover_function_files(*dirs):
    """Recorre las carpetas dadas (recursivamente) y devuelve
    {'probes': [...], 'planes': [...], 'topo': [...]} con las rutas de
    cada archivo clasificado."""
    buckets = {"probes": [], "planes": [], "topo": [], "lines": []}
    seen = set()
    for d in dirs:
        if not d or not os.path.exists(d):
            continue
        if os.path.isfile(d):
            candidates = [d]
        else:
            candidates = []
            for root, _, files in os.walk(d):
                for f in files:
                    candidates.append(os.path.join(root, f))
        for path in candidates:
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                continue
            seen.add(key)
            kind = classify_dict_file(path)
            if kind:
                buckets[kind].append(path)
    return buckets


# --- Parser de probes ---
_POINT_RE = re.compile(r'\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)')


def parse_probes(path):
    groups = {}
    label = "probes"
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("//"):
                text = line.lstrip("/ ").strip()
                text = re.split(r'\s+@\s+', text)[0]
                text = re.split(r'\s+-\s+', text)[0]
                label = text.strip() or label
                continue
            m = _POINT_RE.search(line)
            if m:
                x, y, z = (float(v) for v in m.groups())
                groups.setdefault(label, []).append((x, y, z))
    return groups


# --- Parser de cuttingPlane ---
_NAMED_BLOCK_RE = re.compile(r'(\w+)\s*\{([^{}]*)\}', re.DOTALL)
_POINT_KV_RE = re.compile(r'\bpoint\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)\s*;')
_NORMAL_KV_RE = re.compile(r'\bnormal\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)\s*;')


def parse_cutting_planes(path):
    """Busca bloques 'nombre { ... }' SIN llaves anidadas (asi encuentra solo
    los planos hoja, nunca el diccionario contenedor) que mencionen
    'cuttingPlane', y les saca 'point' y 'normal' donde sea que aparezcan
    dentro del bloque -- tolera comentarios y el orden de los campos, a
    diferencia de la version anterior que exigia 'type cuttingPlane;' pegado
    justo despues de la llave y fallaba en silencio (cero planos, sin
    ningun aviso) si habia un comentario de por medio."""
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    planes = []
    for m in _NAMED_BLOCK_RE.finditer(text):
        name, block = m.group(1), m.group(2)
        if "cuttingPlane" not in block:
            continue
        pm = _POINT_KV_RE.search(block)
        nm = _NORMAL_KV_RE.search(block)
        if not pm or not nm:
            continue
        point = np.array([float(pm.group(i)) for i in (1, 2, 3)])
        normal = np.array([float(nm.group(i)) for i in (1, 2, 3)])
        n_norm = np.linalg.norm(normal)
        if n_norm > 0:
            normal = normal / n_norm
        planes.append({"name": name, "point": point, "normal": normal})
    return planes


# --- Parser de lineas de perfil (functionObject 'sets', type uniform) ---
_LINESET_RE = re.compile(
    r'(\w+)\s*\{\s*(?://[^\n]*\n\s*)*'
    r'type\s+uniform\s*;.*?'
    r'axis\s+(\w+)\s*;.*?'
    r'start\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)\s*;.*?'
    r'end\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)\s*;.*?'
    r'nPoints\s+(\d+)\s*;',
    re.DOTALL,
)


def parse_line_sets(path):
    """Convierte cada bloque 'nombre { type uniform; axis ..; start (..);
    end (..); nPoints N; }' en los N puntos reales que OpenFOAM va a
    samplear (linspace entre start y end) -- el archivo NO trae los puntos
    explicitos como 'probes', solo la receta para generarlos."""
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    lines = []
    for m in _LINESET_RE.finditer(text):
        name, axis = m.group(1), m.group(2)
        start = np.array([float(m.group(i)) for i in (3, 4, 5)])
        end = np.array([float(m.group(i)) for i in (6, 7, 8)])
        n = int(m.group(9))
        if n < 1:
            continue
        pts = np.linspace(start, end, n)
        lines.append({"name": name, "axis": axis, "start": start, "end": end, "points": pts})
    return lines


# --- Parser de topoSetDict (boxToCell) ---
_BLOCK_RE = re.compile(r'\{([^{}]*)\}', re.DOTALL)
_NAME_RE = re.compile(r'name\s+([^\s;]+)\s*;')
_SOURCE_RE = re.compile(r'source\s+([^\s;]+)\s*;')
_BOX_RE = re.compile(
    r'box\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)\s*'
    r'\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)\s*;'
)


def parse_topo_boxes(path):
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    boxes = []
    for m in _BLOCK_RE.finditer(text):
        block = m.group(1)
        box_m = _BOX_RE.search(block)
        if not box_m:
            continue
        name_m = _NAME_RE.search(block)
        source_m = _SOURCE_RE.search(block)
        vals = [float(v) for v in box_m.groups()]
        boxes.append({
            "name": name_m.group(1) if name_m else "zona",
            "source": source_m.group(1) if source_m else "boxToCell",
            "min": np.array(vals[:3]),
            "max": np.array(vals[3:]),
        })
    return boxes


# ===========================================================================
# 3. Geometria auxiliar: planos (recorte real + respaldo relleno) y cajas
# ===========================================================================
def _plane_basis(normal):
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, normal)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return u, v


def load_domain_mesh(path):
    if trimesh is None:
        raise RuntimeError("trimesh no esta instalado")
    m = trimesh.load(path)
    try:
        m.merge_vertices()
    except Exception:
        pass
    return m


def clip_plane_to_domain(domain_mesh, point, normal):
    """Interseccion REAL del plano con la geometria de paredes. Devuelve
    (verts3d, faces) rellenos, o (None, None) si no se pudo (se debe usar
    el respaldo de caja rellena en ese caso -- nunca dejamos solo lineas)."""
    try:
        from trimesh.creation import triangulate_polygon
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        return None, None

    normal = normal / np.linalg.norm(normal)
    try:
        section = domain_mesh.section(plane_origin=point, plane_normal=normal)
    except Exception:
        return None, None
    if section is None:
        return None, None

    outline_segments = section.discrete
    if not outline_segments:
        for tol in (0.005, 0.02, 0.05, 0.1):
            try:
                section.fill_gaps(distance=tol)
            except Exception:
                pass
            outline_segments = section.discrete
            if outline_segments:
                break
    if not outline_segments:
        return None, None

    u, v = _plane_basis(normal)
    loops_2d = []
    for seg in outline_segments:
        rel = np.asarray(seg) - point
        loops_2d.append(np.column_stack([rel @ u, rel @ v]))

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
        return None, None
    if faces is None or len(faces) == 0:
        return None, None

    verts3d = point + np.outer(verts2d[:, 0], u) + np.outer(verts2d[:, 1], v)
    return verts3d, faces


def plane_quad_bounds(point, normal, s_min, s_max, t_min, t_max):
    """Rectangulo relleno definido por limites INDEPENDIENTES en cada
    direccion (s_min..s_max a lo largo de u, t_min..t_max a lo largo de
    v) relativos a point -- a diferencia de un tamano simetrico, esto
    ajusta el plano exactamente al contorno real de la sala sin importar
    que tan lejos del centro este 'point' (ej. un outlet en una esquina)."""
    u, v = _plane_basis(normal / np.linalg.norm(normal))
    corners = np.array([
        point + s_min * u + t_min * v,
        point + s_max * u + t_min * v,
        point + s_max * u + t_max * v,
        point + s_min * u + t_max * v,
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return corners, faces


def required_uv_bounds(point, normal, bbox_min, bbox_max, margin=0.05):
    """Limites REALES (min,max) -- no simetricos -- de la caja
    [bbox_min,bbox_max] proyectada sobre la base (u,v) del plano, relativos
    a 'point'. 'margin' es un colchon fijo en metros (no un factor
    multiplicativo), para que no se distorsione cuando 'point' esta lejos
    del centro de la sala (ej. sobre un outlet en una esquina)."""
    u, v = _plane_basis(normal)
    corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]], [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]], [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]], [bbox_max[0], bbox_max[1], bbox_max[2]],
    ])
    rel = corners - point
    s, t = rel @ u, rel @ v
    return s.min() - margin, s.max() + margin, t.min() - margin, t.max() + margin


def box_mesh(bmin, bmax):
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ])
    return verts, faces


def box_edges(bmin, bmax):
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z0), None,
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1), None,
        (x0, y0, z0), (x0, y0, z1), None, (x1, y0, z0), (x1, y0, z1), None,
        (x1, y1, z0), (x1, y1, z1), None, (x0, y1, z0), (x0, y1, z1),
    ]
    xs = [c[0] if c else None for c in corners]
    ys = [c[1] if c else None for c in corners]
    zs = [c[2] if c else None for c in corners]
    return xs, ys, zs


# ===========================================================================
# 4. Construccion de la figura unificada, con botones por categoria
# ===========================================================================
_STL_COLORS = ["lightgray", "lightblue", "wheat", "lightgreen", "plum", "khaki",
               "lightsalmon", "lightsteelblue"]
_PROBE_COLORS = ["red", "blue", "green", "orange", "purple", "brown",
                  "magenta", "cyan", "darkred", "navy", "darkgreen", "gold"]
_PLANE_COLORS = ["red", "blue", "green", "orange", "purple", "brown", "magenta", "cyan"]
_ZONE_COLORS = ["red", "blue", "green", "orange", "purple", "brown", "magenta", "cyan"]


def build_figure(stl_paths, probe_paths, plane_paths, topo_paths, line_paths=(),
                  max_tris=15000, wall_max_tris=500, stl_opacity=0.18, plane_opacity=0.45,
                  zone_opacity=0.25):
    fig = go.Figure()
    trace_categories = []  # 'stl' | 'probes' | 'planes' | 'topo' | 'lines', paralelo a fig.data

    all_pts = []
    domain_path = None
    for p in stl_paths:
        if "wall" in os.path.basename(p).lower():
            domain_path = p
            break

    # --- STL (siempre visibles) ---
    _WALL_COLOR = "lightslategray"
    other_color_i = 0
    for path in stl_paths:
        # las paredes se recortan mucho mas agresivo que el resto: algunos
        # quirofanos vienen de modelos arquitectonicos reales con cientos de
        # miles de triangulos (mucho detalle de superficie), otros de cajas
        # parametricas simples de apenas 40 -- sin este tope aparte, las
        # paredes se ven con densidad de detalle muy distinta entre casos
        # aunque geometricamente sean todas 'una caja con paredes'. Se
        # verifico que el volumen (y por lo tanto cualquier entrante/recorte
        # de la pared) se preserva intacto incluso muy por debajo de este
        # tope, asi que no se pierde forma real por bajarlo.
        is_wall = "wall" in os.path.basename(path).lower()
        tris_cap = wall_max_tris if is_wall else max_tris
        try:
            x, y, z, ii, jj, kk, pts = read_stl_decimated(path, max_tris=tris_cap)
        except Exception as e:
            print(f"[aviso] no se pudo leer {path}: {e}", file=sys.stderr)
            continue
        all_pts.append(pts)
        # las paredes SIEMPRE con el mismo color fijo, sin importar cuantos
        # otros STL haya ni en que orden -- antes dependia de la posicion en
        # la lista (walls.stl podia caer en cualquier indice segun cuantos
        # archivos hubiera y su orden alfabetico), asi que salia un color
        # distinto en cada quirofano. El resto de los STL si sigue rotando
        # colores entre si, para poder distinguirlos.
        if is_wall:
            mesh_color = _WALL_COLOR
        else:
            mesh_color = _STL_COLORS[other_color_i % len(_STL_COLORS)]
            other_color_i += 1
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=ii, j=jj, k=kk,
            color=mesh_color,
            opacity=stl_opacity,
            name=os.path.basename(path),
            showlegend=True, hoverinfo="skip", flatshading=True,
        ))
        trace_categories.append("stl")

    bbox_min = bbox_max = None
    if all_pts:
        concat = np.concatenate(all_pts, axis=0)
        bbox_min, bbox_max = concat.min(axis=0), concat.max(axis=0)

    domain_mesh = None
    if domain_path is not None:
        try:
            domain_mesh = load_domain_mesh(domain_path)
        except Exception as e:
            print(f"[aviso] no se pudo cargar dominio ({domain_path}): {e}", file=sys.stderr)

    # --- Probes ---
    color_i = 0
    for ppath in probe_paths:
        try:
            groups = parse_probes(ppath)
        except Exception as e:
            print(f"[aviso] no se pudo leer probes {ppath}: {e}", file=sys.stderr)
            continue
        base_name = os.path.basename(ppath)
        for label, pts in groups.items():
            pts = np.array(pts)
            hover = [f"{base_name} / {label}<br>x={p[0]:.4f} y={p[1]:.4f} z={p[2]:.4f}" for p in pts]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                marker=dict(size=4, color=_PROBE_COLORS[color_i % len(_PROBE_COLORS)]),
                name=f"{base_name}: {label} ({len(pts)} pts)",
                text=hover, hoverinfo="text",
            ))
            trace_categories.append("probes")
            color_i += 1

    # --- Lineas de perfil (functionObject 'sets', type uniform) ---
    # Estas NO listan puntos explicitos como 'probes': el archivo solo
    # trae start/end/nPoints, asi que hay que generar los puntos reales
    # (parse_line_sets ya lo hace via linspace) para poder verlas.
    color_i = 0
    for lpath in line_paths:
        try:
            linesets = parse_line_sets(lpath)
        except Exception as e:
            print(f"[aviso] no se pudo leer lineas de perfil {lpath}: {e}", file=sys.stderr)
            continue
        base_name = os.path.basename(lpath)
        for ls in linesets:
            pts = ls["points"]
            color = _PROBE_COLORS[color_i % len(_PROBE_COLORS)]
            hover = [f"{base_name} / {ls['name']} (eje {ls['axis']})<br>"
                     f"x={p[0]:.4f} y={p[1]:.4f} z={p[2]:.4f}" for p in pts]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers+lines",
                marker=dict(size=3, color=color), line=dict(color=color, width=3),
                name=f"{base_name}: {ls['name']} ({len(pts)} pts, eje {ls['axis']})",
                text=hover, hoverinfo="text",
            ))
            trace_categories.append("lines")
            color_i += 1

    # --- Planos (SIEMPRE rellenos con color) ---
    color_i = 0
    for ppath in plane_paths:
        try:
            planes = parse_cutting_planes(ppath)
        except Exception as e:
            print(f"[aviso] no se pudo leer planos {ppath}: {e}", file=sys.stderr)
            continue
        for pl in planes:
            color = _PLANE_COLORS[color_i % len(_PLANE_COLORS)]
            verts3d = faces = None
            if domain_mesh is not None:
                verts3d, faces = clip_plane_to_domain(domain_mesh, pl["point"], pl["normal"])
            recortado = verts3d is not None

            if verts3d is None:
                # Respaldo: SIEMPRE relleno y dimensionado con la caja REAL
                # de los STL cargados -- nunca un numero inventado. Si no
                # hay ningun STL cargado, no hay forma de saber que tamano
                # tiene la sala real, asi que se OMITE el plano en vez de
                # dibujar algo del tamano que sea (eso es lo que generaba
                # planos gigantes que se salian del dominio real).
                if bbox_min is None:
                    print(f"[aviso] '{pl['name']}': no se pudo recortar al dominio NI hay "
                          f"ningun STL cargado para estimar su tamano real -- se omite "
                          f"este plano (revisa que --stl-dir encuentre tus STL).",
                          file=sys.stderr)
                    color_i += 1
                    continue
                s_min, s_max, t_min, t_max = required_uv_bounds(pl["point"], pl["normal"], bbox_min, bbox_max)
                verts3d, faces = plane_quad_bounds(pl["point"], pl["normal"], s_min, s_max, t_min, t_max)

            fig.add_trace(go.Mesh3d(
                x=verts3d[:, 0], y=verts3d[:, 1], z=verts3d[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=plane_opacity,
                name=f"{pl['name']}" + (" (recorte real)" if recortado else " (aprox.)"),
                showlegend=True,
                hovertext=(f"{pl['name']}<br>point=({pl['point'][0]:.3f},{pl['point'][1]:.3f},"
                           f"{pl['point'][2]:.3f})<br>normal=({pl['normal'][0]:.3f},"
                           f"{pl['normal'][1]:.3f},{pl['normal'][2]:.3f})"),
                hoverinfo="text",
            ))
            trace_categories.append("planes")
            color_i += 1

    # --- Topo (cajas boxToCell), sub-clasificadas por banda/esquina/paciente ---
    color_i = 0
    topo_zone_names = []       # nombre de cada zona, en orden de aparicion
    topo_zone_subcats = []     # subcategoria (banda/esquina/paciente) de cada zona, mismo orden
    topo_trace_zone_idx = []   # a que indice de topo_zone_names pertenece cada TRACE topo
    for tpath in topo_paths:
        try:
            boxes = parse_topo_boxes(tpath)
        except Exception as e:
            print(f"[aviso] no se pudo leer topo {tpath}: {e}", file=sys.stderr)
            continue
        for bx in boxes:
            zone_idx = len(topo_zone_names)
            topo_zone_names.append(bx["name"])
            subcat = classify_zone_name(bx["name"])
            topo_zone_subcats.append(subcat)
            color = _ZONE_COLORS[color_i % len(_ZONE_COLORS)]
            dims = bx["max"] - bx["min"]
            verts, faces = box_mesh(bx["min"], bx["max"])
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=zone_opacity,
                name=f"{bx['name']} ({dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f} m)",
                showlegend=True,
                hovertext=(f"{bx['name']} ({bx['source']})<br>"
                           f"min=({bx['min'][0]:.4f},{bx['min'][1]:.4f},{bx['min'][2]:.4f})<br>"
                           f"max=({bx['max'][0]:.4f},{bx['max'][1]:.4f},{bx['max'][2]:.4f})"),
                hoverinfo="text",
            ))
            trace_categories.append("topo")
            topo_trace_zone_idx.append(zone_idx)
            xs, ys, zs = box_edges(bx["min"], bx["max"])
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=color, width=5), showlegend=False, hoverinfo="skip",
            ))
            trace_categories.append("topo")
            topo_trace_zone_idx.append(zone_idx)
            color_i += 1

    # --- Botones por categoria (igual que antes) ---
    def visibility_for(*wanted):
        return [True if (cat == "stl" or cat in wanted) else False for cat in trace_categories]

    n_probes = trace_categories.count("probes")
    n_planes = trace_categories.count("planes")
    n_topo = trace_categories.count("topo")
    n_lines = trace_categories.count("lines")

    buttons = [dict(label="Todos", method="update",
                     args=[{"visible": visibility_for("probes", "planes", "topo", "lines")}])]
    if n_probes:
        buttons.append(dict(label=f"Probes ({n_probes})", method="update",
                             args=[{"visible": visibility_for("probes")}]))
    if n_lines:
        buttons.append(dict(label=f"Perfiles ({n_lines})", method="update",
                             args=[{"visible": visibility_for("lines")}]))
    if n_planes:
        buttons.append(dict(label=f"Planos ({n_planes})", method="update",
                             args=[{"visible": visibility_for("planes")}]))
    if n_topo:
        buttons.append(dict(label=f"Topo: todas ({n_topo // 2})", method="update",
                             args=[{"visible": visibility_for("topo")}]))
    buttons.append(dict(label="Solo geometria", method="update",
                         args=[{"visible": visibility_for()}]))

    # posicion secuencial dentro de los traces topo (0,1,2...) para cada
    # entrada de trace_categories; None si esa entrada no es topo (no se usa)
    topo_trace_pos = []
    _t = 0
    for cat in trace_categories:
        if cat == "topo":
            topo_trace_pos.append(_t)
            _t += 1
        else:
            topo_trace_pos.append(None)

    # --- Sub-botones de Topo: banda / esquina / paciente por separado ---
    # (con las 16 zonas juntas era imposible distinguir una esquina de otra;
    # esto permite ver solo las 3 bandas grandes, o solo las 12 esquinas,
    # o solo la zona del paciente, sin las demas encima)
    topo_subcat_buttons = []
    if n_topo:
        def visibility_for_topo_subcat(subcat):
            vis = []
            for cat, t_i in zip(trace_categories, topo_trace_pos):
                if cat == "stl":
                    vis.append(True)
                elif cat == "topo" and topo_zone_subcats[topo_trace_zone_idx[t_i]] == subcat:
                    vis.append(True)
                else:
                    vis.append(False)
            return vis

        for subcat, label in [("banda", "Bandas"), ("esquina", "Esquinas"), ("paciente", "Paciente")]:
            count = sum(1 for s in topo_zone_subcats if s == subcat)
            if count:
                topo_subcat_buttons.append(dict(
                    label=f"Topo: {label} ({count})", method="update",
                    args=[{"visible": visibility_for_topo_subcat(subcat)}],
                ))

    # --- Dropdown: aislar UNA sola zona topo a la vez ---
    # (util para confirmar que una esquina puntual quedo bien ubicada,
    # sin las otras 15 zonas tapando la vista)
    zone_dropdown_buttons = []
    if topo_zone_names:
        for zi, zname in enumerate(topo_zone_names):
            vis = []
            for cat, t_i in zip(trace_categories, topo_trace_pos):
                if cat == "stl":
                    vis.append(True)
                elif cat == "topo" and topo_trace_zone_idx[t_i] == zi:
                    vis.append(True)
                else:
                    vis.append(False)
            zone_dropdown_buttons.append(dict(label=zname, method="update", args=[{"visible": vis}]))

    # --- Presets de camara: planta / frontal / lateral / isometrica ---
    # (misma logica que se usa para anotar la geometria a mano: vista de
    # planta para ver ubicacion en XY, vista frontal/lateral para alturas)
    camera_buttons = [
        dict(label="Isometrica", method="relayout",
             args=[{"scene.camera": dict(eye=dict(x=1.4, y=1.4, z=1.1), up=dict(x=0, y=0, z=1))}]),
        dict(label="Planta (arriba)", method="relayout",
             args=[{"scene.camera": dict(eye=dict(x=0.0, y=0.0, z=2.6), up=dict(x=0, y=1, z=0))}]),
        dict(label="Frontal", method="relayout",
             args=[{"scene.camera": dict(eye=dict(x=0.0, y=-2.6, z=0.15), up=dict(x=0, y=0, z=1))}]),
        dict(label="Lateral", method="relayout",
             args=[{"scene.camera": dict(eye=dict(x=2.6, y=0.0, z=0.15), up=dict(x=0, y=0, z=1))}]),
    ]

    updatemenus = [dict(
        type="buttons", direction="right", showactive=True,
        x=0.02, y=1.14, xanchor="left", yanchor="top",
        buttons=buttons,
    )]
    if topo_subcat_buttons:
        updatemenus.append(dict(
            type="buttons", direction="right", showactive=True,
            x=0.02, y=1.08, xanchor="left", yanchor="top",
            buttons=topo_subcat_buttons,
        ))
    updatemenus.append(dict(
        type="buttons", direction="right", showactive=True,
        x=0.02, y=1.02, xanchor="left", yanchor="top",
        buttons=camera_buttons,
    ))
    if zone_dropdown_buttons:
        updatemenus.append(dict(
            type="dropdown", direction="down", showactive=True,
            x=0.98, y=1.14, xanchor="right", yanchor="top",
            buttons=[dict(label="Aislar zona topo...", method="skip", args=[])] + zone_dropdown_buttons,
        ))

    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=110, b=0),
        title="Visualizacion del quirofano",
        updatemenus=updatemenus,
    )
    return fig


# ===========================================================================
# 5. CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-dir", default=".",
                    help="Raiz del caso (default: directorio actual)")
    ap.add_argument("--stl-dir", default=None,
                    help="Carpeta con los STL (default: <case-dir>/constant/triSurface)")
    ap.add_argument("--functions-dir", default=None,
                    help="Carpeta con probes/planos (default: <case-dir>/system/specificFunctions)")
    ap.add_argument("--topo-file", default=None,
                    help="Archivo topoSetDict (default: <case-dir>/system/topoSetDict, si existe)")
    ap.add_argument("--output", default="quirofano_view.html")
    ap.add_argument("--max-tris", type=int, default=15000)
    ap.add_argument("--wall-max-tris", type=int, default=500,
                    help="Tope de triangulos SOLO para paredes (default: 500). Las paredes "
                         "vienen de fuentes muy distintas segun el quirofano -- modelos "
                         "arquitectonicos reales con cientos de miles de triangulos, o cajas "
                         "parametricas simples de apenas 40 -- y sin un tope aparte, mas bajo, "
                         "se ven con densidad de detalle muy distinta entre casos aunque "
                         "geometricamente sean todas 'una caja con paredes'.")
    ap.add_argument("--stl-opacity", type=float, default=0.18)
    ap.add_argument("--plane-opacity", type=float, default=0.45)
    ap.add_argument("--zone-opacity", type=float, default=0.25)
    args = ap.parse_args()

    stl_dir = args.stl_dir or os.path.join(args.case_dir, "constant", "triSurface")
    functions_dir = args.functions_dir or os.path.join(args.case_dir, "system", "specificFunctions")
    topo_file = args.topo_file or os.path.join(args.case_dir, "system", "topoSetDict")

    # --- STL ---
    stl_paths = discover_stls(stl_dir)
    print(f"[info] STL encontrados en {stl_dir}: {len(stl_paths)}")
    for p in stl_paths:
        print(f"    - {os.path.basename(p)}")

    dummy_found = [os.path.basename(p) for p in stl_paths if is_dummy_name(os.path.basename(p))]
    if not dummy_found:
        print("\n*** QUIROFANO VACIO: no se detecto paciente/mobiliario (mesa, lampara, "
              "cirujano, ayudante) entre los STL. ***\n")
    else:
        print(f"\nManiquies/mobiliario detectado: {', '.join(dummy_found)}\n")

    # --- Functions (probes + planos) + topo ---
    search_targets = [functions_dir]
    if os.path.isfile(topo_file):
        search_targets.append(topo_file)
    buckets = discover_function_files(*search_targets)

    print(f"[info] Archivos de sondas (probes) detectados: {len(buckets['probes'])}")
    for p in buckets["probes"]:
        print(f"    - {p}")
    print(f"[info] Archivos de lineas de perfil (sets/uniform) detectados: {len(buckets['lines'])}")
    for p in buckets["lines"]:
        print(f"    - {p}")
    print(f"[info] Archivos de planos (cuttingPlane) detectados: {len(buckets['planes'])}")
    for p in buckets["planes"]:
        print(f"    - {p}")
    print(f"[info] Archivos de topoSet (boxToCell) detectados: {len(buckets['topo'])}")
    for p in buckets["topo"]:
        print(f"    - {p}")

    if not (stl_paths or buckets["probes"] or buckets["planes"] or buckets["topo"] or buckets["lines"]):
        print("[error] No se encontro absolutamente nada (ni STL ni funciones). "
              "Verifica que estas parado en la raiz del caso.", file=sys.stderr)
        sys.exit(1)

    fig = build_figure(stl_paths, buckets["probes"], buckets["planes"], buckets["topo"],
                        line_paths=buckets["lines"],
                        max_tris=args.max_tris, wall_max_tris=args.wall_max_tris,
                        stl_opacity=args.stl_opacity,
                        plane_opacity=args.plane_opacity, zone_opacity=args.zone_opacity)
    fig.write_html(args.output, include_plotlyjs=True)
    print(f"\nListo -> {args.output}")


if __name__ == "__main__":
    main()
