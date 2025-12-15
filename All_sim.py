import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider


# ============================================================
# 1) Icosahedron (frequency=1): unit-sphere vertices + faces
# ============================================================
def icosahedron_vertices_faces_unit():
    phi = (1 + np.sqrt(5)) / 2
    V = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ], dtype=float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    F = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ], dtype=int)
    return V, F


# ============================================================
# 2) Inradius of regular icosahedron with circumradius R=1
# ============================================================
def icosahedron_inradius_given_circumradius(R=1.0):
    r_over_R = (1/3) * np.sqrt(3) * (3 + np.sqrt(5)) / np.sqrt(10 + 2*np.sqrt(5))
    return R * r_over_R


# ============================================================
# 3) HEALPix boundaries on a sphere (healpy preferred)
# ============================================================
def healpix_boundaries_xyz(nside, step=2, radius=1.0):
    try:
        import healpy as hp
        npix = hp.nside2npix(nside)
        boundaries = []
        for ipix in range(npix):
            b = hp.boundaries(nside, ipix, step=step, nest=False)  # (3, N)
            boundaries.append(radius * b)
        return boundaries
    except ImportError:
        pass

    try:
        import astropy.units as u
        from astropy_healpix import HEALPix
        from astropy.coordinates import ICRS

        hpix = HEALPix(nside=nside, order="ring", frame=ICRS())
        boundaries = []
        for ipix in range(hpix.npix):
            sc = hpix.boundaries_skycoord(ipix, step=step)
            lon = sc.ra.to_value(u.rad)
            lat = sc.dec.to_value(u.rad)
            x = radius * np.cos(lat) * np.cos(lon)
            y = radius * np.cos(lat) * np.sin(lon)
            z = radius * np.sin(lat)
            boundaries.append(np.vstack([x, y, z]))
        return boundaries
    except ImportError:
        raise ImportError(
            "Need 'healpy' or 'astropy-healpix'. Install one:\n"
            "  pip install healpy\n"
            "or\n"
            "  pip install astropy-healpix\n"
        )


# ============================================================
# 4) Ray -> triangle intersection (Möller–Trumbore)
# ============================================================
def ray_intersect_triangle(origin, direction, v0, v1, v2, eps=1e-12):
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(direction, e2)
    det = np.dot(e1, pvec)
    if abs(det) < eps:
        return None
    inv_det = 1.0 / det
    tvec = origin - v0
    u = np.dot(tvec, pvec) * inv_det
    if u < -eps or u > 1.0 + eps:
        return None
    qvec = np.cross(tvec, e1)
    v = np.dot(direction, qvec) * inv_det
    if v < -eps or (u + v) > 1.0 + eps:
        return None
    t = np.dot(e2, qvec) * inv_det
    if t <= eps:
        return None
    return t


def project_point_to_icosahedron_surface(p, V, F):
    d = p / np.linalg.norm(p)
    origin = np.zeros(3)
    best_t, best_face = None, None
    for fi, (i0, i1, i2) in enumerate(F):
        v0, v1, v2 = V[i0], V[i1], V[i2]
        t = ray_intersect_triangle(origin, d, v0, v1, v2)
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t, best_face = t, fi
    if best_t is None:
        return None, None
    return best_t * d, best_face


# ============================================================
# 5) Natural face-edge splitting (bisection on face transitions)
# ============================================================
def interp_on_sphere(p0, p1, t, radius):
    p = (1 - t) * p0 + t * p1
    return radius * (p / np.linalg.norm(p))


def project_polyline_split_at_edges(boundary_xyz, V, F, radius, bisect_iters=25):
    pts = boundary_xyz.T
    if len(pts) < 2:
        return []
    segments = []
    q0, f0 = project_point_to_icosahedron_surface(pts[0], V, F)
    cur_face = f0
    cur_pts = [q0]

    for i in range(len(pts) - 1):
        pA, pB = pts[i], pts[i + 1]
        qA, fA = project_point_to_icosahedron_surface(pA, V, F)
        qB, fB = project_point_to_icosahedron_surface(pB, V, F)

        if cur_face is None:
            cur_face = fA
            cur_pts = [qA]

        if fA == fB:
            if fB != cur_face:
                if len(cur_pts) >= 2:
                    segments.append(np.array(cur_pts))
                cur_face = fB
                cur_pts = [qA]
            cur_pts.append(qB)
            continue

        lo, hi = 0.0, 1.0
        face_lo = fA
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            pM = interp_on_sphere(pA, pB, mid, radius)
            _, fM = project_point_to_icosahedron_surface(pM, V, F)
            if fM == face_lo:
                lo = mid
            else:
                hi = mid

        pL = interp_on_sphere(pA, pB, lo, radius)
        pH = interp_on_sphere(pA, pB, hi, radius)
        qL, _ = project_point_to_icosahedron_surface(pL, V, F)
        qH, _ = project_point_to_icosahedron_surface(pH, V, F)

        cur_pts.append(qL)
        if len(cur_pts) >= 2:
            segments.append(np.array(cur_pts))

        cur_face = fB
        cur_pts = [qH, qB]

    if len(cur_pts) >= 2:
        segments.append(np.array(cur_pts))

    return segments


def build_projected_wire_polylines_on_icosahedron(nside=8, boundary_step=2, face_split_iters=25):
    V, F = icosahedron_vertices_faces_unit()
    r_in = icosahedron_inradius_given_circumradius(1.0)
    boundaries = healpix_boundaries_xyz(nside=nside, step=boundary_step, radius=r_in)

    polylines = []
    for b in boundaries:
        segs = project_polyline_split_at_edges(b, V, F, radius=r_in, bisect_iters=face_split_iters)
        polylines.extend(segs)
    return polylines


# ============================================================
# 6) Camera model (azimuth/elevation/roll) + FOV (angle-based)
# ============================================================
def camera_basis(azimuth, elevation, roll, eps=1e-12):
    f = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ], dtype=float)
    f /= np.linalg.norm(f)

    up_world = np.array([0.0, 0.0, 1.0], dtype=float)
    r = np.cross(f, up_world)
    nr = np.linalg.norm(r)
    if nr < eps:
        up_world = np.array([0.0, 1.0, 0.0], dtype=float)
        r = np.cross(f, up_world)
        r /= np.linalg.norm(r)
    else:
        r /= nr

    u = np.cross(r, f)
    r2 = np.cos(roll) * r + np.sin(roll) * u
    u2 = -np.sin(roll) * r + np.cos(roll) * u
    return r2, u2, f


def camera_rotation_matrix(azimuth, elevation, roll):
    r, u, f = camera_basis(azimuth, elevation, roll)
    return np.vstack([r, u, f])  # rows


def world_to_camera(p, R):
    return R @ p


def camera_angles(p_cam):
    return np.arctan2(p_cam[0], p_cam[2]), np.arctan2(p_cam[1], p_cam[2])


def is_visible_cam(p_cam, fov_x, fov_y):
    if p_cam[2] <= 0:
        return False
    tx, ty = camera_angles(p_cam)
    return (abs(tx) <= 0.5 * fov_x) and (abs(ty) <= 0.5 * fov_y)


def cam_to_ndc(p_cam, fov_x, fov_y):
    tx, ty = camera_angles(p_cam)
    return np.array([tx / (0.5 * fov_x), ty / (0.5 * fov_y)], dtype=float)


# ============================================================
# 7) Clip polyline to FOV in 3D (segment-wise bisection)
# ============================================================
def clip_segment_to_fov_world(p0, p1, R, fov_x, fov_y, iters=24):
    c0 = world_to_camera(p0, R)
    c1 = world_to_camera(p1, R)
    v0 = is_visible_cam(c0, fov_x, fov_y)
    v1 = is_visible_cam(c1, fov_x, fov_y)

    if v0 and v1:
        return [(c0, c1)]
    if (not v0) and (not v1):
        return []

    lo, hi = (0.0, 1.0) if v0 else (1.0, 0.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pm = (1 - mid) * p0 + mid * p1
        cm_ = world_to_camera(pm, R)
        if is_visible_cam(cm_, fov_x, fov_y):
            lo = mid
        else:
            hi = mid

    pb = (1 - lo) * p0 + lo * p1
    cb = world_to_camera(pb, R)
    return [(c0, cb)] if v0 else [(cb, c1)]


def clip_polyline_to_fov_world(poly_world, R, fov_x, fov_y):
    if len(poly_world) < 2:
        return []
    out_polys = []
    cur = []

    for i in range(len(poly_world) - 1):
        p0, p1 = poly_world[i], poly_world[i + 1]
        segs = clip_segment_to_fov_world(p0, p1, R, fov_x, fov_y)
        if not segs:
            if len(cur) >= 2:
                out_polys.append(np.array(cur))
            cur = []
            continue
        cA, cB = segs[0]
        if not cur:
            cur = [cA, cB]
        else:
            if np.linalg.norm(cur[-1] - cA) > 1e-9:
                cur.append(cA)
            cur.append(cB)

    if len(cur) >= 2:
        out_polys.append(np.array(cur))
    return out_polys


# ============================================================
# 8) 2D clip in unit square [-1,1]^2 (Liang–Barsky for lines)
# ============================================================
def clip_line_to_unit_square(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 + 1, 1 - x0, y0 + 1, 1 - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t

    a = np.array([x0 + u1 * dx, y0 + u1 * dy])
    b = np.array([x0 + u2 * dx, y0 + u2 * dy])
    return a, b


def camera_polyline_to_ndc_segments(poly_cam, fov_x, fov_y):
    segs2d = []
    for i in range(len(poly_cam) - 1):
        a = cam_to_ndc(poly_cam[i], fov_x, fov_y)
        b = cam_to_ndc(poly_cam[i + 1], fov_x, fov_y)
        clipped = clip_line_to_unit_square(a, b)
        if clipped is not None:
            segs2d.append(clipped)
    return segs2d


# ============================================================
# 9) Face highlight: clip triangle in camera space by nonlinear constraints
# ============================================================
EPS_Z = 1e-9

def theta_x(p_cam): return np.arctan2(p_cam[0], p_cam[2])
def theta_y(p_cam): return np.arctan2(p_cam[1], p_cam[2])

def make_constraints(fov_x, fov_y):
    hx, hy = 0.5 * fov_x, 0.5 * fov_y
    return [
        (lambda p: p[2] > EPS_Z,      lambda p: p[2] - EPS_Z),
        (lambda p: theta_x(p) <= hx,  lambda p: hx - theta_x(p)),
        (lambda p: theta_x(p) >= -hx, lambda p: theta_x(p) + hx),
        (lambda p: theta_y(p) <= hy,  lambda p: hy - theta_y(p)),
        (lambda p: theta_y(p) >= -hy, lambda p: theta_y(p) + hy),
    ]

def intersect_bisect(p_in, p_out, signed_fn, iters=30):
    a, b = p_in.copy(), p_out.copy()
    fa, fb = signed_fn(a), signed_fn(b)
    if fa < 0 and fb >= 0:
        a, b = b, a
        fa, fb = fb, fa
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = signed_fn(m)
        if fm >= 0:
            a, fa = m, fm
        else:
            b, fb = m, fm
    return 0.5 * (a + b)

def clip_polygon_cam(poly_cam, inside_fn, signed_fn):
    if not poly_cam:
        return []
    out = []
    prev = poly_cam[-1]
    prev_in = inside_fn(prev)
    for cur in poly_cam:
        cur_in = inside_fn(cur)
        if cur_in:
            if not prev_in:
                out.append(intersect_bisect(cur, prev, signed_fn))
            out.append(cur)
        else:
            if prev_in:
                out.append(intersect_bisect(prev, cur, signed_fn))
        prev, prev_in = cur, cur_in
    return out

def clip_face_to_fov_polygon_cam(tri_cam, fov_x, fov_y):
    poly = list(tri_cam)
    for inside_fn, signed_fn in make_constraints(fov_x, fov_y):
        poly = clip_polygon_cam(poly, inside_fn, signed_fn)
        if len(poly) == 0:
            return []
    return poly

def clip_polygon_2d_unit_square(poly2):
    def clip_edge(poly, inside, intersect):
        if not poly:
            return []
        out = []
        prev = poly[-1]
        prev_in = inside(prev)
        for cur in poly:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            else:
                if prev_in:
                    out.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    def lerp(a, b, t): return a + t * (b - a)
    poly = list(poly2)

    poly = clip_edge(poly, lambda p: p[0] >= -1,
                     lambda a, b: lerp(a, b, (-1 - a[0]) / (b[0] - a[0] + 1e-15)))
    poly = clip_edge(poly, lambda p: p[0] <= 1,
                     lambda a, b: lerp(a, b, (1 - a[0]) / (b[0] - a[0] + 1e-15)))
    poly = clip_edge(poly, lambda p: p[1] >= -1,
                     lambda a, b: lerp(a, b, (-1 - a[1]) / (b[1] - a[1] + 1e-15)))
    poly = clip_edge(poly, lambda p: p[1] <= 1,
                     lambda a, b: lerp(a, b, (1 - a[1]) / (b[1] - a[1] + 1e-15)))
    return poly

def visible_face_polygons_ndc_with_id(V, F, R, fov_x, fov_y):
    out = []
    for fi, (i0, i1, i2) in enumerate(F):
        v0 = world_to_camera(V[i0], R)
        v1 = world_to_camera(V[i1], R)
        v2 = world_to_camera(V[i2], R)

        poly_cam = clip_face_to_fov_polygon_cam([v0, v1, v2], fov_x, fov_y)
        if not poly_cam:
            continue
        poly2 = [cam_to_ndc(p, fov_x, fov_y) for p in poly_cam]
        poly2 = clip_polygon_2d_unit_square(poly2)
        if len(poly2) >= 3:
            out.append((fi, np.array(poly2, dtype=float)))
    return out


# ============================================================
# 10) 3D helpers: visible faces + frustum rays
# ============================================================
def face_is_visible(fi, V, F, R, fov_x, fov_y):
    i0, i1, i2 = F[fi]
    tri_cam = [world_to_camera(V[i0], R), world_to_camera(V[i1], R), world_to_camera(V[i2], R)]
    poly_cam = clip_face_to_fov_polygon_cam(tri_cam, fov_x, fov_y)
    return len(poly_cam) >= 3

def frustum_corner_rays_world(R, fov_x, fov_y):
    hx, hy = 0.5 * fov_x, 0.5 * fov_y
    rays = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            tx = np.tan(sx * hx)
            ty = np.tan(sy * hy)
            d_cam = np.array([tx, ty, 1.0], dtype=float)
            d_cam /= np.linalg.norm(d_cam)
            d_world = (R.T @ d_cam)
            d_world /= np.linalg.norm(d_world)
            rays.append(d_world)
    return rays


# ============================================================
# 11) Ax-based plotting (NO new figures inside)
# ============================================================
def plot_visible_faces_3d_ax(ax, V, F, r_in,
                             azimuth, elevation, roll, fov_x, fov_y,
                             show_inner_sphere=True,
                             show_fov_rays=True,
                             ray_len=1.4,
                             show_face_ids=True):
    R = camera_rotation_matrix(azimuth, elevation, roll)
    ax.cla()
    ax.set_box_aspect((1, 1, 1))

    # inner sphere
    if show_inner_sphere:
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        X = r_in * np.outer(np.cos(u), np.sin(v))
        Y = r_in * np.outer(np.sin(u), np.sin(v))
        Z = r_in * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(X, Y, Z, alpha=0.07, linewidth=0, shade=False)

    visible_faces = []
    invisible_faces = []
    for fi in range(len(F)):
        i0, i1, i2 = F[fi]
        tri = [V[i0], V[i1], V[i2]]
        if face_is_visible(fi, V, F, R, fov_x, fov_y):
            visible_faces.append((fi, tri))
        else:
            invisible_faces.append((fi, tri))

    if invisible_faces:
        coll = Poly3DCollection([tri for _, tri in invisible_faces], alpha=0.10, linewidths=0.3)
        coll.set_facecolor((0.6, 0.6, 0.6, 1.0))
        coll.set_edgecolor((0, 0, 0, 0.25))
        ax.add_collection3d(coll)

    cmap_ = cm.get_cmap("tab20", len(F))
    if visible_faces:
        tris = [tri for _, tri in visible_faces]
        colors = [cmap_(fi) for fi, _ in visible_faces]
        coll2 = Poly3DCollection(tris, alpha=0.45, linewidths=0.8)
        coll2.set_facecolor(colors)
        coll2.set_edgecolor((0, 0, 0, 0.7))
        ax.add_collection3d(coll2)

    if show_face_ids:
        for fi, tri in visible_faces:
            c = (tri[0] + tri[1] + tri[2]) / 3.0
            ax.text(c[0], c[1], c[2], str(fi), fontsize=8)

    if show_fov_rays:
        rays = frustum_corner_rays_world(R, fov_x, fov_y)
        for d in rays:
            ax.plot([0, ray_len*d[0]], [0, ray_len*d[1]], [0, ray_len*d[2]],
                    linewidth=2.0, alpha=0.9, color="black")
        _, _, fwd = camera_basis(azimuth, elevation, roll)
        ax.plot([0, ray_len*fwd[0]], [0, ray_len*fwd[1]], [0, ray_len*fwd[2]],
                linewidth=3.0, alpha=0.9, color="black")

    lim = 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_title("3D: Visible faces + FOV rays")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")


def render_visible_wires_and_faces_2d_ax(ax, wire_polys_world,
                                         azimuth, elevation, roll,
                                         fov_x, fov_y,
                                         V_ico, F_ico,
                                         show_face_ids=True,
                                         match_fov_aspect=True,
                                         wire_linewidth=0.9):
    R = camera_rotation_matrix(azimuth, elevation, roll)
    ax.cla()

    yscale = float(fov_y / fov_x) if match_fov_aspect else 1.0
    def disp(p): return np.array([p[0], p[1]*yscale], dtype=float)

    face_polys = visible_face_polygons_ndc_with_id(V_ico, F_ico, R, fov_x, fov_y)
    cmap_ = cm.get_cmap("tab20", len(F_ico))
    for fi, poly in face_polys:
        poly_d = np.array([disp(p) for p in poly])
        color = cmap_(fi)
        ax.add_patch(MplPolygon(poly_d, closed=True, facecolor=color, alpha=0.22, edgecolor="none"))
        ax.plot(poly_d[:, 0], poly_d[:, 1], linewidth=0.7, alpha=0.65, color=color)
        if show_face_ids:
            ax.text(poly_d[:, 0].mean(), poly_d[:, 1].mean(),
                    str(fi), fontsize=8, ha="center", va="center")

    # frame
    ax.plot([-1, 1, 1, -1, -1], [-yscale, -yscale, yscale, yscale, -yscale],
            linewidth=1.5, color="black")

    # wires
    for poly_w in wire_polys_world:
        polys_cam = clip_polyline_to_fov_world(poly_w, R, fov_x, fov_y)
        for poly_cam in polys_cam:
            segs2d = camera_polyline_to_ndc_segments(poly_cam, fov_x, fov_y)
            for a, b in segs2d:
                a2, b2 = disp(a), disp(b)
                ax.plot([a2[0], b2[0]], [a2[1], b2[1]],
                        color="black", linewidth=wire_linewidth, alpha=0.9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-yscale*1.05, yscale*1.05)
    ax.set_title("2D: FOV projection (NDC)")
    ax.set_xlabel("n_x")
    ax.set_ylabel("n_y (scaled)")


# ============================================================
# 12) Interactive window with sliders (azimuth/elevation live)
# ============================================================
def interactive_view(nside=8, boundary_step=2):
    # Precompute geometry + wires once
    V_ico, F_ico = icosahedron_vertices_faces_unit()
    r_in = icosahedron_inradius_given_circumradius(1.0)
    wires_world = build_projected_wire_polylines_on_icosahedron(
        nside=nside, boundary_step=boundary_step, face_split_iters=25
    )

    # initial camera params (degrees for sliders)
    init_az_deg = 35.0
    init_el_deg = 15.0
    init_roll_deg = 0.0
    init_fovx_deg = 90.0
    init_fovy_deg = 60.0

    # Figure layout: top for plots, bottom for sliders
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[12, 2], hspace=0.25, wspace=0.15)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])

    # Slider axes
    ax_az   = fig.add_subplot(gs[1, 0])
    ax_el   = fig.add_subplot(gs[1, 1])

    # Make room inside each slider axes
    ax_az.set_position([0.08, 0.08, 0.40, 0.03])
    ax_el.set_position([0.55, 0.08, 0.40, 0.03])

    s_az = Slider(ax_az, "Azimuth (deg)", -180.0, 180.0, valinit=init_az_deg)
    s_el = Slider(ax_el, "Elevation (deg)", -89.0, 89.0, valinit=init_el_deg)

    # Optional extra sliders (roll/fov) - uncomment if you want:
    ax_roll = fig.add_axes([0.08, 0.03, 0.25, 0.03])
    ax_fovx = fig.add_axes([0.37, 0.03, 0.25, 0.03])
    ax_fovy = fig.add_axes([0.66, 0.03, 0.25, 0.03])

    s_roll = Slider(ax_roll, "Roll (deg)", -180.0, 180.0, valinit=init_roll_deg)
    s_fovx = Slider(ax_fovx, "FOVx (deg)", 10.0, 170.0, valinit=init_fovx_deg)
    s_fovy = Slider(ax_fovy, "FOVy (deg)", 10.0, 170.0, valinit=init_fovy_deg)

    def redraw(_=None):
        az = np.deg2rad(s_az.val)
        el = np.deg2rad(s_el.val)
        roll = np.deg2rad(s_roll.val)
        fov_x = np.deg2rad(s_fovx.val)
        fov_y = np.deg2rad(s_fovy.val)

        plot_visible_faces_3d_ax(
            ax3d, V_ico, F_ico, r_in,
            az, el, roll, fov_x, fov_y,
            show_inner_sphere=True,
            show_fov_rays=True,
            ray_len=1.4,
            show_face_ids=True
        )

        render_visible_wires_and_faces_2d_ax(
            ax2d, wires_world,
            az, el, roll, fov_x, fov_y,
            V_ico, F_ico,
            show_face_ids=True,
            match_fov_aspect=True,
            wire_linewidth=0.9
        )

        fig.canvas.draw_idle()

    # Hook callbacks
    s_az.on_changed(redraw)
    s_el.on_changed(redraw)
    s_roll.on_changed(redraw)
    s_fovx.on_changed(redraw)
    s_fovy.on_changed(redraw)

    # Initial draw
    redraw()

    plt.show()


# ============================================================
# 13) Main
# ============================================================
if __name__ == "__main__":
    interactive_view(nside=8, boundary_step=2)
