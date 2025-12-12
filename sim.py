import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider

# ---------- 1. Icosahedron & geodesic subdivision ----------

def create_icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        (0, -1, -phi),
        (0, -1,  phi),
        (0,  1, -phi),
        (0,  1,  phi),
        (-1, -phi, 0),
        (-1,  phi, 0),
        (1, -phi, 0),
        (1,  phi, 0),
        (-phi, 0, -1),
        ( phi, 0, -1),
        (-phi, 0, 1),
        ( phi, 0, 1),
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    n = len(verts)
    dists = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    edge_len = np.min(dists[np.nonzero(dists)])
    faces = []
    for i, j, k in itertools.combinations(range(n), 3):
        dij = dists[i, j]; dik = dists[i, k]; djk = dists[j, k]
        if (abs(dij - edge_len) < 1e-6 and
            abs(dik - edge_len) < 1e-6 and
            abs(djk - edge_len) < 1e-6):
            faces.append((i, j, k))
    return verts, np.array(faces, dtype=int)

def subdivide_icosahedron(freq=3):
    ico_verts, ico_faces = create_icosahedron()
    small_triangles = []
    parent_face_idx = []
    for k, (i0, i1, i2) in enumerate(ico_faces):
        v0, v1, v2 = ico_verts[i0], ico_verts[i1], ico_verts[i2]
        for i in range(freq):
            for j in range(freq - i):
                a = i; b = j; c = freq - a - b
                bary_up = np.array([
                    (a,   b,   c),
                    (a+1, b,   c-1),
                    (a,   b+1, c-1),
                ], dtype=float) / freq
                verts_up = np.array([
                    bary_up[0, 0]*v0 + bary_up[0, 1]*v1 + bary_up[0, 2]*v2,
                    bary_up[1, 0]*v0 + bary_up[1, 1]*v1 + bary_up[1, 2]*v2,
                    bary_up[2, 0]*v0 + bary_up[2, 1]*v1 + bary_up[2, 2]*v2,
                ])
                verts_up /= np.linalg.norm(verts_up, axis=1, keepdims=True)
                small_triangles.append(verts_up)
                parent_face_idx.append(k)
                if i + j < freq - 1:
                    bary_dn = np.array([
                        (a+1, b,   c-1),
                        (a+1, b+1, c-2),
                        (a,   b+1, c-1),
                    ], dtype=float) / freq
                    verts_dn = np.array([
                        bary_dn[0, 0]*v0 + bary_dn[0, 1]*v1 + bary_dn[0, 2]*v2,
                        bary_dn[1, 0]*v0 + bary_dn[1, 1]*v1 + bary_dn[1, 2]*v2,
                        bary_dn[2, 0]*v0 + bary_dn[2, 1]*v1 + bary_dn[2, 2]*v2,
                    ])
                    verts_dn /= np.linalg.norm(verts_dn, axis=1, keepdims=True)
                    small_triangles.append(verts_dn)
                    parent_face_idx.append(k)
    return ico_verts, ico_faces, np.array(small_triangles), np.array(parent_face_idx, int)

# ---------- 2. Camera basis ----------

def camera_basis_from_angles(az_deg, el_deg, roll_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    roll = np.deg2rad(roll_deg)
    fx = np.cos(el) * np.cos(az)
    fy = np.cos(el) * np.sin(az)
    fz = np.sin(el)
    forward = np.array([fx, fy, fz], dtype=float)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up); right /= np.linalg.norm(right)
    up = np.cross(right, forward); up /= np.linalg.norm(up)
    cr = np.cos(roll); sr = np.sin(roll)
    right_r =  cr * right + sr * up
    up_r    = -sr * right + cr * up
    return right_r, up_r, forward

# ---------- 3. 2D polygon clipping (Sutherland–Hodgman) ----------

def clip_polygon_against_halfspace(poly, inside_fn, intersect_fn):
    if len(poly) == 0:
        return []
    output = []
    prev = poly[-1]
    prev_inside = inside_fn(prev)
    for curr in poly:
        curr_inside = inside_fn(curr)
        if curr_inside:
            if prev_inside:
                output.append(curr)
            else:
                inter = intersect_fn(prev, curr)
                output.append(inter)
                output.append(curr)
        else:
            if prev_inside:
                inter = intersect_fn(prev, curr)
                output.append(inter)
        prev, prev_inside = curr, curr_inside
    return output

def clip_polygon_to_rect(poly, x_min, x_max, y_min, y_max):
    poly = [np.array(p, float) for p in poly]
    if len(poly) == 0:
        return []
    # left: x >= x_min
    def inside_left(p): return p[0] >= x_min
    def intersect_left(p1, p2):
        x1,y1 = p1; x2,y2 = p2
        dx = x2-x1
        if abs(dx) < 1e-9: return np.array([x_min, y1])
        t = (x_min - x1)/dx
        return np.array([x_min, y1 + t*(y2-y1)])
    poly = clip_polygon_against_halfspace(poly, inside_left, intersect_left)
    if len(poly) == 0: return []
    # right: x <= x_max
    def inside_right(p): return p[0] <= x_max
    def intersect_right(p1, p2):
        x1,y1 = p1; x2,y2 = p2
        dx = x2-x1
        if abs(dx) < 1e-9: return np.array([x_max, y1])
        t = (x_max - x1)/dx
        return np.array([x_max, y1 + t*(y2-y1)])
    poly = clip_polygon_against_halfspace(poly, inside_right, intersect_right)
    if len(poly) == 0: return []
    # bottom: y >= y_min
    def inside_bottom(p): return p[1] >= y_min
    def intersect_bottom(p1, p2):
        x1,y1 = p1; x2,y2 = p2
        dy = y2-y1
        if abs(dy) < 1e-9: return np.array([x1, y_min])
        t = (y_min - y1)/dy
        return np.array([x1 + t*(x2-x1), y_min])
    poly = clip_polygon_against_halfspace(poly, inside_bottom, intersect_bottom)
    if len(poly) == 0: return []
    # top: y <= y_max
    def inside_top(p): return p[1] <= y_max
    def intersect_top(p1, p2):
        x1,y1 = p1; x2,y2 = p2
        dy = y2-y1
        if abs(dy) < 1e-9: return np.array([x1, y_max])
        t = (y_max - y1)/dy
        return np.array([x1 + t*(x2-x1), y_max])
    poly = clip_polygon_against_halfspace(poly, inside_top, intersect_top)
    return poly

# ---------- 4. Projection with clipping ----------

def project_with_clipping(small_tris, parent_idx,
                          az_deg, el_deg, roll_deg,
                          fov_x_deg, fov_y_deg):
    right, up, forward = camera_basis_from_angles(az_deg, el_deg, roll_deg)
    R_cam = np.stack([right, up, forward], axis=0)

    M = small_tris.shape[0]
    verts_flat = small_tris.reshape(-1, 3)
    verts_cam = verts_flat @ R_cam.T
    verts_cam = verts_cam.reshape(M, 3, 3)

    x = verts_cam[..., 0]
    y = verts_cam[..., 1]
    z = verts_cam[..., 2]

    eps = 1e-6
    in_front = z > eps  # 카메라 앞쪽?

    fov_x = np.deg2rad(fov_x_deg)
    fov_y = np.deg2rad(fov_y_deg)

    ang_x = np.arctan2(x, z)
    ang_y = np.arctan2(y, z)

    in_fov_x = np.abs(ang_x) <= (fov_x / 2.0)
    in_fov_y = np.abs(ang_y) <= (fov_y / 2.0)

    # ★ 왼쪽(3D)에서 쓰던 기준:
    #   "정점 중 하나라도 카메라 앞 + FOV 안" → 보여줄 삼각형
    tri_visible_mask = np.any(in_front & in_fov_x & in_fov_y, axis=1)

    # 정규화 평면 좌표 (FOV 기준)
    nx_all = np.tan(ang_x) / np.tan(fov_x / 2.0)
    ny_all = np.tan(ang_y) / np.tan(fov_y / 2.0)

    poly_list = []
    parent_list = []

    for i in range(M):
        # ★ 왼쪽에서 "보이는" 삼각형만 2D로도 그린다
        if not tri_visible_mask[i]:
            continue

        # 삼각형 i를 정규화 평면으로 투영
        poly = np.stack([nx_all[i], ny_all[i]], axis=1)  # (3,2)

        # FOV 사각형 [-1,1]×[-1,1]으로 클리핑
        poly_clipped = clip_polygon_to_rect(poly, -1.0, 1.0, -1.0, 1.0)

        # 수치 오차로 비어버리면 그냥 스킵
        if len(poly_clipped) == 0:
            continue

        poly_list.append(np.array(poly_clipped))
        parent_list.append(parent_idx[i])

    return poly_list, np.array(parent_list, int), tri_visible_mask


# ---------- 5. Static plot + pixel mapping & saving ----------

def render_and_save(ico_verts, ico_faces, small_tris, parent_idx,
                    az_deg=0, el_deg=0, roll_deg=0,
                    fov_x_deg=80, fov_y_deg=60,
                    img_w=1920, img_h=1080,
                    filename="geodesic_view.png"):
    polys_2d, parents_vis, tri_mask = project_with_clipping(
        small_tris, parent_idx, az_deg, el_deg, roll_deg, fov_x_deg, fov_y_deg
    )
    cmap = plt.cm.get_cmap("tab20")

    fig = plt.figure(figsize=(12, 6))

    # --- 왼쪽: 3D 구 + FOV 영역 ---
    ax3d = fig.add_subplot(1,2,1, projection="3d")
    # 전체 지오데식 돔 (연하게)
    for tri in small_tris:
        poly = Poly3DCollection([tri])
        poly.set_edgecolor("lightgray"); poly.set_linewidth(0.2)
        poly.set_facecolor("lightgray"); poly.set_alpha(0.2)
        ax3d.add_collection3d(poly)
    # 하이라이트 (FOV와 교차하는 삼각형)
    for tri, pidx, vis in zip(small_tris, parent_idx, tri_mask):
        if not vis: continue
        poly = Poly3DCollection([tri])
        poly.set_edgecolor("k"); poly.set_linewidth(0.6)
        poly.set_facecolor(cmap(pidx % 20)); poly.set_alpha(0.9)
        ax3d.add_collection3d(poly)
    # 정이십면체 면 경계
    for (i0, i1, i2) in ico_faces:
        for a,b in [(i0,i1),(i1,i2),(i2,i0)]:
            seg = np.stack([ico_verts[a], ico_verts[b]], axis=0)
            ax3d.plot(seg[:,0], seg[:,1], seg[:,2], "k", linewidth=0.6)
    # 카메라 방향 화살표
    right, up, forward = camera_basis_from_angles(az_deg, el_deg, roll_deg)
    origin = np.zeros(3)
    ax3d.quiver(*origin, *forward, length=1.2, arrow_length_ratio=0.1, color="red")
    lim = 1.3
    ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(-lim, lim)
    ax3d.set_box_aspect([1,1,1])
    ax3d.set_title("3D geodesic & camera")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

    # --- 오른쪽: 이미지 평면 (픽셀 좌표) ---
    ax_img = fig.add_subplot(1,2,2)
    for poly2d, pidx in zip(polys_2d, parents_vis):
        xs_norm = poly2d[:,0]; ys_norm = poly2d[:,1]
        # [-1,1] -> [0, img_w-1], [0, img_h-1] (y는 위->아래)
        xs_pix = (xs_norm + 1) * 0.5 * (img_w - 1)
        ys_pix = (1 - ys_norm) * 0.5 * (img_h - 1)
        pts_pix = np.stack([xs_pix, ys_pix], axis=1)
        patch = plt.Polygon(pts_pix, closed=True,
                            edgecolor="k", linewidth=0.3,
                            facecolor=cmap(pidx % 20), alpha=0.9)
        ax_img.add_patch(patch)
    ax_img.set_xlim(0, img_w); ax_img.set_ylim(img_h, 0)  # y축 뒤집기
    ax_img.set_aspect("equal", adjustable="box")
    ax_img.set_title(f"Image plane ({img_w}x{img_h})")
    ax_img.set_xlabel("x (px)"); ax_img.set_ylabel("y (px)")

    plt.tight_layout()
    fig.savefig(filename, dpi=100, bbox_inches="tight")
    plt.show()

# ---------- 6. Interactive GUI with sliders ----------

def interactive_view(freq=3):
    ico_verts, ico_faces, small_tris, parent_idx = subdivide_icosahedron(freq)

    az0, el0, roll0 = 30, 20, 0
    fovx0, fovy0 = 80, 60
    cmap = plt.cm.get_cmap("tab20")

    fig = plt.figure(figsize=(12,6))
    ax3d = fig.add_subplot(1,2,1, projection="3d")
    ax_img = fig.add_subplot(1,2,2)
    tris_all = small_tris  # alias

    def redraw(az, el, roll, fovx, fovy):
        ax3d.cla(); ax_img.cla()

        polys_2d, parents_vis, tri_mask = project_with_clipping(
            tris_all, parent_idx, az, el, roll, fovx, fovy
        )

        # 3D
        for tri in tris_all:
            poly = Poly3DCollection([tri])
            poly.set_edgecolor("lightgray"); poly.set_linewidth(0.2)
            poly.set_facecolor("lightgray"); poly.set_alpha(0.2)
            ax3d.add_collection3d(poly)
        for tri, pidx, vis in zip(tris_all, parent_idx, tri_mask):
            if not vis: continue
            poly = Poly3DCollection([tri])
            poly.set_edgecolor("k"); poly.set_linewidth(0.6)
            poly.set_facecolor(cmap(pidx % 20)); poly.set_alpha(0.9)
            ax3d.add_collection3d(poly)
        for (i0, i1, i2) in ico_faces:
            for a,b in [(i0,i1),(i1,i2),(i2,i0)]:
                seg = np.stack([ico_verts[a], ico_verts[b]], axis=0)
                ax3d.plot(seg[:,0], seg[:,1], seg[:,2], "k", linewidth=0.6)
        right, up, forward = camera_basis_from_angles(az, el, roll)
        origin = np.zeros(3)
        ax3d.quiver(*origin, *forward, length=1.2,
                    arrow_length_ratio=0.1, color="red")
        lim = 1.3
        ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(-lim, lim)
        ax3d.set_box_aspect([1,1,1])
        ax3d.set_title("3D geodesic & camera")
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

        # 2D 이미지 평면
        img_w, img_h = 1920, 1080
        for poly2d, pidx in zip(polys_2d, parents_vis):
            xs_norm = poly2d[:,0]; ys_norm = poly2d[:,1]
            xs_pix = (xs_norm + 1) * 0.5 * (img_w - 1)
            ys_pix = (1 - ys_norm) * 0.5 * (img_h - 1)
            pts_pix = np.stack([xs_pix, ys_pix], axis=1)
            patch = plt.Polygon(pts_pix, closed=True,
                                edgecolor="k", linewidth=0.3,
                                facecolor=cmap(pidx % 20), alpha=0.9)
            ax_img.add_patch(patch)
        ax_img.set_xlim(0, img_w); ax_img.set_ylim(img_h, 0)
        ax_img.set_aspect("equal", adjustable="box")
        ax_img.set_title("Image plane (1920x1080)")
        ax_img.set_xlabel("x (px)"); ax_img.set_ylabel("y (px)")

        fig.canvas.draw_idle()

    # 초기 그림
    interactive_freq = freq
    redraw(az0, el0, roll0, fovx0, fovy0)

    # 슬라이더 영역
    axcolor = "lightgoldenrodyellow"
    ax_az   = plt.axes([0.15, 0.02, 0.65, 0.02], facecolor=axcolor)
    ax_el   = plt.axes([0.15, 0.05, 0.65, 0.02], facecolor=axcolor)
    ax_roll = plt.axes([0.15, 0.08, 0.65, 0.02], facecolor=axcolor)
    ax_fovx = plt.axes([0.15, 0.11, 0.65, 0.02], facecolor=axcolor)
    ax_fovy = plt.axes([0.15, 0.14, 0.65, 0.02], facecolor=axcolor)

    s_az   = Slider(ax_az,   "azimuth", -180, 180, valinit=az0)
    s_el   = Slider(ax_el,   "elev",    -89,  89,  valinit=el0)
    s_roll = Slider(ax_roll, "roll",   -180, 180,  valinit=roll0)
    s_fovx = Slider(ax_fovx, "FOV x",   10, 170,  valinit=fovx0)
    s_fovy = Slider(ax_fovy, "FOV y",   10, 170,  valinit=fovy0)

    def update(_):
        redraw(s_az.val, s_el.val, s_roll.val, s_fovx.val, s_fovy.val)

    s_az.on_changed(update); s_el.on_changed(update)
    s_roll.on_changed(update); s_fovx.on_changed(update); s_fovy.on_changed(update)

    plt.tight_layout(rect=[0,0.16,1,1])
    plt.show()

# ---------- 실행 예시 ----------

if __name__ == "__main__":
    freq = 3  # 너무 크게 하면 GUI가 좀 느려질 수 있음

    ico_verts, ico_faces, small_tris, parent_idx = subdivide_icosahedron(freq)

    # 슬라이더로 실시간 카메라 조절
    interactive_view(freq=freq)
