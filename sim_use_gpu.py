import math
import numpy as np
import glfw
import moderngl
from pyrr import Matrix44
from dataclasses import dataclass
import colorsys


# ========= 0. 색 팔레트 (정이십면체 20면용) =========

def make_palette(n=20):
    colors = []
    for i in range(n):
        h = i / n
        s = 0.65
        v = 0.9
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))
    return colors

PALETTE = make_palette(20)


# ========= 1. 지오데식 돔 생성 =========

def create_icosahedron():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
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
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                dij = dists[i, j]
                dik = dists[i, k]
                djk = dists[j, k]
                if (abs(dij - edge_len) < 1e-6 and
                    abs(dik - edge_len) < 1e-6 and
                    abs(djk - edge_len) < 1e-6):
                    faces.append((i, j, k))
    return verts, np.array(faces, dtype=int)


def subdivide_icosahedron(freq=5):
    ico_verts, ico_faces = create_icosahedron()
    small_triangles = []
    parent_face_idx = []

    for k, (i0, i1, i2) in enumerate(ico_faces):
        v0, v1, v2 = ico_verts[i0], ico_verts[i1], ico_verts[i2]
        for i in range(freq):
            for j in range(freq - i):
                a = i
                b = j
                c = freq - a - b

                # 위 삼각형
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

                # 아래 삼각형
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

    return ico_verts, ico_faces, np.array(small_triangles), np.array(parent_face_idx, dtype=int)


# ========= 2. 카메라/클리핑 =========

def camera_basis_from_angles(az_deg, el_deg, roll_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    roll = math.radians(roll_deg)

    fx = math.cos(el) * math.cos(az)
    fy = math.cos(el) * math.sin(az)
    fz = math.sin(el)
    fwd = np.array([fx, fy, fz], dtype=float)
    fwd /= np.linalg.norm(fwd)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)

    cr, sr = math.cos(roll), math.sin(roll)
    right_r = cr * right + sr * up
    up_r = -sr * right + cr * up

    return right_r, up_r, fwd


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
    poly = [np.array(p, dtype=float) for p in poly]
    if not poly:
        return []

    def inside_left(p): return p[0] >= x_min
    def intersect_left(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dx = x2 - x1
        if abs(dx) < 1e-9: return np.array([x_min, y1])
        t = (x_min - x1) / dx
        return np.array([x_min, y1 + t*(y2-y1)])
    poly = clip_polygon_against_halfspace(poly, inside_left, intersect_left)
    if not poly: return []

    def inside_right(p): return p[0] <= x_max
    def intersect_right(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dx = x2 - x1
        if abs(dx) < 1e-9: return np.array([x_max, y1])
        t = (x_max - x1) / dx
        return np.array([x_max, y1 + t*(y2-y1)])
    poly = clip_polygon_against_halfspace(poly, inside_right, intersect_right)
    if not poly: return []

    def inside_bottom(p): return p[1] >= y_min
    def intersect_bottom(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dy = y2 - y1
        if abs(dy) < 1e-9: return np.array([x1, y_min])
        t = (y_min - y1) / dy
        return np.array([x1 + t*(x2-x1), y_min])
    poly = clip_polygon_against_halfspace(poly, inside_bottom, intersect_bottom)
    if not poly: return []

    def inside_top(p): return p[1] <= y_max
    def intersect_top(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dy = y2 - y1
        if abs(dy) < 1e-9: return np.array([x1, y_max])
        t = (y_max - y1) / dy
        return np.array([x1 + t*(x2-x1), y_max])
    poly = clip_polygon_against_halfspace(poly, inside_top, intersect_top)
    return poly


def compute_visible_triangles_and_2d(tris_world, parent_idx,
                                     right, up, fwd,
                                     fov_x_deg, fov_y_deg):
    R_cam = np.stack([right, up, fwd], axis=0)
    cam = tris_world.reshape(-1, 3) @ R_cam.T
    cam = cam.reshape(tris_world.shape)

    M = tris_world.shape[0]
    x = cam[..., 0]
    y = cam[..., 1]
    z = cam[..., 2]

    eps = 1e-6
    in_front = z > eps

    fov_x = math.radians(fov_x_deg)
    fov_y = math.radians(fov_y_deg)

    ang_x = np.arctan2(x, z)
    ang_y = np.arctan2(y, z)

    in_fov_x = np.abs(ang_x) <= (fov_x / 2.0)
    in_fov_y = np.abs(ang_y) <= (fov_y / 2.0)

    tri_visible_mask = np.any(in_front & in_fov_x & in_fov_y, axis=1)

    nx_all = np.tan(ang_x) / math.tan(fov_x / 2.0)
    ny_all = np.tan(ang_y) / math.tan(fov_y / 2.0)

    polys_2d = []
    poly_parent = []
    for i in range(M):
        if not tri_visible_mask[i]:
            continue
        poly = np.stack([nx_all[i], ny_all[i]], axis=1)
        clipped = clip_polygon_to_rect(poly, -1.0, 1.0, -1.0, 1.0)
        if len(clipped) == 0:
            tri_visible_mask[i] = False
            continue
        polys_2d.append(np.array(clipped, dtype=np.float32))
        poly_parent.append(parent_idx[i])

    return tri_visible_mask, polys_2d, np.array(poly_parent, dtype=int)


# ========= 3. 2D 숫자 라벨 (7-seg) =========

SEGMENTS = {
    0: [0, 1, 2, 3, 4, 5],
    1: [1, 2],
    2: [0, 1, 6, 4, 3],
    3: [0, 1, 6, 2, 3],
    4: [5, 6, 1, 2],
    5: [0, 5, 6, 2, 3],
    6: [0, 5, 4, 3, 2, 6],
    7: [0, 1, 2],
    8: [0, 1, 2, 3, 4, 5, 6],
    9: [0, 1, 2, 3, 5, 6],
}

SEG_POINTS = [
    ((0.0, 1.0), (1.0, 1.0)),   # A
    ((1.0, 1.0), (1.0, 0.5)),   # B
    ((1.0, 0.5), (1.0, 0.0)),   # C
    ((0.0, 0.0), (1.0, 0.0)),   # D
    ((0.0, 0.0), (0.0, 0.5)),   # E
    ((0.0, 0.5), (0.0, 1.0)),   # F
    ((0.0, 0.5), (1.0, 0.5)),   # G
]

def build_digit_lines(digit, scale=0.06, offset=(0.0, 0.0)):
    seg_ids = SEGMENTS.get(digit, [])
    lines = []
    for sid in seg_ids:
        (x0, y0), (x1, y1) = SEG_POINTS[sid]
        x0 = (x0 - 0.5) * scale + offset[0]
        x1 = (x1 - 0.5) * scale + offset[0]
        y0 = (y0 - 0.5) * scale + offset[1]
        y1 = (y1 - 0.5) * scale + offset[1]
        lines.append(((x0, y0), (x1, y1)))
    return lines

def label_lines_for_id(label_id, center, scale=0.08):
    cx, cy = center
    tens = label_id // 10
    ones = label_id % 10

    lines = []
    if tens > 0:
        off = (-0.6 * scale, 0.0)
        for (p0, p1) in build_digit_lines(tens, scale=scale, offset=(cx + off[0], cy + off[1])):
            lines.append((p0, p1))
        off = (0.6 * scale, 0.0)
        for (p0, p1) in build_digit_lines(ones, scale=scale, offset=(cx + off[0], cy + off[1])):
            lines.append((p0, p1))
    else:
        for (p0, p1) in build_digit_lines(ones, scale=scale, offset=(cx, cy)):
            lines.append((p0, p1))
    return lines


# ========= 4. OpenGL 셰이더/윈도우 =========

VERT_SHADER_SRC = """
#version 330
in vec3 in_pos;
in vec4 in_color;
out vec4 v_color;
uniform mat4 mvp;
void main() {
    v_color = in_color;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAG_SHADER_SRC = """
#version 330
in vec4 v_color;
out vec4 f_color;
void main() {
    f_color = v_color;
}
"""


@dataclass
class CameraParams:
    az: float = 30.0
    el: float = 20.0
    roll: float = 0.0
    fov_y: float = 60.0
    fov_x: float = 80.0   # 루프에서 aspect로 업데이트


def create_window(width=1280, height=720, title="Geodesic Dome OpenGL"):
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")
    glfw.make_context_current(window)
    return window


# ========= 5. 메인 루프 =========

def run():
    freq = 20

    ico_verts, ico_faces, small_tris, parent_idx = subdivide_icosahedron(freq)

    tri_count = small_tris.shape[0]
    tri_vertices_world = small_tris.reshape(-1, 3).astype("f4")

    # 면별 기본 색 (3D용)
    base_colors = np.zeros((tri_count * 3, 4), dtype="f4")
    for i, p in enumerate(parent_idx):
        r, g, b = PALETTE[p % len(PALETTE)]
        base_colors[3*i : 3*i+3, :] = (r, g, b, 0.25)

    win_w, win_h = 1280, 720
    window = create_window(win_w, win_h)
    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=VERT_SHADER_SRC, fragment_shader=FRAG_SHADER_SRC)

    # 알파 블렌딩 (반투명 돔용)
    ctx.enable(moderngl.BLEND)
    # moderngl.BLEND_DEFAULT 가 없는 버전이 있어서 직접 지정
    ctx.blend_func = (
        moderngl.SRC_ALPHA,
        moderngl.ONE_MINUS_SRC_ALPHA,
    )


    # 전체 돔 VAO
    vbo_pos_all = ctx.buffer(tri_vertices_world.tobytes())
    vbo_col_all = ctx.buffer(base_colors.tobytes())
    vao_all = ctx.vertex_array(
        prog,
        [(vbo_pos_all, "3f", "in_pos"), (vbo_col_all, "4f", "in_color")]
    )

    # 3D FOV 하이라이트용
    highlight_vbo_pos = ctx.buffer(reserve=tri_vertices_world.nbytes)
    highlight_vbo_col = ctx.buffer(reserve=base_colors.nbytes)
    highlight_vao = ctx.vertex_array(
        prog,
        [(highlight_vbo_pos, "3f", "in_pos"), (highlight_vbo_col, "4f", "in_color")]
    )

    # 3D FOV 와이어프레임용
    visible_line_vbo_pos = ctx.buffer(reserve=tri_vertices_world.nbytes)
    visible_line_vbo_col = ctx.buffer(reserve=tri_vertices_world.nbytes)
    visible_line_vao = ctx.vertex_array(
        prog,
        [(visible_line_vbo_pos, "3f", "in_pos"), (visible_line_vbo_col, "4f", "in_color")]
    )

    cam = CameraParams()
    near, far = 0.1, 10.0
    ortho_2d = Matrix44.orthogonal_projection(-1, 1, -1, 1, -1, 1, dtype="f4")

    print("[키 조작]")
    print("←/→ : azimuth, ↑/↓ : elevation, Q/E : roll")
    print("Z/X : FOV_y, ESC : 종료")

    while not glfw.window_should_close(window):
        glfw.poll_events()

        step_ang = 1.0
        step_fov = 1.0

        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            cam.az -= step_ang
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            cam.az += step_ang
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            cam.el += step_ang
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            cam.el -= step_ang
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            cam.roll -= step_ang
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            cam.roll += step_ang
        if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS:
            cam.fov_y = max(10.0, cam.fov_y - step_fov)
        if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS:
            cam.fov_y = min(170.0, cam.fov_y + step_fov)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        aspect = (win_w / 2) / win_h
        cam.fov_x = math.degrees(
            2.0 * math.atan(math.tan(math.radians(cam.fov_y) / 2.0) * aspect)
        )

        # 2D용 중심 카메라
        right, up, fwd = camera_basis_from_angles(cam.az, cam.el, cam.roll)

        # FOV/2D 계산
        tri_mask, polys_2d, poly_parent = compute_visible_triangles_and_2d(
            small_tris, parent_idx, right, up, fwd, cam.fov_x, cam.fov_y
        )
        visible_tris_world = small_tris[tri_mask]
        visible_parents = parent_idx[tri_mask]

        # 3D 하이라이트 & 와이어프레임 버퍼 채우기
        if visible_tris_world.size > 0:
            # 채우기용 (살짝 확장)
            vis_scaled = visible_tris_world * 1.002
            vis_vertices = vis_scaled.reshape(-1, 3).astype("f4")
            vis_colors = np.zeros((vis_vertices.shape[0], 4), dtype="f4")
            for i, p in enumerate(visible_parents):
                r, g, b = PALETTE[p % len(PALETTE)]
                vis_colors[3*i:3*i+3, :] = (r, g, b, 1.0)
            highlight_vbo_pos.orphan(size=vis_vertices.nbytes)
            highlight_vbo_pos.write(vis_vertices.tobytes())
            highlight_vbo_col.orphan(size=vis_colors.nbytes)
            highlight_vbo_col.write(vis_colors.tobytes())

            # 와이어프레임용 (조금 더 바깥으로 확장해서 z-fighting 방지)
            offset = 1.003
            line_segs = []
            for tri in visible_tris_world:
                v0, v1, v2 = tri * offset
                line_segs.extend([v0, v1,  v1, v2,  v2, v0])
            line_segs = np.array(line_segs, dtype="f4")
            line_cols = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype="f4"),
                                (line_segs.shape[0], 1))
            visible_line_vbo_pos.orphan(size=line_segs.nbytes)
            visible_line_vbo_pos.write(line_segs.tobytes())
            visible_line_vbo_col.orphan(size=line_cols.nbytes)
            visible_line_vbo_col.write(line_cols.tobytes())
        else:
            visible_line_vbo_pos.orphan(size=0)
            visible_line_vbo_col.orphan(size=0)

        # ===== 3D (왼쪽) =====
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.clear(0.05, 0.05, 0.08, 1.0)

        cam_dist = 2.5
        eye = cam_dist * fwd   # 2D 카메라가 보는 방향 쪽 바깥
        target = np.array([0.0, 0.0, 0.0], dtype="f4")
        world_up = np.array([0.0, 0.0, 1.0], dtype="f4")
        if abs(np.dot(fwd, world_up)) > 0.95:
            world_up = np.array([0.0, 1.0, 0.0], dtype="f4")

        view_ext = Matrix44.look_at(eye, target, world_up, dtype="f4")
        proj = Matrix44.perspective_projection(cam.fov_y, aspect, near, far)
        mvp3d = proj * view_ext
        prog["mvp"].write(mvp3d.astype("f4").tobytes())

        ctx.viewport = (0, 0, win_w // 2, win_h)
        vao_all.render(mode=moderngl.TRIANGLES)
        if visible_tris_world.size > 0:
            highlight_vao.render(mode=moderngl.TRIANGLES)
            visible_line_vao.render(mode=moderngl.LINES)

        # ===== 2D (오른쪽) =====
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.viewport = (win_w // 2, 0, win_w // 2, win_h)
        prog["mvp"].write(ortho_2d.astype("f4").tobytes())

        # (1) 패치 채우기
        tri2_pos = []
        tri2_col = []
        for poly, parent in zip(polys_2d, poly_parent):
            if len(poly) < 3:
                continue
            v0 = poly[0]
            r, g, b = PALETTE[parent % len(PALETTE)]
            col = (r, g, b, 1.0)
            for i in range(1, len(poly) - 1):
                tri = [v0, poly[i], poly[i+1]]
                tri2_pos.extend(tri)
                tri2_col.extend([col, col, col])

        if tri2_pos:
            tri2_pos = np.array(tri2_pos, dtype="f4")
            pos2 = np.zeros((tri2_pos.shape[0], 3), dtype="f4")
            pos2[:, 0:2] = tri2_pos
            col2 = np.array(tri2_col, dtype="f4")

            vbo2_pos = ctx.buffer(pos2.tobytes())
            vbo2_col = ctx.buffer(col2.tobytes())
            vao2 = ctx.vertex_array(
                prog, [(vbo2_pos, "3f", "in_pos"), (vbo2_col, "4f", "in_color")]
            )
            vao2.render(mode=moderngl.TRIANGLES)

        # (2) 2D 와이어프레임
        edge_pos = []
        edge_col = []
        for poly in polys_2d:
            if len(poly) < 2:
                continue
            for i in range(len(poly)):
                p0 = poly[i]
                p1 = poly[(i+1) % len(poly)]
                edge_pos.extend([(p0[0], p0[1], 0.0), (p1[0], p1[1], 0.0)])
                edge_col.extend([(0, 0, 0, 1), (0, 0, 0, 1)])

        if edge_pos:
            edge_pos = np.array(edge_pos, dtype="f4")
            edge_col = np.array(edge_col, dtype="f4")
            vbo_edge_pos = ctx.buffer(edge_pos.tobytes())
            vbo_edge_col = ctx.buffer(edge_col.tobytes())
            vao_edge = ctx.vertex_array(
                prog, [(vbo_edge_pos, "3f", "in_pos"), (vbo_edge_col, "4f", "in_color")]
            )
            vao_edge.render(mode=moderngl.LINES)

        # (3) 면 ID 라벨
        label_pos = []
        label_col = []
        if polys_2d:
            centroids = {}
            counts = {}
            for poly, parent in zip(polys_2d, poly_parent):
                c = poly.mean(axis=0)
                if parent not in centroids:
                    centroids[parent] = c.copy()
                    counts[parent] = 1
                else:
                    centroids[parent] += c
                    counts[parent] += 1
            for parent, c_sum in centroids.items():
                c = c_sum / counts[parent]
                lines = label_lines_for_id(parent, c, scale=0.12)
                for (p0, p1) in lines:
                    label_pos.extend([(p0[0], p0[1], 0.0), (p1[0], p1[1], 0.0)])
                    label_col.extend([(0, 0, 0, 1), (0, 0, 0, 1)])

        if label_pos:
            label_pos = np.array(label_pos, dtype="f4")
            label_col = np.array(label_col, dtype="f4")
            vbo_lab_pos = ctx.buffer(label_pos.tobytes())
            vbo_lab_col = ctx.buffer(label_col.tobytes())
            vao_lab = ctx.vertex_array(
                prog, [(vbo_lab_pos, "3f", "in_pos"), (vbo_lab_col, "4f", "in_color")]
            )
            vao_lab.render(mode=moderngl.LINES)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    run()
