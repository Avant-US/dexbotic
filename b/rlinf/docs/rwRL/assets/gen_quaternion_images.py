"""Generate quaternion explanation images with CJK font support."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

# CJK font setup
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_axis(ax, origin, length=1.0, labels=True):
    """Draw 3D coordinate axes."""
    colors = ['r', 'g', 'b']
    dirs = np.eye(3) * length
    axis_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax.quiver(*origin, *dirs[i], color=colors[i], arrow_length_ratio=0.08, linewidth=2)
        if labels:
            ax.text(origin[0]+dirs[i][0]*1.15, origin[1]+dirs[i][1]*1.15, origin[2]+dirs[i][2]*1.15,
                    axis_labels[i], color=colors[i], fontsize=12, fontweight='bold')


def draw_rotation_arrow(ax, axis, angle_deg, radius=0.6, color='purple'):
    """Draw a curved arrow indicating rotation around an axis."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    if abs(axis[2]) < 0.9:
        perp = np.cross(axis, [0, 0, 1])
    else:
        perp = np.cross(axis, [1, 0, 0])
    perp = perp / np.linalg.norm(perp)
    perp2 = np.cross(axis, perp)

    angles = np.linspace(0, np.radians(min(angle_deg, 320)), 40)
    pts = np.array([radius * (np.cos(a) * perp + np.sin(a) * perp2) for a in angles])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=2.5, alpha=0.8)
    if len(pts) > 2:
        direction = pts[-1] - pts[-2]
        ax.quiver(*pts[-1], *direction*3, color=color, arrow_length_ratio=0.4, linewidth=2)


# ============================================================
# IMAGE 1: Quaternion Concept (Axis-Angle Intuition + Formula)
# ============================================================
fig = plt.figure(figsize=(16, 10))
fig.suptitle('四元数 (Quaternion) 核心概念', fontsize=20, fontweight='bold', y=0.98)

# Left panel: 3D axis-angle visualization
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('轴-角直觉：绕轴旋转', fontsize=14, fontweight='bold', pad=15)
draw_axis(ax1, [0, 0, 0], length=1.0)

# Draw rotation axis (tilted)
rot_axis = np.array([0.3, 0.3, 0.9])
rot_axis = rot_axis / np.linalg.norm(rot_axis)
ax1.quiver(0, 0, 0, *rot_axis*1.3, color='purple', arrow_length_ratio=0.06, linewidth=3)
ax1.text(rot_axis[0]*1.4, rot_axis[1]*1.4, rot_axis[2]*1.4,
         'n (旋转轴)', color='purple', fontsize=11, fontweight='bold')

# Draw rotation arrow
draw_rotation_arrow(ax1, rot_axis, 120, radius=0.5, color='orange')
ax1.text(0.5, 0.5, 0.15, r'$\theta$', color='orange', fontsize=16, fontweight='bold')

# Draw a point being rotated
p_start = np.array([0.8, 0.0, 0.0])
ax1.scatter(*p_start, color='blue', s=80, zorder=5)
ax1.text(p_start[0]+0.05, p_start[1]+0.05, p_start[2]-0.1, 'P', color='blue', fontsize=12)

ax1.set_xlim([-0.5, 1.5])
ax1.set_ylim([-0.5, 1.5])
ax1.set_zlim([-0.3, 1.5])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.view_init(elev=25, azim=-60)

# Right panel: Formula breakdown
ax2 = fig.add_subplot(122)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

y = 9.5
ax2.text(5, y, '四元数公式分解', fontsize=16, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#1976D2'))

y = 8.3
ax2.text(0.5, y, '给定旋转轴 n = (nx, ny, nz) 和旋转角 θ :', fontsize=12)

y = 7.4
ax2.text(1.0, y, 'qw = cos(θ/2)', fontsize=14, fontweight='bold', color='#D32F2F',
         fontfamily='monospace')
y = 6.7
ax2.text(1.0, y, 'qx = nx · sin(θ/2)', fontsize=14, fontweight='bold', color='#1976D2',
         fontfamily='monospace')
y = 6.0
ax2.text(1.0, y, 'qy = ny · sin(θ/2)', fontsize=14, fontweight='bold', color='#388E3C',
         fontfamily='monospace')
y = 5.3
ax2.text(1.0, y, 'qz = nz · sin(θ/2)', fontsize=14, fontweight='bold', color='#7B1FA2',
         fontfamily='monospace')

y = 4.2
ax2.text(0.5, y, '约束: qx² + qy² + qz² + qw² = 1', fontsize=13,
         fontweight='bold', color='#E65100',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#E65100'))

y = 3.2
ax2.text(0.5, y, '直觉理解:', fontsize=13, fontweight='bold')
items = [
    ('qw (标量部分)', '编码"转了多少"  —  cos(θ/2)'),
    ('qx, qy, qz (向量部分)', '编码"绕哪个轴转"  —  轴方向 × sin(θ/2)'),
    ('qw = 1, 其余 = 0', '没有旋转 (单位四元数 / Identity)'),
    ('qw = 0', '旋转了 180° (θ/2 = 90°, cos=0)'),
]
for i, (title, desc) in enumerate(items):
    yi = 2.5 - i * 0.65
    ax2.text(1.0, yi, f'• {title}:', fontsize=11, fontweight='bold')
    ax2.text(1.2, yi - 0.3, desc, fontsize=10, color='#555')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, 'quaternion_concept.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[1/3] quaternion_concept.png saved')


# ============================================================
# IMAGE 2: Common Quaternion Examples
# ============================================================
fig = plt.figure(figsize=(16, 12))
fig.suptitle('常见四元数示例', fontsize=20, fontweight='bold', y=0.98)

examples = [
    {
        'title': 'Identity (0, 0, 0, 1)\nθ=0°  —  不旋转',
        'axis': [0, 0, 1], 'angle': 0, 'quat': '(0, 0, 0, 1)',
    },
    {
        'title': '绕Z转90° (0, 0, 0.707, 0.707)\nθ=90°, n=(0,0,1)',
        'axis': [0, 0, 1], 'angle': 90, 'quat': '(0, 0, 0.707, 0.707)',
    },
    {
        'title': '绕X转180° (1, 0, 0, 0)\nθ=180°, n=(1,0,0)',
        'axis': [1, 0, 0], 'angle': 180, 'quat': '(1, 0, 0, 0)',
    },
    {
        'title': '绕Y转90° (0, 0.707, 0, 0.707)\nθ=90°, n=(0,1,0)',
        'axis': [0, 1, 0], 'angle': 90, 'quat': '(0, 0.707, 0, 0.707)',
    },
]

for idx, ex in enumerate(examples):
    ax = fig.add_subplot(2, 2, idx+1, projection='3d')
    ax.set_title(ex['title'], fontsize=11, fontweight='bold', pad=10)
    draw_axis(ax, [0, 0, 0], length=0.8)

    if ex['angle'] > 0:
        # Draw rotation axis
        axis = np.array(ex['axis'], dtype=float)
        ax.quiver(0, 0, 0, *axis*1.0, color='purple', arrow_length_ratio=0.08, linewidth=2.5)
        draw_rotation_arrow(ax, axis, ex['angle'], radius=0.45, color='orange')

        # Show a "before" and "after" arrow
        # Rotate the X-axis unit vector
        theta = np.radians(ex['angle'])
        c, s = np.cos(theta), np.sin(theta)
        nx, ny, nz = axis
        R = np.array([
            [c + nx*nx*(1-c), nx*ny*(1-c)-nz*s, nx*nz*(1-c)+ny*s],
            [ny*nx*(1-c)+nz*s, c + ny*ny*(1-c), ny*nz*(1-c)-nx*s],
            [nz*nx*(1-c)-ny*s, nz*ny*(1-c)+nx*s, c + nz*nz*(1-c)]
        ])
        p0 = np.array([0.7, 0.0, 0.0])
        p1 = R @ p0
        ax.scatter(*p0, color='blue', s=60, zorder=5)
        ax.scatter(*p1, color='red', s=60, zorder=5)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k--', alpha=0.3)
        ax.text(p0[0], p0[1], p0[2]-0.12, 'P', color='blue', fontsize=10)
        ax.text(p1[0]+0.05, p1[1]+0.05, p1[2]+0.05, "P'", color='red', fontsize=10)
    else:
        ax.text(0.3, 0.3, 0.8, '✔ 无旋转', fontsize=14, color='green', fontweight='bold')

    ax.set_xlim([-0.8, 1.0])
    ax.set_ylim([-0.8, 1.0])
    ax.set_zlim([-0.8, 1.0])
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('Z', fontsize=9)
    ax.view_init(elev=25, azim=-55)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 'quaternion_examples.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[2/3] quaternion_examples.png saved')


# ============================================================
# IMAGE 3: Gimbal Lock + R1 Pro Context
# ============================================================
fig = plt.figure(figsize=(16, 10))
fig.suptitle('万向节锁 (Gimbal Lock) 与 R1 Pro 安全系统的四元数应用',
             fontsize=18, fontweight='bold', y=0.98)

# Left: Gimbal Lock explanation
ax1 = fig.add_subplot(121)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('万向节锁问题', fontsize=14, fontweight='bold')

y = 9.2
ax1.text(0.5, y, '欧拉角 (Roll, Pitch, Yaw) 的致命缺陷:', fontsize=13, fontweight='bold')

y = 8.2
ax1.add_patch(mpatches.FancyBboxPatch((0.3, 7.5), 9.2, 1.2,
              boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor='#C62828'))
ax1.text(0.6, 8.3, '当 Pitch = ±90° 时:', fontsize=12, fontweight='bold', color='#C62828')
ax1.text(0.6, 7.8, 'Roll 和 Yaw 绕同一轴旋转 → 丢失一个自由度!', fontsize=11, color='#C62828')

y = 7.0
ax1.text(0.5, y, '对机器人的影响:', fontsize=13, fontweight='bold')

problems = [
    ('• 奇异点附近控制不稳定', '关节速度发散, 抧动剧烈'),
    ('• 路径规划失败', '插值路径可能突然跳变'),
    ('• 安全检查误判', '姿态解算不唯一, 安全边界可能被突破'),
]
for i, (title, desc) in enumerate(problems):
    yi = 6.3 - i * 0.8
    ax1.text(1.0, yi, title, fontsize=11, fontweight='bold', color='#1565C0')
    ax1.text(1.5, yi - 0.35, desc, fontsize=10, color='#555')

y = 3.5
ax1.add_patch(mpatches.FancyBboxPatch((0.3, 2.8), 9.2, 1.2,
              boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='#2E7D32'))
ax1.text(0.6, 3.65, '四元数解决方案:', fontsize=12, fontweight='bold', color='#2E7D32')
ax1.text(0.6, 3.1, '✔ 无奇异点  ✔ 插值平滑 (SLERP)  ✔ 4个数 vs 3×3矩阵',
         fontsize=11, color='#2E7D32')

y = 2.2
ax1.text(0.5, y, '三种旋转表示对比:', fontsize=13, fontweight='bold')
# Mini table
table_data = [
    ['', '欧拉角', '旋转矩阵', '四元数'],
    ['参数数', '3', '9', '4'],
    ['万向节锁', '✘ 有', '✔ 无', '✔ 无'],
    ['插值', '✘ 不平滑', '✘ 复杂', '✔ SLERP'],
    ['组合', '✘ 顺序相关', '✔ 矩阵乘法', '✔ 乘法'],
]
cell_colors = ['#E3F2FD', '#FFF', '#FFF', '#FFF', '#FFF']
for row_idx, row in enumerate(table_data):
    for col_idx, cell in enumerate(row):
        x_pos = 0.5 + col_idx * 2.3
        y_pos = 1.6 - row_idx * 0.38
        if row_idx == 0:
            ax1.text(x_pos, y_pos, cell, fontsize=9, fontweight='bold', ha='left',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='#BBDEFB'))
        else:
            ax1.text(x_pos, y_pos, cell, fontsize=9, ha='left')

# Right: R1 Pro quaternion pipeline
ax2 = fig.add_subplot(122)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('R1 Pro 安全系统四元数处理管线', fontsize=14, fontweight='bold')

# Pipeline boxes
pipeline = [
    ('用户输入', 'setposq 0.40 -0.10 0.30\n  0.0  0.0  0.0  1.0', '#E3F2FD', '#1565C0'),
    ('① 归一化', 'q = q / ||q||\n确保 qx²+qy²+qz²+qw²=1', '#F3E5F5', '#6A1B9A'),
    ('② 四元数→欧拉角', 'quat_to_euler(qx,qy,qz,qw)\n→ (roll, pitch, yaw)', '#FFF3E0', '#E65100'),
    ('③ 安全空间裁剪', 'roll  = clip(roll,  min, max)\npitch = clip(pitch, min, max)\nyaw   = clip(yaw,   min, max)', '#FFEBEE', '#C62828'),
    ('④ 欧拉角→四元数', 'euler_to_quat(r, p, y)\n→ (qx, qy, qz, qw)', '#E8F5E9', '#2E7D32'),
    ('⑤ 发送给机器人', 'PoseStamped.orientation\n= Quaternion(qx,qy,qz,qw)', '#E0F7FA', '#00838F'),
]

for i, (title, desc, bg, fg) in enumerate(pipeline):
    y_pos = 9.0 - i * 1.45
    ax2.add_patch(mpatches.FancyBboxPatch((0.3, y_pos - 0.5), 9.2, 1.1,
                  boxstyle='round,pad=0.2', facecolor=bg, edgecolor=fg, linewidth=1.5))
    ax2.text(0.6, y_pos + 0.25, title, fontsize=11, fontweight='bold', color=fg)
    ax2.text(0.8, y_pos - 0.25, desc, fontsize=9, fontfamily='monospace', color='#333')
    if i < len(pipeline) - 1:
        ax2.annotate('', xy=(4.9, y_pos - 0.55), xytext=(4.9, y_pos - 0.75),
                     arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT_DIR, 'quaternion_gimbal_lock.png'), dpi=150, bbox_inches='tight')
plt.close()
print('[3/3] quaternion_gimbal_lock.png saved')

print('All 3 images generated successfully!')
