import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 几何体生成逻辑 ---

def generate_fancy_top(resolution=30):
    """ 生成陀螺网格 """
    t = np.linspace(0, 1, resolution)
    z = t * 3 
    r = np.zeros_like(z)
    
    for i, val in enumerate(t):
        if val < 0.5: r[i] = 1.5 * (val / 0.5)**0.7
        elif val < 0.6: r[i] = 1.5
        elif val < 0.8: r[i] = 1.5 * (1 - (val - 0.6)/0.2) * 0.3 + 0.2
        else: r[i] = 0.15

    theta = np.linspace(0, 2*np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    r_grid = np.tile(r, (resolution, 1)).T
    
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    colors = np.sin(3 * theta_grid) + np.cos(5 * z_grid)
    
    return x, y, z_grid, colors

def get_rotation_matrix(alpha, beta, gamma):
    """ 欧拉角旋转矩阵 """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    R = np.array([
        [ca*cg - sa*cb*sg, -ca*sg - sa*cb*cg, sa*sb],
        [sa*cg + ca*cb*sg, -sa*sg + ca*cb*cg, -ca*sb],
        [sb*sg,            sb*cg,             cb]
    ])
    return R

# --- 2. 界面与绘图设置 ---

plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25)
ax = fig.add_subplot(111, projection='3d')

# 坐标轴设置
ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
ax.grid(False) 
ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
ax.zaxis.set_pane_color((0.05, 0.05, 0.05, 1.0))

limit = 2.5
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(0, 3.5)

# --- 3. 初始化数据 ---

X0, Y0, Z0, C0 = generate_fancy_top(resolution=25)
shape_dims = X0.shape
coords0 = np.vstack((X0.flatten(), Y0.flatten(), Z0.flatten()))

# === 修改点 1: 降低透明度 (alpha=0.5) ===
surface = ax.plot_surface(X0, Y0, Z0, facecolors=plt.cm.magma(C0), 
                          shade=True, rstride=1, cstride=1, alpha=0.5, linewidth=0)

# 轴线
axis_line, = ax.plot([], [], [], 'cyan', lw=2, alpha=0.4)

# === 修改点 2: 轨迹线增强 ===
# 空中轨迹 (更亮，更粗)
trace_line, = ax.plot([], [], [], 'white', lw=2, alpha=1.0, zorder=10)
# 地面投影轨迹 (新增，方便观察图案)
ground_trace, = ax.plot([], [], [], 'lime', lw=1.5, alpha=0.6, zorder=1)

# 顶点与影子
tip_point, = ax.plot([], [], [], 'wo', markersize=4, zorder=10)
shadow_point, = ax.plot([], [], [], 'gray', marker='o', markersize=4, alpha=0.5)

trace_history = {'x': [], 'y': [], 'z': []}
max_trace = 200 # 增加轨迹长度，看清楚花瓣形状

params = {
    'spin_speed': 8.0,
    'prec_speed': 0.8,
    'nut_speed': 2.0,
    'mean_tilt': 25.0 * np.pi / 180,
    'nut_amp': 10.0 * np.pi / 180
}

# --- 4. 动画核心 ---

def update(frame):
    t = frame * 0.05
    
    gamma = params['spin_speed'] * t
    alpha = params['prec_speed'] * t
    beta = params['mean_tilt'] + params['nut_amp'] * np.cos(params['nut_speed'] * t)
    
    R = get_rotation_matrix(alpha, beta, gamma)
    
    # 1. 更新曲面
    rotated_coords = R @ coords0
    X_new = rotated_coords[0, :].reshape(shape_dims)
    Y_new = rotated_coords[1, :].reshape(shape_dims)
    Z_new = rotated_coords[2, :].reshape(shape_dims)
    
    global surface
    surface.remove()
    # === 关键：保持 alpha=0.5 以便透视 ===
    surface = ax.plot_surface(X_new, Y_new, Z_new, 
                              facecolors=plt.cm.magma(C0), 
                              shade=True, rstride=1, cstride=1, alpha=0.5, linewidth=0)
    
    # 2. 更新中轴线
    top_center = R @ np.array([0, 0, 3])
    axis_line.set_data([0, top_center[0]], [0, top_center[1]])
    axis_line.set_3d_properties([0, top_center[2]])
    
    # 3. 记录轨迹
    trace_history['x'].append(top_center[0])
    trace_history['y'].append(top_center[1])
    trace_history['z'].append(top_center[2])
    
    if len(trace_history['x']) > max_trace:
        trace_history['x'].pop(0)
        trace_history['y'].pop(0)
        trace_history['z'].pop(0)
        
    # 空中轨迹
    trace_line.set_data(trace_history['x'], trace_history['y'])
    trace_line.set_3d_properties(trace_history['z'])

    # === 新增：地面投影轨迹 (Z=0) ===
    # 这条线永远不会被遮挡，能清晰看到章动的“花瓣”
    ground_trace.set_data(trace_history['x'], trace_history['y'])
    ground_trace.set_3d_properties(np.zeros_like(trace_history['z']))

    # 顶点与影子
    tip_point.set_data([top_center[0]], [top_center[1]])
    tip_point.set_3d_properties([top_center[2]])
    
    shadow_point.set_data([top_center[0]], [top_center[1]])
    shadow_point.set_3d_properties([0])
    
    return surface, axis_line, trace_line, ground_trace

# --- 5. 交互控件 ---

slider_color = '#333333'
text_color = 'white'

ax_spin = plt.axes([0.2, 0.12, 0.6, 0.025], facecolor=slider_color)
ax_prec = plt.axes([0.2, 0.09, 0.6, 0.025], facecolor=slider_color)
ax_nut_amp = plt.axes([0.2, 0.06, 0.6, 0.025], facecolor=slider_color)
ax_tilt = plt.axes([0.2, 0.03, 0.6, 0.025], facecolor=slider_color)

s_spin = Slider(ax_spin, 'Spin', 0, 20, valinit=params['spin_speed'], color='#ff5555')
s_prec = Slider(ax_prec, 'Precession', -3, 3, valinit=params['prec_speed'], color='#5555ff')
s_nut_amp = Slider(ax_nut_amp, 'Nutation Amp', 0, 30, valinit=np.degrees(params['nut_amp']), color='#55ff55')
s_tilt = Slider(ax_tilt, 'Tilt', 0, 60, valinit=np.degrees(params['mean_tilt']), color='#ffff55')

for s in [s_spin, s_prec, s_nut_amp, s_tilt]:
    s.label.set_color(text_color)
    s.valtext.set_color(text_color)

def update_params(val):
    params['spin_speed'] = s_spin.val
    params['prec_speed'] = s_prec.val
    params['nut_amp'] = np.radians(s_nut_amp.val)
    params['mean_tilt'] = np.radians(s_tilt.val)

s_spin.on_changed(update_params)
s_prec.on_changed(update_params)
s_nut_amp.on_changed(update_params)
s_tilt.on_changed(update_params)

# --- 6. 运行 ---
fig.text(0.05, 0.92, 'Rigid Body: Transparent Trace', color='white', fontsize=14, weight='bold')
fig.text(0.05, 0.89, 'Green: Ground Projection (No Occlusion)', color='lime', fontsize=10)

ani = FuncAnimation(fig, update, frames=np.arange(0, 1000), interval=40, blit=False)
plt.show()