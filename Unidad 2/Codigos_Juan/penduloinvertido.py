# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import time
from scipy import constants

# =============================================================================
# CONSTANTES DEL SISTEMA
# =============================================================================
M = 3.0
m = 1.0
l = 1.0
g = constants.g

# =============================================================================
# PARTICIONES BORROSAS — 7 conjuntos triangulares
# =============================================================================
CONJUNTOS = ['NGG', 'NG', 'NP', 'Z', 'PP', 'PG', 'PGG']

COLORES = ['#e05252', '#e08c52', '#d4c84a', '#52c47a', '#52a8e0', '#7a6ee0', '#c052e0']
F_UNIVERSE = np.linspace(-300, 300, 60001)

# Particiones definidas a mano para los graficos.
THETA_PLOT_SPECS = [
    ('left', -180.000, -120.000),
    ('tri', -130.0000, -100.0000, -70.0000),
    ('tri', -80.0000, -50.0000, -20.0000),
    ('tri', -30.0000, 0.0000, 30.0000),
    ('tri', 20.0000, 50.0000, 80.0000),
    ('tri', 70.0000, 100.0000, 130.0000),
    ('right', 120.0000, 180.0000),
]

DTHETA_PLOT_SPECS = [
    ('left', -90.0000, -45.0000),
    ('tri', -55.0000, -40.0000, -25.0000),
    ('tri', -35.0000, -20.0000, -5.0000),
    ('tri', -15.0000,   0.0000,  15.0000),
    ('tri',  5.0000,  20.0000,  35.0000),
    ('tri',  25.0000,  40.0000,  55.0000),
    ('right', 45.0000, 90.0000    ),
]

F_PLOT_SPECS = [
    ('left', -300.0000, -240.0000),
    ('tri', -260.0000, -200.0000, -140.0000),
    ('tri', -160.0000, -100.0000, -40.0000),
    ('tri', -60.0000,    0.0000,  60.0000),
    ('tri',  40.0000,  100.0000, 160.0000),
    ('tri',  140.0000,  200.0000, 260.0000),
    ('right', 240.0000, 300.0000),
]

# Particiones manuales equivalentes a la logica del controlador que te
# funcionaba antes. Esto mantiene el comportamiento del control, pero sin
# volver a usar PICOS_/D_ en el codigo.
THETA_CTRL_SPECS = [
    ('left', -130.0000, -110.0000),
    ('tri', -160.0000, -110.0000, -60.0000),
    ('tri', -100.0000, -50.0000, 0.0000), 
    ('tri', -50.0000, 0.0000, 50.0000),
    ('tri', 0.0000, 50.0000, 100.0000),
    ('tri', 60.0000, 110.0000, 160.0000),
    ('right', 110.0000, 130.0000),
]

DTHETA_CTRL_SPECS = [
    ('left', -55.0000, -45.0000),
    ('tri', -55.0000, -40.0000, -25.0000),
    ('tri', -35.0000, -20.0000, -5.0000),
    ('tri', -15.0000,   0.0000,  15.0000),
    ('tri',  5.0000,  20.0000,  35.0000),
    ('tri',  25.0000,  40.0000,  55.0000),
    ('right', 45.0000, 55.0000),
]

F_CTRL_SPECS = [
    ('left', -260.0000, -240.0000),
    ('tri', -260.0000, -200.0000, -140.0000),
    ('tri', -160.0000, -100.0000, -40.0000),
    ('tri', -60.0000,    0.0000,  60.0000),
    ('tri',  40.0000,  100.0000, 160.0000),
    ('tri',  140.0000,  200.0000, 260.0000),
    ('right', 240.0000, 300.0000),
]


def _mu_plot_scalar(x, specs):
    out = np.zeros(len(CONJUNTOS))
    for i, spec in enumerate(specs):
        kind = spec[0]
        if kind == 'left':
            full_until, zero_at = spec[1], spec[2]
            if x <= full_until:
                out[i] = 1.0
            elif x < zero_at:
                out[i] = (zero_at - x) / (zero_at - full_until)
        elif kind == 'right':
            zero_at, full_from = spec[1], spec[2]
            if x >= full_from:
                out[i] = 1.0
            elif x > zero_at:
                out[i] = (x - zero_at) / (full_from - zero_at)
        else:
            left, center, right = spec[1], spec[2], spec[3]
            if left < x <= center:
                out[i] = (x - left) / (center - left)
            elif center < x < right:
                out[i] = (right - x) / (right - center)
            elif x == center:
                out[i] = 1.0
    return out


def _mu_plot_particion(x_arr, specs):
    out = np.zeros((len(CONJUNTOS), len(x_arr)))
    for k, x in enumerate(x_arr):
        out[:, k] = _mu_plot_scalar(float(x), specs)
    return out


MU_F_BASE = _mu_plot_particion(F_UNIVERSE, F_CTRL_SPECS)

# =============================================================================
# TABLA FAM 7x7
# =============================================================================
IDX = {name: i for i, name in enumerate(CONJUNTOS)}

FAM = np.array([
#  NGG         NG          NP          Z           PP          PG          PGG       ← θ
  [IDX['PGG'], IDX['PGG'], IDX['PGG'], IDX['PP'],  IDX['PP'],  IDX['NP'],  IDX['NP']],
  [IDX['PGG'], IDX['PGG'], IDX['PG'],  IDX['PP'],  IDX['Z'],   IDX['NP'],  IDX['NG']],
  [IDX['PGG'], IDX['PGG'], IDX['PP'],  IDX['PP'],  IDX['NP'],  IDX['NP'],  IDX['NGG']],
  [IDX['PGG'], IDX['PG'],  IDX['PP'],  IDX['Z'],   IDX['NP'],  IDX['NG'],  IDX['NGG']],
  [IDX['PGG'], IDX['PP'],  IDX['PP'],  IDX['NP'],  IDX['NP'],  IDX['NGG'], IDX['NGG']],
  [IDX['PG'],  IDX['PP'],  IDX['Z'],   IDX['NP'],  IDX['NG'],  IDX['NGG'], IDX['NGG']],
  [IDX['PP'],  IDX['PP'],  IDX['NP'],  IDX['NP'],  IDX['NGG'], IDX['NGG'], IDX['NGG']],
])

# =============================================================================
# CONTROLADOR DIFUSO
# =============================================================================
def controlador_fam(theta_deg, dtheta_deg):
    dtheta_deg = np.clip(dtheta_deg, -90, 90)
    mu_t = _mu_plot_scalar(theta_deg, THETA_CTRL_SPECS)
    mu_dt = _mu_plot_scalar(dtheta_deg, DTHETA_CTRL_SPECS)
    alpha = np.minimum(mu_dt[:, None], mu_t[None, :])
    mu_salida = np.zeros(len(F_UNIVERSE))
    for i in range(7):
        for j in range(7):
            a = alpha[i, j]
            if a < 1e-9:
                continue
            np.maximum(mu_salida, np.minimum(a, MU_F_BASE[FAM[i, j]]), out=mu_salida)
    total = np.trapezoid(mu_salida, F_UNIVERSE)
    if total < 1e-7:
        return 0.0
    f = -float(np.trapezoid(F_UNIVERSE * mu_salida, F_UNIVERSE) / total)
    return f

# =============================================================================
# FÍSICA
# =============================================================================
def calcula_aceleracion(theta_rad, v_rad, f_N):
    num = (g * np.sin(theta_rad)
           + np.cos(theta_rad) * ((-f_N - m*l*v_rad**2*np.sin(theta_rad)) / (M+m)))
    den = l * (4/3 - m*np.cos(theta_rad)**2 / (M+m))
    return num / den

# =============================================================================
# PLOT ESTÁTICO DE PARTICIONES
# =============================================================================
def graficar_particiones():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle('Particiones borrosas — 7 conjuntos triangulares',
                 color='#c8d0e8', fontsize=12, y=1.01)

    configs = [
        (axes[0], THETA_PLOT_SPECS, 'θ  [°]', -180, 180),
        (axes[1], DTHETA_PLOT_SPECS, "θ'  [°/s]", -90, 90),
        (axes[2], F_PLOT_SPECS, 'F  [N]', -300, 300),
    ]

    for ax, specs, xlabel, xmin, xmax in configs:
        ax.set_facecolor('#1a1d2e')
        ax.tick_params(colors='#8890a8', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2d3a')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel(xlabel, color='#8890a8', fontsize=10)
        ax.set_ylabel('μ', color='#8890a8', fontsize=10)
        ax.axhline(0, color='#2a2d3a', lw=0.8)
        ax.axhline(1, color='#2a2d3a', lw=0.5, ls=':')
        ax.axvline(0, color='#2a2d3a', lw=0.5, ls='--')
        ax.grid(True, alpha=0.08)

        x = np.linspace(xmin, xmax, 1000)
        mu_all = _mu_plot_particion(x, specs)
        for i, (spec, nombre) in enumerate(zip(specs, CONJUNTOS)):
            ax.plot(x, mu_all[i], color=COLORES[i], lw=2, label=nombre)
            if spec[0] == 'left':
                c = spec[1]
            elif spec[0] == 'right':
                c = spec[2]
            else:
                c = spec[2]
            ax.plot(c, 1.0, 'o', color=COLORES[i], ms=5, zorder=5)
            ax.text(c, 1.06, nombre, color=COLORES[i],
                    fontsize=7.5, ha='center', va='bottom', fontweight='bold')

        ax.axhline(0.5, color='#ffffff', lw=0.4, ls=':', alpha=0.3)

    plt.tight_layout()
    plt.show()

# =============================================================================
# SIMULACIÓN CON ANIMACIÓN
# =============================================================================
def simular(delta_t=0.005, theta_0_deg=170.0, v_0_deg=0.0):

    estado = {
        'theta':  np.radians(theta_0_deg),
        'v':      np.radians(v_0_deg),
        't':      0.0,
        'cart_x': 0.0,
        'empuje': 0.0,
    }
    hist_t = [0.0]
    hist_theta = [theta_0_deg]
    hist_f = [0.0]

    VELOCIDAD = 1
    L_vis = 1.5
    VENTANA = 10.0

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor('#0f1117')

    ax_anim = fig.add_axes([0.02, 0.12, 0.52, 0.84])
    ax_th = fig.add_axes([0.58, 0.55, 0.40, 0.38])
    ax_f = fig.add_axes([0.58, 0.12, 0.40, 0.35])

    for ax in (ax_anim, ax_th, ax_f):
        ax.set_facecolor('#1a1d2e')
        ax.tick_params(colors='#8890a8', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2d3a')

    ax_anim.set_xlim(-3.5, 3.5)
    ax_anim.set_ylim(-2.0, 2.0)
    ax_anim.set_aspect('equal')
    ax_anim.axhline(0, color='#2a2d3a', lw=1)
    ax_anim.axvline(0, color='#2a2d3a', lw=0.5, ls='--')
    ax_anim.axhline(-L_vis, color='#2a2d3a', lw=0.5, ls=':')
    ax_anim.set_title('Péndulo invertido — FAM 7×7', color='#c8d0e8', fontsize=11)
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])

    carro = patches.FancyBboxPatch((-0.4, -0.18), 0.8, 0.32,
                                   boxstyle='round,pad=0.02',
                                   facecolor='#1e2235', edgecolor='#4a9eff', lw=1.5)
    ax_anim.add_patch(carro)
    pertiga, = ax_anim.plot([], [], color='#a78bfa', lw=3, solid_capstyle='round')
    masa, = ax_anim.plot([], [], 'o', color='#a78bfa', ms=14,
                         mec='#7c5cfc', mew=2)
    flecha_ln, = ax_anim.plot([], [], lw=2.5)
    flecha_pt, = ax_anim.plot([], [], ms=9)
    texto = ax_anim.text(-3.3, 1.85, '', color='#c8d0e8',
                         fontsize=9, fontfamily='monospace')

    ax_th.set_ylabel('θ [°]', color='#4a9eff', fontsize=9)
    ax_th.axhline(0, color='#4a9eff', lw=0.5, ls='--', alpha=0.4)
    ax_th.axhline(180, color='#3a4060', lw=0.5, ls=':')
    ax_th.axhline(-180, color='#3a4060', lw=0.5, ls=':')
    ax_th.set_ylim(-200, 200)
    ax_th.set_yticks([-180, -90, 0, 90, 180])
    ax_th.grid(True, alpha=0.12)
    linea_th, = ax_th.plot([], [], color='#4a9eff', lw=1.2)

    ax_f.set_ylabel('F [N]', color='#ff6b4a', fontsize=9)
    ax_f.set_xlabel('t [s]', color='#8890a8', fontsize=9)
    ax_f.axhline(0, color='#3a4060', lw=0.8)
    ax_f.set_ylim(-400, 400)
    ax_f.grid(True, alpha=0.12)
    linea_f, = ax_f.plot([], [], color='#ff6b4a', lw=1.2)

    ax_mu_t = fig.add_axes([0.29, 0.13, 0.12, 0.10])
    ax_mu_dt = fig.add_axes([0.29, 0.24, 0.12, 0.10])
    ax_mu_f = fig.add_axes([0.29, 0.35, 0.12, 0.10])

    inset_configs = [
        (ax_mu_t, THETA_PLOT_SPECS, 'θ', -180, 180),
        (ax_mu_dt, DTHETA_PLOT_SPECS, "θ'", -90, 90),
        (ax_mu_f, F_PLOT_SPECS, 'F', -300, 300),
    ]

    for ax_i, specs, lbl, xmin, xmax in inset_configs:
        ax_i.set_facecolor('#0f1117')
        ax_i.tick_params(colors='#555', labelsize=5, length=2, pad=1)
        for sp in ax_i.spines.values():
            sp.set_edgecolor('#2a2d3a')
        ax_i.set_xlim(xmin, xmax)
        ax_i.set_ylim(-0.05, 1.15)
        ax_i.set_yticks([0, 0.5, 1])
        ax_i.axhline(0, color='#2a2d3a', lw=0.5)
        x = np.linspace(xmin, xmax, 400)
        mu_all = _mu_plot_particion(x, specs)
        for i in range(7):
            ax_i.plot(x, mu_all[i], color=COLORES[i], lw=1.0, alpha=0.6)
        ax_i.text(0.02, 0.85, lbl, transform=ax_i.transAxes,
                  color='#8890a8', fontsize=6, va='top')

    linea_cur_t, = ax_mu_t.plot([], [], color='white', lw=0.8, ls='--')
    linea_cur_dt, = ax_mu_dt.plot([], [], color='white', lw=0.8, ls='--')
    linea_cur_f, = ax_mu_f.plot([], [], color='#ff6b4a', lw=1.2, ls='--')
    puntos_t = [ax_mu_t.plot([], [], 'o', color=COLORES[i], ms=3, zorder=5)[0]
                for i in range(7)]
    puntos_dt = [ax_mu_dt.plot([], [], 'o', color=COLORES[i], ms=3, zorder=5)[0]
                 for i in range(7)]

    ax_btn_l = fig.add_axes([0.58, 0.03, 0.12, 0.06])
    ax_btn_r = fig.add_axes([0.72, 0.03, 0.12, 0.06])
    ax_btn_rst = fig.add_axes([0.87, 0.03, 0.10, 0.06])

    btn_l = widgets.Button(ax_btn_l, '← Empujar', color='#1e2235', hovercolor='#2a3050')
    btn_r = widgets.Button(ax_btn_r, 'Empujar →', color='#1e2235', hovercolor='#2a3050')
    btn_rst = widgets.Button(ax_btn_rst, '↺ Reset', color='#1e2235', hovercolor='#2a3050')

    for btn, sp_color, lbl_color in [
        (btn_l, '#4a9eff', '#4a9eff'),
        (btn_r, '#ff6b4a', '#ff6b4a'),
        (btn_rst, '#8890a8', '#8890a8'),
    ]:
        for sp in btn.ax.spines.values():
            sp.set_edgecolor(sp_color)
        btn.label.set_color(lbl_color)
        btn.label.set_fontsize(10)

    EMPUJE_N = 250.0
    EMPUJE_DT = 0.20
    empuje_fin = [0.0]

    def empujar_izq(event):
        estado['empuje'] = -EMPUJE_N
        empuje_fin[0] = estado['t'] + EMPUJE_DT

    def empujar_der(event):
        estado['empuje'] = +EMPUJE_N
        empuje_fin[0] = estado['t'] + EMPUJE_DT

    def reset(event):
        estado['theta'] = np.radians(theta_0_deg)
        estado['v'] = np.radians(v_0_deg)
        estado['t'] = 0.0
        estado['cart_x'] = 0.0
        estado['empuje'] = 0.0
        empuje_fin[0] = 0.0
        hist_t.clear()
        hist_t.append(0.0)
        hist_theta.clear()
        hist_theta.append(theta_0_deg)
        hist_f.clear()
        hist_f.append(0.0)

    btn_l.on_clicked(empujar_izq)
    btn_r.on_clicked(empujar_der)
    btn_rst.on_clicked(reset)

    _last_wall = [time.perf_counter()]

    def update(_):
        now = time.perf_counter()
        elapsed_wall = now - _last_wall[0]
        _last_wall[0] = now
        sim_time = min(elapsed_wall * VELOCIDAD, delta_t * 1000)
        steps = max(1, int(sim_time / delta_t))

        for _ in range(steps):
            if estado['t'] > empuje_fin[0]:
                estado['empuje'] = 0.0
            th_norm = (np.degrees(estado['theta']) + 180) % 360 - 180
            vd_deg = np.degrees(estado['v'])
            f_ctrl = controlador_fam(th_norm, vd_deg)
            f_total = f_ctrl + estado['empuje']
            a = calcula_aceleracion(estado['theta'], estado['v'], f_total)
            estado['v'] += a * delta_t
            estado['theta'] += estado['v'] * delta_t + 0.5 * a * delta_t**2
            estado['t'] += delta_t
            estado['cart_x'] = np.clip(
                estado['cart_x'] - f_total * delta_t * 0.005, -2.8, 2.8)
            hist_t.append(estado['t'])
            hist_theta.append((np.degrees(estado['theta']) + 180) % 360 - 180)
            hist_f.append(f_total)

        cx = estado['cart_x']
        th = estado['theta']
        f_act = hist_f[-1]
        t_act = estado['t']
        th_norm = (np.degrees(th) + 180) % 360 - 180
        vd_deg = np.degrees(estado['v'])

        carro.set_x(cx - 0.4)
        px = cx + L_vis * np.sin(th)
        py = L_vis * np.cos(th)
        pertiga.set_data([cx, px], [0, py])
        masa.set_data([px], [py])

        f_ctrl_act = f_act - estado['empuje']
        color_f = '#4a9eff' if f_ctrl_act >= 0 else '#ff6b4a'
        alen = np.clip(f_ctrl_act / 300 * 1.0, -1.0, 1.0)
        x_tip = cx - alen
        flecha_ln.set_data([cx, x_tip], [-0.12, -0.12])
        flecha_ln.set_color(color_f)
        flecha_pt.set_data([x_tip], [-0.12])
        flecha_pt.set_color(color_f)
        flecha_pt.set_marker('<' if f_ctrl_act >= 0 else '>')

        empuje_str = f"  [EMPUJE {estado['empuje']:+.0f}N]" if estado['empuje'] != 0 else ""
        texto.set_text(
            f"t={t_act:.1f}s   θ={th_norm:.1f}°{empuje_str}\n"
            f"θ'={vd_deg:.1f}°/s   F_ctrl={f_ctrl_act:.2f} N"
        )

        t0 = max(0.0, t_act - VENTANA)
        ax_th.set_xlim(t0, t0 + VENTANA)
        ax_f.set_xlim(t0, t0 + VENTANA)
        linea_th.set_data(hist_t, hist_theta)
        linea_f.set_data(hist_t, hist_f)

        linea_cur_t.set_data([th_norm, th_norm], [-0.05, 1.15])
        mu_t_cur = _mu_plot_scalar(th_norm, THETA_PLOT_SPECS)
        for i, pt in enumerate(puntos_t):
            pt.set_data([th_norm], [mu_t_cur[i]])

        vd_clip = np.clip(vd_deg, -90, 90)
        linea_cur_dt.set_data([vd_clip, vd_clip], [-0.05, 1.15])
        mu_dt_cur = _mu_plot_scalar(vd_clip, DTHETA_PLOT_SPECS)
        for i, pt in enumerate(puntos_dt):
            pt.set_data([vd_clip], [mu_dt_cur[i]])

        linea_cur_f.set_data([f_ctrl_act, f_ctrl_act], [-0.05, 1.15])

        return (carro, pertiga, masa, flecha_ln, flecha_pt,
                linea_th, linea_f, texto,
                linea_cur_t, linea_cur_dt, linea_cur_f,
                *puntos_t, *puntos_dt)

    ani = animation.FuncAnimation(fig, update, frames=None,
                                  interval=20, blit=False,
                                  cache_frame_data=False)
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    graficar_particiones()
    simular(delta_t=0.001, theta_0_deg=170.0, v_0_deg=0.0)
