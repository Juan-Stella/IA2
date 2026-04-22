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
M = 20.0
m = 3.0
l = 1.0
g = constants.g
CART_HALF_WIDTH = 0.4
CART_HEIGHT = 0.32
ARROW_SCALE = 1.0
CONTROL_FORCE_GAIN = 1.0
CART_VISCOSITY = 30.0
STARTUP_KICK_FORCE = 120.0
STARTUP_KICK_TIME = 0.15
ANIM_MAX_HALF_WIDTH = 6.0
COLOR_F_CTRL = '#52a8e0'
COLOR_F_PUSH = '#f59e0b'
COLOR_F_TOTAL = '#f43f5e'

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
    ('tri', -160.0000, -100.0000,  -40.0000),
    ('tri',  -60.0000,    0.0000,   60.0000),
    ('tri',   40.0000,  100.0000,  160.0000),
    ('tri',  140.0000,  200.0000,  260.0000),
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
    ('left', -300.0000, -240.0000),
    ('tri', -260.0000, -200.0000, -140.0000),
    ('tri', -160.0000, -100.0000,  -40.0000),
    ('tri',  -60.0000,    0.0000,   60.0000),
    ('tri',   40.0000,  100.0000,  160.0000),
    ('tri',  140.0000,  200.0000,  260.0000),
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


def _normaliza_angulo_deg(theta_deg):
    return ((theta_deg + 180.0) % 360.0) - 180.0


def fuerza_usuario_a_modelo(f_user):
    # Convención del usuario:
    #   F > 0  => izquierda
    #   F < 0  => derecha
    # Convención del modelo dinámico:
    #   F > 0  => derecha
    #   F < 0  => izquierda
    return -f_user


def _spec_points(spec):
    kind = spec[0]
    if kind == 'left':
        peak = spec[1]
        zeros = [spec[2]]
    elif kind == 'right':
        peak = spec[2]
        zeros = [spec[1]]
    else:
        peak = spec[2]
        zeros = [spec[1], spec[3]]
    return peak, zeros


def _spec_support_interval(spec):
    kind = spec[0]
    if kind == 'left':
        return spec[1], spec[2]
    if kind == 'right':
        return spec[1], spec[2]
    return spec[1], spec[3]


def _adjacent_overlap_percentages(specs):
    porcentajes = []
    for spec_a, spec_b in zip(specs[:-1], specs[1:]):
        left_a, right_a = _spec_support_interval(spec_a)
        left_b, right_b = _spec_support_interval(spec_b)
        overlap = max(0.0, min(right_a, right_b) - max(left_a, left_b))
        ancho_a = right_a - left_a
        ancho_b = right_b - left_b
        ref = min(ancho_a, ancho_b)
        porcentajes.append(0.0 if ref <= 0.0 else 100.0 * overlap / ref)
    return porcentajes


def _internal_adjacent_overlap_percentages(specs):
    porcentajes = _adjacent_overlap_percentages(specs)
    if len(porcentajes) <= 2:
        return porcentajes
    return porcentajes[1:-1]


def crear_arranque(theta_0_deg, v_0_deg):
    theta_norm = _normaliza_angulo_deg(theta_0_deg)
    if np.isclose(abs(theta_norm), 180.0, atol=1e-9) and np.isclose(v_0_deg, 0.0, atol=1e-9):
        signo = float(np.random.choice([-1.0, 1.0]))
        return signo * STARTUP_KICK_FORCE, STARTUP_KICK_TIME
    return 0.0, 0.0


def actualizar_camara_animacion(ax_anim, track_limit, side_margin):
    full_half_width = track_limit + side_margin
    view_half_width = min(full_half_width, ANIM_MAX_HALF_WIDTH)
    ax_anim.set_xlim(-view_half_width, view_half_width)

    tick_step = 1.0 if view_half_width <= 6 else 2.0
    xticks = np.arange(np.floor(-view_half_width), np.ceil(view_half_width) + 0.001, tick_step)
    ax_anim.set_xticks(xticks)


MU_F_BASE = _mu_plot_particion(F_UNIVERSE, F_CTRL_SPECS)

# =============================================================================
# TABLA FAM 7x7
# =============================================================================
IDX = {name: i for i, name in enumerate(CONJUNTOS)}

FAM = np.array([
  [IDX['PGG'], IDX['PGG'], IDX['PGG'], IDX['PP'],  IDX['PP'],  IDX['NP'],  IDX['NP']],
  [IDX['PGG'], IDX['PGG'], IDX['PG'],  IDX['PP'],  IDX['Z'],   IDX['NP'],  IDX['NG']],
  [IDX['PGG'], IDX['PGG'], IDX['PG'],  IDX['PP'],  IDX['NP'],  IDX['NP'],  IDX['NGG']],
  [IDX['PGG'], IDX['PG'],  IDX['PG'],  IDX['Z'],   IDX['NG'],  IDX['NG'],  IDX['NGG']],
  [IDX['PGG'], IDX['PP'],  IDX['PP'],  IDX['NP'],  IDX['NG'],  IDX['NGG'], IDX['NGG']],
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
    # Se devuelve F con la convención del usuario:
    # positivo hacia la izquierda, negativo hacia la derecha.
    f = CONTROL_FORCE_GAIN * float(np.trapezoid(F_UNIVERSE * mu_salida, F_UNIVERSE) / total)
    return f

# =============================================================================
# FÍSICA
# =============================================================================
def calcula_aceleracion(theta_rad, v_rad, f_N):
    num = (g * np.sin(theta_rad)
           + np.cos(theta_rad) * ((-f_N - m*l*v_rad**2*np.sin(theta_rad)) / (M+m)))
    den = l * (4/3 - m*np.cos(theta_rad)**2 / (M+m))
    return num / den


def calcula_aceleracion_carro(theta_rad, v_rad, theta_dd, cart_v, f_N):
    friccion_viscosa = CART_VISCOSITY * cart_v
    fuerza_neta = f_N - friccion_viscosa
    return (fuerza_neta + m * l * (v_rad**2 * np.sin(theta_rad) - theta_dd * np.cos(theta_rad))) / (M + m)


def crear_estado(theta_0_deg, v_0_deg):
    return {
        'theta': np.radians(theta_0_deg),
        'v': np.radians(v_0_deg),
        't': 0.0,
        'cart_x': 0.0,
        'cart_v': 0.0,
        'empuje': 0.0,
        'f_ctrl': 0.0,
        'f_total': 0.0,
    }


def reiniciar_historiales(theta_0_deg):
    return {
        't': [0.0],
        'theta': [theta_0_deg],
        'f_ctrl': [0.0],
        'f_push': [0.0],
        'f_total': [0.0],
        'x': [0.0],
    }


def limitar_fuerza_control(cart_x, track_limit, edge_eps, f_ctrl):
    if cart_x <= -track_limit + edge_eps and f_ctrl > 0.0:
        return 0.0
    if cart_x >= track_limit - edge_eps and f_ctrl < 0.0:
        return 0.0
    return f_ctrl


def integrar_pendulo(estado, f_total, delta_t):
    f_modelo = fuerza_usuario_a_modelo(f_total)
    theta_dd = calcula_aceleracion(estado['theta'], estado['v'], f_modelo)
    x_dd = calcula_aceleracion_carro(estado['theta'], estado['v'], theta_dd, estado['cart_v'], f_modelo)

    v_prev = estado['v']
    cart_v_prev = estado['cart_v']

    estado['theta'] += v_prev * delta_t + 0.5 * theta_dd * delta_t**2
    estado['v'] = v_prev + theta_dd * delta_t
    estado['cart_x'] += cart_v_prev * delta_t + 0.5 * x_dd * delta_t**2
    estado['cart_v'] = cart_v_prev + x_dd * delta_t
    estado['t'] += delta_t


def actualizar_posicion_carro(estado, track_limit, edge_eps):
    en_tope_izq = estado['cart_x'] <= -track_limit + edge_eps
    en_tope_der = estado['cart_x'] >= track_limit - edge_eps

    if en_tope_izq:
        estado['cart_x'] = -track_limit
        if estado['cart_v'] < 0.0:
            estado['cart_v'] = 0.0
    elif en_tope_der:
        estado['cart_x'] = track_limit
        if estado['cart_v'] > 0.0:
            estado['cart_v'] = 0.0


def registrar_historial(hist, estado):
    hist['t'].append(estado['t'])
    hist['theta'].append(_normaliza_angulo_deg(np.degrees(estado['theta'])))
    hist['f_ctrl'].append(estado['f_ctrl'])
    hist['f_push'].append(estado['empuje'])
    hist['f_total'].append(estado['f_total'])
    hist['x'].append(estado['cart_x'])

# =============================================================================
# PLOT ESTÁTICO DE PARTICIONES
# =============================================================================
def graficar_particiones():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle('Particiones borrosas — 7 conjuntos triangulares',
                 color='#c8d0e8', fontsize=14, y=0.97)

    configs = [
        (axes[0], THETA_PLOT_SPECS, 'θ  [°]', -180, 180),
        (axes[1], DTHETA_PLOT_SPECS, "θ'  [°/s]", -90, 90),
        (axes[2], F_PLOT_SPECS, 'F  [N]', -300, 300),
    ]

    for ax, specs, xlabel, xmin, xmax in configs:
        ax.set_facecolor('#1a1d2e')
        ax.tick_params(colors='#8890a8', labelsize=8)
        ax.tick_params(axis='x', colors='#ffffff', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2d3a')
        ax.spines['bottom'].set_edgecolor('#ffffff')
        ax.spines['bottom'].set_linewidth(1.1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.16, 1.15)
        ax.set_xlabel(xlabel, color='#8890a8', fontsize=10)
        ax.set_ylabel('μ', color='#8890a8', fontsize=10)
        ax.axhline(1, color='#2a2d3a', lw=0.5, ls=':')
        ax.axvline(0, color='#2a2d3a', lw=0.5, ls='--')
        ax.grid(True, alpha=0.08)

        x = np.linspace(xmin, xmax, 1000)
        mu_all = _mu_plot_particion(x, specs)
        for i, (spec, nombre) in enumerate(zip(specs, CONJUNTOS)):
            ax.plot(x, mu_all[i], color=COLORES[i], lw=2, label=nombre)
            c, zeros = _spec_points(spec)
            ax.plot(c, 1.0, 'o', color=COLORES[i], ms=5, zorder=5)
            ax.text(c, 1.08, nombre, color=COLORES[i],
                    fontsize=7.5, ha='center', va='bottom', fontweight='bold')
            ax.text(c, 1.015, f'{c:g}', color=COLORES[i],
                    fontsize=7, ha='center', va='bottom')

            for k, x0 in enumerate(zeros):
                ax.plot([x0], [0.0], marker='o', color=COLORES[i], ms=3, zorder=5)
                ax.plot([x0, x0], [0.0, 0.03], color=COLORES[i], lw=0.8, alpha=0.7)
                ax.text(x0, -0.085 - 0.03 * (k % 2), f'{x0:g}',
                        color=COLORES[i], fontsize=6.5, ha='center', va='top')

        ax.axhline(0, color='#ffffff', lw=1.3, alpha=0.9, zorder=10)
        ax.axhline(0.5, color='#ffffff', lw=0.4, ls=':', alpha=0.3)
        solapes = _internal_adjacent_overlap_percentages(specs)
        if np.ptp(solapes) < 1e-6:
            texto_solape = f'Solape interno: {solapes[0]:.1f}%'
        else:
            texto_solape = 'Solape interno: ' + ' | '.join(f'{p:.1f}%' for p in solapes)
        ax.text(0.98, 0.97, texto_solape, transform=ax.transAxes,
                ha='right', va='top', color='#ffffff', fontsize=7.2,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='#0f1117',
                          edgecolor='#ffffff', alpha=0.25))

    plt.tight_layout(rect=[0.0, 0.04, 1.0, 0.92])
    plt.show()

# =============================================================================
# SIMULACIÓN CON ANIMACIÓN
# =============================================================================
def simular(delta_t=0.01, theta_0_deg=180.0, v_0_deg=0.0, track_limit=2.5):

    max_force_user = max(abs(F_UNIVERSE[0]), abs(F_UNIVERSE[-1])) + 150.0
    max_cart_acc = abs(fuerza_usuario_a_modelo(max_force_user)) / M
    EDGE_EPS = max(1e-6, 0.5 * max_cart_acc * delta_t**2)
    startup_force = [0.0]
    startup_until = [0.0]
    startup_force[0], startup_until[0] = crear_arranque(theta_0_deg, v_0_deg)

    estado = crear_estado(theta_0_deg, v_0_deg)
    hist = reiniciar_historiales(theta_0_deg)

    VELOCIDAD = 1
    L_vis = 1.5
    VENTANA = 10.0

    fig = plt.figure(figsize=(14.8, 8.6))
    fig.patch.set_facecolor('#0f1117')

    ax_anim = fig.add_axes([0.03, 0.21, 0.50, 0.74])
    ax_th = fig.add_axes([0.58, 0.71, 0.38, 0.17])
    ax_x = fig.add_axes([0.58, 0.47, 0.38, 0.13])
    ax_f = fig.add_axes([0.58, 0.18, 0.38, 0.21])

    for ax in (ax_anim, ax_th, ax_x, ax_f):
        ax.set_facecolor('#1a1d2e')
        ax.tick_params(colors='#8890a8', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2d3a')

    side_margin = L_vis + CART_HALF_WIDTH + 0.45
    actualizar_camara_animacion(ax_anim, track_limit, side_margin)
    ax_anim.set_ylim(-2.0, 2.0)
    ax_anim.set_aspect('equal')
    ax_anim.axhline(0, color='#2a2d3a', lw=1)
    ax_anim.axvline(0, color='#2a2d3a', lw=0.5, ls='--')
    ax_anim.axhline(-L_vis, color='#2a2d3a', lw=0.5, ls=':')
    ax_anim.axvspan(-track_limit, track_limit, color='#171c2c', alpha=0.35, zorder=0)
    # Topes dinámicos en rojo — se dibujan según track_limit
    ax_anim.axvline(-track_limit, color='#e05252', lw=1.5, ls='--', alpha=0.8)
    ax_anim.axvline(track_limit, color='#e05252', lw=1.5, ls='--', alpha=0.8)
    ax_anim.set_title('Péndulo invertido — FAM 7×7', color='#c8d0e8', fontsize=11)
    # Eje X con ticks de posición
    ax_anim.set_yticks([])
    ax_anim.tick_params(axis='x', colors='#8890a8', labelsize=7, length=3)
    ax_anim.set_xlabel('x [m]', color='#8890a8', fontsize=8, labelpad=1)
    # Indicador de posición del carro en el eje X (punto deslizante)
    ind_x, = ax_anim.plot([0], [-1.95], 'v', color='#4a9eff', ms=7, zorder=6, clip_on=False)
    ax_anim.text(-track_limit, -1.82, f'{-track_limit:.1f}', color='#e05252', fontsize=7, ha='center')
    ax_anim.text(track_limit, -1.82, f'{track_limit:.1f}', color='#e05252', fontsize=7, ha='center')

    carro = patches.FancyBboxPatch((-CART_HALF_WIDTH, -0.18), 2 * CART_HALF_WIDTH, CART_HEIGHT,
                                   boxstyle='round,pad=0.02',
                                   facecolor='#1e2235', edgecolor='#4a9eff', lw=1.5)
    ax_anim.add_patch(carro)
    pertiga, = ax_anim.plot([], [], color='#a78bfa', lw=3, solid_capstyle='round')
    masa, = ax_anim.plot([], [], 'o', color='#a78bfa', ms=14,
                         mec='#7c5cfc', mew=2)
    flecha_ln, = ax_anim.plot([], [], lw=2.5)
    flecha_pt, = ax_anim.plot([], [], ms=9)
    texto = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes,
                         color='#c8d0e8', fontsize=9, fontfamily='monospace',
                         va='top', ha='left')

    ax_th.set_ylabel('θ [°]', color='#4a9eff', fontsize=9)
    ax_th.axhline(0, color='#4a9eff', lw=0.5, ls='--', alpha=0.4)
    ax_th.axhline(180, color='#3a4060', lw=0.5, ls=':')
    ax_th.axhline(-180, color='#3a4060', lw=0.5, ls=':')
    ax_th.set_ylim(-200, 200)
    ax_th.set_yticks([-180, -90, 0, 90, 180])
    ax_th.grid(True, alpha=0.12)
    linea_th, = ax_th.plot([], [], color='#4a9eff', lw=1.2)

    ax_x.set_ylabel('x [m]', color='#7dd3fc', fontsize=9)
    ax_x.axhline(0, color='#3a4060', lw=0.8)
    ax_x.axhline(track_limit, color='#e05252', lw=0.8, ls='--', alpha=0.8)
    ax_x.axhline(-track_limit, color='#e05252', lw=0.8, ls='--', alpha=0.8)
    ax_x.set_ylim(-track_limit - 0.4, track_limit + 0.4)
    ax_x.grid(True, alpha=0.12)
    linea_x, = ax_x.plot([], [], color='#7dd3fc', lw=1.3, label='x')
    ax_x.legend(loc='upper right', frameon=False, fontsize=8, labelcolor='#c8d0e8')

    ax_f.set_ylabel('F [N]', color=COLOR_F_TOTAL, fontsize=9)
    ax_f.set_xlabel('t [s]', color='#8890a8', fontsize=9)
    ax_f.axhline(0, color='#3a4060', lw=0.8)
    ax_f.set_ylim(-400, 400)
    ax_f.grid(True, alpha=0.12)
    linea_f_ctrl, = ax_f.plot([], [], color=COLOR_F_CTRL, lw=1.4, label='F_ctrl')
    linea_f_push, = ax_f.plot([], [], color=COLOR_F_PUSH, lw=1.5, label='F_empuje')
    linea_f_total, = ax_f.plot([], [], color=COLOR_F_TOTAL, lw=1.9, label='F_total')
    ax_f.legend(loc='upper right', frameon=False, fontsize=8, labelcolor='#c8d0e8')

    ax_mu_t = fig.add_axes([0.05, 0.07, 0.14, 0.09])
    ax_mu_dt = fig.add_axes([0.21, 0.07, 0.14, 0.09])
    ax_mu_f = fig.add_axes([0.37, 0.07, 0.14, 0.09])

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
    linea_cur_f, = ax_mu_f.plot([], [], color=COLOR_F_TOTAL, lw=1.2, ls='--')
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

    EMPUJE_N = 100.0
    EMPUJE_DT = 0.5
    empuje_fin = [0.0]

    def empujar_izq(event):
        estado['empuje'] = +EMPUJE_N
        empuje_fin[0] = estado['t'] + EMPUJE_DT

    def empujar_der(event):
        estado['empuje'] = -EMPUJE_N
        empuje_fin[0] = estado['t'] + EMPUJE_DT

    def reset(event):
        estado.update(crear_estado(theta_0_deg, v_0_deg))
        empuje_fin[0] = 0.0
        startup_force[0], startup_until[0] = crear_arranque(theta_0_deg, v_0_deg)
        nuevo_hist = reiniciar_historiales(theta_0_deg)
        for clave, valores in nuevo_hist.items():
            hist[clave].clear()
            hist[clave].extend(valores)

    btn_l.on_clicked(empujar_izq)
    btn_r.on_clicked(empujar_der)
    btn_rst.on_clicked(reset)

    _last_wall = [time.perf_counter()]

    def update(_):
        now = time.perf_counter()
        elapsed_wall = now - _last_wall[0]
        _last_wall[0] = now
        sim_time = min(elapsed_wall * VELOCIDAD, delta_t * 100)
        steps = max(1, int(sim_time / delta_t))

        for _ in range(steps):
            if estado['t'] > empuje_fin[0]:
                estado['empuje'] = 0.0
            th_norm = _normaliza_angulo_deg(np.degrees(estado['theta']))
            vd_deg = np.degrees(estado['v'])
            estado['f_ctrl'] = limitar_fuerza_control(
                estado['cart_x'], track_limit, EDGE_EPS, controlador_fam(th_norm, vd_deg)
            )
            f_start = startup_force[0] if estado['t'] <= startup_until[0] else 0.0
            estado['f_total'] = estado['f_ctrl'] + estado['empuje'] + f_start
            integrar_pendulo(estado, estado['f_total'], delta_t)
            actualizar_posicion_carro(estado, track_limit, EDGE_EPS)
            registrar_historial(hist, estado)

        cx = estado['cart_x']
        th = estado['theta']
        f_ctrl_act = estado['f_ctrl']
        f_push_act = estado['empuje']
        f_total_act = estado['f_total']
        t_act = estado['t']
        th_norm = _normaliza_angulo_deg(np.degrees(th))
        vd_deg = np.degrees(estado['v'])
        cart_v_act = estado['cart_v']
        f_start_act = startup_force[0] if estado['t'] <= startup_until[0] else 0.0

        actualizar_camara_animacion(ax_anim, track_limit, side_margin)
        carro.set_x(cx - CART_HALF_WIDTH)
        ind_x.set_data([cx], [-1.95])  # indicador de posición en eje X
        px = cx + L_vis * np.sin(th)
        py = L_vis * np.cos(th)
        pertiga.set_data([cx, px], [0, py])
        masa.set_data([px], [py])

        color_f = '#4a9eff' if f_total_act >= 0 else '#ff6b4a'
        alen = np.clip(f_total_act / 300 * ARROW_SCALE, -1.0, 1.0)
        x_tip = cx - alen
        flecha_ln.set_data([cx, x_tip], [-0.12, -0.12])
        flecha_ln.set_color(color_f)
        flecha_pt.set_data([x_tip], [-0.12])
        flecha_pt.set_color(color_f)
        flecha_pt.set_marker('<' if f_total_act >= 0 else '>')

        texto.set_text(
            f"t={t_act:.1f}s   x={cx:.2f} m   θ={th_norm:.1f}°\n"
            f"x'={cart_v_act:.2f} m/s   θ'={vd_deg:.1f}°/s   F_ctrl={f_ctrl_act:+.2f} N   F_emp={f_push_act:+.2f} N   F_arr={f_start_act:+.2f} N   F_total={f_total_act:+.2f} N"
        )

        t0 = max(0.0, t_act - VENTANA)
        ax_th.set_xlim(t0, t0 + VENTANA)
        ax_x.set_xlim(t0, t0 + VENTANA)
        ax_f.set_xlim(t0, t0 + VENTANA)
        linea_th.set_data(hist['t'], hist['theta'])
        linea_x.set_data(hist['t'], hist['x'])
        linea_f_ctrl.set_data(hist['t'], hist['f_ctrl'])
        linea_f_push.set_data(hist['t'], hist['f_push'])
        linea_f_total.set_data(hist['t'], hist['f_total'])

        linea_cur_t.set_data([th_norm, th_norm], [-0.05, 1.15])
        mu_t_cur = _mu_plot_scalar(th_norm, THETA_PLOT_SPECS)
        for i, pt in enumerate(puntos_t):
            pt.set_data([th_norm], [mu_t_cur[i]])

        vd_clip = np.clip(vd_deg, -90, 90)
        linea_cur_dt.set_data([vd_clip, vd_clip], [-0.05, 1.15])
        mu_dt_cur = _mu_plot_scalar(vd_clip, DTHETA_PLOT_SPECS)
        for i, pt in enumerate(puntos_dt):
            pt.set_data([vd_clip], [mu_dt_cur[i]])

        linea_cur_f.set_data([f_total_act, f_total_act], [-0.05, 1.15])

        return (carro, pertiga, masa, flecha_ln, flecha_pt, ind_x,
                linea_th, linea_x, linea_f_ctrl, linea_f_push, linea_f_total, texto,
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
    simular(delta_t=0.005, theta_0_deg=180.0, v_0_deg=0.0, track_limit=5)
