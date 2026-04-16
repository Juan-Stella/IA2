# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import matplotlib

if "--no-gui" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.widgets as widgets
from scipy import constants

M = 2.0
m = 1.0
l = 1.0
g = constants.g

CONJUNTOS = ["NGG", "NG", "NP", "Z", "PP", "PG", "PGG"]
IDX = {n: i for i, n in enumerate(CONJUNTOS)}
COLORES = ["#e05252", "#e08c52", "#d4c84a", "#52c47a", "#52a8e0", "#7a6ee0", "#c052e0"]

THETA_SPECS = {
    "NGG": ("left", -180.0, -120.0),
    "NG": ("tri", -130.0, -110.0, -70.0),
    "NP": ("tri", -80.0, -50.0, -20.0),
    "Z": ("tri", -30.0, 0.0, 30.0),
    "PP": ("tri", 20.0, 50.0, 80.0),
    "PG": ("tri", 70.0, 110.0, 130.0),
    "PGG": ("right", 120.0, 180.0),
}
DTHETA_SPECS = {
    "NGG": ("left", -90.0, -60.0),
    "NG": ("tri", -90.0, -60.0, -30.0),
    "NP": ("tri", -60.0, -30.0, 0.0),
    "Z": ("tri", -30.0, 0.0, 30.0),
    "PP": ("tri", 0.0, 30.0, 60.0),
    "PG": ("tri", 30.0, 60.0, 90.0),
    "PGG": ("right", 60.0, 90.0),
}
F_SPECS = {
    "NGG": ("left", -300.0, -200.0),
    "NG": ("tri", -300.0, -200.0, -100.0),
    "NP": ("tri", -200.0, -100.0, 0.0),
    "Z": ("tri", -100.0, 0.0, 100.0),
    "PP": ("tri", 0.0, 100.0, 200.0),
    "PG": ("tri", 100.0, 200.0, 300.0),
    "PGG": ("right", 200.0, 300.0),
}
F_UNIVERSE = np.linspace(-300.0, 300.0, 1201)

FAM = np.array([
    [IDX["PGG"], IDX["PGG"], IDX["PGG"], IDX["PP"],  IDX["PP"],  IDX["NP"],  IDX["NP"]],
    [IDX["PGG"], IDX["PGG"], IDX["PG"],  IDX["PP"],  IDX["Z"],   IDX["NP"],  IDX["NG"]],
    [IDX["PGG"], IDX["PGG"], IDX["PP"],  IDX["PP"],  IDX["NP"],  IDX["NP"],  IDX["NGG"]],
    [IDX["PGG"], IDX["PG"],  IDX["PP"],  IDX["Z"],   IDX["NP"],  IDX["NG"],  IDX["NGG"]],
    [IDX["PGG"], IDX["PP"],  IDX["PP"],  IDX["NP"],  IDX["NP"],  IDX["NGG"], IDX["NGG"]],
    [IDX["PG"],  IDX["PP"],  IDX["Z"],   IDX["NP"],  IDX["NG"],  IDX["NGG"], IDX["NGG"]],
    [IDX["PP"],  IDX["PP"],  IDX["NP"],  IDX["NP"],  IDX["NGG"], IDX["NGG"], IDX["NGG"]],
])


def normalizar_angulo(theta_deg):
    return (theta_deg + 180.0) % 360.0 - 180.0


def mu_left(x, full_until, zero_at):
    if x <= full_until:
        return 1.0
    if x >= zero_at:
        return 0.0
    return (zero_at - x) / (zero_at - full_until)


def mu_right(x, zero_at, full_from):
    if x <= zero_at:
        return 0.0
    if x >= full_from:
        return 1.0
    return (x - zero_at) / (full_from - zero_at)


def mu_tri(x, left, center, right):
    if x <= left or x >= right:
        return 0.0
    if x == center:
        return 1.0
    if x < center:
        return (x - left) / (center - left)
    return (right - x) / (right - center)


def mu_eval(x, spec):
    if spec[0] == "left":
        return mu_left(x, spec[1], spec[2])
    if spec[0] == "right":
        return mu_right(x, spec[1], spec[2])
    return mu_tri(x, spec[1], spec[2], spec[3])


def mu_scalar(x, specs):
    return np.array([mu_eval(x, specs[n]) for n in CONJUNTOS], dtype=float)


def mu_vector(x_arr, specs):
    out = np.zeros((len(CONJUNTOS), len(x_arr)))
    for k, x in enumerate(x_arr):
        out[:, k] = mu_scalar(float(x), specs)
    return out


MU_F_BASE = mu_vector(F_UNIVERSE, F_SPECS)


def controlador_fam(theta_deg, dtheta_deg):
    theta_deg = np.clip(theta_deg, -180.0, 180.0)
    dtheta_deg = np.clip(dtheta_deg, -90.0, 90.0)
    mu_t = mu_scalar(theta_deg, THETA_SPECS)
    mu_dt = mu_scalar(dtheta_deg, DTHETA_SPECS)
    alpha = np.minimum(mu_dt[:, None], mu_t[None, :])
    mu_salida = np.zeros(len(F_UNIVERSE))
    for i in range(7):
        for j in range(7):
            a = alpha[i, j]
            if a > 1e-9:
                np.maximum(mu_salida, np.minimum(a, MU_F_BASE[FAM[i, j]]), out=mu_salida)
    area = np.trapezoid(mu_salida, F_UNIVERSE)
    if area < 1e-9:
        return 0.0
    f = -float(np.trapezoid(F_UNIVERSE * mu_salida, F_UNIVERSE) / area)
    
    return f


def calcula_aceleracion(theta_rad, v_rad, f_N):
    num = g * np.sin(theta_rad) + np.cos(theta_rad) * ((-f_N - m * l * v_rad**2 * np.sin(theta_rad)) / (M + m))
    den = l * (4.0 / 3.0 - m * np.cos(theta_rad) ** 2 / (M + m))
    return num / den


def graficar_particiones():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#0f1117")
    cfgs = [(axes[0], THETA_SPECS, "theta [deg]", -180, 180), (axes[1], DTHETA_SPECS, "theta' [deg/s]", -90, 90), (axes[2], F_SPECS, "F [N]", -300, 300)]
    for ax, specs, xlabel, xmin, xmax in cfgs:
        ax.set_facecolor("#1a1d2e")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.08)
        ax.axhline(0, color="#2a2d3a", lw=0.8)
        ax.axhline(1, color="#2a2d3a", lw=0.5, ls=":")
        ax.axvline(0, color="#2a2d3a", lw=0.5, ls="--")
        ax.tick_params(colors="#8890a8", labelsize=8)
        ax.set_xlabel(xlabel, color="#8890a8")
        ax.set_ylabel("mu", color="#8890a8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2d3a")
        x = np.linspace(xmin, xmax, 1000)
        mu_all = mu_vector(x, specs)
        for i, nombre in enumerate(CONJUNTOS):
            ax.plot(x, mu_all[i], color=COLORES[i], lw=2)
            ax.text(0.02 + 0.12 * i, 0.95, nombre, transform=ax.transAxes, color=COLORES[i], fontsize=7, va="top")
    plt.tight_layout()
    plt.show()


def simular(delta_t=0.005, theta_0_deg=170.0, v_0_deg=0.0):
    estado = {"theta": np.radians(theta_0_deg), "v": np.radians(v_0_deg), "t": 0.0, "cart_x": 0.0, "empuje": 0.0}
    hist_t, hist_theta, hist_f = [0.0], [normalizar_angulo(theta_0_deg)], [0.0]
    velocidad, l_vis, ventana = 1.0, 1.5, 10.0

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#0f1117")
    ax_anim = fig.add_axes([0.02, 0.12, 0.52, 0.84])
    ax_th = fig.add_axes([0.58, 0.55, 0.40, 0.38])
    ax_f = fig.add_axes([0.58, 0.12, 0.40, 0.35])
    for ax in (ax_anim, ax_th, ax_f):
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#8890a8", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2d3a")

    ax_anim.set_xlim(-3.5, 3.5)
    ax_anim.set_ylim(-2.0, 2.0)
    ax_anim.set_aspect("equal")
    ax_anim.axhline(0, color="#2a2d3a", lw=1)
    ax_anim.axvline(0, color="#2a2d3a", lw=0.5, ls="--")
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])
    ax_anim.set_title("Pendulo invertido - FAM 7x7", color="#c8d0e8")
    carro = patches.FancyBboxPatch((-0.4, -0.18), 0.8, 0.32, boxstyle="round,pad=0.02", facecolor="#1e2235", edgecolor="#4a9eff", lw=1.5)
    ax_anim.add_patch(carro)
    pertiga, = ax_anim.plot([], [], color="#a78bfa", lw=3)
    masa, = ax_anim.plot([], [], "o", color="#a78bfa", ms=14, mec="#7c5cfc", mew=2)
    flecha_ln, = ax_anim.plot([], [], lw=2.5)
    flecha_pt, = ax_anim.plot([], [], ms=9)
    texto = ax_anim.text(-3.3, 1.8, "", color="#c8d0e8", fontsize=9, fontfamily="monospace")

    ax_th.set_ylabel("theta [deg]", color="#4a9eff")
    ax_th.set_ylim(-200, 200)
    ax_th.grid(True, alpha=0.12)
    linea_th, = ax_th.plot([], [], color="#4a9eff", lw=1.2)
    ax_f.set_ylabel("F [N]", color="#ff6b4a")
    ax_f.set_xlabel("t [s]", color="#8890a8")
    ax_f.set_ylim(-400, 400)
    ax_f.grid(True, alpha=0.12)
    linea_f, = ax_f.plot([], [], color="#ff6b4a", lw=1.2)

    ax_mu_t = fig.add_axes([0.29, 0.13, 0.12, 0.10])
    ax_mu_dt = fig.add_axes([0.29, 0.24, 0.12, 0.10])
    for ax_i, specs, xmin, xmax, lbl in [(ax_mu_t, THETA_SPECS, -180, 180, "theta"), (ax_mu_dt, DTHETA_SPECS, -90, 90, "theta'")]:
        ax_i.set_facecolor("#0f1117")
        ax_i.set_xlim(xmin, xmax)
        ax_i.set_ylim(-0.05, 1.15)
        ax_i.tick_params(colors="#555", labelsize=5, length=2, pad=1)
        for sp in ax_i.spines.values():
            sp.set_edgecolor("#2a2d3a")
        x = np.linspace(xmin, xmax, 400)
        mu_all = mu_vector(x, specs)
        for i in range(7):
            ax_i.plot(x, mu_all[i], color=COLORES[i], lw=1.0, alpha=0.6)
        ax_i.text(0.03, 0.84, lbl, transform=ax_i.transAxes, color="#8890a8", fontsize=6)
    linea_cur_t, = ax_mu_t.plot([], [], color="white", lw=0.8, ls="--")
    linea_cur_dt, = ax_mu_dt.plot([], [], color="white", lw=0.8, ls="--")
    puntos_t = [ax_mu_t.plot([], [], "o", color=COLORES[i], ms=3)[0] for i in range(7)]
    puntos_dt = [ax_mu_dt.plot([], [], "o", color=COLORES[i], ms=3)[0] for i in range(7)]

    ax_btn_l = fig.add_axes([0.58, 0.03, 0.12, 0.06])
    ax_btn_r = fig.add_axes([0.72, 0.03, 0.12, 0.06])
    ax_btn_rst = fig.add_axes([0.87, 0.03, 0.10, 0.06])
    btn_l = widgets.Button(ax_btn_l, "<- Empujar", color="#1e2235", hovercolor="#2a3050")
    btn_r = widgets.Button(ax_btn_r, "Empujar ->", color="#1e2235", hovercolor="#2a3050")
    btn_rst = widgets.Button(ax_btn_rst, "Reset", color="#1e2235", hovercolor="#2a3050")
    for btn, c in [(btn_l, "#4a9eff"), (btn_r, "#ff6b4a"), (btn_rst, "#8890a8")]:
        for sp in btn.ax.spines.values():
            sp.set_edgecolor(c)
        btn.label.set_color(c)

    empuje_n, empuje_dt, empuje_fin = 250.0, 0.15, [0.0]

    def empujar_izq(event):
        estado["empuje"] = -empuje_n
        empuje_fin[0] = estado["t"] + empuje_dt

    def empujar_der(event):
        estado["empuje"] = empuje_n
        empuje_fin[0] = estado["t"] + empuje_dt

    def reset(event):
        estado["theta"] = np.radians(theta_0_deg)
        estado["v"] = np.radians(v_0_deg)
        estado["t"] = 0.0
        estado["cart_x"] = 0.0
        estado["empuje"] = 0.0
        empuje_fin[0] = 0.0
        hist_t[:] = [0.0]
        hist_theta[:] = [normalizar_angulo(theta_0_deg)]
        hist_f[:] = [0.0]

    btn_l.on_clicked(empujar_izq)
    btn_r.on_clicked(empujar_der)
    btn_rst.on_clicked(reset)

    _last_wall = [time.perf_counter()]

    def update(_):
        now = time.perf_counter()
        sim_time = min((now - _last_wall[0]) * velocidad, delta_t * 10)
        _last_wall[0] = now
        steps = max(1, int(sim_time / delta_t))
        for _ in range(steps):
            if estado["t"] > empuje_fin[0]:
                estado["empuje"] = 0.0
            th_norm = normalizar_angulo(np.degrees(estado["theta"]))
            vd_deg = np.degrees(estado["v"])
            f_ctrl = controlador_fam(th_norm, vd_deg)
            f_total = f_ctrl + estado["empuje"]
            a = calcula_aceleracion(estado["theta"], estado["v"], f_total)
            v_prev = estado["v"]
            estado["theta"] += v_prev * delta_t + 0.5 * a * delta_t**2
            estado["v"] += a * delta_t
            estado["t"] += delta_t
            estado["cart_x"] = np.clip(estado["cart_x"] - f_total * delta_t * 0.005, -2.8, 2.8)
            hist_t.append(estado["t"])
            hist_theta.append(normalizar_angulo(np.degrees(estado["theta"])))
            hist_f.append(f_total)

        cx = estado["cart_x"]
        th = estado["theta"]
        th_norm = normalizar_angulo(np.degrees(th))
        vd_deg = np.degrees(estado["v"])
        f_ctrl_act = hist_f[-1] - estado["empuje"]
        carro.set_x(cx - 0.4)
        px, py = cx + l_vis * np.sin(th), l_vis * np.cos(th)
        pertiga.set_data([cx, px], [0, py])
        masa.set_data([px], [py])
        color_f = "#4a9eff" if f_ctrl_act >= 0 else "#ff6b4a"
        x_tip = cx - np.clip(f_ctrl_act / 300.0, -1.0, 1.0)
        flecha_ln.set_data([cx, x_tip], [-0.12, -0.12])
        flecha_ln.set_color(color_f)
        flecha_pt.set_data([x_tip], [-0.12])
        flecha_pt.set_color(color_f)
        flecha_pt.set_marker("<" if f_ctrl_act >= 0 else ">")
        extra = f"  [EMPUJE {estado['empuje']:+.0f}N]" if estado["empuje"] != 0 else ""
        texto.set_text(f"t={estado['t']:.1f}s   theta={th_norm:.1f} deg{extra}\ntheta'={vd_deg:.1f} deg/s   F_ctrl={f_ctrl_act:.0f} N")
        t0 = max(0.0, estado["t"] - ventana)
        ax_th.set_xlim(t0, t0 + ventana)
        ax_f.set_xlim(t0, t0 + ventana)
        linea_th.set_data(hist_t, hist_theta)
        linea_f.set_data(hist_t, hist_f)
        linea_cur_t.set_data([th_norm, th_norm], [-0.05, 1.15])
        linea_cur_dt.set_data([np.clip(vd_deg, -90, 90), np.clip(vd_deg, -90, 90)], [-0.05, 1.15])
        mu_t_cur = mu_scalar(th_norm, THETA_SPECS)
        mu_dt_cur = mu_scalar(np.clip(vd_deg, -90, 90), DTHETA_SPECS)
        for i in range(7):
            puntos_t[i].set_data([th_norm], [mu_t_cur[i]])
            puntos_dt[i].set_data([np.clip(vd_deg, -90, 90)], [mu_dt_cur[i]])
        return carro, pertiga, masa, flecha_ln, flecha_pt, linea_th, linea_f, texto, linea_cur_t, linea_cur_dt, *puntos_t, *puntos_dt

    ani = animation.FuncAnimation(fig, update, frames=None, interval=20, blit=False, cache_frame_data=False)
    return fig, ani


if __name__ == "__main__":
    if "--no-gui" in sys.argv:
        print("Chequeo rapido del controlador")
        for theta in [0, 10, 20, 33, 45, 60]:
            print(f"theta={theta:5.1f} deg -> F={controlador_fam(theta, 0.0):8.3f} N")
    else:
        graficar_particiones()
        fig, ani = simular(delta_t=0.005, theta_0_deg=170.0, v_0_deg=0.0)
        plt.show()
