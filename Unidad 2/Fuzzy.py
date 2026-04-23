import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tkinter import Tk, Label, Entry, Button

# ============================================================
# 1. PARÁMETROS FÍSICOS
# ============================================================
m, M, l, g = 0.5, 1.5, 0.7, 9.81
dt_frame = 0.02
substeps = 10
dt = dt_frame / substeps

# ============================================================
# 2. FUNCIÓN DE PERTENENCIA TRIANGULAR
# ============================================================
def trimf(x, a, b, c):
    """Función triangular: a=inicio, b=pico, c=fin"""
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

# ============================================================
# 3. FUNCIONES PARA VISUALIZAR CONJUNTOS DIFUSOS (TRIANGULARES)
# ============================================================
def get_angle_membership_functions():
    x = np.linspace(-180, 180, 1000)
    sets = {
        'MuyNegativo':   trimf(x, -180, (-180-50)/2, -50),
        'MedioNegativo': trimf(x, -75, (-75-8)/2, -8),
        'PocoNegativo':  trimf(x, -25, (-25+2)/2, 2),
        'PocoPositivo':  trimf(x, -2, ((-2)+25)/2, 25),
        'MedioPositivo': trimf(x, 8, (8+75)/2, 75),
        'MuyPositivo':   trimf(x, 50, (50+180)/2, 180)
    }
    return x, sets

def get_velocity_membership_functions():
    x = np.linspace(-500, 500, 1000)
    sets = {
        'RapidoNegativo': trimf(x, -350, (-350-120)/2, -120),
        'MedioNegativo':  trimf(x, -170, (-170-40)/2, -40),
        'LentoNegativo':  trimf(x, -60, (-60-10)/2, -10),
        'Quieto':         trimf(x, -20, 0, 20),
        'LentoPositivo':  trimf(x, 10, (10+60)/2, 60),
        'MedioPositivo':  trimf(x, 40, (40+170)/2, 170),
        'RapidoPositivo': trimf(x, 120, (120+350)/2, 350)
    }
    return x, sets

# ============================================================
# 4. CONTROLADOR DIFUSO
# ============================================================
def fuzzy_control(theta_deg, omega_deg):
    theta_norm = (theta_deg + 180) % 360 - 180
    
    # Fuzzificación de Ángulo
    mu_ang = {
        'MuyNegativo':   trimf(theta_norm, -180, (-180-50)/2, -50),
        'MedioNegativo': trimf(theta_norm, -70, (-70-35)/2, -35),
        'PocoNegativo':  trimf(theta_norm, -25, (-25-9)/2, -9),
        'PocoPositivo':  trimf(theta_norm, 0, (0+25)/2, 25),
        'MedioPositivo': trimf(theta_norm, 8, (8+70)/2, 70),
        'MuyPositivo':   trimf(theta_norm, 60, (60+180)/2  , 180),
        'Demasiado':     1.0 if (70 <= abs(theta_deg) <= 180) else 0.0
    }
    
    # Fuzzificación de Velocidad
    mu_vel = {
        'RapidoNegativo': trimf(omega_deg, -350, (-350-120)/2, -120),
        'MedioNegativo':  trimf(omega_deg, -170, (-170-40)/2, -40),
        'LentoNegativo':  trimf(omega_deg, -60, (-60-10)/2, -10),
        'Quieto':         trimf(omega_deg, -20, 0, 20),
        'LentoPositivo':  trimf(omega_deg, 10, (10+60)/2, 60),
        'MedioPositivo':  trimf(omega_deg, 40, (40+170)/2, 170),
        'RapidoPositivo': trimf(omega_deg, 120, (120+350)/2, 350)
    }
    
    F = {
        'FMuyNegativa': -200.0, 'FMuyNegativaplus': -300.0,
        'FNegativa': -100.0, 'FNegativaplus': -200.0,
        'FMedioNegativa': -80.0, 'FMedioNegativaplus': -180.0,
        'FPocoNegativa': -50.0, 'FPocoNegativaplus': -150.0,
        'NoFuerza': 0.0,
        'FPocoPositiva': 50.0, 'FPocoPositivaplus': 150.0,
        'FMedioPositiva': 80.0, 'FMedioPositivaplus': 180.0,
        'FPositiva': 100.0, 'FPositivaplus': 200.0,
        'FMuyPositiva': 200.0, 'FMuyPositivaplus': 300.0,
        'subir': 400.0
    }
    
    FAM = {
        ('RapidoNegativo', 'MuyNegativo'): 'NoFuerza',
        ('RapidoNegativo', 'MedioNegativo'): 'FMuyPositiva',
        ('RapidoNegativo', 'PocoNegativo'): 'FPositiva',
        ('RapidoNegativo', 'PocoPositivo'): 'FMedioPositiva',
        ('RapidoNegativo', 'MedioPositivo'): 'NoFuerza',
        ('RapidoNegativo', 'MuyPositivo'): 'NoFuerza',
        ('RapidoNegativo', 'Demasiado'): 'FMuyPositivaplus',
        
        ('MedioNegativo', 'MuyNegativo'): 'FMuyPositiva',
        ('MedioNegativo', 'MedioNegativo'): 'FPositiva',
        ('MedioNegativo', 'PocoNegativo'): 'FMedioPositiva',
        ('MedioNegativo', 'PocoPositivo'): 'FPocoPositiva',
        ('MedioNegativo', 'MedioPositivo'): 'FPocoNegativa',
        ('MedioNegativo', 'MuyPositivo'): 'FPocoNegativa',
        ('MedioNegativo', 'Demasiado'): 'FPositivaplus',
        
        ('LentoNegativo', 'MuyNegativo'): 'FPositiva',
        ('LentoNegativo', 'MedioNegativo'): 'FMedioPositiva',
        ('LentoNegativo', 'PocoNegativo'): 'FPocoPositiva',
        ('LentoNegativo', 'PocoPositivo'): 'NoFuerza',
        ('LentoNegativo', 'MedioPositivo'): 'FMedioNegativa',
        ('LentoNegativo', 'MuyPositivo'): 'FMedioNegativa',
        ('LentoNegativo', 'Demasiado'): 'FPocoPositivaplus',
        
        ('LentoPositivo', 'MuyNegativo'): 'FMedioPositiva',
        ('LentoPositivo', 'MedioNegativo'): 'FMedioPositiva',
        ('LentoPositivo', 'PocoNegativo'): 'NoFuerza',
        ('LentoPositivo', 'PocoPositivo'): 'FPocoNegativa',
        ('LentoPositivo', 'MedioPositivo'): 'FMedioNegativa',
        ('LentoPositivo', 'MuyPositivo'): 'FNegativa',
        ('LentoPositivo', 'Demasiado'): 'FPocoNegativaplus',
        
        ('MedioPositivo', 'MuyNegativo'): 'FPocoPositiva',
        ('MedioPositivo', 'MedioNegativo'): 'FPocoPositiva',
        ('MedioPositivo', 'PocoNegativo'): 'FPocoNegativa',
        ('MedioPositivo', 'PocoPositivo'): 'FMedioNegativa',
        ('MedioPositivo', 'MedioPositivo'): 'FNegativa',
        ('MedioPositivo', 'MuyPositivo'): 'FMuyNegativa',
        ('MedioPositivo', 'Demasiado'): 'FNegativaplus',
        
        ('RapidoPositivo', 'MuyNegativo'): 'NoFuerza',
        ('RapidoPositivo', 'MedioNegativo'): 'NoFuerza',
        ('RapidoPositivo', 'PocoNegativo'): 'FMedioNegativa',
        ('RapidoPositivo', 'PocoPositivo'): 'FNegativa',
        ('RapidoPositivo', 'MedioPositivo'): 'FMuyNegativa',
        ('RapidoPositivo', 'MuyPositivo'): 'NoFuerza',
        ('RapidoPositivo', 'Demasiado'): 'FMuyNegativaplus',
        
        ('Quieto', 'MuyNegativo'): 'FMuyPositiva',
        ('Quieto', 'MedioNegativo'): 'FMedioPositiva',
        ('Quieto', 'PocoNegativo'): 'FPocoPositiva',
        ('Quieto', 'PocoPositivo'): 'FPocoNegativa',
        ('Quieto', 'MedioPositivo'): 'FMedioNegativa',
        ('Quieto', 'MuyPositivo'): 'FMuyNegativa',
        ('Quieto', 'Demasiado'): 'subir'
    }
    
    ang_activos = [t for t in mu_ang if mu_ang[t] > 0.01]
    vel_activos = [t for t in mu_vel if mu_vel[t] > 0.01]
    
    numerador = 0.0
    denominador = 0.0
    
    for vt in vel_activos:
        for at in ang_activos:
            regla = (vt, at)
            if regla in FAM:
                fuerza_term = FAM[regla]
                w = min(mu_vel[vt], mu_ang[at])
                numerador += w * F[fuerza_term]
                denominador += w
    
    if denominador > 1e-6:
        return numerador / denominador
    else:
        return 0.0

# [El resto de las funciones de dinámica, interfaz e inicio de simulación se mantienen igual que en tu archivo original]

# ... (Aquí sigue el código de angular_acceleration, la interfaz Tkinter y el bucle de animación)

# ============================================================
# 5. DINÁMICA DEL PÉNDULO
# ============================================================
def angular_acceleration(theta, omega, F):
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    F_term = (F - m * l * omega**2 * sin_t) / (M + m)
    numerator = g * sin_t + cos_t * F_term
    denominator = l * (4/3 - (m * cos_t**2) / (M + m))
    return numerator / denominator

# ============================================================
# 6. INTERFAZ DE ENTRADA
# ============================================================
start_p = {}

def start():
    global start_p
    start_p = {
        'th': float(e_th.get()),
        'om': float(e_om.get()),
        'x': float(e_x.get())
    }
    root.destroy()

root = Tk()
root.title("Péndulo Invertido - Control Difuso")
root.configure(bg='#2c3e50')

Label(root, text="Ángulo Inicial (°) [0° = arriba, + = izquierda]:", bg='#2c3e50', fg='white').pack(pady=5)
e_th = Entry(root)
e_th.insert(0, "70")
e_th.pack()

Label(root, text="Velocidad Inicial (°/s) [+ = antihorario]:", bg='#2c3e50', fg='white').pack(pady=5)
e_om = Entry(root)
e_om.insert(0, "0")
e_om.pack()

Label(root, text="Posición X inicial (m):", bg='#2c3e50', fg='white').pack(pady=5)
e_x = Entry(root)
e_x.insert(0, "0")
e_x.pack()

Button(root, text="Lanzar Simulación", command=start, bg='#e74c3c', fg='white').pack(pady=10)

root.mainloop()

# ============================================================
# 7. SIMULACIÓN CON VISUALIZACIÓN DE PERTENENCIAS
# ============================================================
state = [start_p['x'], 0.0, np.radians(start_p['th']), np.radians(start_p['om'])]

history = {'t': [], 'th': [], 'f': [], 'omega': []}

# Configuración de gráficas
fig = plt.figure(figsize=(16, 10), facecolor='#121417')
gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])

# Péndulo
ax_sim = fig.add_subplot(gs[:, 0], facecolor='#1e2126')
ax_sim.set_aspect('equal')
ax_sim.set_xlim(-1.5, 1.5)
ax_sim.set_ylim(-0.5, 1.2)
ax_sim.set_xlabel("Posición X (m)", color='white')
ax_sim.set_ylabel("Altura Y (m)", color='white')
ax_sim.set_title("Péndulo Invertido", color='white')
ax_sim.tick_params(colors='white')
ax_sim.grid(alpha=0.3)

# Ángulo
ax_th = fig.add_subplot(gs[0, 1], facecolor='#1e2126')
ax_th.set_ylim(-180, 180)
ax_th.set_xlabel("Tiempo (s)", color='white')
ax_th.set_ylabel("Ángulo (°)", color='white')
ax_th.set_title("Ángulo del Péndulo", color='white')
ax_th.tick_params(colors='white')
ax_th.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax_th.axhline(y=70, color='orange', linestyle=':', alpha=0.5, label='Límite Demasiado')
ax_th.axhline(y=-70, color='orange', linestyle=':', alpha=0.5)
ax_th.legend()
ax_th.grid(alpha=0.3)

# Fuerza
ax_f = fig.add_subplot(gs[1, 1], facecolor='#1e2126')
ax_f.set_ylim(-450, 450)
ax_f.set_xlabel("Tiempo (s)", color='white')
ax_f.set_ylabel("Fuerza (N)", color='white')
ax_f.set_title("Fuerza Aplicada", color='white')
ax_f.tick_params(colors='white')
ax_f.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax_f.grid(alpha=0.3)

# Funciones de pertenencia - Ángulo
ax_mf_ang = fig.add_subplot(gs[0, 2], facecolor='#1e2126')
ax_mf_ang.set_xlim(-180, 180)
ax_mf_ang.set_ylim(-0.1, 1.1)
ax_mf_ang.set_xlabel("Ángulo (°)", color='white')
ax_mf_ang.set_ylabel("Pertenencia", color='white')
ax_mf_ang.set_title("Conjuntos Difusos - Ángulo", color='white')
ax_mf_ang.tick_params(colors='white')
ax_mf_ang.grid(alpha=0.3)

# Funciones de pertenencia - Velocidad
ax_mf_vel = fig.add_subplot(gs[1, 2], facecolor='#1e2126')
ax_mf_vel.set_xlim(-400, 400)
ax_mf_vel.set_ylim(-0.1, 1.1)
ax_mf_vel.set_xlabel("Velocidad (°/s)", color='white')
ax_mf_vel.set_ylabel("Pertenencia", color='white')
ax_mf_vel.set_title("Conjuntos Difusos - Velocidad", color='white')
ax_mf_vel.tick_params(colors='white')
ax_mf_vel.grid(alpha=0.3)

# Elementos visuales
cart = plt.Rectangle((-0.25, -0.1), 0.5, 0.2, fc='#3498db', zorder=3)
ax_sim.add_patch(cart)
pole, = ax_sim.plot([], [], 'o-', lw=4, color='#e74c3c', markersize=10, zorder=4)
f_arrow = ax_sim.quiver(0, 0, 0, 0, color='#00ff88', scale=800, width=0.015)
line_th, = ax_th.plot([], [], '#4e9af1', lw=2)
line_f, = ax_f.plot([], [], '#f05454', lw=2)
txt = ax_sim.text(0.02, 0.95, '', transform=ax_sim.transAxes, color='white', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Graficar conjuntos difusos estáticos
ang_x, ang_sets = get_angle_membership_functions()
vel_x, vel_sets = get_velocity_membership_functions()

colors_ang = ['#ff6b6b', '#feca57', '#ff9ff3', '#48dbfb', '#1dd1a1', '#5f27cd']
colors_vel = ['#ff6b6b', '#feca57', '#ff9ff3', '#48dbfb', '#1dd1a1', '#5f27cd', '#00cec9']

for i, (name, values) in enumerate(ang_sets.items()):
    ax_mf_ang.plot(ang_x, values, color=colors_ang[i % len(colors_ang)], label=name, alpha=0.8, linewidth=1.5)

for i, (name, values) in enumerate(vel_sets.items()):
    ax_mf_vel.plot(vel_x, values, color=colors_vel[i % len(colors_vel)], label=name, alpha=0.8, linewidth=1.5)

ax_mf_ang.legend(loc='upper right', fontsize=7, facecolor='#1e2126', labelcolor='white')
ax_mf_vel.legend(loc='upper right', fontsize=7, facecolor='#1e2126', labelcolor='white')

# Líneas verticales para valores actuales (se actualizarán)
ang_line = ax_mf_ang.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
vel_line = ax_mf_vel.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1)

def update(frame):
    global state
    
    for _ in range(substeps):
        x, vx, theta, omega = state
        theta_deg = np.degrees(theta)
        omega_deg = np.degrees(omega)
        
        F_user = fuzzy_control(theta_deg, omega_deg)
        alpha = angular_acceleration(theta, omega, F_user)
        
        omega_new = omega + alpha * dt
        theta_new = theta + omega * dt + 0.5 * alpha * dt * dt
        
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        acc_x = (F_user - m * l * (omega**2 * sin_t - alpha * cos_t)) / (M + m)
        
        vx_new = vx + acc_x * dt
        x_new = x + vx * dt + 0.5 * acc_x * dt * dt
        
        state = [x_new, vx_new, theta_new, omega_new]
    
    t = frame * dt_frame
    th_deg = (np.degrees(state[2]) + 180) % 360 - 180
    omega_deg = np.degrees(state[3])
    
    history['t'].append(t)
    history['th'].append(th_deg)
    history['f'].append(F_user)
    history['omega'].append(omega_deg)
    
    # Actualizar péndulo
    ax_sim.set_xlim(state[0] - 1.5, state[0] + 1.5)
    cart.set_xy((state[0] - 0.25, -0.1))
    px = state[0] - l * np.sin(state[2])
    py = 0.1 + l * np.cos(state[2])
    pole.set_data([state[0], px], [0.1, py])
    f_arrow.set_offsets([state[0], 0.2])
    f_arrow.set_UVC(F_user / 2.0, 0)
    
    # Actualizar gráficas
    line_th.set_data(history['t'], history['th'])
    ax_th.set_xlim(max(0, t - 5), t + 0.5)
    line_f.set_data(history['t'], history['f'])
    ax_f.set_xlim(max(0, t - 5), t + 0.5)
    
    # Actualizar líneas verticales de pertenencia
    ang_line.set_xdata([th_deg, th_deg])
    vel_line.set_xdata([omega_deg, omega_deg])
    
    # Resaltar conjuntos activos (opcional: cambiar opacidad)
    en_demasiado = 70 <= th_deg <= 290
    txt.set_text(f"Tiempo: {t:.2f} s\nÁngulo: {th_deg:.1f}°\nVelocidad: {omega_deg:.1f} °/s\nFuerza: {F_user:.1f} N\nDemasiado: {'SÍ' if en_demasiado else 'NO'}")
    
    return cart, pole, f_arrow, line_th, line_f, txt, ang_line, vel_line

ani = animation.FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()