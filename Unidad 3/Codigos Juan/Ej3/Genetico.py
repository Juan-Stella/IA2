import A_Star as A
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import Temple

ordenes = []

with open("ordenes.csv", "r") as archivo:
    lector = csv.reader(archivo)
    for fila in lector:
        if fila:  # ignorar filas vacías al final del CSV
            ordenes.append([int(x) for x in fila])

MAPA_ESTANTERIAS = {
    1: (1, 2),  2: (1, 3),   3: (2, 2),  4: (2, 3),
    5: (3, 2),  6: (3, 3),   7: (4, 2),  8: (4, 3),
    9: (1, 6),  10: (1, 7),  11: (2, 6), 12: (2, 7),
    13: (3, 6), 14: (3, 7),  15: (4, 6), 16: (4, 7),
    17: (1, 10), 18: (1, 11), 19: (2, 10), 20: (2, 11),
    21: (3, 10), 22: (3, 11), 23: (4, 10), 24: (4, 11),
    25: (6, 2), 26: (6, 3),  27: (7, 2), 28: (7, 3),
    29: (8, 2), 30: (8, 3),  31: (9, 2), 32: (9, 3),
    33: (6, 6), 34: (6, 7),  35: (7, 6), 36: (7, 7),
    37: (8, 6), 38: (8, 7),  39: (9, 6), 40: (9, 7),
    41: (6, 10), 42: (6, 11), 43: (7, 10), 44: (7, 11),
    45: (8, 10), 46: (8, 11), 47: (9, 10), 48: (9, 11),
}


def generar_poblacion(tamano_poblacion):
    # En la imagen, el "Modelo de individuo" es un arreglo de 48 elementos
    # que representa dónde va cada producto en el almacén.
    poblacion = []
    productos = list(range(1, 49)) # IDs de los 48 productos
    
    for i in range(tamano_poblacion):
        individuo = productos.copy()
        random.shuffle(individuo) # Generamos una configuración aleatoria
        poblacion.append(individuo)
    return poblacion

ENTRADA = (5, 0)  # Fila, columna de la estación de carga

def fitness(poblacion, frecuencias, dist_slots):
    """Costo = suma(freq[producto] * distancia_slot_a_entrada).
    Recibe frecuencias y dist_slots pre-calculados para evitar repetirlos."""
    costos = []
    for individuo in poblacion:
        costo = 0
        for slot_idx, producto in enumerate(individuo):
            slot_id = slot_idx + 1
            costo += frecuencias.get(producto, 0) * dist_slots[slot_id]
        costos.append(costo)
    return costos

def seleccion(poblacion, costos, n):
    seleccionados = []
    costos_sel = []
    # Trabajamos con copias para poder remover candidatos ya elegidos
    candidatos = list(zip(costos, poblacion))
    for i in range(n):
        idx = int(np.argmin([c for c, _ in candidatos]))
        costo, individuo = candidatos.pop(idx)
        seleccionados.append(individuo)
        costos_sel.append(costo)
    return seleccionados, costos_sel

def crossover(seleccionados, costos_sel, n):
    poblacion_nueva = []
    # Invert costs so lower cost = higher selection probability
    inversos = [1 / c if c > 0 else 1.0 for c in costos_sel]
    total = sum(inversos)
    probabilidad = [v / total for v in inversos]
    for i in range(n // 2):
        padre1 = random.choices(seleccionados, weights=probabilidad, k=1)[0]
        padre2 = random.choices(seleccionados, weights=probabilidad, k=1)[0]
        hijo1 = ox_crossover(padre1, padre2)
        hijo2 = ox_crossover(padre2, padre1)
        poblacion_nueva.append(hijo1)
        poblacion_nueva.append(hijo2)
    # Pad con padres seleccionados para mantener tamano n
    for padre in seleccionados:
        poblacion_nueva.append(padre)
    return poblacion_nueva[:n]

def ox_crossover(padre1, padre2):
    """Cruce de orden (OX): preserva segmento de padre1, rellena el resto
    en el orden en que aparecen en padre2, comenzando desde el segundo
    punto de corte (circular), sin duplicar valores."""
    size = len(padre1)
    a, b = sorted(random.sample(range(size), 2))
    hijo = [None] * size
    hijo[a:b] = padre1[a:b]          # segmento preservado de padre1
    seg = set(padre1[a:b])
    # Secuencia de padre2 circular desde posicion b, excluyendo el segmento
    seq = [padre2[(b + i) % size] for i in range(size) if padre2[(b + i) % size] not in seg]
    # Rellenar hijo desde posicion b circularmente
    for i in range(size - (b - a)):
        hijo[(b + i) % size] = seq[i]
    return hijo
        
def mutacion(poblacion_nueva, prob_mutacion):
    for i in range(len(poblacion_nueva)):
        if random.random() < prob_mutacion:
            # Seleccionar una cantidad aleatoria de genes para mutar (por ejemplo, entre 2 y len(hijos[i])//2)
            num_genes = random.randint(2, max(2, len(poblacion_nueva[i])//2))
            posiciones = random.sample(range(len(poblacion_nueva[i])), num_genes)
            valores = [poblacion_nueva[i][p] for p in posiciones]
            random.shuffle(valores)
            for j in range(num_genes):
                poblacion_nueva[i][posiciones[j]] = valores[j]
    return poblacion_nueva

def precalcular_distancias(astar):
    dist = np.zeros((49, 49), dtype=int)
    for i in range(49):
        coord_i = ENTRADA if i == 0 else MAPA_ESTANTERIAS[i]
        for j in range(i+1, 49):
            coord_j = ENTRADA if j == 0 else MAPA_ESTANTERIAS[j]
            camino = astar.busqueda(coord_i, coord_j)
            d = len(camino) - 1 if camino else 999
            dist[i, j] = d
            dist[j, i] = d
    return dist

def ruteo_orden_nn(orden, dist_matrix, layout):
    prod_a_slot = {p: slot+1 for slot, p in enumerate(layout)}
    actual = 0
    costo_total = 0
    por_visitar = set(prod_a_slot[p] for p in orden)
    while por_visitar:
        siguiente = min(por_visitar, key=lambda s: dist_matrix[actual, s])
        costo_total += dist_matrix[actual, siguiente]
        actual = siguiente
        por_visitar.remove(siguiente)
    costo_total += dist_matrix[actual, 0]
    return costo_total

def costo_layout_real(layout, ordenes, dist_matrix):
    costo = 0
    for orden in ordenes:
        costo += ruteo_orden_nn(orden, dist_matrix, layout)
    return costo


def visualizar_almacen(mejor_solucion, ordenes, titulo_extra=""):
    # --- Frecuencia de cada producto en las órdenes ---
    frecuencias = {}
    for orden in ordenes:
        for prod in orden:
            frecuencias[prod] = frecuencias.get(prod, 0) + 1

    FILAS, COLS = 11, 13

    grilla_freq = np.full((FILAS, COLS), np.nan)
    grilla_prod = {}

    for slot_idx, producto in enumerate(mejor_solucion):
        shelf_id = slot_idx + 1
        fila, col = MAPA_ESTANTERIAS[shelf_id]
        grilla_freq[fila, col] = frecuencias.get(producto, 0)
        grilla_prod[(fila, col)] = producto

    # --- Figura ---
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    ax.imshow(np.zeros((FILAS, COLS)), cmap='Greys', vmin=0, vmax=1, aspect='equal', alpha=0.15)

    masked = np.ma.masked_invalid(grilla_freq)
    heatmap = ax.imshow(
        masked,
        cmap='YlOrRd',
        aspect='equal',
        vmin=0,
        vmax=np.nanmax(grilla_freq) if np.nanmax(grilla_freq) > 0 else 1,
        interpolation='nearest',
        alpha=0.9
    )

    for (fila, col), producto in grilla_prod.items():
        freq = frecuencias.get(producto, 0)
        ax.text(col, fila - 0.18, f'P{producto}',
                ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')
        ax.text(col, fila + 0.22, f'f={freq}',
                ha='center', va='center', fontsize=5.5, color='#ffffffcc')

    ax.plot(0, 5, marker='*', markersize=18, color='#00d4ff', zorder=5)
    ax.text(0.5, 5, 'Entrada', ha='left', va='center', fontsize=8, color='#00d4ff', fontweight='bold')

    for shelf_id, (fila, col) in MAPA_ESTANTERIAS.items():
        rect = plt.Rectangle((col - 0.5, fila - 0.5), 1, 1,
                              linewidth=0.8, edgecolor='#ffffff55', facecolor='none')
        ax.add_patch(rect)

    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Frecuencia de pedidos', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(FILAS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(FILAS))
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.set_xlabel('Columna', color='#aaaaaa', fontsize=10)
    ax.set_ylabel('Fila', color='#aaaaaa', fontsize=10)
    ax.set_title(f'Almacén — Configuración óptima {titulo_extra}\\nMapa de calor: frecuencia de pedidos por producto',
                 color='white', fontsize=13, pad=14)
    ax.spines[:].set_color('#444466')

    plt.tight_layout()
    plt.show()

def main():
    grafo = np.zeros((11, 13), dtype=int)

    for row_start in [1, 6]:
        for col_start in [2, 6, 10]:
            for r in range(row_start, row_start + 4):
                for c in range(col_start, col_start + 2):
                    grafo[r, c] = 1

    astar = A.A_Star(grafo)

    # --- Pre-calcular una sola vez (48 búsquedas A* en total) ---
    frecuencias = {}
    for orden in ordenes:
        for prod in orden:
            frecuencias[prod] = frecuencias.get(prod, 0) + 1

    dist_slots = {}
    for slot_id, (fila, col) in MAPA_ESTANTERIAS.items():
        camino = astar.busqueda(ENTRADA, (fila, col))
        dist_slots[slot_id] = len(camino) - 1 if camino else 999

    poblacion = generar_poblacion(100)

    historial_costos = []
    costo_inicial = min(fitness(poblacion, frecuencias, dist_slots))
    for i in range(1000):
        costos = fitness(poblacion, frecuencias, dist_slots)
        historial_costos.append(min(costos))
        seleccionados, costos_sel = seleccion(poblacion, costos, 20)
        poblacion_nueva = crossover(seleccionados, costos_sel, 100)
        poblacion_nueva = mutacion(poblacion_nueva, 0.1)
        poblacion = poblacion_nueva
    
    costos = fitness(poblacion, frecuencias, dist_slots)
    mejor_idx = np.argmin(costos)
    mejor_solucion_ga = poblacion[mejor_idx]
    print("Costo heurístico GA (Trivial):", costos[mejor_idx])

    print("\nCalculando matriz de distancias reales (A* para todos los pares)...")
    dist_matrix = precalcular_distancias(astar)

    costo_real_ga = costo_layout_real(mejor_solucion_ga, ordenes, dist_matrix)
    print(f"-> Costo RUTEO REAL del layout Trivial (Genético): {costo_real_ga} pasos totales")

    print("\nOptimizando layout con Temple Simulado (Considerando agrupación de órdenes)...")
    layout_inicial = list(range(1, 49))
    random.shuffle(layout_inicial)
    fn_costo = lambda layout: costo_layout_real(layout, ordenes, dist_matrix)
    mejor_solucion_temple, costo_real_temple, hist_temple = Temple.temple_generico(
        layout_inicial, fn_costo, iteraciones=10000, T=500.0)

    print(f"-> Costo RUTEO REAL del layout por agrupación (Temple): {costo_real_temple} pasos totales")
    
    ahorro = costo_real_ga - costo_real_temple
    porcentaje = (ahorro / costo_real_ga) * 100 if costo_real_ga > 0 else 0
    print(f"\nAhorro total: {ahorro} pasos equivalentes al {porcentaje:.2f}% de mejora.\n")

    # --- Gráficos de convergencia comparados (normalizados) ---
    base_ga        = historial_costos[0] if historial_costos else 1
    base_temple    = hist_temple[0]      if hist_temple      else 1
    hist_ga_n      = [v / base_ga        for v in historial_costos]
    hist_temple_n  = [v / base_temple    for v in hist_temple]
    ga_final_n     = costo_real_ga    / base_ga
    temple_final_n = costo_real_temple / base_temple

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')

    # -- GA --
    ax1.set_facecolor('#1a1a2e')
    ax1.plot(hist_ga_n, color='#00d4ff', linewidth=2, label='Costo normalizado')
    ax1.fill_between(range(len(hist_ga_n)), hist_ga_n, alpha=0.15, color='#00d4ff')
    ax1.axhline(1.0, color='#ff7043', linewidth=1.5, linestyle='--', label='Costo inicial (1.0)')
    ax1.axhline(ga_final_n, color='#aaffaa', linewidth=1.2, linestyle=':', label=f'Ruteo real: {ga_final_n:.3f}')
    ax1.legend(facecolor='#2a2a4e', labelcolor='white', fontsize=9)
    ax1.set_xlabel('Generación', color='#aaaaaa', fontsize=11)
    ax1.set_ylabel('Costo normalizado', color='#aaaaaa', fontsize=11)
    ax1.set_title('Convergencia — Algoritmo Genético\n(Heurística: frec × distancia)', color='white', fontsize=12, pad=10)
    ax1.tick_params(colors='#aaaaaa')
    ax1.spines[:].set_color('#444466')
    ax1.set_ylim(min(hist_ga_n) * 0.97, 1.03)

    # -- Temple --
    ax2.set_facecolor('#1a1a2e')
    ax2.plot(hist_temple_n, color='#ff9f43', linewidth=2, label='Mejor costo normalizado')
    ax2.fill_between(range(len(hist_temple_n)), hist_temple_n, alpha=0.15, color='#ff9f43')
    ax2.axhline(1.0, color='#ff7043', linewidth=1.5, linestyle='--', label='Costo inicial (1.0)')
    ax2.axhline(temple_final_n, color='#aaffaa', linewidth=1.2, linestyle=':', label=f'Costo final: {temple_final_n:.3f}')
    ax2.legend(facecolor='#2a2a4e', labelcolor='white', fontsize=9)
    ax2.set_xlabel('Iteración', color='#aaaaaa', fontsize=11)
    ax2.set_ylabel('Costo normalizado', color='#aaaaaa', fontsize=11)
    ax2.set_title('Convergencia — Temple Simulado\n(Optimización por agrupación de órdenes)', color='white', fontsize=12, pad=10)
    ax2.tick_params(colors='#aaaaaa')
    ax2.spines[:].set_color('#444466')
    ax2.set_ylim(min(hist_temple_n) * 0.97, 1.03)

    fig.suptitle('Comparación de convergencia: Genético vs Temple', color='white', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    print("Mostrando Almacén: Trivial (Genético)")
    visualizar_almacen(mejor_solucion_ga, ordenes, "(Genético: Solo por Frecuencia)")
    print("Mostrando Almacén: Agrupado por Órdenes (Temple)")
    visualizar_almacen(mejor_solucion_temple, ordenes, "(Temple: Agrupado por Orden)")

    plt.show() # Bloquea al final para mantener ventanas abiertas

if __name__ == "__main__":
    main()
