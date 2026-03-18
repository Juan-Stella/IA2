import A_Star as A
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import Temple

ordenes = []

with open(r"D:\01. Facultad\ia2\IA2\IA2\Unidad 1\Codigos_Juan\ordenes.csv", "r") as archivo:
    lector = csv.reader(archivo)
    for fila in lector:
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

def visualizar_almacen(mejor_solucion, ordenes):
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
    ax.set_title('Almacén — Configuración óptima (Algoritmo Genético)\nMapa de calor: frecuencia de pedidos por producto',
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
    for i in range(5000):
        costos = fitness(poblacion, frecuencias, dist_slots)
        historial_costos.append(min(costos))
        seleccionados, costos_sel = seleccion(poblacion, costos, 20)
        poblacion_nueva = crossover(seleccionados, costos_sel, 100)
        poblacion_nueva = mutacion(poblacion_nueva, 0.1)
        poblacion = poblacion_nueva
    
    costos = fitness(poblacion, frecuencias, dist_slots)
    mejor_idx = np.argmin(costos)
    mejor_solucion = poblacion[mejor_idx]
    print("Mejor solución encontrada:", mejor_solucion)
    print("Costo total:", costos[mejor_idx])

    # --- Gráfico de convergencia ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.plot(historial_costos, color='#00d4ff', linewidth=2, label='Costo mínimo por generación')
    ax.fill_between(range(len(historial_costos)), historial_costos, alpha=0.15, color='#00d4ff')
    ax.axhline(costo_inicial, color='#ff7043', linewidth=1.5, linestyle='--', label=f'Costo inicial: {costo_inicial:.1f}')
    ax.legend(facecolor='#2a2a4e', labelcolor='white', fontsize=10)
    ax.set_xlabel('Iteración', color='#aaaaaa', fontsize=11)
    ax.set_ylabel('Costo mínimo', color='#aaaaaa', fontsize=11)
    ax.set_title('Convergencia del Algoritmo Genético', color='white', fontsize=13, pad=12)
    ax.tick_params(colors='#aaaaaa')
    ax.spines[:].set_color('#444466')
    plt.tight_layout()
    plt.show()

    visualizar_almacen(mejor_solucion, ordenes)

if __name__ == "__main__":
    main()
