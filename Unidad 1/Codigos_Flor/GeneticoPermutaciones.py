import csv
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

# --- IMPORTAMOS TUS CLASES ---
from Estanteria import Almacen
from A_estrella import A_estrella

# ---------------------------------------------------------------------------
# 1. FUNCIONES AUXILIARES (Pre-cálculo)
# ---------------------------------------------------------------------------
def precalcular_distancias_a_carga(almacen, astar):
    """
    Usa tu A_estrella para calcular la distancia en pasos desde 'C' a cada estante.
    """
    distancias = {}
    inicio = almacen.estacion_carga
    
    for id_estante in almacen.estantes_coords.keys():
        camino = astar.buscar_camino_astar(almacen, inicio, id_estante)
        if camino:
            # La distancia es la cantidad de pasos (nodos en el camino - 1)
            distancias[id_estante] = len(camino) - 1
        else:
            distancias[id_estante] = float('inf') # Por si hay algún error de ruteo
    return distancias

def cargar_frecuencias(ruta_csv):
    """Lee el histórico de órdenes.csv y cuenta la frecuencia de cada producto."""
    productos_pedidos = []
    try:
        with open(ruta_csv, "r") as archivo:
            lector = csv.reader(archivo)
            for fila in lector:
                productos_pedidos.extend([int(x) for x in fila])
        return Counter(productos_pedidos)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_csv}")
        return Counter()

# ---------------------------------------------------------------------------
# 2. MOTOR DEL ALGORITMO GENÉTICO (Permutaciones)
# ---------------------------------------------------------------------------
class GeneticoPermutaciones:
    def __init__(self, tamano_poblacion, distancias_a_c, frecuencias, num_productos=48):
        self.tamano_poblacion = tamano_poblacion
        self.num_productos = num_productos
        self.distancias_a_c = distancias_a_c
        self.frecuencias = frecuencias
        self.poblacion = self._generar_poblacion_inicial()

    def _generar_poblacion_inicial(self):
        """Genera individuos al azar. Cada individuo es una lista de 48 IDs."""
        poblacion = []
        productos_base = list(range(1, self.num_productos + 1))
        for _ in range(self.tamano_poblacion):
            individuo = copy.copy(productos_base)
            random.shuffle(individuo)
            poblacion.append(individuo)
        return poblacion

    def calcular_fitness(self, individuo):
        """Calcula el costo: suma de (Distancia del estante i * Frecuencia del producto en i)."""
        costo_total = 0
        for indice, id_producto in enumerate(individuo):
            id_estante = indice + 1 # El índice 0 corresponde al estante 1
            distancia = self.distancias_a_c.get(id_estante, 0)
            frecuencia = self.frecuencias.get(id_producto, 0)
            costo_total += (distancia * frecuencia)
        return costo_total

    def seleccion_torneo(self, pop_fitness, k=3):
        """Selecciona al mejor de un torneo de tamaño k."""
        participantes = random.sample(pop_fitness, k)
        ganador = min(participantes, key=lambda x: x[1])
        return ganador[0]

    def cruce_pmx(self, padre1, padre2):
        """Partially Mapped Crossover para mantener permutaciones válidas."""
        size = len(padre1)
        hijo1, hijo2 = [-1]*size, [-1]*size
        pt1, pt2 = sorted(random.sample(range(size), 2))

        hijo1[pt1:pt2+1] = padre1[pt1:pt2+1]
        hijo2[pt1:pt2+1] = padre2[pt1:pt2+1]

        for i in range(pt1, pt2 + 1):
            if padre2[i] not in hijo1:
                pos = i
                while pt1 <= pos <= pt2:
                    pos = padre2.index(padre1[pos])
                hijo1[pos] = padre2[i]

        for i in range(pt1, pt2 + 1):
            if padre1[i] not in hijo2:
                pos = i
                while pt1 <= pos <= pt2:
                    pos = padre1.index(padre2[pos])
                hijo2[pos] = padre1[i]

        for i in range(size):
            if hijo1[i] == -1: hijo1[i] = padre2[i]
            if hijo2[i] == -1: hijo2[i] = padre1[i]

        return hijo1, hijo2

    def mutacion_intercambio(self, individuo, prob_mutacion=0.05):
        """Mutación Swap con una probabilidad dada."""
        if random.random() < prob_mutacion:
            a, b = random.sample(range(len(individuo)), 2)
            individuo[a], individuo[b] = individuo[b], individuo[a]
        return individuo

    def evolucionar(self, iteraciones=100, prob_mutacion=0.1):
        """Ejecuta las generaciones con Elitismo."""
        historial_costos = []

        for i in range(iteraciones):
            pop_fitness = [(ind, self.calcular_fitness(ind)) for ind in self.poblacion]
            pop_fitness.sort(key=lambda x: x[1])
            
            mejor_actual = pop_fitness[0][0]
            mejor_costo = pop_fitness[0][1]
            historial_costos.append(mejor_costo)

            # Elitismo: preservamos al mejor
            nueva_generacion = [copy.copy(mejor_actual)] 
            
            while len(nueva_generacion) < self.tamano_poblacion:
                p1 = self.seleccion_torneo(pop_fitness)
                p2 = self.seleccion_torneo(pop_fitness)
                
                h1, h2 = self.cruce_pmx(p1, p2)
                
                h1 = self.mutacion_intercambio(h1, prob_mutacion)
                h2 = self.mutacion_intercambio(h2, prob_mutacion)
                
                nueva_generacion.append(h1)
                if len(nueva_generacion) < self.tamano_poblacion:
                    nueva_generacion.append(h2)
                    
            self.poblacion = nueva_generacion
            
            # Imprimimos progreso cada 10 iteraciones
            if (i+1) % 10 == 0 or i == 0:
                print(f"Generación {i+1}: Mejor Costo = {mejor_costo}")

        pop_fitness = [(ind, self.calcular_fitness(ind)) for ind in self.poblacion]
        pop_fitness.sort(key=lambda x: x[1])
        return pop_fitness[0][0], pop_fitness[0][1], historial_costos

# ---------------------------------------------------------------------------
# 3. VISUALIZACIÓN: MAPA DE CALOR
# ---------------------------------------------------------------------------
def dibujar_mapa_calor(almacen, configuracion, frecuencias, titulo):
    """
    Dibuja el almacén usando las coordenadas de tu clase Almacen.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    max_freq = max(frecuencias.values()) if frecuencias else 1
    
    # Dibujar "C" (Estación de Carga)
    # Tu clase usa (5, 0). Para Matplotlib (x,y), x=columna, y=fila. Invertimos y.
    # Usaremos 11 - fila para que la fila 0 quede arriba.
    c_f, c_c = almacen.estacion_carga
    rect_c = patches.Rectangle((c_c, 11 - c_f - 1), 1, 1, color='yellow')
    ax.add_patch(rect_c)
    plt.text(c_c + 0.5, 11 - c_f - 0.5, 'C', ha='center', va='center', fontweight='bold', fontsize=12)

    # Dibujar Estantes con colores
    for i, id_producto in enumerate(configuracion):
        id_estante = i + 1
        if id_estante in almacen.estantes_coords:
            r, c = almacen.estantes_coords[id_estante]
            freq = frecuencias.get(id_producto, 0)
            
            # Color: Frecuencia 0 -> Rosa claro (1, 0.8, 0.8). Max Frec -> Rojo intenso (0.8, 0, 0)
            intensidad = freq / max_freq
            # Usamos un gradiente personalizado para que se vea bien
            color = (1.0 - (0.2 * intensidad), 0.8 - (0.8 * intensidad), 0.8 - (0.8 * intensidad))
            
            # 11 - r - 1 para ajustar al eje Y de Matplotlib
            rect = patches.Rectangle((c, 11 - r - 1), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            plt.text(c + 0.5, 11 - r - 0.5, str(id_producto), ha='center', va='center', fontsize=9)

    plt.xlim(0, almacen.columnas)
    plt.ylim(0, almacen.filas)
    plt.title(titulo)
    
    # Ajustamos la grilla visual para que coincida con tus celdas
    ax.set_xticks(range(almacen.columnas + 1))
    ax.set_yticks(range(almacen.filas + 1))
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    
    plt.show()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Inicializando Almacén y A*")
    almacen = Almacen()
    astar = A_estrella()
    
    print("Calculando distancias base y leyendo CSV")
    distancias_a_c = precalcular_distancias_a_carga(almacen, astar)
    
    # Asegurate de que ordenes.csv esté en la misma carpeta
    frecuencias = cargar_frecuencias("ordenes.csv") 

    if not frecuencias:
        print("Error: No se cargaron frecuencias. Revisa el archivo CSV.")
        return

    # Creamos el Genético (Población de 50)
    ag = GeneticoPermutaciones(tamano_poblacion=50, distancias_a_c=distancias_a_c, frecuencias=frecuencias)
    
    print("\n Mapa de Calor Inicial")
    # Creamos el diseño original: producto 1 en estante 1, 2 en 2, ..., 48 en 48
    layout_original = list(range(1, 49)) 
    
    # Calculamos el costo de este estado original para compararlo después
    costo_original = ag.calcular_fitness(layout_original)
    print(f"Costo del layout original : {costo_original}")
    
    # Dibujamos el mapa
    
    dibujar_mapa_calor(almacen, layout_original, frecuencias, "Mapa de Calor - ESTADO INICIAL ")

    # 200 iteraciones suele ser un buen número para ver convergencia en este problema
    mejor_layout, mejor_costo, historial = ag.evolucionar(iteraciones=200, prob_mutacion=0.1)

    print(f"\n¡Evolución terminada! Mejor costo final: {mejor_costo}")

    print("\nResultados")
    # Gráfico de convergencia
    plt.figure(figsize=(8, 4))
    plt.plot(historial, color='tab:red', linewidth=2)
    plt.title('Evolución del Costo Genético')
    plt.xlabel('Generación')
    plt.ylabel('Costo Total (Distancia * Frecuencia)')
    plt.grid(True)
    plt.show()

    # Mapa final optimizado
    dibujar_mapa_calor(almacen, mejor_layout, frecuencias, "Mapa de Calor - ESTADO FINAL ")

if __name__ == "__main__":
    main()