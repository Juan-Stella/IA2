import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import time
from collections import deque

class AlmacenOptimizadoGA:
    def __init__(self, ruta_csv=None):
        # 1. Definición del Layout (Basado en tu imagen de 6 estanterías de 8)
        # Estación de carga 'C' en la base del pasillo central
        self.punto_carga = (0, 5) 
        self.pos_estanterias, self.puntos_acceso = self._generar_layout()
        
        # 2. Carga de Frecuencias
        self.frecuencias = self._cargar_frecuencias(ruta_csv)
        self.productos_ids = sorted(list(self.frecuencias.keys()))
        
        # 3. Historial para gráficos
        self.historial_mejor = []
        self.historial_peor = []
        self.historial_promedio = []
        self.historial_mejores_individuos = []
        
        # 4. Configuración de visualización
        self.ventana_suavizado = 5  # Para suavizar las curvas

    def _generar_layout(self):
        """
        Crea las coordenadas de los 48 estantes y sus puntos de acceso en pasillos.
        Las estanterías están en bloques. Pasillos en X: 1, 4, 7, 10
        """
        pos = {}
        acceso = {}
        # Bloques de estanterías en columnas X (2-3), (5-6), (8-9)
        columnas_bloques_x = [2, 5, 8]
        n = 0
        for start_x in columnas_bloques_x:
            # Filas de estanterías (dejando pasillo central en Y=5)
            filas_y = [1, 2, 3, 4, 6, 7, 8, 9]
            for y in filas_y:
                for x_offset in [0, 1]:
                    n += 1
                    x = start_x + x_offset
                    pos[n] = (x, y)
                    # Punto de acceso: Si está a la izq del bloque, accede por x-1. Si no, x+1.
                    x_acc = x - 1 if x_offset == 0 else x + 1
                    acceso[n] = (x_acc, y)
        return pos, acceso

    def _cargar_frecuencias(self, ruta):
        try:
            df = pd.read_csv(ruta)
            # Aplanar el DataFrame y contar frecuencias
            freq = {}
            for col in df.columns:
                for val in df[col].dropna():
                    if val in freq:
                        freq[val] += 1
                    else:
                        freq[val] = 1
        except Exception as e:
            print(f"Error cargando CSV: {e}. Usando datos aleatorios...")
            # Datos aleatorios si no hay CSV - distribución más realista
            freq = {}
            # Productos populares (top 10)
            for i in range(1, 11):
                freq[i] = random.randint(80, 150)
            # Productos medios (11-30)
            for i in range(11, 31):
                freq[i] = random.randint(30, 79)
            # Productos poco populares (31-48)
            for i in range(31, 49):
                freq[i] = random.randint(1, 29)
        
        # Asegurar que existan los 48 productos
        for i in range(1, 49):
            if i not in freq: 
                freq[i] = 0
                
        return freq

    def calcular_distancia_navegable(self, p1, p2):
        """
        Calcula la distancia evitando pasar por arriba de las estanterías.
        El montacargas viaja por los pasillos (X=1, 4, 7, 10 o Y=5).
        """
        # En este layout simplificado, la distancia Manhattan entre puntos de PASILLO
        # suele ser segura, ya que los pasillos conectan todos los accesos.
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def calcular_fitness(self, cromosoma):
        """
        Costo = Sumatoria de (Frecuencia * Distancia a Carga).
        Cromosoma: lista donde el índice es el estante y el valor es el ID del producto.
        MENOR costo = MEJOR fitness
        """
        costo_total = 0
        for i, prod_id in enumerate(cromosoma):
            estante_idx = i + 1
            punto_destino = self.puntos_acceso[estante_idx]
            dist = self.calcular_distancia_navegable(self.punto_carga, punto_destino)
            costo_total += self.frecuencias[prod_id] * dist
        return costo_total

    def plot_evolucion(self, mejor_ind, gen, mejor_fitness, peor_fitness, prom_fitness):
        """
        Visualización mejorada con 3 subplots:
        1. Mapa de calor del almacén
        2. Evolución de costos
        3. Distribución de la población actual
        """
        # Crear figura con 3 subplots
        fig = plt.figure(figsize=(18, 10))
        
        # ===== SUBPLOT 1: MAPA DE CALOR DEL ALMACÉN =====
        ax1 = plt.subplot(2, 2, 1)
        grid = np.zeros((11, 12))  # Matriz para visualización
        
        # Llenar estanterías con calor de frecuencia
        for i, prod_id in enumerate(mejor_ind):
            x, y = self.pos_estanterias[i+1]
            grid[y, x] = self.frecuencias[prod_id]
        
        # Máscara para que los pasillos se vean blancos
        mask_pasillos = grid == 0
        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        
        sns.heatmap(grid, annot=False, cmap=cmap, cbar=False, 
                   ax=ax1, square=True, mask=mask_pasillos)
        
        # Marcar la Estación de Carga
        ax1.scatter(self.punto_carga[0]+0.5, self.punto_carga[1]+0.5, 
                   marker="s", s=300, color="blue", edgecolors='white', 
                   linewidth=2, zorder=5, label="Estación C")
        ax1.text(self.punto_carga[0]+0.5, self.punto_carga[1]+0.5, 'C', 
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        
        # Agregar números de productos
        for i, prod_id in enumerate(mejor_ind):  # Solo mostrar algunos para no saturar
            x, y = self.pos_estanterias[i+1]
            ax1.text(x+0.5, y+0.5, str(prod_id), ha='center', va='center', 
                    color='white' if grid[y, x] > 50 else 'black', 
                    fontsize=8, fontweight='bold')
        
        ax1.set_title(f"Distribución Óptima - Generación {gen}", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Columnas (Pasillos en 1, 4, 7, 10)")
        ax1.set_ylabel("Filas (Pasillo Central en 5)")
        
        # ===== SUBPLOT 2: EVOLUCIÓN DE COSTOS =====
        ax2 = plt.subplot(2, 2, 2)
        
        # Suavizar las curvas con media móvil
        def suavizar(datos, ventana):
            if len(datos) < ventana:
                return datos
            suave = np.convolve(datos, np.ones(ventana)/ventana, mode='valid')
            return np.concatenate((datos[:ventana-1], suave))
        
        gens = range(len(self.historial_mejor))
        mejor_suave = suavizar(self.historial_mejor, self.ventana_suavizado)
        peor_suave = suavizar(self.historial_peor, self.ventana_suavizado)
        prom_suave = suavizar(self.historial_promedio, self.ventana_suavizado)
        
        ax2.plot(gens, peor_suave, 'r-', alpha=0.5, linewidth=1, label='Máximo')
        ax2.plot(gens, prom_suave, 'b-', linewidth=2, label='Promedio')
        ax2.plot(gens, mejor_suave, 'g-', linewidth=2, label='Mínimo (Mejor)')
        
        # Marcar el mejor valor actual
        ax2.scatter(gen, mejor_fitness, c='green', s=100, zorder=5, 
                   edgecolors='white', linewidth=2)
        
        ax2.set_xlabel('Generación', fontsize=11)
        ax2.set_ylabel('Costo Total', fontsize=11)
        ax2.set_title('Evolución del Costo', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Añadir texto con valores actuales
        ax2.text(0.02, 0.98, f"Mejor: {mejor_fitness:.0f}\nProm: {prom_fitness:.0f}\nPeor: {peor_fitness:.0f}", 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # ===== SUBPLOT 3: DISTRIBUCIÓN DE COSTOS ACTUAL =====
        ax3 = plt.subplot(2, 2, 3)
        
        # Calcular costos de toda la población actual si está disponible
        if hasattr(self, 'poblacion_actual'):
            costos_actuales = [self.calcular_fitness(ind) for ind in self.poblacion_actual]
            ax3.hist(costos_actuales, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            ax3.axvline(mejor_fitness, color='g', linestyle='--', linewidth=2, 
                       label=f'Mín: {mejor_fitness:.0f}')
            ax3.axvline(prom_fitness, color='b', linestyle='--', linewidth=2, 
                       label=f'Prom: {prom_fitness:.0f}')
            ax3.axvline(peor_fitness, color='r', linestyle='--', linewidth=2, 
                       label=f'Máx: {peor_fitness:.0f}')
            ax3.set_xlabel('Costo')
            ax3.set_ylabel('Frecuencia')
            ax3.set_title('Distribución de Costos en Población Actual', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        # ===== SUBPLOT 4: TOP PRODUCTOS MÁS FRECUENTES =====
        ax4 = plt.subplot(2, 2, 4)
        
        # Obtener top 10 productos más frecuentes
        top_productos = sorted(self.frecuencias.items(), key=lambda x: x[1], reverse=True)[:10]
        productos = [f"P{p}" for p, _ in top_productos]
        frecuencias = [f for _, f in top_productos]
        
        # Encontrar dónde están ubicados en la mejor solución
        distancias_top = []
        for prod_id, _ in top_productos:
            for i, p in enumerate(mejor_ind):
                if p == prod_id:
                    estante_idx = i + 1
                    dist = self.calcular_distancia_navegable(self.punto_carga, self.puntos_acceso[estante_idx])
                    distancias_top.append(dist)
                    break
        
        # Gráfico de barras
        x = np.arange(len(productos))
        width = 0.35
        ax4.bar(x - width/2, frecuencias, width, label='Frecuencia', color='orange', alpha=0.7)
        ax4.bar(x + width/2, [d*10 for d in distancias_top], width, label='Distancia x10', color='blue', alpha=0.7)
        
        ax4.set_xlabel('Producto')
        ax4.set_ylabel('Valor')
        ax4.set_title('Top 10 Productos: Frecuencia vs Distancia', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(productos, rotation=45)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        plt.pause(0.01)

    def ejecutar(self, generaciones=150, pop_size=50):
        """
        Ejecuta el algoritmo genético con visualización mejorada.
        """
        print("\n" + "="*60)
        print(" INICIANDO ALGORITMO GENÉTICO")
        print("="*60)
        print(f"Generaciones: {generaciones}")
        print(f"Tamaño población: {pop_size}")
        print(f"Total productos: {len(self.productos_ids)}")
        print(f"Rango frecuencias: {min(self.frecuencias.values())} - {max(self.frecuencias.values())}")
        print("="*60 + "\n")
        
        # Población inicial (permutaciones de IDs de productos)
        poblacion = [random.sample(self.productos_ids, 48) for _ in range(pop_size)]
        
        for gen in range(generaciones):
            # Evaluar toda la población
            scores = [(self.calcular_fitness(ind), ind) for ind in poblacion]
            scores.sort(key=lambda x: x[0])  # Ordenar por costo (menor es mejor)
            
            # Guardar población actual para visualización
            self.poblacion_actual = poblacion
            
            # Estadísticas de la generación
            mejor_fitness, mejor_ind = scores[0]
            peor_fitness, _ = scores[-1]
            prom_fitness = sum(s[0] for s in scores) / len(scores)
            
            # Guardar historial
            self.historial_mejor.append(mejor_fitness)
            self.historial_peor.append(peor_fitness)
            self.historial_promedio.append(prom_fitness)
            self.historial_mejores_individuos.append(mejor_ind)
            
            # Visualización cada N generaciones
            if gen % 5 == 0 or gen == generaciones - 1:
                self.plot_evolucion(mejor_ind, gen, mejor_fitness, peor_fitness, prom_fitness)
                time.sleep(0.1)
            
            # Mostrar progreso en consola cada 10 generaciones
            if gen % 10 == 0:
                mejora = ((self.historial_mejor[0] - mejor_fitness) / self.historial_mejor[0]) * 100
                print(f"Gen {gen:3d} | Mejor: {mejor_fitness:6.0f} | Prom: {prom_fitness:6.0f} | Peor: {peor_fitness:6.0f} | Mejora: {mejora:5.1f}%")

            # Crear nueva población
            nueva_pob = [mejor_ind]  # Elitismo (conservar el mejor)
            
            while len(nueva_pob) < pop_size:
                # Selección por torneo (de los mejores)
                p1 = self._torneo(scores)
                p2 = self._torneo(scores)
                
                # Crossover de orden (preserva unicidad de productos)
                hijo = self._ordered_crossover(p1, p2)
                
                # Mutación por intercambio
                if random.random() < 0.2:
                    idx1, idx2 = random.sample(range(48), 2)
                    hijo[idx1], hijo[idx2] = hijo[idx2], hijo[idx1]
                
                nueva_pob.append(hijo)
            
            poblacion = nueva_pob
        
        # Resultados finales
        print("\n" + "="*60)
        print(" ALGORITMO GENÉTICO COMPLETADO")
        print("="*60)
        print(f"Mejor costo final: {self.historial_mejor[-1]:.0f}")
        mejora_total = ((self.historial_mejor[0] - self.historial_mejor[-1]) / self.historial_mejor[0]) * 100
        print(f"Mejora total: {mejora_total:.1f}%")
        print("="*60)
        
        # Mostrar gráfico final de evolución
        self._plot_resultados_finales()
        
        return self.historial_mejor[-1], mejor_ind

    def _plot_resultados_finales(self):
        """Muestra un gráfico final con la evolución completa."""
        plt.figure(figsize=(12, 6))
        
        gens = range(len(self.historial_mejor))
        plt.plot(gens, self.historial_peor, 'r-', alpha=0.5, label='Máximo', linewidth=1)
        plt.plot(gens, self.historial_promedio, 'b-', label='Promedio', linewidth=2)
        plt.plot(gens, self.historial_mejor, 'g-', label='Mínimo (Mejor)', linewidth=2)
        
        plt.xlabel('Generación', fontsize=12)
        plt.ylabel('Costo Total', fontsize=12)
        plt.title('Evolución Completa del Algoritmo Genético', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Añadir anotaciones
        plt.text(0.02, 0.98, f"Mejor final: {self.historial_mejor[-1]:.0f}\nMejora: {((self.historial_mejor[0] - self.historial_mejor[-1]) / self.historial_mejor[0] * 100):.1f}%", 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def _torneo(self, scores, k=3):
        """Selección por torneo (elige al mejor de k individuos aleatorios)."""
        seleccionados = random.sample(scores, k)
        return min(seleccionados, key=lambda x: x[0])[1]

    def _ordered_crossover(self, p1, p2):
        """
        Crossover de orden (OX1) para permutaciones.
        Preserva la unicidad de los productos.
        """
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        hijo = [None] * size
        hijo[a:b] = p1[a:b]
        
        p2_rest = [item for item in p2 if item not in hijo]
        idx = 0
        for i in range(size):
            if hijo[i] is None:
                hijo[i] = p2_rest[idx]
                idx += 1
        return hijo

# Ejecución
if __name__ == "__main__":
    # Intentar cargar el archivo CSV
    try:
        ga = AlmacenOptimizadoGA(ruta_csv="ordenes.csv")
    except:
        print("Usando datos aleatorios...")
        ga = AlmacenOptimizadoGA(ruta_csv=None)
    
    # Ejecutar el algoritmo
    mejor_costo, mejor_solucion = ga.ejecutar(generaciones=150, pop_size=50)
    
    # Mostrar top 10 productos en la solución final
    print("\n📊 TOP 10 PRODUCTOS MÁS FRECUENTES - UBICACIÓN FINAL:")
    print("-" * 60)
    print(f"{'Producto':^10} | {'Frecuencia':^12} | {'Estante':^10} | {'Distancia':^10}")
    print("-" * 60)
    
    top_productos = sorted(ga.frecuencias.items(), key=lambda x: x[1], reverse=True)[:10]
    for prod_id, freq in top_productos:
        for i, p in enumerate(mejor_solucion):
            if p == prod_id:
                estante = i + 1
                dist = ga.calcular_distancia_navegable(ga.punto_carga, ga.puntos_acceso[estante])
                print(f"{prod_id:^10} | {freq:^12} | Estante {estante:2d} | {dist:^10}")
                break