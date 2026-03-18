import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import time

class AlmacenOptimizadoGA:
    def __init__(self, ruta_csv=None):
        # 1. Definición del Layout (Basado en tu imagen de 6 estanterías de 8)
        # Estación de carga 'C' en la base del pasillo central
        self.punto_carga = (0, 5) 
        self.pos_estanterias, self.puntos_acceso = self._generar_layout()
        
        # 2. Carga de Frecuencias
        self.frecuencias = self._cargar_frecuencias(ruta_csv)
        self.productos_ids = sorted(list(self.frecuencias.keys()))

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
            freq = df.stack().value_counts().to_dict()
        except:
            # Datos aleatorios si no hay CSV
            freq = {i: np.random.randint(1, 100) for i in range(1, 49)}
        
        # Asegurar que existan los 48 productos
        for i in range(1, 49):
            if i not in freq: freq[i] = 0
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
        """
        costo_total = 0
        for i, prod_id in enumerate(cromosoma):
            estante_idx = i + 1
            punto_destino = self.puntos_acceso[estante_idx]
            dist = self.calcular_distancia_navegable(self.punto_carga, punto_destino)
            costo_total += self.frecuencias[prod_id] * dist
        return costo_total

    def plot_evolucion(self, mejor_ind, gen, fitness):
        grid = np.zeros((11, 12)) # Matriz para visualización
        # Llenar estanterías con calor de frecuencia
        for i, prod_id in enumerate(mejor_ind):
            x, y = self.pos_estanterias[i+1]
            grid[y, x] = self.frecuencias[prod_id]
        
        clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        # Máscara para que los pasillos se vean blancos
        sns.heatmap(grid, annot=False, cmap="YlOrRd", cbar_kws={'label': 'Frecuencia'})
        
        # Marcar la Estación de Carga
        plt.scatter(self.punto_carga[0]+0.5, self.punto_carga[1]+0.5, marker="s", s=200, color="blue", label="Carga")
        
        plt.title(f"Evolución GA - Generación {gen}\nCosto de Desplazamiento: {fitness}")
        plt.xlabel("Columnas (Pasillos en 1, 4, 7, 10)")
        plt.ylabel("Filas (Pasillo Central en 5)")
        plt.show()

    def ejecutar(self, generaciones=100, pop_size=30):
        # Población inicial (permutaciones de IDs de productos)
        poblacion = [random.sample(self.productos_ids, 48) for _ in range(pop_size)]
        
        for gen in range(generaciones):
            # Evaluar
            scores = [(self.calcular_fitness(ind), ind) for ind in poblacion]
            scores.sort(key=lambda x: x[0])
            
            mejor_fitness, mejor_ind = scores[0]
            
            if gen % 5 == 0:
                self.plot_evolucion(mejor_ind, gen, mejor_fitness)
                time.sleep(0.05)

            # Nueva población
            nueva_pob = [mejor_ind] # Elitismo
            
            while len(nueva_pob) < pop_size:
                # Selección por torneo
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

    def _torneo(self, scores, k=3):
        seleccionados = random.sample(scores, k)
        return min(seleccionados, key=lambda x: x[0])[1]

    def _ordered_crossover(self, p1, p2):
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
# Asegúrate de que 'ordenes.csv' esté en la misma carpeta o usa la ruta de AlgGen.py
ga = AlmacenOptimizadoGA(ruta_csv="ordenes.csv")
ga.ejecutar(generaciones=150, pop_size=50)