import A_Star as A
import numpy as np
import matplotlib.pyplot as plt
import csv
import random

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

def estanteria_a_coord(num):
    return MAPA_ESTANTERIAS.get(num)

cache_distancias = {}

def obtener_distancia(est_origen, est_destino, astar):
    if est_origen == est_destino:
        return 0
    par = tuple(sorted((est_origen, est_destino)))
    
    if par not in cache_distancias:
        start = estanteria_a_coord(par[0])
        finish = estanteria_a_coord(par[1])
        camino = astar.busqueda(start, finish)
        cache_distancias[par] = len(camino) - 1 if camino else float('inf')
        
    return cache_distancias[par]

def calcular_costo_ruta(ruta, astar):
    costo = 0
    for j in range(len(ruta) - 1):
        costo += obtener_distancia(ruta[j], ruta[j + 1], astar)
    return costo

def temple_simulado(solucion_inicial, iteraciones, T):
    grafo = np.zeros((11, 13), dtype=int)

    for row_start in [1, 6]:
        for col_start in [2, 6, 10]:
            for r in range(row_start, row_start + 4):
                for c in range(col_start, col_start + 2):
                    grafo[r, c] = 1

    astar = A.A_Star(grafo)
    solucion_actual = solucion_inicial.copy()
    
    costo_actual = calcular_costo_ruta(solucion_actual, astar)

    historial_costos = []
    historial_temperatura = []

    for i in range(iteraciones):
        T = T * 0.96
        if T < 0.0001:
            break

        vecino = solucion_actual.copy()

        # cambio chico: swap de dos posiciones
        a, b = random.sample(range(len(vecino)), 2)
        vecino[a], vecino[b] = vecino[b], vecino[a]

        costo_vecino = calcular_costo_ruta(vecino, astar)

        if costo_vecino < costo_actual:
            solucion_actual = vecino
            costo_actual = costo_vecino
        else:
            try:
                p = np.exp((costo_actual - costo_vecino) / T)
            except OverflowError:
                p = 0
                
            if np.random.rand() < p:
                solucion_actual = vecino
                costo_actual = costo_vecino

        historial_costos.append(costo_actual)
        historial_temperatura.append(T)

        print("Iteración:", i, "Costo:", costo_actual, "T:", T)

    return solucion_actual, costo_actual, historial_costos, historial_temperatura

def graficar_curvas(historial_costos, historial_temperatura):
    plt.figure(figsize=(10, 5))
    plt.plot(historial_costos, label="Costo")
    plt.plot(historial_temperatura, label="Temperatura")
    plt.xlabel("Iteración")
    plt.ylabel("Valor")
    plt.title("Evolución del costo y enfriamiento")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    solucion, costo, historial_costos, historial_temperatura = temple_simulado(ordenes[1], 1000, 100)

    print("Mejor solución encontrada:", solucion)
    print("Costo total:", costo)

    graficar_curvas(historial_costos, historial_temperatura)

if __name__ == "__main__":
    main()