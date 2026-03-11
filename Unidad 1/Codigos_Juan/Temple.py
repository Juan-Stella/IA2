import A_Star as A
import numpy as np
import matplotlib.pyplot as plt
import csv
import random

ordenes = []

with open(r"d:\Facultad\IA2\Unidad 1\Codigos_Juan\ordenes.csv", "r") as archivo:
    lector = csv.reader(archivo)

    for fila in lector:
        ordenes.append([int(x) for x in fila])


def temple_simulado (solucion_inicial,iter):
    # Crear grilla
    grafo = np.zeros((11, 13), dtype=int)

    for row_start in [1, 6]:
        for col_start in [2, 6, 10]:
            for r in range(row_start, row_start + 4):
                for c in range(col_start, col_start + 2):
                    grafo[r, c] = 1

    astar = A.A_Star(grafo)
    costo = 0
    T = 100
    print(solucion_inicial)
    E = np.zeros(iter)
    for i in range(iter):
        T = 100-1
        for j in range(len(solucion_inicial)-1):
            start = estanteria_a_coord(solucion_inicial[j])
            finish = estanteria_a_coord(solucion_inicial[j+1])
            camino = astar.busqueda(start, finish)
            camino = astar.busqueda(start, finish)
            camino = astar.busqueda(start, finish)
            costo += len(camino) - 1
        E[i] = costo
        if E[i] < E[i-1]:
            p = np.exp((E[i-1] - E[i]) / T)
            if np.random.rand() < p:
                solucion_elegida = solucion_inicial[i]
            else:
                solucion_elegida = solucion_inicial[i-1]
        else:
            solucion_elegida = solucion_inicial[i]

        random.shuffle(solucion_inicial)
    
    return solucion_elegida
        

    pass

def estanteria_a_coord(num):
    mapa = {
        1:(1,2),  2:(1,3),
        3:(2,2),  4:(2,3),
        5:(3,2),  6:(3,3),
        7:(4,2),  8:(4,3),

        9:(1,6),  10:(1,7),
        11:(2,6), 12:(2,7),
        13:(3,6), 14:(3,7),
        15:(4,6), 16:(4,7),

        17:(1,10), 18:(1,11),
        19:(2,10), 20:(2,11),
        21:(3,10), 22:(3,11),
        23:(4,10), 24:(4,11),

        25:(6,2), 26:(6,3),
        27:(7,2), 28:(7,3),
        29:(8,2), 30:(8,3),
        31:(9,2), 32:(9,3),

        33:(6,6), 34:(6,7),
        35:(7,6), 36:(7,7),
        37:(8,6), 38:(8,7),
        39:(9,6), 40:(9,7),

        41:(6,10), 42:(6,11),
        43:(7,10), 44:(7,11),
        45:(8,10), 46:(8,11),
        47:(9,10), 48:(9,11),
    }

    return mapa.get(num)


def main():
    temple_simulado(ordenes[0], 100)


if __name__ == "__main__":
    main()