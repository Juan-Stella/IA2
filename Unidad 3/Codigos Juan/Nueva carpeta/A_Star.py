import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Nodo:

    def __init__(self, posicion, padre=None):
        self.posicion = posicion
        self.padre = padre

        self.g = 0
        self.h = 0
        self.f = 0


class A_Star:

    def __init__(self, grafo):
        self.grafo = grafo;
        pass
    

    def heuristica(self,nodo,objetivo):
        #Utilizaremos distancia Manhattan, para sub-estimar lo menos posible. Además, suponemos movimientos solamente
        #En horizontal o vertical, no en diagonal
        x1, y1 = nodo.posicion
        x2, y2 = objetivo.posicion
        h = abs(x1-x2) + abs(y1-y2)
        return h
    
    def funcion_eval(self,h,g):
        #Si bien el costo será 1, pero la hacemos para que el programa sea general, en otro caso g podría no serlo
        f = g + h
        return f
    
    def objetivo_accesible(self, pos_objetivo):
        fila, col = pos_objetivo
        filas, cols = self.grafo.shape

        # Si ya es accesible, lo devuelve igual
        if self.grafo[fila, col] == 0:
            return pos_objetivo

        # Si es obstáculo, probar izquierda primero
        if 0 <= col - 1 < cols and self.grafo[fila, col - 1] == 0:
            return (fila, col - 1)

        # Si no, probar derecha
        if 0 <= col + 1 < cols and self.grafo[fila, col + 1] == 0:
            return (fila, col + 1)

        # Si no hay acceso horizontal
        return None

    def buscar_vecinos(self,nodo):
        x,y = nodo.posicion
        filas, cols = self.grafo.shape
        movimientos_posibles = [
            (x+1,y),
            (x-1,y),
            (x,y+1),
            (x,y-1),
        ]
        vecinos = []
        for nx, ny in movimientos_posibles:
            if 0 <= nx < filas and 0 <= ny < cols:
            # 2) celda libre (0)
                if self.grafo[nx, ny] == 0:
                    vecinos.append(Nodo((nx, ny), nodo))
        return vecinos


    def reconstruir_camino(self,nodo):
        camino = []
        actual = nodo
        while actual is not None:
            camino.append(actual.posicion)
            actual = actual.padre
        return camino[::-1] 
    
    def busqueda(self, pos_inicio, pos_final):
        inicio = Nodo(pos_inicio)

        pos_meta_real = pos_final
        pos_meta_accesible = self.objetivo_accesible(pos_meta_real)

        if pos_meta_accesible is None:
            return None

        meta = Nodo(pos_meta_accesible)

        lista_abierta = []
        lista_cerrada = set()    
        abierta_pos = set()      

        # importante: inicializar f del inicio
        inicio.g = 0
        inicio.h = self.heuristica(inicio, meta)
        inicio.f = inicio.g + inicio.h

        lista_abierta.append(inicio)
        abierta_pos.add(inicio.posicion)

        while lista_abierta:
            actual = min(lista_abierta, key=lambda nodo: nodo.f)
            lista_abierta.remove(actual)
            abierta_pos.remove(actual.posicion)

            # marcar como cerrado por POSICIÓN
            lista_cerrada.add(actual.posicion)

            if actual.posicion == meta.posicion:
                return self.reconstruir_camino(actual)

            vecinos = self.buscar_vecinos(actual)
            for vecino in vecinos:
                if vecino.posicion in lista_cerrada:
                    continue
                if vecino.posicion in abierta_pos:
                    continue

                vecino.g = actual.g + 1
                vecino.h = self.heuristica(vecino, meta)
                vecino.f = vecino.g + vecino.h

                lista_abierta.append(vecino)
                abierta_pos.add(vecino.posicion)

        return None
            

def plot_path(grafo, camino, inicio=None, final=None, title="Grilla con camino"):
    # Copia para no modificar el grafo original
    grid = grafo.copy()

    # Pintar camino
    if camino:
        for (r, c) in camino:
            if grid[r, c] == 0:   # no pises obstáculos
                grid[r, c] = 2

    # Marcar inicio / final
    if inicio is not None:
        r, c = inicio
        grid[r, c] = 3
    if final is not None:
        r, c = final
        grid[r, c] = 4

    # Colores: libre, obstáculo, camino, inicio, final
    cmap = ListedColormap(["#f4f4f4", "#030303", "#df8bdd", "#3df021", "#de1c1c"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4)

    # Grilla
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color="black", linewidth=1)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color="black", linewidth=1)

    # Texto dentro de las celdas
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):

            # Obstáculos
            if grafo[i, j] == 1:
                ax.text(j, i, "X",
                        ha="center", va="center",
                        color="white", fontsize=12, fontweight="bold")

            # Start
            if inicio is not None and (i, j) == inicio:
                ax.text(j, i, "START",
                        ha="center", va="center",
                        color="black", fontsize=8, fontweight="bold")

            # Finish
            if final is not None and (i, j) == final:
                ax.text(j, i, "FINISH",
                        ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)

    plt.tight_layout()
    plt.show()


def main():
    # Crear la grilla
    grafo = np.zeros((11, 13), dtype=int)

    for row_start in [1,6]:
        for col_start in [2, 6, 10]:
            for r in range(row_start, row_start + 4):
                for c in range(col_start, col_start + 2):
                    grafo[r, c] = 1

    astar = A_Star(grafo)

    inicio = (0, 0)   # (fila, col)
    final  = (9, 11)

    if grafo[inicio] == 1:
        print("Error: START está en un obstáculo.")
        return


    camino = astar.busqueda(inicio, final)

    if camino is None:
        print("No existe camino posible entre START y FINISH.")
    else:
        print(camino)
        print("Objetivo real:", final)
        print("Objetivo accesible:", camino[-1])
        plot_path(grafo, camino, inicio=inicio, final=camino[-1])


if __name__ == "__main__":
    main()