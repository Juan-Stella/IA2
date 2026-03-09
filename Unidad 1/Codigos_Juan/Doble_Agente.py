import A_Star as A
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch

def plot_two_paths(grafo, camino1, camino2, start1, goal1, start2, goal2, celda_colision=None):
    grid = grafo.copy()

    set1 = set(camino1) if camino1 else set()
    set2 = set(camino2) if camino2 else set()
    compartidas = (set1 & set2)

    # Pintar camino del agente 1 solo donde NO coincide con el otro
    if camino1:
        for r, c in camino1:
            if grid[r, c] == 0 and (r, c) not in compartidas:
                grid[r, c] = 2

    # Pintar camino del agente 2 solo donde NO coincide con el otro
    if camino2:
        for r, c in camino2:
            if grid[r, c] == 0 and (r, c) not in compartidas:
                grid[r, c] = 3

    # Marcar inicio y fin de cada agente
    if start1 is not None:
        grid[start1] = 4
    if goal1 is not None:
        grid[goal1] = 5
    if start2 is not None:
        grid[start2] = 6
    if goal2 is not None:
        grid[goal2] = 7

    cmap = ListedColormap([
        "#f4f4f4",  # 0 libre
        "#030303",  # 1 obstáculo
        "#df8bdd",  # 2 camino agente 1
        "#00bcd4",  # 3 camino agente 2
        "#3df021",  # 4 start agente 1
        "#dece1c",  # 5 finish agente 1
        "#ffd700",  # 6 start agente 2
        "#f0580c",  # 7 finish agente 2
        "#f4f4f4",  # 8 neutro
        "#ff0000"   # 9 colisión
    ])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)

    # Dibujar celdas compartidas mitad y mitad
    for r, c in compartidas:
        x = c - 0.5
        y = r - 0.5

        ax.add_patch(Rectangle((x, y), 0.5, 1,
                               facecolor="#df8bdd", edgecolor="none"))
        ax.add_patch(Rectangle((x + 0.5, y), 0.5, 1,
                               facecolor="#00bcd4", edgecolor="none"))

    # Dibujar la celda de colisión en rojo
    if celda_colision is not None:
        r, c = celda_colision
        x = c - 0.5
        y = r - 0.5
        ax.add_patch(Rectangle((x, y), 1, 1,
                               facecolor="red", edgecolor="black", linewidth=1.5))
        ax.text(c, r, "X",
                ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")

    # Dibujar grilla
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color="black", linewidth=1)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color="black", linewidth=1)

    # Agregar texto en las celdas
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                ax.text(j, i, "X",
                        ha="center", va="center",
                        color="white", fontsize=12, fontweight="bold")

            if start1 is not None and (i, j) == start1:
                ax.text(j, i, "S1",
                        ha="center", va="center",
                        color="black", fontsize=8, fontweight="bold")

            if goal1 is not None and (i, j) == goal1:
                ax.text(j, i, "F1",
                        ha="center", va="center",
                        color="black", fontsize=8, fontweight="bold")

            if start2 is not None and (i, j) == start2:
                ax.text(j, i, "S2",
                        ha="center", va="center",
                        color="black", fontsize=8, fontweight="bold")

            if goal2 is not None and (i, j) == goal2:
                ax.text(j, i, "F2",
                        ha="center", va="center",
                        color="black", fontsize=8, fontweight="bold")

    legend_elements = [
        Patch(facecolor="#030303", edgecolor="black", label="Obstáculo"),
        Patch(facecolor="#df8bdd", edgecolor="black", label="Camino agente 1"),
        Patch(facecolor="#00bcd4", edgecolor="black", label="Camino agente 2"),
        Patch(facecolor="#3df021", edgecolor="black", label="Start agente 1"),
        Patch(facecolor="#dece1c", edgecolor="black", label="Finish agente 1"),
        Patch(facecolor="#ffd700", edgecolor="black", label="Start agente 2"),
        Patch(facecolor="#270de8", edgecolor="black", label="Finish agente 2"),
        Patch(facecolor="#ff0000", edgecolor="black", label="Colisión"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

    ax.set_title("Grilla con caminos de ambos agentes", fontsize=14, fontweight="bold")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)

    plt.tight_layout()
    plt.show()


def main():
    # Posiciones de los agentes
    start_agente1 = (0, 0)
    start_agente2 = (10, 12)

    goal_agente1 = (8, 8)
    goal_agente2 = (2, 4)

    # Crear grilla
    grafo = np.zeros((11, 13), dtype=int)

    for row_start in [1, 6]:
        for col_start in [2, 6, 10]:
            for r in range(row_start, row_start + 4):
                for c in range(col_start, col_start + 2):
                    grafo[r, c] = 1

    astar = A.A_Star(grafo)

    camino_agente1 = astar.busqueda(start_agente1, goal_agente1)
    camino_agente2 = astar.busqueda(start_agente2, goal_agente2)

    if camino_agente1 is None:
        print("No existe camino posible para el agente 1")

    if camino_agente2 is None:
        print("No existe camino posible para el agente 2")

    costo_agente1 = len(camino_agente1) - 1
    costo_agente2 = len(camino_agente2) - 1
    print("Costo agente 1: ", costo_agente1)
    print("Costo agente 2: ", costo_agente2)

    celda_colision = None
    hay_colision = True

    while hay_colision:
        hay_colision = False

        costo_agente1 = len(camino_agente1) - 1
        costo_agente2 = len(camino_agente2) - 1

        for t in range(min(len(camino_agente1), len(camino_agente2))):
            if camino_agente1[t] == camino_agente2[t]:
                print("Colisión en:", camino_agente1[t])
                celda_colision = camino_agente1[t]
                grafo[camino_agente1[t]] = 9
                astar = A.A_Star(grafo)

                if costo_agente1 >= costo_agente2:
                    camino_agente1 = astar.busqueda(start_agente1, goal_agente1)
                    if camino_agente1 is None:
                        print("Agente 1 se quedó sin camino")
                        break
                else:
                    camino_agente2 = astar.busqueda(start_agente2, goal_agente2)
                    if camino_agente2 is None:
                        print("Agente 2 se quedó sin camino")
                        break

                costo_agente1 = len(camino_agente1) - 1 if camino_agente1 is not None else None
                costo_agente2 = len(camino_agente2) - 1 if camino_agente2 is not None else None
                print("Costo agente 1: ", costo_agente1)
                print("Costo agente 2: ", costo_agente2)

                hay_colision = True
                break

    plot_two_paths(
        grafo,
        camino_agente1,
        camino_agente2,
        start_agente1,
        goal_agente1,
        start_agente2,
        goal_agente2,
        celda_colision=celda_colision
    )


if __name__ == "__main__":
    main()