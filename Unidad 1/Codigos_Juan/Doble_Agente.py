import A_Star as A
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch

def plot_two_paths(ax, grafo, camino1, camino2, start1, goal1, start2, goal2, pos1=None, pos2=None, celdas_colision=None):
    ax.clear()
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

    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)

    # Dibujar celdas compartidas mitad y mitad
    for r, c in compartidas:
        x = c - 0.5
        y = r - 0.5

        ax.add_patch(Rectangle((x, y), 0.5, 1,
                               facecolor="#df8bdd", edgecolor="none"))
        ax.add_patch(Rectangle((x + 0.5, y), 0.5, 1,
                               facecolor="#00bcd4", edgecolor="none"))

    # Dibujar las celdas de colisión en rojo
    if celdas_colision:
        for r, c in celdas_colision:
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

    # Dibujar a los agentes temporalmente
    if pos1 is not None:
        ax.add_patch(plt.Circle((pos1[1], pos1[0]), 0.3, color='magenta', zorder=10))
        ax.text(pos1[1], pos1[0], "A1", ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=11)
        
    if pos2 is not None:
        ax.add_patch(plt.Circle((pos2[1], pos2[0]), 0.3, color='blue', zorder=10))
        ax.text(pos2[1], pos2[0], "A2", ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=11)

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


def main():
    # Posiciones de los agentes
    start_agente1 = (0, 0)
    start_agente2 = (10, 12)

    goal_agente1 = (9, 9)
    goal_agente2 = (2, 1)

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

    camino_real1 = [start_agente1]
    camino_real2 = [start_agente2]

    pos1 = start_agente1
    pos2 = start_agente2
    
    celdas_colision = []
    pasos_max = 1000

    print("Iniciando simulación interactiva...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            c = int(round(event.xdata))
            r = int(round(event.ydata))
            if 0 <= r < grafo.shape[0] and 0 <= c < grafo.shape[1]:
                if grafo[r, c] == 0:
                    grafo[r, c] = 1
                    print(f"\\n[EVENTO] Nuevo obstáculo añadido en: ({r}, {c})")
                elif grafo[r, c] == 1:
                    grafo[r, c] = 0
                    print(f"\\n[EVENTO] Obstáculo quitado en: ({r}, {c})")

    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Dibujo inicial
    plot_two_paths(ax, grafo, camino_real1, camino_real2, start_agente1, goal_agente1, start_agente2, goal_agente2, pos1, pos2, celdas_colision)
    plt.pause(0.5)

    while (pos1 != goal_agente1 or pos2 != goal_agente2) and pasos_max > 0:
        pasos_max -= 1
        
        # 1. Determinar el próximo paso de cada agente de acuerdo a su plan
        if pos1 != goal_agente1 and camino_agente1 and len(camino_agente1) > 1:
            next1 = camino_agente1[1]
        else:
            next1 = pos1

        if pos2 != goal_agente2 and camino_agente2 and len(camino_agente2) > 1:
            next2 = camino_agente2[1]
        else:
            next2 = pos2

        # 1.5 Evitar obstáculos en el entorno (añadidos por click)
        if grafo[next1] == 1:
            print(f"Agente 1 detecta obstáculo interactivo en {next1}!")
            astar_temp = A.A_Star(grafo)
            nuevo_camino1 = astar_temp.busqueda(pos1, goal_agente1)
            if nuevo_camino1:
                camino_agente1 = nuevo_camino1
                next1 = camino_agente1[1] if len(camino_agente1) > 1 else pos1
            else:
                print(f"Agente 1 atrapado por obstáculo. Espera en {pos1}.")
                next1 = pos1
                
        if grafo[next2] == 1:
            print(f"Agente 2 detecta obstáculo interactivo en {next2}!")
            astar_temp = A.A_Star(grafo)
            nuevo_camino2 = astar_temp.busqueda(pos2, goal_agente2)
            if nuevo_camino2:
                camino_agente2 = nuevo_camino2
                next2 = camino_agente2[1] if len(camino_agente2) > 1 else pos2
            else:
                print(f"Agente 2 atrapado por obstáculo. Espera en {pos2}.")
                next2 = pos2

        # 2. El sensor detecta colisiones entre agentes
        colision = False
        celda_bloqueo = None
        
        if next1 == next2 and next1 != pos1 and next2 != pos2:
            colision = True
            celda_bloqueo = next1
            print(f"Colisión inminente detectada en {celda_bloqueo}! Ambos intentan acceder a la misma celda.")
        elif next1 == pos2 and next2 == pos1 and pos1 != pos2:
            colision = True
            celda_bloqueo = pos2
            print(f"Choque frontal inminente entre {pos1} y {pos2}!")
        elif next1 == pos2 and next2 == pos2 and pos1 != pos2:
            colision = True
            celda_bloqueo = pos2
            print(f"Agente 1 choca con Agente 2 que está detenido en {pos2}!")
        elif next2 == pos1 and next1 == pos1 and pos1 != pos2:
            colision = True
            celda_bloqueo = pos1
            print(f"Agente 2 choca con Agente 1 que está detenido en {pos1}!")

        # 3. Recalcular si hay colisión
        if colision:
            if celda_bloqueo not in celdas_colision:
                celdas_colision.append(celda_bloqueo)

            len1 = len(camino_agente1) if camino_agente1 else 0
            len2 = len(camino_agente2) if camino_agente2 else 0

            if len1 >= len2:
                # Agente 1 recalcula asumiendo celda_bloqueo es obstáculo
                grafo_temp = grafo.copy()
                grafo_temp[celda_bloqueo] = 1
                astar_temp = A.A_Star(grafo_temp)
                nuevo_camino1 = astar_temp.busqueda(pos1, goal_agente1)
                
                if nuevo_camino1:
                    camino_agente1 = nuevo_camino1
                    next1_attempt = camino_agente1[1] if len(camino_agente1) > 1 else pos1
                    if grafo[next1_attempt] == 0:
                        next1 = next1_attempt
                        print(f"Agente 1 recalculó su ruta desde {pos1}.")
                    else:
                        print(f"Agente 1 recalculó pero su primer paso está bloqueado por el entorno, espera en {pos1}.")
                        next1 = pos1
                else:
                    print(f"Agente 1 no encontró ruta alternativa desde {pos1} y espera.")
                    next1 = pos1 # Espera
            else:
                # Agente 2 recalcula
                grafo_temp = grafo.copy()
                obs2 = next1 if next1 != next2 else pos1
                grafo_temp[obs2] = 1
                astar_temp = A.A_Star(grafo_temp)
                nuevo_camino2 = astar_temp.busqueda(pos2, goal_agente2)
                
                if nuevo_camino2:
                    camino_agente2 = nuevo_camino2
                    next2_attempt = camino_agente2[1] if len(camino_agente2) > 1 else pos2
                    if grafo[next2_attempt] == 0:
                        next2 = next2_attempt
                        print(f"Agente 2 recalculó su ruta desde {pos2}.")
                    else:
                        print(f"Agente 2 recalculó pero su primer paso está bloqueado por el entorno, espera en {pos2}.")
                        next2 = pos2
                else:
                    print(f"Agente 2 no encontró ruta alternativa desde {pos2} y espera.")
                    next2 = pos2 # Espera

        # 4. Mover a los agentes
        pos1 = next1
        pos2 = next2
        
        # Guardar en rutas efectivas (solo si cambian para evitar repetidos)
        if camino_real1[-1] != pos1:
            camino_real1.append(pos1)
        if camino_real2[-1] != pos2:
            camino_real2.append(pos2)

        # Actualizar los planes restando el paso que acabamos de dar
        if camino_agente1 and len(camino_agente1) > 1 and pos1 == camino_agente1[1]:
            camino_agente1 = camino_agente1[1:]
        if camino_agente2 and len(camino_agente2) > 1 and pos2 == camino_agente2[1]:
            camino_agente2 = camino_agente2[1:]

        plot_two_paths(ax, grafo, camino_real1, camino_real2, start_agente1, goal_agente1, start_agente2, goal_agente2, pos1, pos2, celdas_colision)
        plt.pause(0.5)

    if pasos_max == 0:
        print("Se alcanzó el límite de pasos, posible encierro infinito.")
    else:
        print("Ambos agentes han llegado a su destino o no pueden avanzar más.")

    print("Costo agente 1 efectivo: ", len(camino_real1) - 1)
    print("Costo agente 2 efectivo: ", len(camino_real2) - 1)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()