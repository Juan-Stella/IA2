import pygame
import sys
import time
from Estanteria import Almacen, TAM_CELDA
from Nodo import Nodo
from A_estrella import A_estrella

def main():
    pygame.init()
    almacen = Almacen()
    buscador = A_estrella()
    
    ventana = pygame.display.set_mode((almacen.columnas * TAM_CELDA, almacen.filas * TAM_CELDA))
    pygame.display.set_caption(" Multi-Agente,Entorno Dinámico-TP1 E2")

    # --- CONFIGURACIÓN INICIAL ---
    pos1, meta1 = (5, 0), 18
    pos2, meta2 = (5, 12), 1
    
    # Cálculo inicial
    camino1 = buscador.buscar_camino_astar(almacen, pos1, meta1)
    camino2 = buscador.buscar_camino_astar(almacen, pos2, meta2)

    # Imprimimos caminos iniciales como pediste
    #print(f"Camino Inicial Agente 1 (Azul): {camino1}")
    #print(f"Camino Inicial Agente 2 (Rojo): {camino2}")

    # Listas para pintar el rastro
    rastro1, rastro2 = [], []
    
    paso1, paso2 = 0, 0
    reloj = pygame.time.Clock()
    ejecutando = True

    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False

        # --- LÓGICA AGENTE 1 ---
        if paso1 < len(camino1) - 1:
            sig1 = camino1[paso1 + 1]
            if sig1 == pos2: # SENSADO
                print(f"gente 1 detectó colisión en {sig1}. RECALCULANDO...")
                nuevo = buscador.buscar_camino_astar(almacen, pos1, meta1, obstaculos_dinamicos=[pos2])
                if nuevo:
                    camino1 = nuevo
                    paso1 = 0
                    #print(f"--- Nuevo camino Agente 1: {camino1}")
            else:
                paso1 += 1
                pos1 = camino1[paso1]
                if pos1 not in rastro1: rastro1.append(pos1)

        # --- LÓGICA AGENTE 2 ---
        if paso2 < len(camino2) - 1:
            sig2 = camino2[paso2 + 1]
            if sig2 != pos1:
                paso2 += 1
                pos2 = camino2[paso2]
                if pos2 not in rastro2: rastro2.append(pos2)
            else:
                print("Agente 2 esperando...")

        # --- DIBUJO ---
        ventana.fill((0, 0, 0))
        almacen.renderizar(ventana)

        # Dibujar rastros (transparentes o colores suaves)
        for r in rastro1:
            pygame.draw.rect(ventana, (100, 100, 255), (r[1]*50, r[0]*50, 50, 50)) # Rastro Azul
        for r in rastro2:
            pygame.draw.rect(ventana, (255, 100, 100), (r[1]*50, r[0]*50, 50, 50)) # Rastro Rojo

        # Dibujar Agentes (Círculos)
        pygame.draw.circle(ventana, (0, 0, 255), (pos1[1]*50+25, pos1[0]*50+25), 18) # Agente 1
        pygame.draw.circle(ventana, (255, 0, 0), (pos2[1]*50+25, pos2[0]*50+25), 18) # Agente 2

        pygame.display.flip()
        time.sleep(0.2)
        reloj.tick(10)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()