import pygame
import sys
import time
from Estanteria import Almacen, TAM_CELDA   
from Nodo import Nodo                       
from A_estrella import A_estrella          

def main():
    pygame.init()
    
    #Objetos
    almacen = Almacen()
    buscador = A_estrella()
    
    #Entorno
    ANCHO = almacen.columnas * TAM_CELDA
    ALTO = almacen.filas * TAM_CELDA
    ventana = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Simulación Montacargas A* - TP1-E1")
    
    #  Definir Objetivo 
    estante_destino = 18 #Estante que quiero ir
    inicio = almacen.estacion_carga
    
    #  Calcular el camino antes de empezar la animación 
    print(f"Calculando ruta desde {inicio} hacia estante {estante_destino}...")
    camino = buscador.buscar_camino_astar(almacen, inicio, estante_destino)
    
    if not camino:
        print("No se encontró un camino válido.")
        pygame.quit()
        return
    #print(f"Camino encontrado ({len(camino)} pasos): {camino}")
    

    # 4. Bucle de Animación
    reloj = pygame.time.Clock()
    paso_actual = 0
    ejecutando = True
    camino_recorrido = [] # Aquí guardamos las huellas
    COLOR_HUELLA = (0, 0, 255) # azul 
    

    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
        
        # Dibujar el mapa base [cite: 85]
        ventana.fill((0, 0, 0))
        almacen.renderizar(ventana)
        
        # 1. Dibujar el rastro (huella) de lo que ya caminó
        for pos_huella in camino_recorrido:
            rect_huella = pygame.Rect(pos_huella[1] * TAM_CELDA, 
                                     pos_huella[0] * TAM_CELDA, 
                                     TAM_CELDA, TAM_CELDA)
            pygame.draw.rect(ventana, COLOR_HUELLA, rect_huella)
            # Dibujamos el borde para que no se pierda la grilla
            pygame.draw.rect(ventana, (200, 200, 200), rect_huella, 1)
        
        
        # Dibujar el Agente (Montacargas) en su posición actual
        if paso_actual < len(camino):
            pos = camino[paso_actual]
            # Agregamos la posición actual al rastro
            if pos not in camino_recorrido:
                camino_recorrido.append(pos)
            # Convertir (fila, col) a píxeles (centro de la celda)
            centro_x = pos[1] * TAM_CELDA + TAM_CELDA // 2
            centro_y = pos[0] * TAM_CELDA + TAM_CELDA // 2
            
            
            # Dibujamos al agente como un círculo azul [cite: 10]
            pygame.draw.circle(ventana, (0, 0, 255), (centro_x, centro_y), TAM_CELDA // 3)
            
            # Avanzar al siguiente paso (ajustar velocidad aquí)
            paso_actual += 1
            time.sleep(0.2) # Pausa para que el ojo humano vea el movimiento
        else:
            # Si llegó, se queda quieto en el destino
            pos = camino[-1]
            centro_x = pos[1] * TAM_CELDA + TAM_CELDA // 2
            centro_y = pos[0] * TAM_CELDA + TAM_CELDA // 2
            pygame.draw.circle(ventana, (0, 200, 0), (centro_x, centro_y), TAM_CELDA // 3)
        

        pygame.display.flip()
        reloj.tick(10) # Bajamos los FPS para que la animación sea pausada

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    
    main()