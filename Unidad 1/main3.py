import pygame
import sys
import time 
import random
import matplotlib.pyplot as plt
from Estanteria import Almacen, TAM_CELDA 
from A_estrella import A_estrella
from TempleSimulado import TempleSimulado

def leer_ordenes(archivo):
    with open(archivo, 'r') as f:
        lineas = f.readlines()
    return [list(map(int, l.strip().split(','))) for l in lineas]

def main():
    almacen = Almacen()
    buscador = A_estrella()
    ts = TempleSimulado(almacen, buscador)
    
    # 1. Cargar órdenes
    ordenes = leer_ordenes('ordenes.csv')
    
    print("1. Elegir pedido aleatorio")
    print("2. Ingresar pedido manual (ej: 14,10,46,5)")
    opcion = input("Seleccione una opción: ")
    
    if opcion == "1":
        pedido = random.choice(ordenes)
    else:
        pedido = list(map(int, input("Ingrese los IDs separados por coma: ").split(',')))
        
    print(f"\nOrden original: {pedido}")
    costo_ini = ts.calcular_costo_total(pedido)
    print(f"Costo inicial: {costo_ini}")

    # 2. Optimización (Búsqueda Local) 
    iteraciones_totales = 300
    mejor_orden, mejor_costo, historial = ts.optimizar(pedido, iteraciones=iteraciones_totales)
    
    print(f"Mejor orden encontrado: {mejor_orden}")
    print(f"Costo final: {mejor_costo}")

    # 3. Gráfico Doble (Costo y Temperatura) [cite: 120, 121]
    temp_historial = [ts.T * (ts.enfriamiento ** i) for i in range(len(historial))]
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Costo (Pasos A*)', color='tab:blue')
    ax1.plot(historial, color='tab:blue', label='Costo de la ruta')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Temperatura', color='tab:orange')
    ax2.plot(temp_historial, color='tab:orange', label='Temperatura', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Evolución de Costo y Enfriamiento')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # El programa se pausará aquí hasta que cierres la ventana del gráfico
    plt.show() 

    
    print("\nIniciando simulación visual del mejor camino...")
    animar_ruta_optima(almacen, buscador, mejor_orden)
    
    
def animar_ruta_optima(almacen, buscador, mejor_orden):
    pygame.init()
    ANCHO = almacen.columnas * TAM_CELDA
    ALTO = almacen.filas * TAM_CELDA
    ventana = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Simulación de Picking Optimizado (IA II)")

    # Definimos colores nuevos
    COLOR_ESTANTE_OBJETIVO = (255, 255, 0) # Amarillo brillante
    COLOR_HUELLA = (0, 200, 0) # Verde suave

    # Lista para guardar el rastro
    camino_recorrido = []
    
    pos_actual = almacen.estacion_carga
    reloj = pygame.time.Clock()
    
    # Recorremos cada estante en el orden optimizado por el Temple Simulado
    for id_estante in mejor_orden:
        camino = buscador.buscar_camino_astar(almacen, pos_actual, id_estante)
        
        if camino:
            for paso in camino:
                for evento in pygame.event.get():
                    if evento.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Agregamos la posición actual al rastro
                if paso not in camino_recorrido:
                    camino_recorrido.append(paso)

                # --- RENDERIZADO (El orden importa) ---
                ventana.fill((0, 0, 0)) # Fondo negro
                almacen.renderizar(ventana) # Estantes grises y pasillos blancos

                # 1. Dibujar los Estantes que forman parte del Pedido (Amarillo)
                for estante_pedido_id in mejor_orden:
                    e_pos = almacen.estantes_coords[estante_pedido_id]
                    rect_estante = pygame.Rect(e_pos[1] * TAM_CELDA, e_pos[0] * TAM_CELDA, TAM_CELDA, TAM_CELDA)
                    pygame.draw.rect(ventana, COLOR_ESTANTE_OBJETIVO, rect_estante)
                    pygame.draw.rect(ventana, (255, 255, 255), rect_estante, 1) # Borde blanco

                # 2. Dibujar la huella del camino recorrido (Verde)
                for pos_h in camino_recorrido:
                    rect_h = pygame.Rect(pos_h[1] * TAM_CELDA, pos_h[0] * TAM_CELDA, TAM_CELDA, TAM_CELDA)
                    pygame.draw.rect(ventana, COLOR_HUELLA, rect_h)
                    pygame.draw.rect(ventana, (200, 200, 200), rect_h, 1) # Borde suave

                # 3. Dibujar al agente (Círculo Azul)
                centro = (paso[1] * TAM_CELDA + TAM_CELDA // 2, paso[0] * TAM_CELDA + TAM_CELDA // 2)
                pygame.draw.circle(ventana, (0, 0, 255), centro, TAM_CELDA // 3)
                
                pygame.display.flip()
                reloj.tick(5) # Velocidad
            
            # Al llegar, el fin de este tramo es el inicio del siguiente
            pos_actual = camino[-1]
            time.sleep(0.5) # Pausa al recolectar
            print(f"Producto {id_estante} recolectado.")

    print("Orden completada")
    esperando = True
    while esperando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                esperando = False
        
        # Opcional: podés dejar un mensaje en consola
        pygame.display.flip()
        reloj.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()