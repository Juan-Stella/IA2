
import pygame
import sys


# --- CONFIGURACIÓN GLOBAL ---
TAM_CELDA = 50
COLOR_PASILLO = (255, 255, 255) #BLANCO
COLOR_ESTANTE = (50, 50, 50) #GRIS OSCURO
COLOR_CARGA = (255, 255, 0) #AMARILLO
COLOR_LINEA = (200, 200, 200) #GRIS CLARO

class Almacen:
    def __init__(self):
        self.filas = 11
        self.columnas = 13
        # Inicializamos matriz de pasillos (0)
        self.grid = [[0 for _ in range(self.columnas)] for _ in range(self.filas)] #Todos cero. self.grid[fila][columna]
        self.estacion_carga = (5, 0) # Casilla 'C'
        self.estantes_coords = {}    # Diccionario nro -> (f, c)
        self._configurar_escenario()

    def _configurar_escenario(self):
        """Mapea los 6 bloques de estantes"""
        filas_bloques = [(1, 4), (6, 9)]
        columnas_bloques = [(2, 3), (6, 7), (10, 11)]
        
        contador = 1 #a 48
        for fb in filas_bloques: #(1,4)
            for cb in columnas_bloques:  #(2,3)
                for f in range(fb[0], fb[1] + 1): #(1 a 4+1)= 1,2,3,4
                    for c in range(cb[0], cb[1] + 1): #(2 a 3+1)= 2,3
                        self.grid[f][c] = 1 # 1 = Obstáculo - Antes tenía 0
                        self.estantes_coords[contador] = (f, c) #Numeramos los estantes 1-48
                        contador += 1

    def renderizar(self, ventana):
        """Dibuja la grilla y los elementos en pantalla."""
        for f in range(self.filas): #0-12 _ fila 0
            for c in range(self.columnas): #0-11 _ columna 0
                rect = pygame.Rect(c * TAM_CELDA, f * TAM_CELDA, TAM_CELDA, TAM_CELDA) #(x,y, ancho,alto )
                
                # Definir color según el tipo de celda
                if (f, c) == self.estacion_carga:
                    color = COLOR_CARGA
                elif self.grid[f][c] == 1:
                    color = COLOR_ESTANTE
                else:
                    color = COLOR_PASILLO
                
                pygame.draw.rect(ventana, color, rect) #recuadro
                pygame.draw.rect(ventana, COLOR_LINEA, rect, 1) #cuadrilla
                
        ##MAIN1.PY        
    """def obtener_vecinos(self, nodo_pos):
        #Retorna celdas adyacentes que sean pasillos (valor 0) 
        f, c = nodo_pos
        vecinos = []
        # Definimos los movimientos: derecha, izquierda, abajo, arriba
        for df, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nf, nc = f + df, c + dc
            # 1. Verificar que el vecino esté dentro de los límites del almacén 
            if 0 <= nf < self.filas and 0 <= nc < self.columnas:
                # 2. Verificar que la celda sea un pasillo (no un estante) 
                if self.grid[nf][nc] == 0:
                    vecinos.append((nf, nc))
        return vecinos"""
    
    
    ##MAIN2.PY
    def obtener_vecinos(self, nodo_pos, posiciones_obstaculos_dinamicos=None):
        """
        Retorna celdas adyacentes que sean pasillos.
        posiciones_obstaculos_dinamicos: lista de tuplas [(f, c)] con posiciones de otros agentes.
        """
        f, c = nodo_pos
        vecinos = []
        
        # Si no nos pasan obstáculos dinámicos, usamos una lista vacía
        if posiciones_obstaculos_dinamicos is None:
            posiciones_obstaculos_dinamicos = []

        for df, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nf, nc = f + df, c + dc
            
            # 1. Verificar límites del mapa
            if 0 <= nf < self.filas and 0 <= nc < self.columnas:
                # 2. Verificar que no sea un estante (grid[nf][nc] == 1)
                # 3. VERIFICAR QUE NO SEA UN AGENTE (obstáculo dinámico)
                if self.grid[nf][nc] == 0 and (nf, nc) not in posiciones_obstaculos_dinamicos:
                    vecinos.append((nf, nc))
        return vecinos

"""def main():
    pygame.init()
    
    # Dimensiones: 13 col * 50px = 650 ancho | 11 filas * 50px = 550 alto
    ventana = pygame.display.set_mode((13 * TAM_CELDA, 11 * TAM_CELDA))
    pygame.display.set_caption("TP 1 - Almacén")
    
    almacen = Almacen()
    reloj = pygame.time.Clock()

    while True:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        ventana.fill((0, 0, 0))
        almacen.renderizar(ventana)
        
        pygame.display.flip()
        reloj.tick(60)

if __name__ == "__main__":
    main()"""