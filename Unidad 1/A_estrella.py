import math
from Nodo import Nodo
from Estanteria import Almacen


class A_estrella:
        
    def distancia_euclidiana(self, pos1, pos2):
        """Calcula h(n) entre dos coordenadas (f, c)"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    


    def buscar_camino_astar(self, almacen, inicio, id_estante_objetivo, obstaculos_dinamicos=None):
        # 1. Determinar el destino real (pasillo adyacente al estante)
        pos_estante = almacen.estantes_coords[id_estante_objetivo]
        
        # Probamos izquierda o derecha del estante (el que sea pasillo)
        objetivo = None
        for dc in [-1, 1]:
            pos_posible = (pos_estante[0], pos_estante[1] + dc)
            # Verificamos que esté en el mapa y sea pasillo (valor 0)
            if 0 <= pos_posible[1] < almacen.columnas and almacen.grid[pos_posible[0]][pos_posible[1]] == 0:
                objetivo = pos_posible
                break
                
        if not objetivo: return None

        # 2. Inicializar listas
        lista_frontera = [] # Nodos por explorar
        lista_explorados = set() # Posiciones ya visitadas
        
        nodo_raiz = Nodo(inicio)
        lista_frontera.append(nodo_raiz)

        # 3. Bucle de búsqueda [cite: 175, 191]
        while lista_frontera:
            # Elegir el nodo con menor f(n) [cite: 192, 206]
            lista_frontera.sort() # Gracias al __lt__ de la clase Nodo
            nodo_actual = lista_frontera.pop(0)
            
            # ¿Llegamos al objetivo?
            if nodo_actual.posicion == objetivo:
                return self.reconstruir_camino(nodo_actual)
                
            lista_explorados.add(nodo_actual.posicion) 

            # 4. Expansión de vecinos
            for v_pos in almacen.obtener_vecinos(nodo_actual.posicion, obstaculos_dinamicos):
                if v_pos in lista_explorados:
                    continue
                
                vecino = Nodo(v_pos, nodo_actual)
                vecino.g = nodo_actual.g + 1 # Costo uniforme = 1
                vecino.h = self.distancia_euclidiana(vecino.posicion, objetivo)
                vecino.f = vecino.g + vecino.h 
                
                # Si ya está en la frontera con un costo mejor, lo ignoramos
                if any(n for n in lista_frontera if n == vecino and n.g <= vecino.g):
                    continue
                    
                lista_frontera.append(vecino)
                
        return None # No hay camino

    def reconstruir_camino(self, nodo):
        camino = []
        actual = nodo
        while actual:
            camino.append(actual.posicion)
            actual = actual.padre
        return camino[::-1] # Invertir para que vaya de inicio a fin