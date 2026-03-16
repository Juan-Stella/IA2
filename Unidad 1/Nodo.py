import math

class Nodo:
    def __init__(self, posicion, padre=None):
        self.posicion = posicion # (fila, columna)
        self.padre = padre
        
        self.g = 0 # Costo del camino 
        self.h = 0 # Heurística 
        self.f = 0 # Costo total 

    # Para poder comparar nodos en las listas (Frontera/Explorados) 
    def __eq__(self, otro):
        return self.posicion == otro.posicion

    # Para que el algoritmo sepa cuál es el "mejor" nodo (menor f) [cite: 192]
    def __lt__(self, otro):
        return self.f < otro.f