import math
import random
import copy

class TempleSimulado:
    def __init__(self, almacen, buscador, temp_inicial=100, enfriamiento=0.95):
        self.almacen = almacen
        self.buscador = buscador
        self.T = temp_inicial
        self.enfriamiento = enfriamiento

    def calcular_costo_total(self, orden):
        """Calcula la distancia total recorriendo la lista en el orden dado."""
        costo_total = 0
        pos_actual = self.almacen.estacion_carga # Empezamos en C
        
        for id_estante in orden:
            camino = self.buscador.buscar_camino_astar(self.almacen, pos_actual, id_estante)
            if camino:
                costo_total += len(camino)
                pos_actual = camino[-1] # El fin de este tramo es el inicio del siguiente
        return costo_total

    def generar_vecino(self, orden):
        """Genera un vecino intercambiando dos productos al azar."""
        nueva_orden = copy.copy(orden)
        i, j = random.sample(range(len(nueva_orden)), 2)
        nueva_orden[i], nueva_orden[j] = nueva_orden[j], nueva_orden[i]
        return nueva_orden

    def optimizar(self, orden_inicial, iteraciones=100):
        actual = orden_inicial
        mejor = actual
        costo_actual = self.calcular_costo_total(actual)
        mejor_costo = costo_actual
        
        historial_costos = [costo_actual]
        
        temp = self.T
        for i in range(iteraciones):
            vecino = self.generar_vecino(actual)
            costo_vecino = self.calcular_costo_total(vecino)
            
            delta_e = costo_vecino - costo_actual 
            
            # Si es mejor (menor costo) o por probabilidad si es peor 
            if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
                actual = vecino
                costo_actual = costo_vecino
                
                if costo_actual < mejor_costo:
                    mejor = actual
                    mejor_costo = costo_actual
            
            historial_costos.append(costo_actual)
            temp *= self.enfriamiento # Enfriamiento 
            
        return mejor, mejor_costo, historial_costos