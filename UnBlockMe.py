# --- Unblock Me: esqueleto para A* (compatible con tu Aestrella actual) ---

from dataclasses import dataclass
from heapq import heappush, heappop

# Parámetros del tablero estándar
N_FILAS = 6
N_COLS  = 6
ID_ROJO = 0        # asumimos que el bloque rojo tiene id = 0
EXIT_COL = N_COLS - 1   # salida al borde derecho

@dataclass(frozen=True)
class Bloque:
    id: int          # 0 = rojo, 1..n = otros
    fila: int        # fila superior (para H) o fila superior (para V)
    col: int         # columna izquierda (para H) o columna superior (para V)
    largo: int       # 2 o 3 típicamente
    orient: str      # 'H' o 'V'

    def celdas(self):
        """Celdas ocupadas por el bloque (lista de (f,c))."""
        if self.orient == 'H':
            return [(self.fila, self.col + k) for k in range(self.largo)]
        else:
            return [(self.fila + k, self.col) for k in range(self.largo)]

class NodoUnblock:
    def __init__(self, estado, padre=None, g=0):
        """
        estado: tupla inmutable de Bloque(s), ordenada por id para hashing estable.
        padre : backpointer
        g     : costo acumulado (# movimientos)
        """
        # Normalizamos: ordena bloques por id, garantiza inmutabilidad
        self.estado = tuple(sorted(estado, key=lambda b: b.id))
        self.padre  = padre
        self.g      = g
        self.h      = self.heuristica()           # h admisible
        self.f      = self.g + self.h

    # Para heapq (orden por f)
    def __lt__(self, otro):
        return self.f < otro.f

    # Igualdad por estado (clave para evitar repetidos)
    def __eq__(self, otro):
        return isinstance(otro, NodoUnblock) and self.estado == otro.estado

    def __hash__(self):
        return hash(self.estado)

    def __str__(self):
        """Pretty print del tablero actual (6x6 por defecto)."""
        grid = [['.' for _ in range(N_COLS)] for _ in range(N_FILAS)]
        for b in self.estado:
            ch = 'R' if b.id == ID_ROJO else str(b.id)
            for (f,c) in b.celdas():
                grid[f][c] = ch
        return '\n'.join(' '.join(row) for row in grid)

    # ---------- Lógica del problema ----------
    def esMeta(self):
        """La pieza roja alcanzó la salida (su borde derecho tocó EXIT_COL)."""
        rojo = self.estado[0]  # por normalización, id 0 va primero
        assert rojo.id == ID_ROJO and rojo.orient == 'H', "El rojo debe ser H y tener id=0"
        borde_derecho = rojo.col + rojo.largo - 1
        return borde_derecho == EXIT_COL

    def heuristica(self):
        """
        h = distancia horizontal restante del rojo hasta la salida
            + número de bloques que bloquean directamente su salida.
        Admisible si el costo es 1 por movimiento (deslizar cualquier # de celdas = 1).
        """
        rojo = self.estado[0]
        # 1) distancia mínima de celdas que le faltan al rojo para tocar la salida
        dist = (EXIT_COL - (rojo.col + rojo.largo - 1))
        if dist < 0:
            dist = 0  # defensivo

        # 2) bloqueadores: bloques que ocupan la misma fila y están a la derecha del rojo
        fila_r = rojo.fila
        col_r_fin = rojo.col + rojo.largo - 1

        # Construye ocupación rápida (opcionalmente puedes evitarlo y chequear por celdas)
        bloques_bloqueadores = 0
        vistos = set()
        for b in self.estado[1:]:
            for (f,c) in b.celdas():
                if f == fila_r and c > col_r_fin:
                    # hay al menos una celda del bloque en el camino del rojo
                    if b.id not in vistos:
                        vistos.add(b.id)
                        bloques_bloqueadores += 1
                    break

        return dist + bloques_bloqueadores

    def sucesores(self, ABIERTOS, CERRADOS):
        """
        Genera todos los estados alcanzables en 1 movimiento:
        - Cada bloque se puede deslizar ±k celdas en su eje hasta chocar.
        - Cada deslizamiento (una o más celdas en una dirección) cuesta 1.
        Filtra estados ya en ABIERTOS/CERRADOS (comparación por estado).
        """
        suc = []
        ocupadas = self._mapa_ocupacion()

        for idx, b in enumerate(self.estado):
            if b.orient == 'H':
                # Mover a la izquierda
                pasos = 1
                while self._libre(b.fila, b.col - pasos, ocupadas, b, horizontal=True):
                    nuevo = self._mueve(b, -pasos, horizontal=True)
                    nodo = self._nuevo_estado(idx, nuevo)
                    if nodo not in ABIERTOS and nodo not in CERRADOS:
                        suc.append(nodo)
                    pasos += 1
                # Mover a la derecha
                pasos = 1
                while self._libre(b.fila, b.col + b.largo - 1 + pasos, ocupadas, b, horizontal=True):
                    nuevo = self._mueve(b, pasos, horizontal=True)
                    nodo = self._nuevo_estado(idx, nuevo)
                    if nodo not in ABIERTOS and nodo not in CERRADOS:
                        suc.append(nodo)
                    pasos += 1
            else:  # 'V'
                # Mover hacia arriba
                pasos = 1
                while self._libre(b.fila - pasos, b.col, ocupadas, b, horizontal=False):
                    nuevo = self._mueve(b, -pasos, horizontal=False)
                    nodo = self._nuevo_estado(idx, nuevo)
                    if nodo not in ABIERTOS and nodo not in CERRADOS:
                        suc.append(nodo)
                    pasos += 1
                # Mover hacia abajo
                pasos = 1
                while self._libre(b.fila + b.largo - 1 + pasos, b.col, ocupadas, b, horizontal=False):
                    nuevo = self._mueve(b, pasos, horizontal=False)
                    nodo = self._nuevo_estado(idx, nuevo)
                    if nodo not in ABIERTOS and nodo not in CERRADOS:
                        suc.append(nodo)
                    pasos += 1

        return suc

    # ---------- Helpers internos ----------
    def _mapa_ocupacion(self):
        """Devuelve un set de celdas ocupadas {(f,c), ...} para el estado actual."""
        occ = set()
        for b in self.estado:
            for fc in b.celdas():
                occ.add(fc)
        return occ

    def _libre(self, f, c, ocupadas, bloque, horizontal=True):
        """¿Está libre la celda (f,c) y dentro de tablero para extender el bloque en ese sentido?"""
        if not (0 <= f < N_FILAS and 0 <= c < N_COLS):
            return False
        # Si la celda pertenece al propio bloque en su extremo que se mueve, no cuenta como ocupada
        # pero en nuestro chequeo de avance horizontal/vertical miramos la nueva "punta".
        return (f, c) not in ocupadas

    def _mueve(self, b, pasos, horizontal=True):
        """Devuelve un nuevo Bloque trasladado 'pasos' celdas en su eje."""
        if horizontal:
            return Bloque(b.id, b.fila, b.col + pasos, b.largo, b.orient)
        else:
            return Bloque(b.id, b.fila + pasos, b.col, b.largo, b.orient)

    def _nuevo_estado(self, idx_bloque, bloque_nuevo):
        """Crea un NodoUnblock con el bloque idx reemplazado por bloque_nuevo, g+1."""
        lista = list(self.estado)
        lista[idx_bloque] = bloque_nuevo
        return NodoUnblock(tuple(lista), padre=self, g=self.g + 1)


# ---------- Ejemplo de uso con tu A* existente ----------
# Puedes usar tu misma función Aestrella(inicial) si espera atributos .f, .esMeta(), .sucesores(...)
# inicial = NodoUnblock(estado_inicial)
# solucion, expandidos = Aestrella(inicial)
