# -*- coding: utf-8 -*-
"""
Unblock Me - Menú con 3 niveles fijos (6x6, 8x8, 20x20) y A*

"""

import time
import random
from heapq import heappush, heappop

# ---------------- CONFIG DEL PROBLEMA ----------------
ROWS, COLS = 6, 6   # por defecto (se ajusta al elegir nivel)
EXPLORATION_RATE = 0.2  # Probabilidad de exploración aleatoria
RANDOM_HEURISTIC = 0.3  # Componente aleatorio en heurística


class Block:
    def __init__(self, x, y, length, orientation, is_red=False):
        self.x = x
        self.y = y
        self.length = length
        self.orientation = orientation  # 'H' o 'V'
        self.is_red = is_red


def blocks_to_state(blocks):
    return tuple((b.x, b.y) for b in blocks)


def apply_state_to_blocks(state, blocks):
    for (x, y), b in zip(state, blocks):
        b.x, b.y = x, y


# --------- Nivel fácil (6x6) ---------
START_BLOCKS_EASY = [
    Block(1, 2, 2, 'H', True),   # rojo
    Block(0, 0, 2, 'V'),
    Block(3, 0, 3, 'V'),
    Block(5, 0, 3, 'V'),
    Block(0, 4, 3, 'H'),
    Block(2, 5, 3, 'H'),
]

# --------- Nivel medio (8x8) ---------
START_BLOCKS_MEDIUM = [
    Block(1, 3, 2, 'H', True),   # rojo
    Block(0, 0, 3, 'V'),
    Block(2, 0, 2, 'V'),
    Block(5, 0, 3, 'V'),
    Block(7, 0, 3, 'V'),
    Block(0, 2, 3, 'H'),
    Block(3, 1, 2, 'H'),
    Block(4, 2, 2, 'V'),
    Block(6, 2, 2, 'H'),
    Block(2, 4, 3, 'H'),
    Block(0, 5, 2, 'H'),
    Block(3, 5, 3, 'V'),
    Block(5, 5, 3, 'H'),
    Block(1, 6, 2, 'V'),
    Block(6, 6, 2, 'H'),
]

# --------- Nivel difícil (20x20) - fijo ---------
START_BLOCKS_HARD20 = [
    Block(2, 10, 2, 'H', True),  # rojo (índice 0)
    Block(0, 0, 3, 'V'), Block(4, 0, 2, 'V'), Block(7, 0, 3, 'V'),
    Block(10, 0, 3, 'H'), Block(14, 0, 3, 'H'), Block(18, 0, 2, 'V'),
    Block(1, 3, 3, 'H'), Block(5, 2, 2, 'H'), Block(9, 2, 2, 'V'),
    Block(12, 2, 3, 'V'), Block(15, 2, 2, 'V'),
    Block(3, 5, 3, 'V'), Block(0, 6, 2, 'H'), Block(4, 6, 3, 'H'),
    Block(8, 6, 2, 'V'), Block(11, 5, 3, 'H'), Block(14, 6, 3, 'V'),
    Block(17, 5, 2, 'H'),
    Block(2, 8, 3, 'H'), Block(6, 9, 2, 'V'), Block(5, 11, 3, 'H'),
    Block(9, 11, 2, 'V'), Block(12, 10, 2, 'V'), Block(15, 9, 3, 'V'),
    Block(18, 9, 2, 'V'),
    Block(0, 13, 3, 'H'), Block(4, 13, 2, 'V'), Block(7, 12, 3, 'H'),
    Block(11, 13, 2, 'V'), Block(14, 13, 3, 'H'),
    Block(2, 16, 3, 'V'), Block(6, 15, 3, 'H'), Block(10, 16, 2, 'V'),
    Block(13, 17, 2, 'H'), Block(16, 16, 3, 'V'),
]


START_BLOCKS = []
LEVEL_META = []
RED_IDX = 0


def choose_level():
    global ROWS, COLS, START_BLOCKS
    print("\n====== SELECCIÓN DE NIVEL ======")
    print("  1) Fácil (6x6)")
    print("  2) Medio (8x8)")
    print("  3) Difícil (20x20)")
    while True:
        try:
            op = int(input("Elige nivel (1-3): ").strip())
            if op == 1:
                ROWS, COLS = 6, 6
                START_BLOCKS = START_BLOCKS_EASY
                return
            elif op == 2:
                ROWS, COLS = 8, 8
                START_BLOCKS = START_BLOCKS_MEDIUM
                return
            elif op == 3:
                ROWS, COLS = 20, 20
                START_BLOCKS = START_BLOCKS_HARD20
                return
        except:
            pass
        print("  Opción inválida.")


def init_level_meta():
    global LEVEL_META, RED_IDX
    LEVEL_META = [(b.length, b.orientation, b.is_red) for b in START_BLOCKS]
    RED_IDX = 0
    for i, meta in enumerate(LEVEL_META):
        if meta[2]:
            RED_IDX = i
            break


def build_grid(state):
    grid = [[-1 for _ in range(COLS)] for _ in range(ROWS)]
    for idx, (x, y) in enumerate(state):
        length, orient, _ = LEVEL_META[idx]
        if orient == 'H':
            for i in range(length):
                grid[y][x+i] = idx
        else:
            for i in range(length):
                grid[y+i][x] = idx
    return grid


def is_goal(state):
    rx, ry = state[RED_IDX]
    rlen, _, _ = LEVEL_META[RED_IDX]
    red_right = rx + rlen - 1
    return red_right == COLS - 1


def move_cost(_from, _to):
    return 1


def heuristic(state):
    if is_goal(state):
        return 0
    rx, ry = state[RED_IDX]
    rlen, _, _ = LEVEL_META[RED_IDX]
    red_right = rx + rlen - 1
    blockers = set()
    for cx in range(red_right + 1, COLS):
        for j, (bx, by) in enumerate(state):
            blen, borient, _ = LEVEL_META[j]
            if j == RED_IDX:
                continue
            if borient == 'H':
                if by == ry and bx <= cx <= bx + blen - 1:
                    blockers.add(j)
            else:
                if bx == cx and by <= ry <= by + blen - 1:
                    blockers.add(j)
    
    # Componente aleatorio en la heurística
    random_component = random.uniform(0, RANDOM_HEURISTIC)
    return len(blockers) + 1 + random_component


def successors(state):
    succs = []
    grid = build_grid(state)
    
    # Priorizar el bloque rojo con cierta probabilidad
    indices = list(range(len(state)))
    if random.random() < 0.4:  # 40% de probabilidad de priorizar el rojo
        if RED_IDX in indices:
            indices.remove(RED_IDX)
            indices.insert(0, RED_IDX)
    
    for i in indices:
        x, y = state[i]
        length, orient, _ = LEVEL_META[i]
        if orient == 'H':
            # Movimiento hacia izquierda
            step = 1
            while x - step >= 0 and grid[y][x - step] == -1:
                new = list(state)
                new[i] = (x - step, y)
                succs.append(tuple(new))
                step += 1
            
            # Movimiento hacia derecha
            step = 1
            right_end = x + length - 1
            while right_end + step <= COLS - 1 and grid[y][right_end + step] == -1:
                new = list(state)
                new[i] = (x + step, y)
                succs.append(tuple(new))
                step += 1
        else:
            # Movimiento hacia arriba
            step = 1
            while y - step >= 0 and grid[y - step][x] == -1:
                new = list(state)
                new[i] = (x, y - step)
                succs.append(tuple(new))
                step += 1
            
            # Movimiento hacia abajo
            step = 1
            bottom_end = y + length - 1
            while bottom_end + step <= ROWS - 1 and grid[bottom_end + step][x] == -1:
                new = list(state)
                new[i] = (x, y + step)
                succs.append(tuple(new))
                step += 1
    
    # Mezclar aleatoriamente los sucesores para exploración diversa
    random.shuffle(succs)
    return succs


def reconstruct(parent_map, goal_state):
    path = [goal_state]
    cur = goal_state
    while parent_map[cur] is not None:
        cur = parent_map[cur]
        path.append(cur)
    path.reverse()
    return path


def astar(start_state, max_exp=400000):
    class Node:
        __slots__ = ("state", "g", "f")

        def __init__(self, state, g):
            self.state = state
            self.g = g
            self.f = g + heuristic(state)

        def __lt__(self, other):
            if self.f != other.f:
                return self.f < other.f
            return self.g > other.g

    open_heap = []
    heappush(open_heap, Node(start_state, 0))
    best_g = {start_state: 0}
    parent = {start_state: None}
    expansions = 0
    expanded_at = {}

    while open_heap and expansions <= max_exp:
        # Exploración aleatoria ocasional
        if random.random() < EXPLORATION_RATE and len(open_heap) > 1:
            # Elegir un nodo aleatorio (no necesariamente el mejor)
            temp_nodes = []
            for _ in range(min(5, len(open_heap))):
                temp_nodes.append(heappop(open_heap))
            
            current = random.choice(temp_nodes)
            # Devolver los otros nodos a la heap
            for node in temp_nodes:
                if node != current:
                    heappush(open_heap, node)
        else:
            current = heappop(open_heap)
        
        s = current.state
        if current.g > best_g.get(s, float('inf')):
            continue
        
        expansions += 1
        expanded_at[s] = expansions
        
        if is_goal(s):
            return reconstruct(parent, s), expansions, expanded_at
        
        for nxt in successors(s):
            new_g = current.g + move_cost(s, nxt)
            if new_g < best_g.get(nxt, float('inf')):
                best_g[nxt] = new_g
                parent[nxt] = s
                heappush(open_heap, Node(nxt, new_g))
    
    return None, expansions, expanded_at


def print_board(state, show_exit=True):
    grid = [['.' for _ in range(COLS)] for _ in range(ROWS)]
    for idx, (x, y) in enumerate(state):
        length, orient, is_red = LEVEL_META[idx]
        ch = 'R' if is_red else str(idx % 10)
        if orient == 'H':
            for i in range(length):
                grid[y][x+i] = ch
        else:
            for i in range(length):
                grid[y+i][x] = ch

    rx, ry = state[RED_IDX]
    print("\n  +" + "--"*COLS + "+")
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            cell = grid[r][c]
            if cell == 'R':
                row.append("\033[91mR\033[0m")
            else:
                row.append(cell)
        exit_mark = "\033[92m→\033[0m" if show_exit and r == ry else " "
        print("  |" + " ".join(row) + "|" + exit_mark)
    print("  +" + "--"*COLS + "+")
    if is_goal(state):
        print("\033[92mMeta alcanzada: el rojo llegó a la salida.\033[0m")


def print_status(sol, expansions, elapsed):
    if sol is None:
        print(
            f"\n[A*] Sin solución (expandidos={expansions}, tiempo={elapsed:.3f}s)")
    else:
        moves = max(0, len(sol) - 1)
        print(f"\n[A*] ¡Solución hallada!")
        print(f"  - Movimientos (transiciones): {moves}")
        print(f"  - Nodos expandidos: {expansions}")
        print(f"  - Tiempo: {elapsed:.3f} s")
        print(f"  - Tasa de exploración: {EXPLORATION_RATE*100}%")
        print(f"  - Aleatoriedad en heurística: ±{RANDOM_HEURISTIC}")


def solve_with_astar(start_state):
    t0 = time.perf_counter()
    sol, exp, expanded_at = astar(start_state)
    t1 = time.perf_counter()
    return sol, exp, (t1 - t0), expanded_at


def main():
    choose_level()
    init_level_meta()
    blocks = [Block(b.x, b.y, b.length, b.orientation, b.is_red)
              for b in START_BLOCKS]
    start_state = blocks_to_state(blocks)

    # Mostrar estado inicial
    print("\n=== Estado inicial ===")
    print("Coordenadas iniciales:", start_state)
    print("Configuración aleatoria:")

    print_board(start_state)

    solution, expansions, elapsed, expanded_at = solve_with_astar(start_state)
    print_status(solution, expansions, elapsed)

    if solution:
        print(f"\nMostrando solución completa ({len(solution)-1} movimientos):\n")
        for step_idx, st in enumerate(solution):
            apply_state_to_blocks(st, blocks)
            nodes_visited = expanded_at.get(st, 0)
            print(f"Paso {step_idx}: Nodos visitados = {nodes_visited}")
            print(f"Coordenadas: {st}")
            print_board(st)
            if step_idx < len(solution) - 1:
                print("--- Movimiento ---")
        if is_goal(solution[-1]):
            print("\n¡Meta alcanzada con éxito!\n")


if __name__ == "__main__":
    main()