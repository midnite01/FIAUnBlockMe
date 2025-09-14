"""
Juego tipo Unblock Me 
 - BFS y UCS implementados de forma robusta.
 - Menú para elegir nivel (fácil o difícil).
 - Tablero ajusta tamaño dinámicamente (6x6 o 8x8).
 - Bloque rojo se imprime en rojo y la meta en verde.
"""

import time
from heapq import heappush, heappop
from collections import deque

# ---------------- CONFIG DEL PROBLEMA ----------------
ROWS, COLS = 6, 6   # por defecto (se ajusta al elegir nivel)


class Block:
    def __init__(self, x, y, length, orientation, is_red=False):
        self.x = x
        self.y = y
        self.length = length
        self.orientation = orientation  # 'H' o 'V'
        self.is_red = is_red

    def cells(self):
        if self.orientation == 'H':
            return [(self.x + i, self.y) for i in range(self.length)]
        else:
            return [(self.x, self.y + i) for i in range(self.length)]

# Estado = tupla de (x,y) para cada bloque en el mismo orden que START_BLOCKS


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

# --------- Nivel difícil (8x8) ---------
START_BLOCKS_HARD = [
    Block(1, 3, 2, 'H', True),   # rojo (fila 3, debe salir a la derecha)
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

START_BLOCKS = []  # se setea al elegir nivel
LEVEL_META = []    # lista de (length, orient, is_red)
RED_IDX = 0        # índice del bloque rojo


def choose_level():
    global ROWS, COLS, START_BLOCKS
    print("\n====== SELECCIÓN DE NIVEL ======")
    print("  1) Fácil (6x6)")
    print("  2) Difícil (8x8)")
    while True:
        try:
            op = int(input("Elige nivel (1-2): ").strip())
            if op == 1:
                ROWS, COLS = 6, 6
                START_BLOCKS = START_BLOCKS_EASY
                return
            elif op == 2:
                ROWS, COLS = 8, 8
                START_BLOCKS = START_BLOCKS_HARD
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

# -------- COSTO UNIFORME --------


def move_cost(_from, _to):
    return 1

# -------- Heurística A* --------


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
    return len(blockers) + 1

# -------- Generación de sucesores --------


def successors(state):
    succs = []
    grid = build_grid(state)
    for i, (x, y) in enumerate(state):
        length, orient, _ = LEVEL_META[i]
        if orient == 'H':
            step = 1
            while x - step >= 0 and grid[y][x - step] == -1:
                new = list(state)
                new[i] = (x - step, y)
                succs.append(tuple(new))
                step += 1
            step = 1
            right_end = x + length - 1
            while right_end + step <= COLS - 1 and grid[y][right_end + step] == -1:
                new = list(state)
                new[i] = (x + step, y)
                succs.append(tuple(new))
                step += 1
        else:
            step = 1
            while y - step >= 0 and grid[y - step][x] == -1:
                new = list(state)
                new[i] = (x, y - step)
                succs.append(tuple(new))
                step += 1
            step = 1
            bottom_end = y + length - 1
            while bottom_end + step <= ROWS - 1 and grid[bottom_end + step][x] == -1:
                new = list(state)
                new[i] = (x, y + step)
                succs.append(tuple(new))
                step += 1
    return succs

# -------- Reconstrucción de camino --------


def reconstruct(parent_map, goal_state):
    path = [goal_state]
    cur = goal_state
    while parent_map[cur] is not None:
        cur = parent_map[cur]
        path.append(cur)
    path.reverse()
    return path

# ---------------- Algoritmos de búsqueda ----------------


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

    while open_heap and expansions <= max_exp:
        current = heappop(open_heap)
        s = current.state
        if is_goal(s):
            return reconstruct(parent, s), expansions
        if current.g > best_g.get(s, float('inf')):
            continue
        expansions += 1
        for nxt in successors(s):
            new_g = current.g + move_cost(s, nxt)
            if new_g < best_g.get(nxt, float('inf')):
                best_g[nxt] = new_g
                parent[nxt] = s
                heappush(open_heap, Node(nxt, new_g))
    return None, expansions


def bfs(start_state, max_exp=600000):
    if is_goal(start_state):
        return [start_state], 0
    Q = deque([start_state])
    parent = {start_state: None}
    expansions = 0
    while Q and expansions <= max_exp:
        s = Q.popleft()
        expansions += 1
        for nxt in successors(s):
            if nxt not in parent:
                parent[nxt] = s
                if is_goal(nxt):
                    return reconstruct(parent, nxt), expansions
                Q.append(nxt)
    return None, expansions


def dfs(start_state, max_exp=600000):
    stack = [start_state]
    parent = {start_state: None}
    visited = {start_state}
    expansions = 0
    while stack and expansions <= max_exp:
        s = stack.pop()
        if is_goal(s):
            return reconstruct(parent, s), expansions
        expansions += 1
        for nxt in successors(s):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = s
                stack.append(nxt)
    return None, expansions


def ucs(start_state, max_exp=600000):
    heap = []
    heappush(heap, (0, start_state))
    best_g = {start_state: 0}
    parent = {start_state: None}
    expansions = 0
    while heap and expansions <= max_exp:
        g, s = heappop(heap)
        if g > best_g.get(s, float('inf')):
            continue
        if is_goal(s):
            return reconstruct(parent, s), expansions
        expansions += 1
        for nxt in successors(s):
            new_g = g + move_cost(s, nxt)
            if new_g < best_g.get(nxt, float('inf')):
                best_g[nxt] = new_g
                parent[nxt] = s
                heappush(heap, (new_g, nxt))
    return None, expansions

# ---------------- Impresión en terminal ----------------


def print_board(state):
    grid = [['.' for _ in range(COLS)] for _ in range(ROWS)]
    for idx, (x, y) in enumerate(state):
        length, orient, is_red = LEVEL_META[idx]
        ch = 'R' if is_red else str(idx)
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
                row.append("\033[91mR\033[0m")  # Rojo para el bloque objetivo
            else:
                row.append(cell)
        exit_mark = "\033[92m→\033[0m" if r == ry else " "
        print("  |" + " ".join(row) + "|" + exit_mark)
    print("  +" + "--"*COLS + "+")
    if is_goal(state):
        print("\033[92mMeta alcanzada: el rojo llegó a la salida.\033[0m")


def print_status(alg_name, sol, expansions, elapsed):
    if sol is None:
        print(
            f"\n[{alg_name}] Sin solución (expandidos={expansions}, tiempo={elapsed:.3f}s)")
    else:
        moves = max(0, len(sol) - 1)
        print(f"\n[{alg_name}] ¡Solución hallada!")
        print(f"  - Movimientos (transiciones): {moves}")
        print(f"  - Nodos expandidos: {expansions}")
        print(f"  - Tiempo: {elapsed:.3f} s")


def choose_algorithm_cli():
    algos = ["ASTAR", "BFS", "DFS", "UCS"]
    print("\n====== SELECCIÓN DE ALGORITMO ======")
    for i, a in enumerate(algos, 1):
        print(f"  {i}) {a}")
    while True:
        try:
            op = int(input("Elige (1-4): ").strip())
            if 1 <= op <= 4:
                return algos[op-1]
        except:
            pass
        print("  Opción inválida.")


def solve_with(alg, start_state):
    t0 = time.perf_counter()
    if alg == "ASTAR":
        sol, exp = astar(start_state)
    elif alg == "BFS":
        sol, exp = bfs(start_state)
    elif alg == "DFS":
        sol, exp = dfs(start_state)
    elif alg == "UCS":
        sol, exp = ucs(start_state)
    else:
        sol, exp = None, 0
    t1 = time.perf_counter()
    return sol, exp, (t1 - t0)


def main_menu():
    print("\n====== UNBLOCK ME (CLI) ======")
    print("  1) Resolver (mostrar todo el recorrido)")
    print("  2) Reiniciar")
    print("  3) salir")


def main():
    choose_level()
    init_level_meta()
    blocks = [Block(b.x, b.y, b.length, b.orientation, b.is_red)
              for b in START_BLOCKS]
    start_state = blocks_to_state(blocks)

    selected = choose_algorithm_cli()
    solution = None
    expansions = 0
    elapsed = 0.0
    step_idx = 0

    while True:
        print_board(blocks_to_state(blocks))
        if solution is not None:
            print(f"\nMovimientos: {max(0,len(solution)-1)}")
            print(f"Nodos expandidos: {expansions} | Tiempo: {elapsed:.3f}s")
        main_menu()
        try:
            op = int(input("Opción: ").strip())
        except:
            print("Opción inválida.")
            continue

        if op == 1:
            start_state = blocks_to_state(blocks)
            solution, expansions, elapsed = solve_with(selected, start_state)
            print_status(selected, solution, expansions, elapsed)
            step_idx = 0
            if solution:
                print(
                    f"\nMostrando solución completa ({len(solution)-1} movimientos):\n")
                apply_state_to_blocks(solution[0], blocks)
                print_board(solution[0])
                for i, st in enumerate(solution[1:], start=1):
                    apply_state_to_blocks(st, blocks)
                    step_idx = i

                    print_board(st)
                    time.sleep(0.35)
                if is_goal(solution[-1]):
                    print("\n¡Meta alcanzada con éxito!\n")

        elif op == 2:
            blocks = [Block(b.x, b.y, b.length, b.orientation, b.is_red)
                      for b in START_BLOCKS]
            solution = None
            step_idx = 0
            print("\nTablero reiniciado.")

        elif op == 3:
            print("\nNos vemos pronto amigo!")
            break

        else:
            print("Opción inválida.")


if __name__ == "__main__":
    main()
