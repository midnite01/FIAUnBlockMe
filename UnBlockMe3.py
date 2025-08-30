# -*- coding: utf-8 -*-
import pygame
import sys
import time
from heapq import heappush, heappop
from collections import deque

# ---------------- CONFIG ----------------
ROWS, COLS = 6, 6
CELL_SIZE = 96  # 96*6 = 576 px tablero
UI_H = 180      # un poco más alto: caben métricas + botones
WIDTH, HEIGHT = COLS*CELL_SIZE, ROWS*CELL_SIZE + UI_H
FPS = 60

# Colores
WHITE = (250, 250, 250)
BLACK = (25, 25, 25)
RED   = (220, 60, 60)
BLUE  = (70, 90, 200)
DARK  = (40, 40, 60)
GREY  = (210, 210, 210)
LIGHT = (235, 235, 245)
PALE  = (250, 250, 252)

# ---------------- MODELO DE BLOQUES ----------------
class Block:
    def __init__(self, x, y, length, orientation, is_red=False):
        self.x = x
        self.y = y
        self.length = length
        self.orientation = orientation  # 'H' o 'V'
        self.is_red = is_red

    def rect(self):
        if self.orientation == 'H':
            return pygame.Rect(self.x*CELL_SIZE, self.y*CELL_SIZE, self.length*CELL_SIZE, CELL_SIZE)
        else:
            return pygame.Rect(self.x*CELL_SIZE, self.y*CELL_SIZE, CELL_SIZE, self.length*CELL_SIZE)

    def cells(self):
        if self.orientation == 'H':
            return [(self.x + i, self.y) for i in range(self.length)]
        else:
            return [(self.x, self.y + i) for i in range(self.length)]

# Estado <-> lista de posiciones (mantiene el orden de los bloques)
def blocks_to_state(blocks):
    return tuple((b.x, b.y) for b in blocks)

def apply_state_to_blocks(state, blocks):
    for (x, y), b in zip(state, blocks):
        b.x, b.y = x, y

# ---------------- METADATOS DE NIVEL ----------------
LEVEL_META = []
RED_IDX = 0

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

def successors(state):
    """Genera TODOS los estados alcanzables con un movimiento (desplazar un bloque k celdas)."""
    succs = []
    grid = build_grid(state)
    for i, (x, y) in enumerate(state):
        length, orient, _ = LEVEL_META[i]
        if orient == 'H':
            # Izquierda
            step = 1
            while x - step >= 0 and grid[y][x - step] == -1:
                new = list(state)
                new[i] = (x - step, y)
                succs.append(tuple(new))
                step += 1
            # Derecha
            step = 1
            right_end = x + length - 1
            while right_end + step <= COLS - 1 and grid[y][right_end + step] == -1:
                new = list(state)
                new[i] = (x + step, y)
                succs.append(tuple(new))
                step += 1
        else:
            # Arriba
            step = 1
            while y - step >= 0 and grid[y - step][x] == -1:
                new = list(state)
                new[i] = (x, y - step)
                succs.append(tuple(new))
                step += 1
            # Abajo
            step = 1
            bottom_end = y + length - 1
            while bottom_end + step <= ROWS - 1 and grid[bottom_end + step][x] == -1:
                new = list(state)
                new[i] = (x, y + step)
                succs.append(tuple(new))
                step += 1
    return succs

def move_cost(from_state, to_state):
    """Costo usado por A* y UCS: cantidad de celdas desplazadas en ese movimiento."""
    for k, (ox, oy) in enumerate(from_state):
        nx, ny = to_state[k]
        if nx != ox or ny != oy:
            return abs(nx - ox) + abs(ny - oy)
    return 0

# ---------------- HEURÍSTICA A* ----------------
def heuristic(state):
    """
    h = celdas restantes del rojo hasta la salida + # de bloqueadores en su fila.
    Admisible si el costo por movimiento es 1 (o cota inferior si se mide en celdas).
    """
    rx, ry = state[RED_IDX]
    rlen, _, _ = LEVEL_META[RED_IDX]
    red_right = rx + rlen - 1
    dist = (COLS - 1) - red_right
    if dist < 0:
        dist = 0

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
    return dist + len(blockers)

# ---------------- A* ----------------
class NodeA:
    def __init__(self, state, g, parent=None):
        self.state = state
        self.g = g
        self.parent = parent
        self.f = self.g + heuristic(self.state)

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        return self.g > other.g  # desempate: más g primero

def reconstruct(node):
    path = []
    n = node
    while n:
        path.append(n.state)
        n = n.parent
    return list(reversed(path))

def astar(start_state, max_expansions=200000):
    start = NodeA(start_state, 0, None)
    open_heap = []
    heappush(open_heap, start)
    best_g = {start_state: 0}
    expansions = 0

    while open_heap and expansions <= max_expansions:
        current = heappop(open_heap)
        if is_goal(current.state):
            return reconstruct(current), expansions
        if current.g > best_g.get(current.state, float('inf')):
            continue
        expansions += 1
        for nxt in successors(current.state):
            new_g = current.g + move_cost(current.state, nxt)
            if new_g < best_g.get(nxt, float('inf')):
                best_g[nxt] = new_g
                heappush(open_heap, NodeA(nxt, new_g, current))
    return None, expansions

# ---------------- BFS (no informada) ----------------
def bfs(start_state, max_expansions=400000):
    """BFS por capas (cada arista = 1). Ignora costos reales."""
    if is_goal(start_state):
        return [start_state], 0
    Q = deque([start_state])
    parent = {start_state: None}
    expansions = 0
    visited = set([start_state])

    while Q and expansions <= max_expansions:
        s = Q.popleft()
        expansions += 1
        for nxt in successors(s):
            if nxt not in visited:
                parent[nxt] = s
                if is_goal(nxt):
                    # reconstruir
                    path = [nxt]
                    cur = nxt
                    while parent[cur] is not None:
                        cur = parent[cur]
                        path.append(cur)
                    path.reverse()
                    return path, expansions
                visited.add(nxt)
                Q.append(nxt)
    return None, expansions

# ---------------- DFS (no informada) ----------------
def dfs(start_state, max_expansions=400000):
    """
    DFS en profundidad con pila y 'visited' global.
    No garantiza camino más corto; útil para ver diferencia con BFS/UCS/A*.
    """
    stack = [start_state]
    parent = {start_state: None}
    visited = set([start_state])
    expansions = 0

    while stack and expansions <= max_expansions:
        s = stack.pop()
        if is_goal(s):
            # reconstruir
            path = [s]
            cur = s
            while parent[cur] is not None:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path, expansions
        expansions += 1
        # Sucesores al final (orden natural); puedes invertir para otro orden
        for nxt in successors(s):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = s
                stack.append(nxt)
    return None, expansions

# ---------------- UCS (costo uniforme) ----------------
def ucs(start_state, max_expansions=400000):
    """
    UCS con prioridad por g(n) = costo acumulado real (en este juego: celdas desplazadas).
    """
    heap = []
    heappush(heap, (0, start_state, None))  # (g, state, parent_state)
    best_g = {start_state: 0}
    parent = {start_state: None}
    expansions = 0

    while heap and expansions <= max_expansions:
        g, s, _ = heappop(heap)
        if g > best_g.get(s, float('inf')):
            continue
        if is_goal(s):
            # reconstruir
            path = [s]
            cur = s
            while parent[cur] is not None:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path, expansions
        expansions += 1
        for nxt in successors(s):
            new_g = g + move_cost(s, nxt)
            if new_g < best_g.get(nxt, float('inf')):
                best_g[nxt] = new_g
                parent[nxt] = s
                heappush(heap, (new_g, nxt, s))
    return None, expansions

# ---------------- UI (Pygame) ----------------
class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

    def draw(self, surf, font):
        pygame.draw.rect(surf, LIGHT, self.rect, border_radius=10)
        pygame.draw.rect(surf, GREY, self.rect, width=2, border_radius=10)
        label = font.render(self.text, True, BLACK)
        surf.blit(label, label.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.rect.collidepoint(pos)

def draw_grid(screen, blocks):
    for r in range(ROWS):
        for c in range(COLS):
            pygame.draw.rect(screen, GREY, (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE), width=1)
    # salida a la derecha en la fila del rojo (apertura)
    exit_y = blocks[RED_IDX].y
    ex_rect = pygame.Rect(WIDTH-8, exit_y*CELL_SIZE + CELL_SIZE*0.25, 8, CELL_SIZE*0.5)
    pygame.draw.rect(screen, (120, 200, 120), ex_rect)

def draw_blocks(screen, blocks):
    for b in blocks:
        color = RED if b.is_red else BLUE
        pygame.draw.rect(screen, color, b.rect(), border_radius=12)
        pygame.draw.rect(screen, DARK, b.rect(), width=2, border_radius=12)

def compute_drag_limits(blocks, idx):
    b = blocks[idx]
    grid = [[-1 for _ in range(COLS)] for _ in range(ROWS)]
    for j, bb in enumerate(blocks):
        for (cx, cy) in bb.cells():
            grid[cy][cx] = j
    if b.orientation == 'H':
        left = b.x
        while left - 1 >= 0 and grid[b.y][left - 1] == -1:
            left -= 1
        right = b.x
        while (right + b.length) <= COLS - 1 and grid[b.y][right + b.length] == -1:
            right += 1
        return (left, right)
    else:
        up = b.y
        while up - 1 >= 0 and grid[up - 1][b.x] == -1:
            up -= 1
        down = b.y
        while (down + b.length) <= ROWS - 1 and grid[down + b.length][b.x] == -1:
            down += 1
        return (up, down)

# --------- Nivel de ejemplo (soluble) ---------
START_BLOCKS = [
    Block(1, 2, 2, 'H', True),   # rojo (índice 0)
    Block(3, 0, 3, 'V'),
    Block(0, 0, 2, 'V'),
    Block(5, 0, 3, 'V'),
    Block(0, 4, 3, 'H'),
    Block(2, 5, 3, 'H'),
]

def init_level_meta():
    global LEVEL_META, RED_IDX
    LEVEL_META = [(b.length, b.orientation, b.is_red) for b in START_BLOCKS]
    for i, meta in enumerate(LEVEL_META):
        if meta[2]:
            RED_IDX = i
            break

# ---------------- Menú de selección de algoritmo ----------------
def choose_algorithm(screen, fonts):
    font_title, font, font_small = fonts
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.fill(PALE)
    pygame.draw.rect(overlay, GREY, (40, 40, WIDTH-80, HEIGHT-80), width=2, border_radius=16)

    title = font_title.render("Elige el algoritmo de resolución", True, DARK)
    subtitle = font.render("Podrás comparar tiempos, nodos expandidos y longitud del plan.", True, DARK)

    # Botones de algoritmo
    labels = ["A* (Informado)", "BFS (No informado)", "DFS (No informado)", "UCS (Costo uniforme)"]
    n = len(labels)
    gap = 16
    btn_w = 300
    btn_h = 52
    start_y = 150
    buttons = []
    cx = WIDTH // 2 - btn_w // 2
    for i, lab in enumerate(labels):
        by = start_y + i * (btn_h + gap)
        buttons.append(Button((cx, by, btn_w, btn_h), lab))

    # Instrucciones
    info_lines = [
        "A*: usa heurística (distancia salida + bloqueadores).",
        "BFS: ignora costos; expande por capas (cada transición vale 1).",
        "DFS: explora en profundidad (no garantiza camino más corto).",
        "UCS: óptimo con el costo real (celdas desplazadas por movimiento).",
        "Luego podrás usar: Resolver, Paso, Auto y Reiniciar."
    ]

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                pos = ev.pos
                for i, b in enumerate(buttons):
                    if b.hit(pos):
                        return ["ASTAR", "BFS", "DFS", "UCS"][i]

        screen.blit(overlay, (0, 0))
        overlay.blit(title, title.get_rect(center=(WIDTH//2, 80)))
        overlay.blit(subtitle, subtitle.get_rect(center=(WIDTH//2, 110)))

        for b in buttons:
            b.draw(overlay, font)

        # Pinta instrucciones
        yinfo = start_y + n*(btn_h + gap) + 12
        for line in info_lines:
            surf = font_small.render("• " + line, True, DARK)
            overlay.blit(surf, (60, yinfo))
            yinfo += 22

        pygame.display.flip()

# ---------------- Loop principal ----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Unblock Me — Algoritmos: A*, BFS, DFS, UCS")
    clock = pygame.time.Clock()

    # Fuentes
    font_small = pygame.font.SysFont(None, 18)
    font = pygame.font.SysFont(None, 22)
    font_big = pygame.font.SysFont(None, 28)

    # Estado inicial
    blocks = [Block(b.x, b.y, b.length, b.orientation, b.is_red) for b in START_BLOCKS]
    init_level_meta()

    # ---- MENÚ AL INICIO ----
    selected_alg = choose_algorithm(screen, (font_big, font, font_small))

    # Botones de control
    n_buttons = 4
    gap = 12
    btn_h = 40
    btn_w = int((WIDTH - gap * (n_buttons + 1)) / n_buttons)
    metrics_h = 86
    metrics_rect = pygame.Rect(0, ROWS*CELL_SIZE, WIDTH, metrics_h)
    yb = ROWS*CELL_SIZE + metrics_h + 8

    buttons = {
        'solve': Button((gap, yb, btn_w, btn_h), f'Resolver {selected_alg}'),
        'step':  Button((gap*2 + btn_w, yb, btn_w, btn_h), 'Paso ▶'),
        'auto':  Button((gap*3 + btn_w*2, yb, btn_w, btn_h), 'Auto ▶▶'),
        'reset': Button((gap*4 + btn_w*3, yb, btn_w, btn_h), 'Reiniciar'),
    }

    # Solver/animación
    solution_states = None
    step_idx = 0
    auto = False
    auto_timer = 0
    auto_delay = 12  # frames por paso

    # Métricas
    solve_time = None
    solve_expansions = 0
    moves_total = 0
    manual_moves = 0

    # Arrastre
    dragging = None  # (idx, offset, start_x, start_y)
    drag_limits = None

    # Utilidad: resolver con el algoritmo elegido
    def solve_with_algorithm(name, start_state):
        t0 = time.perf_counter()
        if name == "ASTAR":
            sol, exp = astar(start_state)
        elif name == "BFS":
            sol, exp = bfs(start_state)
        elif name == "DFS":
            sol, exp = dfs(start_state)
        elif name == "UCS":
            sol, exp = ucs(start_state)
        else:
            sol, exp = None, 0
        t1 = time.perf_counter()
        return sol, exp, t1 - t0

    # Loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            elif event.type == pygame.KEYDOWN:
                # Permite cambiar rápido de algoritmo con teclas (opcional)
                if event.key == pygame.K_1:
                    selected_alg = "ASTAR"
                    buttons['solve'].text = 'Resolver ASTAR'
                elif event.key == pygame.K_2:
                    selected_alg = "BFS"
                    buttons['solve'].text = 'Resolver BFS'
                elif event.key == pygame.K_3:
                    selected_alg = "DFS"
                    buttons['solve'].text = 'Resolver DFS'
                elif event.key == pygame.K_4:
                    selected_alg = "UCS"
                    buttons['solve'].text = 'Resolver UCS'

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if pos[1] >= ROWS*CELL_SIZE:
                    # clic en botones
                    if buttons['solve'].hit(pos):
                        start_state = blocks_to_state(blocks)
                        solution_states, solve_expansions, solve_time = solve_with_algorithm(selected_alg, start_state)
                        moves_total = (len(solution_states) - 1) if solution_states else 0
                        step_idx = 0
                        auto = False
                    elif buttons['step'].hit(pos):
                        if solution_states is None:
                            solution_states, solve_expansions, solve_time = solve_with_algorithm(selected_alg, blocks_to_state(blocks))
                            moves_total = (len(solution_states) - 1) if solution_states else 0
                            step_idx = 0
                        if solution_states and step_idx < len(solution_states):
                            apply_state_to_blocks(solution_states[step_idx], blocks)
                            step_idx += 1
                    elif buttons['auto'].hit(pos):
                        if solution_states is None:
                            solution_states, solve_expansions, solve_time = solve_with_algorithm(selected_alg, blocks_to_state(blocks))
                            moves_total = (len(solution_states) - 1) if solution_states else 0
                            step_idx = 0
                        auto = True if solution_states else False
                    elif buttons['reset'].hit(pos):
                        # Reiniciar tablero y métricas
                        blocks = [Block(b.x, b.y, b.length, b.orientation, b.is_red) for b in START_BLOCKS]
                        solution_states = None
                        step_idx = 0
                        auto = False
                        solve_time = None
                        solve_expansions = 0
                        moves_total = 0
                        manual_moves = 0
                else:
                    # seleccionar bloque para arrastrar (prioriza topmost)
                    for i in range(len(blocks)-1, -1, -1):
                        if blocks[i].rect().collidepoint(pos):
                            offset = (pos[0] - blocks[i].rect().x, pos[1] - blocks[i].rect().y)
                            dragging = (i, offset, blocks[i].x, blocks[i].y)
                            drag_limits = compute_drag_limits(blocks, i)
                            break

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if dragging is not None:
                    idx, _, sx, sy = dragging
                    b = blocks[idx]
                    b.x = int(round(b.rect().x / CELL_SIZE))
                    b.y = int(round(b.rect().y / CELL_SIZE))
                    b.x = max(0, min(COLS - (b.length if b.orientation == 'H' else 1), b.x))
                    b.y = max(0, min(ROWS - (1 if b.orientation == 'H' else b.length), b.y))
                    if b.x != sx or b.y != sy:
                        manual_moves += 1
                dragging = None
                drag_limits = None

            elif event.type == pygame.MOUSEMOTION and dragging is not None:
                idx, offset, _, _ = dragging
                b = blocks[idx]
                mx, my = event.pos
                rx = mx - offset[0]
                ry = my - offset[1]
                if b.orientation == 'H':
                    cx = int(round(rx / CELL_SIZE))
                    minx, maxx = drag_limits
                    cx = max(minx, min(maxx, cx))
                    b.x = cx
                else:
                    cy = int(round(ry / CELL_SIZE))
                    miny, maxy = drag_limits
                    cy = max(miny, min(maxy, cy))
                    b.y = cy

        # Auto-avance
        if auto and solution_states:
            auto_timer += 1
            if auto_timer >= auto_delay and step_idx < len(solution_states):
                apply_state_to_blocks(solution_states[step_idx], blocks)
                step_idx += 1
                auto_timer = 0
            if step_idx >= len(solution_states):
                auto = False

        # Dibujo general
        screen.fill(WHITE)
        draw_grid(screen, blocks)
        draw_blocks(screen, blocks)

        # Área de métricas (panel)
        pygame.draw.rect(screen, PALE, metrics_rect)
        pygame.draw.rect(screen, GREY, metrics_rect, width=1)

        # Métricas
        mx = 12
        my = ROWS*CELL_SIZE + 8
        alg_text = f"Algoritmo: {selected_alg}"
        time_text = "Tiempo: -- s" if solve_time is None else f"Tiempo: {solve_time:.3f} s"
        nodes_text = f"Nodos expandidos: {solve_expansions}"
        moves_text = f"Mov. solución: {moves_total}"
        manual_text = f"Mov. jugador: {manual_moves}"

        screen.blit(font_small.render(alg_text, True, DARK), (mx, my))
        screen.blit(font_small.render(time_text, True, DARK), (mx + 180, my))
        screen.blit(font_small.render(nodes_text, True, DARK), (mx + 360, my))
        screen.blit(font_small.render(moves_text, True, DARK), (mx + 610, my))
        screen.blit(font_small.render(manual_text, True, DARK), (mx + 790, my))

        # Mensaje principal
        msg = "Arrastra bloques. Usa Resolver / Paso / Auto."
        state_now = blocks_to_state(blocks)
        if is_goal(state_now):
            msg = "¡Nivel resuelto! El bloque rojo llegó a la salida."
        elif solution_states:
            moves_disp = moves_total
            msg = f"Solución: {moves_disp} movimientos | Paso {min(step_idx, moves_disp)}/{moves_disp}"

        msg_surf = font.render(msg, True, DARK)
        msg_rect = msg_surf.get_rect(center=(WIDTH // 2, ROWS*CELL_SIZE + metrics_h // 2 + 6))
        screen.blit(msg_surf, msg_rect)

        # Dibujar botones
        for b in buttons.values():
            b.draw(screen, font)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
