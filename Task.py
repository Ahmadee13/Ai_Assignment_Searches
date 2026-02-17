import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
import heapq
from matplotlib.colors import ListedColormap

# ==============================
# CONFIGURATION
# ==============================
ROWS = 10
COLS = 10
DELAY = 0.1

# Clockwise order including diagonals
MOVES = [
    (-1, 0),   # Up
    (0, 1),    # Right
    (1, 1),    # Bottom-Right
    (1, 0),    # Bottom
    (0, -1),   # Left
    (-1, -1)   # Top-Left
]

# ==============================
# CREATE GRID
# ==============================
def create_grid():
    return np.zeros((ROWS, COLS))

# ==============================
# VALID MOVE
# ==============================
def valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

# ==============================
# GUI UPDATE
# ==============================
def update_gui(grid, start, goal, explored, frontier, path):
    display = np.copy(grid)

    for r, c in explored:
        display[r][c] = 2  # explored

    for r, c in frontier:
        display[r][c] = 3  # frontier

    for r, c in path:
        display[r][c] = 4  # path

    display[start] = 5   # start
    display[goal] = 6    # goal

    plt.clf()
    plt.title("GOOD  PERFORMANCE  TIME APP", fontsize=14, fontweight="bold")

    cmap = ListedColormap([
        "white",   # 0 empty
        "black",   # 1 (unused)
        "yellow",  # 2 explored
        "blue",    # 3 frontier
        "purple",  # 4 final path
        "green",   # 5 start
        "red"      # 6 goal
    ])

    plt.imshow(display, cmap=cmap)

    # Draw grid lines
    plt.xticks(np.arange(-.5, COLS, 1), [])
    plt.yticks(np.arange(-.5, ROWS, 1), [])
    plt.grid(color='gray', linestyle='-', linewidth=0.5)

    plt.pause(DELAY)

# ==============================
# RECONSTRUCT PATH
# ==============================
def reconstruct(parent, goal):
    path = []
    while goal in parent:
        path.append(goal)
        goal = parent[goal]
    return path[::-1]

# ==============================
# BFS
# ==============================
def bfs(grid, start, goal):
    queue = deque([start])
    visited = set([start])
    parent = {}
    explored = []

    while queue:
        current = queue.popleft()
        explored.append(current)

        if current == goal:
            return reconstruct(parent, goal)

        for move in MOVES:
            nr, nc = current[0] + move[0], current[1] + move[1]
            if valid(nr, nc) and (nr, nc) not in visited:
                queue.append((nr, nc))
                visited.add((nr, nc))
                parent[(nr, nc)] = current

        update_gui(grid, start, goal, explored, list(queue), [])

    return []

# ==============================
# DFS
# ==============================
def dfs(grid, start, goal):
    stack = [start]
    visited = set()
    parent = {}
    explored = []

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            explored.append(current)

            if current == goal:
                return reconstruct(parent, goal)

            for move in reversed(MOVES):
                nr, nc = current[0] + move[0], current[1] + move[1]
                if valid(nr, nc) and (nr, nc) not in visited:
                    stack.append((nr, nc))
                    parent[(nr, nc)] = current

        update_gui(grid, start, goal, explored, stack, [])

    return []

# ==============================
# UCS
# ==============================
def ucs(grid, start, goal):
    pq = []
    heapq.heappush(pq, (0, start))
    visited = set()
    parent = {}
    cost = {start: 0}
    explored = []

    while pq:
        current_cost, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        explored.append(current)

        if current == goal:
            return reconstruct(parent, goal)

        for move in MOVES:
            nr, nc = current[0] + move[0], current[1] + move[1]
            if valid(nr, nc):
                new_cost = current_cost + 1
                if (nr, nc) not in cost or new_cost < cost[(nr, nc)]:
                    cost[(nr, nc)] = new_cost
                    parent[(nr, nc)] = current
                    heapq.heappush(pq, (new_cost, (nr, nc)))

        update_gui(grid, start, goal, explored, [node[1] for node in pq], [])

    return []

# ==============================
# IDDFS (Includes DLS)
# ==============================
def dls(current, goal, limit, parent, explored):
    if current == goal:
        return True
    if limit <= 0:
        return False

    explored.append(current)

    for move in MOVES:
        nr, nc = current[0] + move[0], current[1] + move[1]
        if valid(nr, nc) and (nr, nc) not in parent:
            parent[(nr, nc)] = current
            if dls((nr, nc), goal, limit-1, parent, explored):
                return True
    return False

def iddfs(grid, start, goal):
    for depth in range(ROWS * COLS):
        parent = {}
        explored = []
        if dls(start, goal, depth, parent, explored):
            return reconstruct(parent, goal)
        update_gui(grid, start, goal, explored, [], [])
    return []

# ==============================
# BIDIRECTIONAL SEARCH
# ==============================
def bidirectional(grid, start, goal):
    q_start = deque([start])
    q_goal = deque([goal])

    visited_start = {start: None}
    visited_goal = {goal: None}

    explored = []

    while q_start and q_goal:
        s = q_start.popleft()
        g = q_goal.popleft()
        explored.append(s)
        explored.append(g)

        for move in MOVES:
            nr, nc = s[0] + move[0], s[1] + move[1]
            nxt = (nr, nc)
            if valid(nr, nc) and nxt not in visited_start:
                visited_start[nxt] = s
                q_start.append(nxt)
                if nxt in visited_goal:
                    return build_path(visited_start, visited_goal, nxt)

        for move in MOVES:
            nr, nc = g[0] + move[0], g[1] + move[1]
            nxt = (nr, nc)
            if valid(nr, nc) and nxt not in visited_goal:
                visited_goal[nxt] = g
                q_goal.append(nxt)
                if nxt in visited_start:
                    return build_path(visited_start, visited_goal, nxt)

        update_gui(grid, start, goal, explored, list(q_start)+list(q_goal), [])

    return []

def build_path(vs, vg, meet):
    path = []
    node = meet
    while node:
        path.append(node)
        node = vs[node]
    path.reverse()

    node = vg[meet]
    while node:
        path.append(node)
        node = vg[node]

    return path

# ==============================
# MAIN
# ==============================
def main():
    grid = create_grid()
    start = (0, 0)
    goal = (ROWS-1, COLS-1)

    plt.figure(figsize=(6,6))

    # Choose algorithm:
    path = bfs(grid, start, goal)
    update_gui(grid, start, goal, [], [], path)
    plt.show()
    time.sleep(3)

    path = dfs(grid, start, goal)
    update_gui(grid, start, goal, [], [], path)
    plt.show()
    time.sleep(3)

    path = ucs(grid, start, goal)
    update_gui(grid, start, goal, [], [], path)
    plt.show()
    time.sleep(3)

    path = iddfs(grid, start, goal)
    update_gui(grid, start, goal, [], [], path)
    plt.show()
    time.sleep(3)

    path = bidirectional(grid, start, goal)
    update_gui(grid, start, goal, [], [], path)
    plt.show()

if __name__ == "__main__":
    main()