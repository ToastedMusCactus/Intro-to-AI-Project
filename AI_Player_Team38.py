"""
AI_Player_Team38.py
Practical 3 - Search in Games (4-player Halma)

- Maxn search (depth 2)
- Pruning of repeated positions
- Branch ordering using heuristics
- Alpha-Beta shallow pruning
- treelib visualization -> Team38_Tree.png
"""

import math
import os
import random
from typing import List, Tuple

import graphviz
from treelib import Tree

from halma import check_legal_move, win_cells_all, random_bot, initial_pos


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
PLAYERS = (1, 2, 3, 4)

MOVE_DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-2, 0), (2, 0), (0, -2), (0, 2),
]

SEARCH_DEPTH = 2
VIS_NODE_LIMIT = 300
MAX_SEARCH_NODES = 4000


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _state_key(state) -> str:
    chars = []
    for row in state:
        for cell in row:
            chars.append(str(cell))
    return "".join(chars)


def _fmt(pos: Tuple[int, int]) -> str:
    return f"{chr(ord('A') + pos[1])}{pos[0] + 1}"


def _validate_board(board: List[List[int]]) -> None:
    if len(board) != 5 or any(len(row) != 5 for row in board):
        raise ValueError("Board must be 5x5")
    for row in board:
        for cell in row:
            if cell not in (0, 1, 2, 3, 4):
                raise ValueError(f"Invalid cell value: {cell}")


def _is_goal(state: tuple, player: int) -> bool:
    return all(state[r][c] == player for r, c in win_cells_all[player])


# ----------------------------------------------------------------------
# Heuristic (kept from teammate's version)
# ----------------------------------------------------------------------
def _heuristic(state: tuple, player: int) -> int:
    win_cells = win_cells_all[player]

    piece_positions = []
    for r in range(5):
        for c in range(5):
            if state[r][c] == player:
                piece_positions.append((r, c))

    if len(piece_positions) < 3:
        return 0

    def dist(p, w) -> int:
        return abs(p[0] - w[0]) + abs(p[1] - w[1])

    p1, p2, p3 = piece_positions[0], piece_positions[1], piece_positions[2]
    w1, w2, w3 = win_cells[0], win_cells[1], win_cells[2]

    best = math.inf
    for perm in [
        (w1, w2, w3), (w1, w3, w2),
        (w2, w1, w3), (w2, w3, w1),
        (w3, w1, w2), (w3, w2, w1),
    ]:
        total = dist(p1, perm[0]) + dist(p2, perm[1]) + dist(p3, perm[2])
        if total < best:
            best = total
    return best


def _evaluate_all(state: tuple) -> Tuple[float, float, float, float]:
    return tuple(-float(_heuristic(state, p)) for p in PLAYERS)


# ----------------------------------------------------------------------
# Move generation
# ----------------------------------------------------------------------
def _legal_destinations(state: tuple, r: int, c: int) -> List[Tuple[int, int]]:
    destinations = []

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == 0:
            destinations.append((nr, nc))

    for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 5 and 0 <= nc < 5):
            continue
        if state[nr][nc] != 0:
            continue
        mr, mc = (r + nr) // 2, (c + nc) // 2
        if state[mr][mc] != 0:
            destinations.append((nr, nc))

    return destinations


def _get_legal_moves(state: tuple, player: int):
    moves = []
    for r in range(5):
        for c in range(5):
            if state[r][c] == player:
                for dest in _legal_destinations(state, r, c):
                    moves.append(((r, c), dest))
    return moves


def _apply_move(state: tuple, old_pos, new_pos, player: int) -> tuple:
    board = [list(row) for row in state]
    board[new_pos[0]][new_pos[1]] = player
    board[old_pos[0]][old_pos[1]] = 0
    return tuple(tuple(row) for row in board)


# ----------------------------------------------------------------------
# Maxn search with shallow Alpha-Beta pruning
# ----------------------------------------------------------------------
def _maxn(
    state: tuple,
    player_to_move: int,
    depth: int,
    visited: set,
    alpha: List[float],
    beta: List[float],
    node_budget: List[int],
) -> Tuple[float, float, float, float]:

    for p in PLAYERS:
        if _is_goal(state, p):
            scores = [-1000.0] * 4
            scores[p - 1] = 1000.0
            return tuple(scores)

    if depth == 0 or node_budget[0] <= 0:
        return _evaluate_all(state)

    node_budget[0] -= 1

    moves = _get_legal_moves(state, player_to_move)
    if not moves:
        return _maxn(state, player_to_move % 4 + 1, depth - 1,
                     visited, alpha, beta, node_budget)

    my_index = player_to_move - 1
    best_scores = None

    def score_move(mv):
        old_pos, new_pos = mv
        ns = _apply_move(state, old_pos, new_pos, player_to_move)
        return _heuristic(ns, player_to_move)

    ordered_moves = sorted(moves, key=score_move)

    for old_pos, new_pos in ordered_moves:
        new_state = _apply_move(state, old_pos, new_pos, player_to_move)
        key = _state_key(new_state)

        if key in visited:
            continue
        visited.add(key)

        child_scores = _maxn(new_state, player_to_move % 4 + 1,
                             depth - 1, visited, alpha, beta, node_budget)

        if best_scores is None or child_scores[my_index] > best_scores[my_index]:
            best_scores = child_scores

        if best_scores is not None:
            if best_scores[my_index] >= beta[my_index]:
                break
            if best_scores[my_index] > alpha[my_index]:
                alpha[my_index] = best_scores[my_index]

    if best_scores is None:
        return _evaluate_all(state)
    return best_scores


# ----------------------------------------------------------------------
# Visualization (build DOT manually, works on all treelib versions)
# ----------------------------------------------------------------------
def _visualize(state: tuple, player: int) -> None:
    """
    Build a search tree with treelib, then write it to Team38_Tree.png
    by constructing the DOT source manually (avoids treelib's
    to_graphviz() version quirks) and rendering it via graphviz.
    """
    tree = Tree()
    tree.create_node("Start", "Start")

    counter = [0]

    def expand(state_now: tuple, player_now: int, parent_id: str, depth: int):
        if depth == 0 or counter[0] >= VIS_NODE_LIMIT:
            return
        moves = _get_legal_moves(state_now, player_now)

        def score(mv):
            old_pos, new_pos = mv
            ns = _apply_move(state_now, old_pos, new_pos, player_now)
            return _heuristic(ns, player_now)

        ordered = sorted(moves, key=score)[:5]

        for old_pos, new_pos in ordered:
            if counter[0] >= VIS_NODE_LIMIT:
                return
            counter[0] += 1
            move_str = f"{_fmt(old_pos)}->{_fmt(new_pos)}"
            node_id = f"n{counter[0]}_{move_str}"
            tree.create_node(move_str, node_id, parent=parent_id)
            ns = _apply_move(state_now, old_pos, new_pos, player_now)
            expand(ns, player_now % 4 + 1, node_id, depth - 1)

    expand(state, player, "Start", SEARCH_DEPTH)

    # ---- Build DOT source manually ----
    lines = ["digraph tree {", '    node [shape=circle, fontsize=10];']
    for node in tree.all_nodes():
        label = str(node.tag).replace('"', '\\"')
        lines.append(f'    "{node.identifier}" [label="{label}"];')
    for node in tree.all_nodes():
        if node.bpointer is not None:
            lines.append(f'    "{node.bpointer}" -> "{node.identifier}";')
    lines.append("}")
    dot_string = "\n".join(lines)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "Team38_Tree")

    try:
        src = graphviz.Source(dot_string)
        src.render(output_path, format="png", cleanup=True)
        print(f">>> SUCCESS: Team38_Tree.png saved to {output_path}.png <<<")
    except Exception as e:
        print(f"[Warning] Graphviz render failed: {e}")
        print("Falling back to ASCII tree.")
        tree.show()


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def AI_Player_Team38(
    board: List[List[int]],
    player: int,
    visualize_tree: bool,
) -> Tuple[str, str]:

    _validate_board(board)
    if player not in PLAYERS:
        raise ValueError(f"Player must be in {PLAYERS}")

    start_state = tuple(tuple(row) for row in board)

    player_moves = _get_legal_moves(start_state, player)
    if not player_moves:
        return random_bot(board, player, visualize_tree)

    def root_score(mv):
        old_pos, new_pos = mv
        ns = _apply_move(start_state, old_pos, new_pos, player)
        return _heuristic(ns, player)

    ordered_moves = sorted(player_moves, key=root_score)

    best_move = None
    best_score = -math.inf
    node_budget = [MAX_SEARCH_NODES]

    for old_pos, new_pos in ordered_moves:
        state_after = _apply_move(start_state, old_pos, new_pos, player)
        visited = {_state_key(state_after)}

        alpha = [-math.inf] * 4
        beta = [math.inf] * 4

        scores = _maxn(state_after, player % 4 + 1, SEARCH_DEPTH - 1,
                       visited, alpha, beta, node_budget)

        my_score = scores[player - 1]
        if my_score > best_score:
            best_score = my_score
            best_move = (old_pos, new_pos)

    if best_move is None:
        best_move = ordered_moves[0]

    if visualize_tree:
        _visualize(start_state, player)

    return _fmt(best_move[0]), _fmt(best_move[1])


# ----------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_board = [row[:] for row in initial_pos]
    print("Starting Maxn search (player 1)...")
    old_ref, new_ref = AI_Player_Team38(test_board, 1, visualize_tree=True)
    print(f"\nRecommended move: {old_ref} -> {new_ref}")