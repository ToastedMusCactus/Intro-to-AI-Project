import math
import random
import heapq
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from treelib import Tree

from halma import check_legal_move, win_cells_all, random_bot, initial_pos

# Player move directions: one-step + two-step jumps
MOVE_DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-2, 0), (2, 0), (0, -2), (0, 2),
]

# Opponent move directions: one-step only (no jumps)
SIMPLE_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _state_key(state) -> str:
    key_chars = []
    for row in state:
        for cell in row:
            key_chars.append(str(cell))
    return "".join(key_chars)


def _is_goal(state: tuple, player: int, win_cells) -> bool:
    return all(state[r][c] == player for r, c in win_cells)


def _heuristic(state: tuple, player: int) -> int:
    """
    Calculates the sum of Manhattan distances by assigning each of the player pieces
    to unique winning cells to ensure the pieces do not go to the same cell
    """
    win_cells = win_cells_all[player]

    piece_positions = []
    for r in range(5):
        for c in range(5):
            if state[r][c] == player:
                piece_positions.append((r, c))

    if not piece_positions:
        return 0

    def dist(p, w) -> int:
        return abs(p[0] - w[0]) + abs(p[1] - w[1])

    total_distances = []

    p1, p2, p3 = piece_positions[0], piece_positions[1], piece_positions[2]
    w1, w2, w3 = win_cells[0], win_cells[1], win_cells[2]

    total_distances.append(dist(p1, w1) + dist(p2, w2) + dist(p3, w3))
    total_distances.append(dist(p1, w1) + dist(p2, w3) + dist(p3, w3))
    total_distances.append(dist(p1, w2) + dist(p2, w1) + dist(p3, w3))
    total_distances.append(dist(p1, w2) + dist(p2, w3) + dist(p3, w1))
    total_distances.append(dist(p1, w3) + dist(p2, w1) + dist(p3, w2))
    total_distances.append(dist(p1, w3) + dist(p2, w2) + dist(p3, w1))

    best_total_distance = total_distances[0]
    for d in total_distances:
        if d < best_total_distance:
            best_total_distance = d

    return best_total_distance


def _eval_vector(state: tuple) -> Tuple[int, int, int, int]:
    """Returns a 4-element score vector (s1, s2, s3, s4) for all players."""
    scores = []
    for p in range(1, 5):
        # Higher score is better: 100 minus distance to target
        scores.append(100 - _heuristic(state, p))
    return (scores[0], scores[1], scores[2], scores[3])


def _get_legal_moves(state: tuple, player: int):
    """
    Legal player moves: single-step and jump
    """
    board = [list(row) for row in state]
    moves = []
    for r in range(5):
        for c in range(5):
            if state[r][c] == player:
                for nr in range(5):
                    for nc in range(5):
                        if check_legal_move(board, (r, c), (nr, nc)):
                            moves.append(((r, c), (nr, nc)))
    return moves


def _apply_move(state: tuple, old_pos, new_pos, player: int) -> tuple:
    board = [list(row) for row in state]
    board[new_pos[0]][new_pos[1]] = player
    board[old_pos[0]][old_pos[1]] = 0
    return tuple(tuple(row) for row in board)


def _maxn(
    state: tuple,
    current_player: int,
    ply_depth: int,
    max_ply_depth: int,
    visited: set,
    tree: Optional[Tree] = None,
    parent_id: Optional[str] = None,
    node_count: Optional[List[int]] = None,
) -> Tuple[Tuple[int, int, int, int], Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Recursive Max^n search for 4 players with repeated position pruning & branch ordering."""
    if ply_depth == max_ply_depth:
        return _eval_vector(state), None

    state_key = (state, current_player)
    if state_key in visited:
        return _eval_vector(state), None
    visited.add(state_key)

    moves = _get_legal_moves(state, current_player)
    next_player = (current_player % 4) + 1

    if not moves:
        scores, _ = _maxn(state, next_player, ply_depth + 1, max_ply_depth, visited, tree, parent_id, node_count)
        visited.remove(state_key)
        return scores, None

    # Branch ordering: sort moves by heuristic score for current player
    p_idx = current_player - 1
    moves.sort(key=lambda m: 100 - _heuristic(_apply_move(state, m[0], m[1], current_player), current_player), reverse=True)

    best_vector = None
    best_move = None

    for old_pos, new_pos in moves:
        next_state = _apply_move(state, old_pos, new_pos, current_player)

        node_id = None
        if tree is not None and parent_id is not None and node_count is not None and node_count[0] < 300:
            node_count[0] += 1
            move_str = f"P{current_player}: {chr(ord('A') + old_pos[1])}{old_pos[0] + 1}->{chr(ord('A') + new_pos[1])}{new_pos[0] + 1}"
            node_id = f"node_{node_count[0]}_{move_str}"
            tree.create_node(move_str, node_id, parent=parent_id)

        vector, _ = _maxn(next_state, next_player, ply_depth + 1, max_ply_depth, visited, tree, node_id, node_count)

        if best_vector is None or vector[p_idx] > best_vector[p_idx]:
            best_vector = vector
            best_move = (old_pos, new_pos)

    visited.remove(state_key)
    return best_vector if best_vector is not None else _eval_vector(state), best_move


def AI_Player_Team38(board: List[List[int]], player: int, visualize_tree: bool) -> Tuple[str, str]:
    # Input validation
    if not isinstance(board, list) or len(board) != 5 or not all(isinstance(r, list) and len(r) == 5 for r in board):
        raise ValueError("Invalid board format: Must be a 5x5 grid.")
    if player not in [1, 2, 3, 4]:
        raise ValueError("Invalid player: Must be 1, 2, 3, or 4.")

    start_state = tuple(tuple(row) for row in board)

    tree = None
    node_count = [0]
    if visualize_tree:
        tree = Tree()
        tree.create_node("Start", "Start")

    player_moves = _get_legal_moves(start_state, player)
    if not player_moves:
        return random_bot(board, player, visualize_tree)

    visited = set()
    MAX_PLY_DEPTH = 4  # 4 plies = 1 full round across 4 players (use 8 for 2 full rounds)

    _, best_move = _maxn(
        state=start_state,
        current_player=player,
        ply_depth=0,
        max_ply_depth=MAX_PLY_DEPTH,
        visited=visited,
        tree=tree,
        parent_id="Start" if visualize_tree else None,
        node_count=node_count if visualize_tree else None,
    )

    if visualize_tree and tree is not None:
        try:
            font = ImageFont.load_default()
            lines = []
            for node_id in tree.expand_tree():
                node = tree.get_node(node_id)
                lines.append(f"{'  ' * tree.depth(node_id)}{node.tag}")

            line_height = 18
            image_width = max(320, max(len(line) for line in lines) * 8 + 20)
            image = Image.new("RGB", (image_width, line_height * len(lines) + 20), "white")
            draw = ImageDraw.Draw(image)
            for index, line in enumerate(lines):
                draw.text((10, 10 + index * line_height), line, fill="black", font=font)

            output_path = os.path.join(os.path.dirname(__file__), "Team38 Tree.png")
            image.save(output_path)
            print("Tree exported successfully to Team38 Tree.png")

        except Exception as e:
            print(f"[Warning] Tree export failed ({e}). Rendering ASCII tree instead:")
            tree.show()

    if best_move is None:
        best_move = player_moves[0]

    first_old, first_new = best_move
    old_ref = f"{chr(ord('A') + first_old[1])}{first_old[0] + 1}"
    new_ref = f"{chr(ord('A') + first_new[1])}{first_new[0] + 1}"
    return old_ref, new_ref


# Standalone test of the search algorithm
if __name__ == "__main__":
    test_board = [row[:] for row in initial_pos]
    print("Starting search (player 1)...")
    old_ref, new_ref = AI_Player_Team38(test_board, 1, visualize_tree=True)
    print(f"\nRecommended move: {old_ref} -> {new_ref}")