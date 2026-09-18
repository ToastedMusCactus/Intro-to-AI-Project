import math
import random
import heapq
import os
from typing import Dict, List, Tuple
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
    key_chars =[]
    for row in state:
        for cell in row:
            key_chars.append(str(cell))
    return "".join(key_chars)


def _is_goal(state: tuple, player: int, win_cells) -> bool:
    return all(state[r][c] == player for r, c in win_cells)


def _heuristic(state: tuple, player: int) -> int:
    """
    Calculates the sum of Manhattan distances by assigning each of the player pieces
    to unique wiining cells to ensure the pieces do not go to the same cell
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


def _get_legal_moves(state: tuple, player: int):
    """
    Legal player moves: single-step and jump
    Key rule: A player can only perform a jump to a cell two positions away 
    if the middle cell contains a piece from any player (including his/her own).

    This guarantees C1->E1 only appears when D1 holds a friendly piece.
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


"""def _get_simple_moves(state: tuple, player: int):
        ## Opponent moves: one-step only.
    moves = []
    for r in range(5):
        for c in range(5):
            if state[r][c] != player:
                continue
            for dr, dc in SIMPLE_DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == 0:
                    moves.append(((r, c), (nr, nc)))
    return moves"""
# The above was simply for facing Ai that would not be able to jump


def _apply_move(state: tuple, old_pos, new_pos, player: int) -> tuple:
    board = [list(row) for row in state]
    board[new_pos[0]][new_pos[1]] = player
    board[old_pos[0]][old_pos[1]] = 0
    return tuple(tuple(row) for row in board)


def _opponent_step(state: tuple, opponent: int):
    moves = _get_legal_moves(state, opponent)
    if not moves:
        return state, None
    best_state = state
    best_move = None
    best_h = _heuristic(state, opponent)
    for old_pos, new_pos in moves:
        new_state = _apply_move(state, old_pos, new_pos, opponent)
        h = _heuristic(new_state, opponent)
        if h < best_h:
            best_h = h
            best_state = new_state
            best_move = (old_pos, new_pos)
    return best_state, best_move


def AI_Player_Team38(board: List[List[int]], player: int, visualize_tree: bool) -> Tuple[str, str]:

    opponent = (player % 4) + 1
    start_state = tuple(tuple(row) for row in board)

    tree = None
    if visualize_tree:
        tree = Tree()
        tree.create_node("Start", "Start")

    player_moves = _get_legal_moves(start_state, player)
    if not player_moves:
        return random_bot(board, player, visualize_tree)

    best_move = None
    best_score = math.inf
    node_count = 0

    for old_pos, new_pos in player_moves:
        state_after_player_move = _apply_move(start_state, old_pos, new_pos, player)

        state_after_opp_move, _ = _opponent_step(state_after_player_move, opponent)
        score = _heuristic(state_after_opp_move, player)

        move_str = (
            f"{chr(ord('A') + old_pos[1])}{old_pos[0] + 1}"
            f"->{chr(ord('A') + new_pos[1])}{new_pos[0] + 1}"
        )

        if visualize_tree and node_count < 300:
            node_count += 1
            node_id = f"node_{node_count}_{move_str}"
            tree.create_node(move_str, node_id, parent="Start")

        if score < best_score:
            best_score = score
            best_move = (old_pos, new_pos)

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
    