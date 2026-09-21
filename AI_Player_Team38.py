import math
import random
import heapq
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont
from treelib import Tree

from halma import check_legal_move, win_cells_all, random_bot, initial_pos

# Player move directions: either one-step or two-step jumps
MOVE_DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-2, 0), (2, 0), (0, -2), (0, 2),
]

# Simple move directions for practical 2 AI
SIMPLE_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _is_goal(state: tuple, player: int, win_cells) -> bool:
    return all(state[r][c] == player for r, c in win_cells)

HEURISTIC_CACHE = {}


def _heuristic(state: tuple, player: int) -> int:
    """
    Calculates the sum of Manhattan distances by assigning each of the player pieces
    to unique winning cells to ensure the pieces do not go to the same cell
    """

    cache_key = (state, player)
    if cache_key in HEURISTIC_CACHE:
        return HEURISTIC_CACHE[cache_key]
    
    win_cells = win_cells_all[player]

    board = [list(row) for row in state]

    piece_positions = []
    for r in range(5):
        for c in range(5):
            if board[r][c] == player:
                piece_positions.append((r, c))

    if not piece_positions:
        return 0
    pieces_outside = [p for p in piece_positions if p not in win_cells]

    if not pieces_outside:
        HEURISTIC_CACHE[cache_key] = 0
        return 0

    unnoccupied_goals = [w for w in win_cells if state[w[0]][w[1]] == 0]

    while len(unnoccupied_goals) < len(pieces_outside):
        unnoccupied_goals.append(win_cells[0])

    def dist(p,w) -> int:
        return abs(p[0] - w[0]) + abs(p[1] - w[1])

    n_outside = len(pieces_outside)
    if n_outside == 1:
        best_total_distance = dist(pieces_outside[0], unnoccupied_goals[0])
    elif n_outside == 2:
        p1, p2 = pieces_outside[0], pieces_outside[1]
        w1, w2 = unnoccupied_goals[0], unnoccupied_goals[1]
        best_total_distance = min(
            dist(p1, w1) + dist(p2, w2),
            dist(p1, w2) + dist(p2, w1)
        )
    else:
        p1, p2, p3 = pieces_outside[0], pieces_outside[1], pieces_outside[2]
        w1, w2, w3 = unnoccupied_goals[0], unnoccupied_goals[1], unnoccupied_goals[2]

        best_total_distance = min(
            dist(p1, w1) + dist(p2, w2) + dist(p3, w3),
            dist(p1, w1) + dist(p2, w3) + dist(p3, w2),
            dist(p1, w2) + dist(p2, w1) + dist(p3, w3),
            dist(p1, w2) + dist(p2, w3) + dist(p3, w1),
            dist(p1, w3) + dist(p2, w1) + dist(p3, w2),
            dist(p1, w3) + dist(p2, w2) + dist(p3, w1)
        )

    score = best_total_distance + (n_outside * 40)
    HEURISTIC_CACHE[cache_key] = score
    return score


def _eval_scores(state: tuple) -> Tuple[int, int, int, int]:
    """Returns the "scores" of each player in a 4 element vector"""
    scores = []
    for p in range(1, 5):
        # Higher score is better: 1000 minus distance to target
        scores.append(1000 - _heuristic(state, p))
    return (scores[0], scores[1], scores[2], scores[3])


def _get_legal_moves(state: tuple, player: int):
    """
    Legal player moves: single-step and jump
    """
    board = [list(row) for row in state]
    moves = []

    for r in range(5):
        for c in range(5):
            if board[r][c] == player:
                for new_r in range(5):
                    for new_c in range(5):
                        if check_legal_move(board, (r, c), (new_r, new_c)):
                            moves.append(((r, c), (new_r, new_c)))

    return moves


def _apply_move(state: tuple, old_pos, new_pos, player: int) -> tuple:
    if old_pos[0] < 0 or old_pos[0] > 4 or old_pos[1] < 0 or old_pos[1] > 4:
        print("Invalid starting position")
        return None;

    board = [list(row) for row in state]
    board[new_pos[0]][new_pos[1]] = player
    board[old_pos[0]][old_pos[1]] = 0
    return tuple(tuple(row) for row in board)


def _maxn(state: tuple, current_player: int, play_depth: int, max_play_depth: int, visited: set,
    tree: Optional[Tree] = None, parent_id: Optional[str] = None, node_count: Optional[List[int]] = None,
) -> Tuple[Tuple[int, int, int, int], Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Recursive Max^n search for 4 players with repeated position pruning & branch ordering."""
    if play_depth == max_play_depth:
        return _eval_scores(state), None

    state_key = (state, current_player)
    if state_key in visited:
        return _eval_scores(state), None
    visited.add(state_key)

    moves = _get_legal_moves(state, current_player)
    next_player = (current_player % 4) + 1

    if not moves:
        scores, _ = _maxn(state, next_player, play_depth + 1, max_play_depth, visited, tree, parent_id, node_count)
        visited.remove(state_key)
        return scores, None

    # Branch ordering: sort moves by heuristic score for current player
    win_cells = win_cells_all[current_player]

    scored_moves = []
    for m in moves:
        if m[0] in win_cells and state[m[0][0]][m[0][1]] == current_player:
            if m[1] not in win_cells:
                continue
        
        next_state = _apply_move(state, m[0], m[1], current_player)
        score = 1000 - _heuristic(next_state, current_player)
        scored_moves.append((score, m[0], m[1], next_state))

    scored_moves.sort(key = lambda x: x[0], reverse = True)

    p_idx = current_player - 1

    best_moves = scored_moves[:3]

    best_vector = None
    best_move = None

    for score, old_pos, new_pos, next_state in best_moves:

        node_id = None
        if tree is not None and parent_id is not None and node_count is not None and node_count[0] < 300:
            node_count[0] += 1
            move_str = f"P{current_player}: {chr(ord('A') + old_pos[1])}{old_pos[0] + 1}->{chr(ord('A') + new_pos[1])}{new_pos[0] + 1}"
            node_id = f"node_{node_count[0]}_{move_str}"
            tree.create_node(move_str, node_id, parent=parent_id)

        vector, _ = _maxn(next_state, next_player, play_depth + 1, max_play_depth, visited, tree, node_id, node_count)

        if best_vector is None or vector[p_idx] > best_vector[p_idx]:
            best_vector = vector
            best_move = (old_pos, new_pos)

            if best_vector[p_idx] >= 1000:
                break

    visited.remove(state_key)
    return best_vector if best_vector is not None else _eval_scores(state), best_move


def AI_Player_Team38(board: List[List[int]], player: int, visualize_tree: bool) -> Tuple[str, str]:
    # Clear cache at every turn
    HEURISTIC_CACHE.clear()
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
    MAX_PLAY_DEPTH = 8  # 4 plies = 1 full round across 4 players (use 8 for 2 full rounds)

    _, best_move = _maxn(
        state=start_state,
        current_player = player,
        play_depth = 0,
        max_play_depth = MAX_PLAY_DEPTH,
        visited = visited,
        tree = tree,
        parent_id = "Start" if visualize_tree else None,
        node_count = node_count if visualize_tree else None,
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