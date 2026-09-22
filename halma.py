from typing import Dict, List, NamedTuple, Tuple
import random
import collections
import heapq
from treelib import Tree

'''
Helper Class to contain the results
'''
class GameResult(NamedTuple):
    status: str
    winners: List[int]
    tied_players: List[int]
    losers: List[int]

'''
Initial board states for 1v1v1v1 and 1v1
'''
initial_pos: List[List[int]] = [
    [1,1,0,2,2],
    [1,0,0,0,2],
    [0,0,0,0,0],
    [4,0,0,0,3],
    [4,4,0,3,3]
]

initial_pos_1v1: List[List[int]] = [
    [1,1,0,0,0],
    [1,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,2],
    [0,0,0,2,2]
]

'''
Win cells per player for 1v1v1v1 and 1v1
'''
win_cells_all: Dict[int, List[Tuple[int, int]]] = {
    1: [(3, 4), (4, 3), (4, 4)],
    2: [(3, 0), (4, 0), (4, 1)],
    3: [(0, 0), (0, 1), (1, 0)],
    4: [(0, 3), (0, 4), (1, 4)]
}

win_cells_1v1: Dict[int, List[Tuple[int, int]]] = {
    1: [(3, 4), (4, 3), (4, 4)],
    2: [(0, 0), (0, 1), (1, 0)]
}

'''
Utility function to parse input string and transform it into Tuple
'''
def parse_position(position: str) -> Tuple[int, int]:
    position = position.strip().lower()
    if len(position) != 2:
        raise ValueError(f"Invalid position '{position}'. Expected something like 'a4'.")
    column_character: str = position[0]
    row_character: str = position[1]
    if column_character not in "abcde":
        raise ValueError("Column must be between 'a' and 'e'.")
    if row_character not in "12345":
        raise ValueError("Row must be between 1 and 5.")
    column: int = ord(column_character) - ord("a")
    row: int = int(row_character) - 1
    return row, column

'''
This function detects if the move is legal according to the rules specified in the assignment
'''
def check_legal_move(board: List[List[int]], oldPos: Tuple[int, int], newPos: Tuple[int, int]) -> bool:
    rest: Tuple[int, int] = tuple(map(lambda i, j: abs(i - j), oldPos, newPos))
    if board[newPos[0]][newPos[1]] != 0:
        return False
    if rest in [(0, 1), (1, 0)]:
        return True
    if rest in [(0, 2), (2, 0)]:
        middlePos: Tuple[int, int] = ((oldPos[0] + newPos[0]) // 2, (oldPos[1] + newPos[1]) // 2)
        if board[middlePos[0]][middlePos[1]] != 0:
            return True
    return False

'''
This function executes the move
'''
def move(board: List[List[int]], oldPos: Tuple[int, int], newPos: Tuple[int, int], player: int) -> bool:
    try:
        assert player in [1, 2, 3, 4], f"Player {player} is not a valid player"
        assert len(board) > 0 and len(board[0]) > 0, "The board is empty"
        assert (0 <= oldPos[0] < len(board) and 0 <= oldPos[1] < len(board[oldPos[0]])), "Old position is out of bounds"
        assert (0 <= newPos[0] < len(board) and 0 <= newPos[1] < len(board[newPos[0]])), "New position is out of bounds"
        assert board[oldPos[0]][oldPos[1]] == player, f"Player {player} selected a cell containing {board[oldPos[0]][oldPos[1]]}"
    except AssertionError as e:
        print(e)
        return False
    legal: bool = check_legal_move(board, oldPos, newPos)
    if not legal:
        return False
    board[newPos[0]][newPos[1]] = player
    board[oldPos[0]][oldPos[1]] = 0
    return True

'''
This Function checks for the win or tie conditions
'''
def check_win_condition(board: List[List[int]], move_count: int, maximum_move_limit: int, all_players: bool) -> GameResult:
    if maximum_move_limit <= 0:
        raise ValueError("Maximum move limit must be greater than zero")
    if move_count < 0:
        raise ValueError("Move count cannot be negative")
    if all_players:
        win_cells: Dict[int, List[Tuple[int, int]]] = win_cells_all
    else:
        win_cells = win_cells_1v1
    active_players: List[int] = list(win_cells.keys())
    winners: List[int] = []
    for player in active_players:
        player_has_won: bool = all(board[row][column] == player for row, column in win_cells[player])
        if player_has_won:
            winners.append(player)
    if winners:
        return GameResult(status="winner", winners=winners, tied_players=[],
                          losers=[p for p in active_players if p not in winners])
    if move_count < maximum_move_limit:
        return GameResult(status="ongoing", winners=[], tied_players=[], losers=[])
    losers: List[int] = []
    for player in active_players:
        is_blocking: bool = False
        for other_player in active_players:
            if player == other_player:
                continue
            for row, column in win_cells[other_player]:
                if board[row][column] == player:
                    is_blocking = True
                    break
            if is_blocking:
                break
        if is_blocking:
            losers.append(player)
    tied_players: List[int] = [p for p in active_players if p not in losers]
    return GameResult(status="move_limit", winners=[], tied_players=tied_players, losers=losers)

def random_bot(board: List[List[int]], player: int, visualize_tree: bool) -> Tuple[str, str]:
    if player not in [1, 2, 3, 4]:
        raise ValueError(f"Player {player} is not a valid player")
    if len(board) != 5 or any(len(row) != 5 for row in board):
        raise ValueError("Board must be 5 by 5")
    legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for row in range(5):
        for column in range(5):
            if board[row][column] != player:
                continue
            oldPos = (row, column)
            for new_row in range(5):
                for new_column in range(5):
                    newPos = (new_row, new_column)
                    if check_legal_move(board, oldPos, newPos):
                        legal_moves.append((oldPos, newPos))
    if not legal_moves:
        raise ValueError(f"Player {player} has no legal moves")
    oldPos, newPos = random.choice(legal_moves)
    if visualize_tree:
        print("Random bot: no minimax search tree to visualize.")
    old_reference = chr(ord("A") + oldPos[1]) + str(oldPos[0] + 1)
    new_reference = chr(ord("A") + newPos[1]) + str(newPos[0] + 1)
    return old_reference, new_reference

def illegal_bot(board: List[List[int]], player: int, visualize_tree: bool) -> Tuple[str, str]:
    if player not in [1, 2, 3, 4]:
        raise ValueError(f"Player {player} is not valid")
    for row in range(5):
        for column in range(5):
            if board[row][column] == player:
                position = chr(ord("A") + column) + str(row + 1)
                if visualize_tree:
                    print(f"Illegal bot attempts: {position} -> {position}")
                return position, position
    raise ValueError(f"Player {player} has no pieces")


# ===================================================================
#                        Search bot: search_bot
# ===================================================================

# Player move directions: one-step + two-step jumps
MOVE_DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-2, 0), (2, 0), (0, -2), (0, 2),
]

# Opponent move directions: one-step only (no jumps)
SIMPLE_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _state_key(state: tuple) -> str:
    return "".join(str(cell) for row in state for cell in row)


def _is_goal(state: tuple, player: int, win_cells) -> bool:
    return all(state[r][c] == player for r, c in win_cells)


def _heuristic(state: tuple, player: int, win_cells) -> int:
    total = 0
    for r in range(5):
        for c in range(5):
            if state[r][c] == player:
                total += min(abs(r - wr) + abs(c - wc) for wr, wc in win_cells)
    return total


def _get_legal_moves(state: tuple, player: int):
    """
    Player moves: one-step + jumps.
    Key rule: a jump is only legal when the middle cell contains
    the player's OWN piece.
    -> This guarantees C1->E1 only appears when D1 holds a friendly piece.
    """
    moves = []
    for r in range(5):
        for c in range(5):
            if state[r][c] != player:
                continue
            for dr, dc in MOVE_DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < 5 and 0 <= nc < 5):
                    continue
                if state[nr][nc] != 0:
                    continue
                # Two-step jump: middle cell must be own piece
                if abs(dr) == 2 or abs(dc) == 2:
                    mr, mc = (r + nr) // 2, (c + nc) // 2
                    if state[mr][mc] != player:
                        continue
                moves.append(((r, c), (nr, nc)))
    return moves


def _get_simple_moves(state: tuple, player: int):
    """Opponent moves: one-step only."""
    moves = []
    for r in range(5):
        for c in range(5):
            if state[r][c] != player:
                continue
            for dr, dc in SIMPLE_DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == 0:
                    moves.append(((r, c), (nr, nc)))
    return moves


def _apply_move(state: tuple, old_pos, new_pos, player: int) -> tuple:
    board = [list(row) for row in state]
    board[new_pos[0]][new_pos[1]] = player
    board[old_pos[0]][old_pos[1]] = 0
    return tuple(tuple(row) for row in board)


def _opponent_step(state: tuple, opponent: int, opp_win_cells):
    moves = _get_simple_moves(state, opponent)
    if not moves:
        return state, None
    best_state = state
    best_move = None
    best_h = _heuristic(state, opponent, opp_win_cells)
    for old_pos, new_pos in moves:
        new_state = _apply_move(state, old_pos, new_pos, opponent)
        h = _heuristic(new_state, opponent, opp_win_cells)
        if h < best_h:
            best_h = h
            best_state = new_state
            best_move = (old_pos, new_pos)
    return best_state, best_move


def search_bot(board: List[List[int]], player: int, visualize_tree: bool) -> Tuple[str, str]:
    win_cells = win_cells_1v1[player]
    opponent = 3 - player
    opp_win_cells = win_cells_1v1[opponent]

    start_state = tuple(tuple(row) for row in board)
    start_key = _state_key(start_state)

    tree = None
    state_to_node: Dict[str, str] = {}
    node_counter = [0]
    if visualize_tree:
        tree = Tree()
        tree.create_node("Start", "Start")
        state_to_node[start_key] = "Start"

    h0 = _heuristic(start_state, player, win_cells)
    counter = 1
    pq = [(h0, 0, 0, start_state, [])]
    visited = {start_key}
    depth_limit = 40
    max_nodes = 200000

    final_path = None
    final_state = None
    explored = 0

    while pq and explored < max_nodes:
        f, g, _, current_state, path = heapq.heappop(pq)
        explored += 1

        if _is_goal(current_state, player, win_cells):
            final_path = path
            final_state = current_state
            break

        if g >= depth_limit:
            continue

        player_moves = _get_legal_moves(current_state, player)

        children = []
        for old_pos, new_pos in player_moves:
            state_after_player = _apply_move(current_state, old_pos, new_pos, player)
            if _is_goal(state_after_player, player, win_cells):
                children.append((0, old_pos, new_pos, state_after_player))
                continue
            state_after_opp, _ = _opponent_step(state_after_player, opponent, opp_win_cells)
            h = _heuristic(state_after_opp, player, win_cells)
            children.append((h, old_pos, new_pos, state_after_opp))
        children.sort(key=lambda x: x[0])

        for h, old_pos, new_pos, new_state in children:
            key = _state_key(new_state)
            if key in visited:
                continue
            visited.add(key)

            move_str = (
                f"{chr(ord('A') + old_pos[1])}{old_pos[0] + 1}"
                f"->{chr(ord('A') + new_pos[1])}{new_pos[0] + 1}"
            )
            new_path = path + [(old_pos, new_pos)]

            if visualize_tree and node_counter[0] < 400:
                parent_id = state_to_node.get(_state_key(current_state), "Start")
                node_counter[0] += 1
                new_node_id = f"n{node_counter[0]}_{move_str}"
                tree.create_node(move_str, new_node_id, parent=parent_id)
                state_to_node[key] = new_node_id

            new_g = g + 1
            heapq.heappush(pq, (new_g + h, new_g, counter, new_state, new_path))
            counter += 1

    if final_path is not None:
        # Highlight the shortest path inside the search tree
        if visualize_tree:
            end_key = _state_key(final_state)
            current_id = state_to_node.get(end_key)
            while current_id and current_id != "Start":
                node = tree.get_node(current_id)
                if node and "★" not in node.tag:
                    node.tag = "★ " + node.tag
                parent = tree.parent(current_id)
                current_id = parent.identifier if parent else None
            print(f"\nSearched {explored} nodes, found a path of {len(final_path)} turns")
            print("=== Search tree (★ = shortest path) ===")
            tree.show()
            print("======================================\n")

        first_old, first_new = final_path[0]
        old_ref = f"{chr(ord('A') + first_old[1])}{first_old[0] + 1}"
        new_ref = f"{chr(ord('A') + first_new[1])}{first_new[0] + 1}"

        print(f"Recommended move: {old_ref} -> {new_ref}")
        print(f"(Searched {explored} nodes, shortest path has {len(final_path)} steps)")
        print("All steps of the shortest path:")
        for i, (old_pos, new_pos) in enumerate(final_path, 1):
            step_old = f"{chr(ord('A') + old_pos[1])}{old_pos[0] + 1}"
            step_new = f"{chr(ord('A') + new_pos[1])}{new_pos[0] + 1}"
            print(f"  ★ {i:2d}. {step_old} -> {step_new}")

        return old_ref, new_ref
    else:
        if visualize_tree:
            print(f"Searched {explored} nodes, no path found, falling back to random move.")
        return random_bot(board, player, visualize_tree)


# Standalone test of the search algorithm
if __name__ == "__main__":
    test_board = [row[:] for row in initial_pos_1v1]
    print("Starting search (player 1)...")
    old_ref, new_ref = search_bot(test_board, 1, visualize_tree=True)
    print(f"\nRecommended move: {old_ref} -> {new_ref}")