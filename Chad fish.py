import chess
import berserk
import random
import time
import threading
import math
from functools import lru_cache
from datetime import timedelta

# Lichess Config

API_TOKEN = ""  

MAX_DEPTH = 10                   

BASE_THINK_TIME = 15.0               

# LICHESS

session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)

# Piece Values

PE_VAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 350,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000
}


# Piece-square tables
pe_pos_opening = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -20, -20, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,0,0,0,0,-20,-40,
        -30,0,10,15,15,10,0,-30,
        -30,5,15,20,20,15,5,-30,
        -30,0,15,20,20,15,0,-30,
        -30,5,10,15,15,10,5,-30,
        -40,-20,0,5,5,0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,0,0,0,0,0,0,-10,
        -10,0,5,10,10,5,0,-10,
        -10,5,5,10,10,5,5,-10,
        -10,0,10,10,10,10,0,-10,
        -10,10,10,10,10,10,10,-10,
        -10,5,0,0,0,0,5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
        0,0,0,5,5,0,0,0,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        5,10,10,10,10,10,10,5,
        0,0,0,0,0,0,0,0
    ],
    chess.QUEEN: [
        -20,-10,-10,-5,-5,-10,-10,-20,
        -10,0,0,0,0,0,0,-10,
        -10,0,5,5,5,5,0,-10,
        -5,0,5,5,5,5,0,-5,
        0,0,5,5,5,5,0,0,
        -10,5,5,5,5,5,0,-10,
        -10,0,5,0,0,0,0,-10,
        -20,-10,-10,-5,-5,-10,-10,-20
    ],
    chess.KING: [
        20,30,10,0,0,10,30,20,
        20,20,0,0,0,0,20,20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30
    ]
}

pe_pos_middlegame= {
    chess.PAWN: [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 5, 5, 5, 5, 5, 5, 5,
    1, 1, 2, 5, 5, 2, 1, 1,
    0.5, 0.5, 1, 3, 3, 1, 0.5, 0.5,
    0, 0, 0, 2, 2, 0, 0, 0,
    0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5,
    0.5, 1, 1, -2, -2, 1, 1, 0.5,
    0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT: [
        -5,-4,-3,-3,-3,-3,-4,-5,
        -4,-2,0,0,0,0,-2,-4,
        -3,0,1,1.5,1.5,1,0,-3,
        -3,0.5,1.5,2,2,1.5,0.5,-3,
        -3,0,1.5,2,2,1.5,0,-3,
        -3,0.5,1,1.5,1.5,1,0.5,-3,
        -4,-2,0,0.5,0.5,0,-2,-4,
        -5,-4,-3,-3,-3,-3,-4,-5
    ],
    chess.BISHOP: [
        -2,-1,-1,-1,-1,-1,-1,-2,
        -1,0,0,0,0,0,0,-1,
        -1,0,0.5,1,1,0.5,0,-1,
        -1,0.5,0.5,1,1,0.5,0.5,-1,
        -1,0,1,1,1,1,0,-1,
        -1,1,1,1,1,1,1,-1,
        -1,0.5,0,0,0,0,0.5,-1,
        -2,-1,-1,-1,-1,-1,-1,-2
    ],
    chess.ROOK: [
        0,0,0,0,0,0,0,0,
        0.5,1,1,1,1,1,1,0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        0,0,0,0.5,0.5,0,0,0
    ],
    chess.QUEEN: [
        -2,-1,-1,-0.5,-0.5,-1,-1,-2,
        -1,0,0,0,0,0,0,-1,
        -1,0,0.5,0.5,0.5,0.5,0,-1,
        -0.5,0,0.5,0.5,0.5,0.5,0,-0.5,
        0,0,0.5,0.5,0.5,0.5,0,0,
        -1,0.5,0.5,0.5,0.5,0.5,0,-1,
        -1,0,0.5,0,0,0,0,-1,
        -2,-1,-1,-0.5,-0.5,-1,-1,-2
    ],
    chess.KING: [
        2,3,1,0,0,1,3,2,
        2,2,0,0,0,0,2,2,
        -1,-2,-2,-2,-2,-2,-2,-1,
        -2,-3,-3,-4,-4,-3,-3,-2,
        -3,-4,-4,-5,-5,-4,-4,-3,
        -3,-4,-4,-5,-5,-4,-4,-3,
        -3,-4,-4,-5,-5,-4,-4,-3,
        -3,-4,-4,-5,-5,-4,-4,-3
    ]
}
pe_pos_endgame = {
    chess.PAWN: [
        0,0,0,0,0,0,0,0,
        5,10,10,-20,-20,10,10,5,
        5,-5,-10,0,0,-10,-5,5,
        0,0,0,20,20,0,0,0,
        5,5,10,25,25,10,5,5,
        10,10,20,30,30,20,10,10,
        50,50,50,50,50,50,50,50,
        0,0,0,0,0,0,0,0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,0,0,0,0,-20,-40,
        -30,0,10,15,15,10,0,-30,
        -30,5,15,20,20,15,5,-30,
        -30,0,15,20,20,15,0,-30,
        -30,5,10,15,15,10,5,-30,
        -40,-20,0,5,5,0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,0,0,0,0,0,0,-10,
        -10,0,5,10,10,5,0,-10,
        -10,5,5,10,10,5,5,-10,
        -10,0,10,10,10,10,0,-10,
        -10,10,10,10,10,10,10,-10,
        -10,5,0,0,0,0,5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
        0,0,0,5,5,0,0,0,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5,
        5,10,10,10,10,10,10,5,
        0,0,0,0,0,0,0,0
    ],
    chess.QUEEN: [
        -20,-10,-10,-5,-5,-10,-10,-20,
        -10,0,0,0,0,0,0,-10,
        -10,0,5,5,5,5,0,-10,
        -5,0,5,5,5,5,0,-5,
        0,0,5,5,5,5,0,0,
        -10,5,5,5,5,5,0,-10,
        -10,0,5,0,0,0,0,-10,
        -20,-10,-10,-5,-5,-10,-10,-20
    ],
    chess.KING: [
        -50,-30,-30,-30,-30,-30,-30,-50,
        -30,-30,0,0,0,0,-30,-30,
        -30,0,10,20,20,10,0,-30,
        -30,0,20,30,30,20,0,-30,
        -30,0,20,30,30,20,0,-30,
        -30,0,10,20,20,10,0,-30,
        -30,-30,0,0,0,0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ]
}

# Opening Book (simple hardcoded version)
OPENING_BOOK = {
    "": ["e2e4", "d2d4", "c2c4", "g1f3"],  # first move options
    "e2e4": ["e7e5", "c7c5", "e7e6"],
    "d2d4": ["d7d5", "g8f6", "e7e6"],
    "c2c4": ["e7e5", "g8f6", "c7c5"],
    "g1f3": ["d7d5", "g8f6", "c7c5"],
}

# Get opening 

def king_safety(board: chess.Board) -> int:
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            continue
        file = chess.square_file(king_sq)
        # Check pawn shield (simple version: pawns on 3 files around king)
        pawn_shield = 0
        for f in range(max(0, file-1), min(7, file+2)):
            for rank in ([1,2] if color == chess.WHITE else [6,7]):
                if board.piece_at(chess.square(f, rank)) == chess.Piece(chess.PAWN, color):
                    pawn_shield += 1
        bonus = 20 * pawn_shield
        score += bonus if color == chess.WHITE else -bonus
    return score
def pawn_structure(board: chess.Board, phase: str) -> int:
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        files = [chess.square_file(p) for p in pawns]

        # Doubled pawns
        for f in set(files):
            if files.count(f) > 1:
                penalty = -15 * (files.count(f)-1)
                score += penalty if color == chess.WHITE else -penalty

        # Isolated pawns
        for f in files:
            if (f-1 not in files) and (f+1 not in files):
                penalty = -10
                score += penalty if color == chess.WHITE else -penalty

        # Passed pawns
        for sq in pawns:
            if not board.attackers(not color, sq):  # naive check
                bonus = 20 if phase == "endgame" else 10
                score += bonus if color == chess.WHITE else -bonus

    return score

def mobility(board: chess.Board) -> int:
    white_moves = sum(1 for m in board.legal_moves if board.turn == chess.WHITE)
    board.push(chess.Move.null())  # pass move to count opponent’s moves
    black_moves = sum(1 for m in board.legal_moves if board.turn == chess.BLACK)
    board.pop()
    return (white_moves - black_moves) * 2
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
def center_control(board: chess.Board) -> int:
    score = 0
    for sq in CENTER:
        piece = board.piece_at(sq)
        if piece:
            bonus = 15
            score += bonus if piece.color == chess.WHITE else -bonus
    return score    
def get_opening_move(board: chess.Board) -> chess.Move | None:
    moves_played = " ".join([m.uci() for m in board.move_stack])
    if moves_played in OPENING_BOOK:
        choice = random.choice(OPENING_BOOK[moves_played])
        return chess.Move.from_uci(choice)
    return None
# Game phrase
def get_phase(board: chess.Board) -> str:
    """Roughly estimate game phase by material balance."""
    total_material = sum(
        len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
        for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    )

    if total_material > 12:   # many pieces → opening
        return "opening"
    elif total_material > 6:  # medium pieces → middlegame
        return "middlegame"
    else:                     # few pieces → endgame
        return "endgame"

# Evaluation Function
def static_exchange_eval(board: chess.Board, move: chess.Move) -> int:
    """Static Exchange Evaluation (SEE) for a capture move.
    Returns net material balance (positive = good for side to move).
    """
    if not board.is_capture(move):
        return 0

    target_square = move.to_square
    captured_piece = board.piece_at(target_square)
    if not captured_piece:
        return 0

    # start with value of captured piece
    gain = [PE_VAL[captured_piece.piece_type]]

    # attackers for both sides
    attackers = [[], []]  # [WHITE, BLACK]
    for color in [chess.WHITE, chess.BLACK]:
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            for sq in board.pieces(pt, color):
                if target_square in board.attacks(sq):
                    attackers[color].append((PE_VAL[pt], sq))

    # sort attackers by least valuable first
    attackers[chess.WHITE].sort()
    attackers[chess.BLACK].sort()

    side = board.turn
    depth = 0

    while attackers[side]:
        value, sq = attackers[side].pop(0)

        # next capture swings balance
        gain.append(value - gain[-1])

        side = not side
        depth += 1
        if depth > 20:  # safety cutoff
            break

    # minimax backward resolution
    for i in range(len(gain) - 2, -1, -1):
        gain[i] = max(-gain[i + 1], gain[i])

    return gain[0]
    
def capture_gain(board: chess.Board) -> int:
    """Reward captures using SEE (Static Exchange Evaluation)."""
    score = 0
    for move in board.legal_moves:
        if board.is_capture(move):
            see_value = static_exchange_eval(board, move)
            score += see_value // 2  # weight it (not full value to avoid over-tactics)
    return score


def hanging_penalty(board: chess.Board) -> int:
    """Harsher penalty for undefended pieces under attack."""
    penalty = 0
    for square, piece in board.piece_map().items():
        attackers = board.attackers(not piece.color, square)
        defenders = board.attackers(piece.color, square)
        if attackers and not defenders:
            penalty -= PE_VAL[piece.piece_type]  # full value penalty
    return penalty


def threat_penalty(board: chess.Board) -> int:
    """Smaller penalty if piece is attacked but has defenders."""
    penalty = 0
    for square, piece in board.piece_map().items():
        attackers = board.attackers(not piece.color, square)
        defenders = board.attackers(piece.color, square)
        if attackers:
            if defenders:
                penalty -= PE_VAL[piece.piece_type] // 4  # light penalty if defended
            else:
                penalty -= PE_VAL[piece.piece_type] // 2  # harsher if undefended
    return penalty


def evaluate_board(board: chess.Board) -> int:
    if board.is_checkmate():
        return -99999 if board.turn else 99999
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        return 0

    phase = get_phase(board)
    pst = {
        "opening": pe_pos_opening,
        "middlegame": pe_pos_middlegame,
        "endgame": pe_pos_endgame,
    }[phase]

    score = 0
    for piece_type in PE_VAL:
        for square in board.pieces(piece_type, chess.WHITE):
            score += PE_VAL[piece_type]
            if piece_type in pst:
                score += pst[piece_type][square]
        for square in board.pieces(piece_type, chess.BLACK):
            score -= PE_VAL[piece_type]
            if piece_type in pst:
                score -= pst[piece_type][chess.square_mirror(square)]

    # Tactical evaluations
    score += hanging_penalty(board)
    score += threat_penalty(board)
    score += capture_gain(board)  # SEE

    # Positional evaluations
    score += king_safety(board)
    score += pawn_structure(board, phase)
    score += mobility(board)
    score += center_control(board)

    return score
# tactics 

def tactics(board: chess.Board, alpha: float, beta: float) -> int:
    stand_pat = evaluate_board(board)

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        # --- skip suicidal checks ---
        if board.gives_check(move):
            attackers = board.attackers(board.turn, move.to_square)
            defenders = board.attackers(not board.turn, move.to_square)
            if defenders and not attackers:
                continue

        # --- quiescence: only search captures and checks ---
        if board.is_capture(move) or board.gives_check(move):
            board.push(move)
            score = -tactics(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

    return alpha
# Alpha-Beta Search

transposition = {}

def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float,
               maximizing: bool, start_time: float, time_limit: float) -> int:
    # --- stop if out of time ---
    if (time.time() - start_time) >= time_limit:
        return evaluate_board(board)

    # --- base case: depth reached or game over ---
    if depth == 0 or board.is_game_over():
        return tactics(board, alpha, beta)   # ✅ quiescence search

    if maximizing:
        value = -math.inf
        for move in board.legal_moves:
            board.push(move)
            value = max(value, alpha_beta(board, depth - 1, alpha, beta,
                                          False, start_time, time_limit))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:  # cutoff
                break
        return value
    else:
        value = math.inf
        for move in board.legal_moves:
            board.push(move)
            value = min(value, alpha_beta(board, depth - 1, alpha, beta,
                                          True, start_time, time_limit))
            board.pop()
            beta = min(beta, value)
            if alpha >= beta:  # cutoff
                break
        return value
# Iterative Deepening

def find_best_move(board: chess.Board, max_depth: int = 4, time_limit: float = 5.0) -> chess.Move:
    global transposition
    transposition = {}

    start_time = time.time()
    best_move = random.choice(list(board.legal_moves))  # fallback move

    for depth in range(1, max_depth + 1):
        if time.time() - start_time >= time_limit:
            break  # stop if out of time

        move, score = None, -math.inf
        for m in board.legal_moves:
            board.push(m)
            eval = alpha_beta(
                board,
                depth - 1,
                -math.inf,
                math.inf,
                False,
                start_time,
                time_limit
            )
            board.pop()

            if eval > score:
                score = eval
                move = m

        if move:
            best_move = move

    return best_move

# Time Management

def choose_think_time(my_time_sec: float) -> float:
    if my_time_sec >= 600:   # 10+ minutes
        return 5.0
    elif my_time_sec >= 300: # 5+ minutes
        return 3.0
    elif my_time_sec >= 60:  # 1+ minute
        return 1.0
    elif my_time_sec >= 30:  # 30+ sec
        return 0.5
    else:                    # panic mode
        return 0.2
# Play One Game

def play_game(game_id: str):
    global transposition
    transposition = {}
    board = chess.Board()
    my_id = client.account.get()["id"]
    my_color = None

    print(f"Started game {game_id}")

    for event in client.bots.stream_game_state(game_id):
        if event["type"] == "gameFull":
            my_color = chess.WHITE if event["white"]["id"] == my_id else chess.BLACK
            print("Color detected:", "White" if my_color == chess.WHITE else "Black")

            # load already played moves
            moves = event.get("state", {}).get("moves", "").split()
            board = chess.Board()
            for m in moves:
                board.push_uci(m)

            # if it's our move at the start
            if board.turn == my_color:
                book_move = get_opening_move(board)
                if book_move:
                    client.bots.make_move(game_id, book_move.uci())
                    print(f"My book move: {book_move.uci()}")
                    continue

                my_time_ms = event["state"].get("wtime" if my_color == chess.WHITE else "btime")
                my_time_sec = my_time_ms / 1000.0 if isinstance(my_time_ms, int) else 60.0
                think_time = choose_think_time(my_time_sec)
                move = find_best_move(board, max_depth=MAX_DEPTH, time_limit=think_time)
                client.bots.make_move(game_id, move.uci())
                print(f"My first move ({think_time:.1f}s): {move.uci()}")

        elif event["type"] == "gameState":
            moves = event["moves"].split()
            board = chess.Board()
            for m in moves:
                board.push_uci(m)

            if board.turn == my_color:
                book_move = get_opening_move(board)
                if book_move:
                    client.bots.make_move(game_id, book_move.uci())
                    print(f"My book move: {book_move.uci()}")
                    continue

                my_time_ms = event.get("wtime" if my_color == chess.WHITE else "btime")
                my_time_sec = my_time_ms / 1000.0 if isinstance(my_time_ms, int) else 60.0
                think_time = choose_think_time(my_time_sec)
                move = find_best_move(board, max_depth=MAX_DEPTH, time_limit=think_time)
                client.bots.make_move(game_id, move.uci())
                print(f"My move ({think_time:.1f}s): {move.uci()}")
# Run Bot Forever

def run_bot():
    print("Bot running...")
    for event in client.bots.stream_incoming_events():
        if event["type"] == "challenge":
            challenge_id = event["challenge"]["id"]
            client.bots.accept_challenge(challenge_id)
            print("Challenge accepted:", challenge_id)

        elif event["type"] == "gameStart":
            game_id = event["game"]["id"]
            print("Game started:", game_id)
            threading.Thread(target=play_game, args=(game_id,), daemon=True).start()

if __name__ == "__main__":
    run_bot()