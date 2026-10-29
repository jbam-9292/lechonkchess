"""Chess game engine wrapper around python-chess.

Provides board display, move parsing/validation, game state persistence,
and game-over detection. Used by both the interactive CLI and the
agent-driven game loop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import chess


def load_env() -> None:
    """Load environment variables from .env file if present."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


load_env()

# Unicode chess pieces for display
PIECE_DISPLAY: dict[chess.PieceType, dict[chess.Color, str]] = {
    chess.PAWN: {chess.WHITE: "♙", chess.BLACK: "♟"},
    chess.KNIGHT: {chess.WHITE: "♘", chess.BLACK: "♞"},
    chess.BISHOP: {chess.WHITE: "♗", chess.BLACK: "♝"},
    chess.ROOK: {chess.WHITE: "♖", chess.BLACK: "♜"},
    chess.QUEEN: {chess.WHITE: "♕", chess.BLACK: "♛"},
    chess.KING: {chess.WHITE: "♔", chess.BLACK: "♚"},
}

EMPTY_SQUARE = "·"

STATE_FILE = Path(__file__).parent / "game_state.json"


@dataclass
class GameState:
    """Persisted game state."""

    board_fen: str = chess.STARTING_FEN
    history: list[str] = field(default_factory=list)  # SAN moves
    white_player: str = "human"  # "human" or "ai"
    black_player: str = "ai"
    move_count: int = 0

    def to_board(self) -> chess.Board:
        board = chess.Board(self.board_fen)
        return board

    def get_last_move(self) -> chess.Move | None:
        """Reconstruct the last move from the board position.

        Boards loaded from FEN have empty move stacks, so we can't use peek().
        Instead, we parse the last SAN from history and find the matching move.
        """
        if not self.history:
            return None
        board = self.to_board()
        # Undo the last move to get the previous position, then find the move
        # Actually simpler: just parse the last SAN from the current board's
        # perspective — but SAN needs the position BEFORE the move.
        # Reconstruct by replaying all moves except the last.
        temp = chess.Board()
        for san in self.history[:-1]:
            temp.push_san(san)
        # Now find which legal move produces the last SAN
        last_san = self.history[-1]
        for move in temp.legal_moves:
            if temp.san(move) == last_san:
                return move
        return None

    @classmethod
    def from_board(cls, board: chess.Board, history: list[str],
                   white_player: str = "human", black_player: str = "ai") -> "GameState":
        return cls(
            board_fen=board.fen(),
            history=history,
            white_player=white_player,
            black_player=black_player,
            move_count=len(history),
        )


def save_state(state: GameState, path: Path = STATE_FILE) -> None:
    path.write_text(json.dumps({
        "board_fen": state.board_fen,
        "history": state.history,
        "white_player": state.white_player,
        "black_player": state.black_player,
        "move_count": state.move_count,
    }, indent=2))


def load_state(path: Path = STATE_FILE) -> GameState | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return GameState(
        board_fen=data["board_fen"],
        history=data["history"],
        white_player=data.get("white_player", "human"),
        black_player=data.get("black_player", "ai"),
        move_count=data.get("move_count", len(data["history"])),
    )


def new_game(white_player: str = "human", black_player: str = "ai",
             path: Path = STATE_FILE) -> GameState:
    """Start a fresh game and persist it."""
    state = GameState(white_player=white_player, black_player=black_player)
    save_state(state, path)
    return state


def render_board(board: chess.Board, *, flip: bool = False) -> str:
    """Render the board as a pretty string with Unicode pieces.

    If flip is True, render from Black's perspective (rank 1 at top).
    """
    lines: list[str] = []
    files = "abcdefgh"
    ranks = range(8, 0, -1) if not flip else range(1, 9)

    # Column header
    header = "    " + "   ".join(files)
    lines.append(header)
    lines.append("  +" + "---+" * 8)

    for rank in ranks:
        row_parts = []
        for file_idx in range(8):
            square = chess.square(file_idx, rank - 1)
            piece = board.piece_at(square)
            if piece is None:
                row_parts.append(f" {EMPTY_SQUARE} ")
            else:
                symbol = PIECE_DISPLAY[piece.piece_type][piece.color]
                row_parts.append(f" {symbol} ")
        lines.append(f"{rank} |" + "|".join(row_parts) + f"| {rank}")
        lines.append("  +" + "---+" * 8)

    lines.append(header)
    return "\n".join(lines)


def render_board_with_coords(board: chess.Board, *, flip: bool = False,
                              last_move: chess.Move | None = None) -> str:
    """Render board with coordinate labels and optional last-move highlight."""
    lines: list[str] = []
    files = "abcdefgh"
    ranks = range(8, 0, -1) if not flip else range(1, 9)

    # Highlight squares
    highlight: set[chess.Square] = set()
    if last_move is not None:
        highlight.add(last_move.from_square)
        highlight.add(last_move.to_square)

    header = "    " + "   ".join(files)
    lines.append(header)
    lines.append("  +" + "---+" * 8)

    for rank in ranks:
        row_parts = []
        for file_idx in range(8):
            square = chess.square(file_idx, rank - 1)
            piece = board.piece_at(square)
            if piece is None:
                cell = f" {EMPTY_SQUARE} "
            else:
                symbol = PIECE_DISPLAY[piece.piece_type][piece.color]
                cell = f" {symbol} "
            # Mark highlighted squares
            if square in highlight:
                cell = f"[{cell.strip()}]"
                cell = f" {cell} "  # re-pad
                cell = cell[:5]  # ensure 5 chars: " [X] "
                # Actually let's keep it simple - just use the piece
            row_parts.append(cell)
        lines.append(f"{rank} |" + "|".join(row_parts) + f"| {rank}")
        lines.append("  +" + "---+" * 8)

    lines.append(header)
    return "\n".join(lines)


def parse_move(board: chess.Board, move_str: str) -> chess.Move | None:
    """Parse a move string (SAN like 'Nf3' or UCI like 'g1f3') into a Move.

    Returns the Move if legal, None otherwise.
    """
    move_str = move_str.strip()
    if not move_str:
        return None

    # Try UCI first (e.g., "e2e4", "e7e8q")
    try:
        move = chess.Move.from_uci(move_str.lower())
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    # Try SAN (e.g., "Nf3", "O-O", "exd5", "e8=Q")
    try:
        move = board.parse_san(move_str)
        return move
    except ValueError:
        pass

    return None


def apply_move(state: GameState, move_str: str, path: Path = STATE_FILE) -> tuple[bool, str, chess.Move | None]:
    """Apply a move to the game state.

    Returns (success, message, move).
    """
    board = state.to_board()

    move = parse_move(board, move_str)
    if move is None:
        # Provide helpful error with legal moves
        legal = sorted(board.san(m) for m in board.legal_moves)
        return False, f"Illegal move: '{move_str}'. Legal moves: {', '.join(legal)}", None

    san = board.san(move)
    board.push(move)

    # Update state
    state.board_fen = board.fen()
    state.history.append(san)
    state.move_count = len(state.history)

    save_state(state, path)

    # Check game over
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return True, f"{san} — CHECKMATE! {winner} wins!", move
    elif board.is_stalemate():
        return True, f"{san} — STALEMATE! Draw.", move
    elif board.is_insufficient_material():
        return True, f"{san} — Draw by insufficient material.", move
    elif board.is_seventyfive_moves():
        return True, f"{san} — Draw by 75-move rule.", move
    elif board.is_fivefold_repetition():
        return True, f"{san} — Draw by fivefold repetition.", move

    # Check / normal
    if board.is_check():
        return True, f"{san} — Check!", move

    return True, san, move


def get_legal_moves_san(board: chess.Board) -> list[str]:
    """Get all legal moves in SAN notation, sorted."""
    return sorted(board.san(m) for m in board.legal_moves)


def get_legal_moves_uci(board: chess.Board) -> list[str]:
    """Get all legal moves in UCI notation, sorted."""
    return sorted(m.uci() for m in board.legal_moves)


def game_status(board: chess.Board) -> str:
    """Return a human-readable game status string."""
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate — {winner} wins"
    if board.is_stalemate():
        return "Stalemate — draw"
    if board.is_insufficient_material():
        return "Draw — insufficient material"
    if board.is_seventyfive_moves():
        return "Draw — 75-move rule"
    if board.is_fivefold_repetition():
        return "Draw — fivefold repetition"
    if board.is_check():
        side = "White" if board.turn == chess.WHITE else "Black"
        return f"Check — {side} to move"
    side = "White" if board.turn == chess.WHITE else "Black"
    return f"{side} to move"


def whose_turn(state: GameState) -> str:
    """Return 'white' or 'black' based on whose turn it is."""
    board = state.to_board()
    return "white" if board.turn == chess.WHITE else "black"


def is_ai_turn(state: GameState) -> bool:
    """Check if it's the AI's turn."""
    turn = whose_turn(state)
    if turn == "white":
        return state.white_player == "ai"
    return state.black_player == "ai"


def is_game_over(state: GameState) -> bool:
    """Check if the game is over."""
    board = state.to_board()
    return board.is_game_over()


def format_history(history: list[str]) -> str:
    """Format move history in standard algebraic notation pairs."""
    lines = []
    for i in range(0, len(history), 2):
        move_num = i // 2 + 1
        white_move = history[i]
        black_move = history[i + 1] if i + 1 < len(history) else "..."
        lines.append(f"  {move_num}. {white_move}  {black_move}")
    return "\n".join(lines)


def board_to_text_for_ai(board: chess.Board) -> str:
    """Produce a compact text representation of the board for the AI to reason about.

    Returns a string with:
    - ASCII board layout
    - FEN
    - Side to move
    - Castling rights
    - Legal moves in both SAN and UCI
    - Move history
    """
    lines = []
    lines.append("=== CHESS POSITION ===")
    lines.append("")

    # ASCII board (rank 8 at top)
    lines.append("Board (rank 8 at top, file a at left):")
    lines.append("")
    for rank in range(7, -1, -1):
        row = f"  {rank + 1} "
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row += ". "
            else:
                # Use piece symbol: uppercase = white, lowercase = black
                symbol = piece.symbol()
                row += symbol.upper() + " " if piece.color == chess.WHITE else symbol + " "
        lines.append(row)
    lines.append("    a b c d e f g h")
    lines.append("")

    lines.append(f"FEN: {board.fen()}")
    lines.append(f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}")
    lines.append(f"Castling rights: {board.fen().split()[2]}")
    lines.append(f"En passant target: {board.fen().split()[3]}")
    lines.append("")

    # Legal moves
    san_moves = get_legal_moves_san(board)
    uci_moves = get_legal_moves_uci(board)
    lines.append(f"Legal moves ({len(san_moves)}):")
    lines.append(f"  SAN: {', '.join(san_moves)}")
    lines.append(f"  UCI: {', '.join(uci_moves)}")
    lines.append("")

    # Status
    lines.append(f"Status: {game_status(board)}")

    return "\n".join(lines)


def display_game(state: GameState, *, show_history: bool = True) -> str:
    """Produce a full display of the current game state."""
    board = state.to_board()
    lines = []

    # Determine perspective: flip if AI is black and it's a 2-player game
    # For now, always show from White's perspective
    lines.append(render_board(board))
    lines.append("")
    lines.append(f"Move {state.move_count // 2 + 1}: {game_status(board)}")

    if show_history and state.history:
        lines.append("")
        lines.append("Move history:")
        lines.append(format_history(state.history))

    return "\n".join(lines)
