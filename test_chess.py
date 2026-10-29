#!/usr/bin/env python3
"""Tests for the chess game modules.

Run with: uv run python test_chess.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import chess

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from chess_game import (
    GameState,
    apply_move,
    display_game,
    format_history,
    game_status,
    get_legal_moves_san,
    is_game_over,
    load_state,
    new_game,
    parse_move,
    render_board,
    save_state,
    whose_turn,
)

PASS = 0
FAIL = 0


def test(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


def test_starting_position() -> None:
    print("\n--- Starting Position ---")
    board = chess.Board()
    test("20 legal moves at start", len(list(board.legal_moves)) == 20)
    test("White to move", board.turn == chess.WHITE)
    test("Not game over", not board.is_game_over())
    test("FEN is starting", board.fen() == chess.STARTING_FEN)

    san_moves = get_legal_moves_san(board)
    test("e4 is legal", "e4" in san_moves)
    test("Nf3 is legal", "Nf3" in san_moves)
    test("Ke2 is illegal", "Ke2" not in san_moves)


def test_move_parsing() -> None:
    print("\n--- Move Parsing ---")
    board = chess.Board()

    # SAN parsing
    move = parse_move(board, "e4")
    test("Parse 'e4' SAN", move is not None and move.uci() == "e2e4")

    move = parse_move(board, "Nf3")
    test("Parse 'Nf3' SAN", move is not None and move.uci() == "g1f3")

    # UCI parsing
    move = parse_move(board, "e2e4")
    test("Parse 'e2e4' UCI", move is not None and move.uci() == "e2e4")

    move = parse_move(board, "g1f3")
    test("Parse 'g1f3' UCI", move is not None and move.uci() == "g1f3")

    # Illegal move
    move = parse_move(board, "Ke2")
    test("Reject illegal 'Ke2'", move is None)

    move = parse_move(board, "e5")
    test("Reject 'e5' when it's White's turn (no pawn on e4)", move is None)

    move = parse_move(board, "")
    test("Reject empty string", move is None)


def test_special_moves() -> None:
    print("\n--- Special Moves ---")

    # Castling
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    move = parse_move(board, "O-O")
    test("Kingside castling (O-O)", move is not None and move.uci() == "e1g1")

    move = parse_move(board, "O-O-O")
    test("Queenside castling (O-O-O)", move is not None and move.uci() == "e1c1")

    # En passant
    board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    move = parse_move(board, "exf6")
    test("En passant capture", move is not None and move.uci() == "e5f6")

    # Promotion
    board = chess.Board("8/P7/8/8/8/8/k6K/8 w - - 0 1")
    move = parse_move(board, "a8=Q")
    test("Pawn promotion to Queen", move is not None and move.uci() == "a7a8q")

    move = parse_move(board, "a8=N")
    test("Pawn promotion to Knight", move is not None and move.uci() == "a7a8n")


def test_check_checkmate() -> None:
    print("\n--- Check & Checkmate ---")

    # Fool's mate
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")
    test("Not checkmate yet", not board.is_checkmate())
    board.push_san("Qh4#")
    test("Fool's mate is checkmate", board.is_checkmate())
    test("Black wins", board.turn == chess.WHITE)  # White is checkmated
    test("Game is over", board.is_game_over())
    test("Status says checkmate", "checkmate" in game_status(board).lower())

    # Scholar's mate
    board = chess.Board()
    for san in ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]:
        board.push_san(san)
    test("Scholar's mate is checkmate", board.is_checkmate())


def test_stalemate() -> None:
    print("\n--- Stalemate ---")
    # Classic stalemate position
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    test("Stalemate position detected", board.is_stalemate())
    test("Game over on stalemate", board.is_game_over())


def test_game_state_persistence() -> None:
    print("\n--- Game State Persistence ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "test_state.json"

        # New game
        state = new_game(path=state_file)
        test("New game created", state_file.exists())
        test("New game has starting FEN", state.board_fen == chess.STARTING_FEN)
        test("New game is White's turn", whose_turn(state) == "white")

        # Apply moves
        success, msg, _ = apply_move(state, "e4", path=state_file)
        test("Apply e4", success)
        test("History has 1 move", len(state.history) == 1)
        test("History is ['e4']", state.history == ["e4"])

        success, msg, _ = apply_move(state, "e5", path=state_file)
        test("Apply e5", success)
        test("History has 2 moves", len(state.history) == 2)

        # Reload state
        loaded = load_state(state_file)
        test("Loaded state matches", loaded is not None and loaded.board_fen == state.board_fen)
        test("Loaded history matches", loaded.history == ["e4", "e5"])

        # Illegal move (king can't jump to a1)
        success, msg, _ = apply_move(state, "Ka1", path=state_file)
        test("Illegal move rejected", not success)


def test_board_rendering() -> None:
    print("\n--- Board Rendering ---")
    board = chess.Board()
    rendered = render_board(board)
    test("Board renders with Unicode pieces", "♜" in rendered and "♖" in rendered)
    test("Board has coordinate labels", "a" in rendered and "8" in rendered)

    display = display_game(GameState())
    test("Display includes status", "White to move" in display)

    history_text = format_history(["e4", "e5", "Nf3"])
    test("History formatting", "1." in history_text and "e4" in history_text)


def test_board_image_rendering() -> None:
    print("\n--- Board Image Rendering ---")
    try:
        from board_renderer import board_to_png, render_board_image
        from PIL import Image
        import io

        board = chess.Board()
        png_bytes = board_to_png(board)
        test("PNG generated", len(png_bytes) > 1000)
        test("PNG magic bytes", png_bytes[:4] == b"\x89PNG")

        img = Image.open(io.BytesIO(png_bytes))
        test("Image dimensions reasonable", img.width > 500 and img.height > 500)

        # After moves
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        png_bytes = board_to_png(board, last_move=board.peek())
        test("PNG with moves generated", len(png_bytes) > 1000)

        print("  (Image rendering works)")
    except ImportError:
        print("  SKIP: Pillow not available")


def test_ai_move_parsing() -> None:
    print("\n--- AI Move Response Parsing ---")
    from ai_player import VisionChessAI

    # We can't instantiate without API key, but we can test the parsing logic
    # by creating a dummy object
    class DummyAI:
        _parse_move_from_response = VisionChessAI._parse_move_from_response

    dummy = DummyAI()
    board = chess.Board()

    # Test various response formats
    san, uci, _ = dummy._parse_move_from_response("e4", board)
    test("Parse 'e4'", san == "e4" and uci == "e2e4")

    san, uci, _ = dummy._parse_move_from_response("Nf3", board)
    test("Parse 'Nf3'", san == "Nf3" and uci == "g1f3")

    san, uci, _ = dummy._parse_move_from_response("The best move is e4", board)
    test("Parse 'The best move is e4'", san == "e4")

    san, uci, _ = dummy._parse_move_from_response("I play Nf3", board)
    test("Parse 'I play Nf3'", san == "Nf3")

    san, uci, _ = dummy._parse_move_from_response("e2e4", board)
    test("Parse UCI 'e2e4'", san == "e4" and uci == "e2e4")

    san, uci, _ = dummy._parse_move_from_response("g1f3", board)
    test("Parse UCI 'g1f3'", san == "Nf3" and uci == "g1f3")

    san, uci, _ = dummy._parse_move_from_response("xyz invalid", board)
    test("Reject invalid response", san is None)

    # Castling
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    san, uci, _ = dummy._parse_move_from_response("O-O", board)
    test("Parse castling O-O", san == "O-O" and uci == "e1g1")

    san, uci, _ = dummy._parse_move_from_response("0-0", board)
    test("Parse castling 0-0", san == "O-O" and uci == "e1g1")

    # Structured JSON responses (move + reasoning)
    board = chess.Board()
    san, uci, reasoning = dummy._parse_move_from_response(
        '{"move": "e4", "reasoning": "Controls the center."}', board
    )
    test("Parse JSON move", san == "e4" and uci == "e2e4")
    test("Parse JSON reasoning", reasoning == "Controls the center.")

    san, uci, reasoning = dummy._parse_move_from_response(
        '```json\n{"move": "Nf3", "reasoning": "Develops a piece."}\n```', board
    )
    test("Parse fenced JSON move", san == "Nf3" and uci == "g1f3")
    test("Parse fenced JSON reasoning", reasoning == "Develops a piece.")

    san, uci, reasoning = dummy._parse_move_from_response(
        '{"move": "e4"}', board
    )
    test("Parse JSON without reasoning", san == "e4" and reasoning is None)

    san, uci, reasoning = dummy._parse_move_from_response(
        '{"move": "Ke4", "reasoning": "Illegal move"}', board
    )
    test("Reject JSON with illegal move", san is None and uci is None)


def test_text_ai_input() -> None:
    print("\n--- Text AI Input Building (no vision) ---")
    from ai_player import MODES, TextChessAI, VisionChessAI

    # Subclass relationship: text mode shares parsing/retry logic with vision
    test("TextChessAI subclasses VisionChessAI", issubclass(TextChessAI, VisionChessAI))
    test("TextChessAI defaults to text mode", TextChessAI.default_mode == "text")
    test("TextChessAI uses a text-only system prompt",
         TextChessAI.system_prompt is not VisionChessAI.system_prompt
         and "TEXT" in TextChessAI.system_prompt)
    test("MODES are text/hybrid/image", MODES == ("text", "hybrid", "image"))

    # Build the text-mode input without an API key via a dummy object
    class DummyTextAI:
        mode = "text"
        _build_input = VisionChessAI._build_input
        _render_board_image = VisionChessAI._render_board_image
        _encode_image = VisionChessAI._encode_image

    dummy = DummyTextAI()
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")

    test("Text mode renders no image", dummy._render_board_image(board, 80) is None)

    memory = [{
        "move_number": 1,
        "side": "white",
        "move_san": "e4",
        "reasoning": "Controls the center.",
        "fen": chess.STARTING_FEN,
    }]
    inputs = dummy._build_input(
        board, None, move_history=["e4", "e5"], ai_memory=memory
    )
    test("Single user message", len(inputs) == 1 and inputs[0].role == "user")
    content = inputs[0].content
    test("No image part in message",
         all(getattr(c, "type", None) != "input_image" for c in content))
    full_text = "\n".join(c.text for c in content if hasattr(c, "text"))
    test("Contains FEN", board.fen() in full_text)
    test("Contains text board with Unicode pieces", "♜" in full_text and "♙" in full_text)
    test("Contains side to move", "It is White's turn." in full_text)
    test("Contains legal moves", "Legal moves:" in full_text and "e4" not in full_text.split("Legal moves:")[1].split("\n")[0])
    test("Contains move history", "e4 e5" in full_text)
    test("Contains AI memory block", "MEMORY" in full_text and "Controls the center." in full_text)

    # Move parsing is inherited from the vision player
    class DummyParser:
        _parse_move_from_response = TextChessAI._parse_move_from_response

    parser = DummyParser()
    san, uci, reasoning = parser._parse_move_from_response(
        '{"move": "Nf3", "reasoning": "Develops."}', board
    )
    test("Text AI parses JSON move (inherited)", san == "Nf3" and uci == "g1f3")
    test("Text AI parses JSON reasoning (inherited)", reasoning == "Develops.")


def test_ai_input_modes() -> None:
    print("\n--- LLM Input Modes (text / image / hybrid) ---")
    from ai_player import MODE_SYSTEM_PROMPTS, MODES, VisionChessAI

    class DummyAI:
        _build_input = VisionChessAI._build_input
        _render_board_image = VisionChessAI._render_board_image
        _encode_image = VisionChessAI._encode_image

    board = chess.Board()

    # Invalid mode is rejected at construction
    try:
        VisionChessAI(api_key="dummy", mode="bogus")
        test("Invalid mode rejected", False)
    except ValueError:
        test("Invalid mode rejected", True)

    # Every mode has a system prompt
    test("All modes have prompts", all(MODE_SYSTEM_PROMPTS[m] for m in MODES))

    # Image-only mode: image part sent, no FEN/legal moves/board text
    d = DummyAI()
    d.mode = "image"
    test("Image mode renders PNG", d._render_board_image(board, 80) is not None)
    inputs = d._build_input(board, b"fakepng", move_history=["e4"], ai_memory=[])
    parts = inputs[0].content
    txt = "\n".join(c.text for c in parts if getattr(c, "type", None) == "input_text")
    test("Image mode sends an image part",
         any(getattr(c, "type", None) == "input_image" for c in parts))
    test("Image mode omits FEN/legal moves/text board",
         "FEN:" not in txt and "Legal moves:" not in txt and "♜" not in txt)
    test("Image mode states side to move", "It is White's turn." in txt)

    # Hybrid mode: image AND full text description
    d2 = DummyAI()
    d2.mode = "hybrid"
    inputs2 = d2._build_input(board, b"fakepng", move_history=["e4"], ai_memory=[])
    parts2 = inputs2[0].content
    txt2 = "\n".join(c.text for c in parts2 if getattr(c, "type", None) == "input_text")
    test("Hybrid sends an image part",
         any(getattr(c, "type", None) == "input_image" for c in parts2))
    test("Hybrid includes FEN, legal moves and text board",
         "FEN:" in txt2 and "Legal moves:" in txt2 and "♜" in txt2)

    # Text mode via constructor default of TextChessAI (no image rendered)
    from ai_player import TextChessAI
    ai = TextChessAI(api_key="dummy")
    test("TextChessAI instance mode is text", ai.mode == "text")
    test("TextChessAI instance renders no image", ai._render_board_image(board, 80) is None)
    test("TextChessAI instance uses text prompt", ai.system_prompt is MODE_SYSTEM_PROMPTS["text"])
    ai_h = VisionChessAI(api_key="dummy", mode="image")
    test("VisionChessAI image instance uses image prompt", ai_h.system_prompt is MODE_SYSTEM_PROMPTS["image"])


def main() -> None:
    print("=" * 50)
    print("  Chess Engine Tests")
    print("=" * 50)

    test_starting_position()
    test_move_parsing()
    test_special_moves()
    test_check_checkmate()
    test_stalemate()
    test_game_state_persistence()
    test_board_rendering()
    test_board_image_rendering()
    test_ai_move_parsing()
    test_text_ai_input()
    test_ai_input_modes()

    print()
    print("=" * 50)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 50)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
