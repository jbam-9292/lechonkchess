#!/usr/bin/env python3
"""Interactive chess CLI — Human vs Mistral Large 4 (with computer vision).

Usage:
    uv run python play.py [--new] [--ai-white] [--vision] [--model MODEL]

Modes:
    --vision       Use Mistral vision model (pixtral-large) to 'see' the board image
    (default)      Interactive: AI prompts for moves on stdin (for agent-driven play)

By default: Human plays White, AI (Mistral Large 4) plays Black.

Move input formats accepted:
    - SAN:  Nf3, e4, O-O, exd5, e8=Q
    - UCI:  g1f3, e2e4, e1g1, e7e8q
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chess

from chess_game import (
    STATE_FILE,
    GameState,
    apply_move,
    board_to_text_for_ai,
    display_game,
    format_history,
    game_status,
    get_legal_moves_san,
    is_ai_turn,
    is_game_over,
    load_state,
    new_game,
    render_board,
    whose_turn,
)


def get_vision_ai_move(state: GameState, model: str, verbose: bool = True) -> str:
    """Get a move from the Mistral vision AI."""
    from ai_player import VisionChessAI

    board = state.to_board()
    side = "White" if board.turn == chess.WHITE else "Black"

    print()
    print("=" * 50)
    print(f"  MISTRAL LARGE 4 (VISION) — {side}'s turn")
    print("=" * 50)
    print()

    ai = VisionChessAI(model=model)
    result = ai.get_move(board, move_history=state.history, verbose=verbose)

    if result.success and result.move_san:
        print(f"  AI plays: {result.move_san}")
        if verbose:
            print(f"  (model: {result.model_used}, attempts: {result.attempts})")
        print()
        return result.move_san
    else:
        # Fallback: pick a reasonable move
        print(f"  Vision AI failed: {result.error}")
        print(f"  Falling back to interactive mode...")
        print()
        return get_ai_move_interactive(state)


def get_ai_move_interactive(state: GameState) -> str:
    """Prompt for the AI's move interactively (fallback mode)."""
    board = state.to_board()

    print()
    print("=" * 50)
    print("  MISTRAL LARGE 4 — YOUR TURN")
    print("=" * 50)
    print()
    print(board_to_text_for_ai(board))
    print()

    legal = get_legal_moves_san(board)
    print(f"  You have {len(legal)} legal moves.")
    print(f"  Enter your move (SAN or UCI), e.g. 'Nf3' or 'g1f3':")
    print()

    while True:
        try:
            move_str = input("  AI> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not move_str:
            continue

        success, msg, _ = apply_move(state, move_str)
        if success:
            print(f"  AI plays: {msg}")
            print()
            return move_str
        else:
            print(f"  {msg}")
            print()


def get_human_move(state: GameState) -> str:
    """Prompt the human for a move."""
    board = state.to_board()
    legal = get_legal_moves_san(board)

    print(f"  You have {len(legal)} legal moves.")
    if len(legal) <= 20:
        print(f"  Legal: {', '.join(legal)}")
    print()

    while True:
        try:
            move_str = input("  Your move (SAN or UCI)> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Game saved. Run again to continue.")
            sys.exit(0)

        if not move_str:
            continue

        if move_str.lower() in ("quit", "exit", "q"):
            print("Game saved. Run again to continue.")
            sys.exit(0)

        if move_str.lower() in ("help", "h", "?"):
            print()
            print("  Move formats:")
            print("    SAN:  e4, Nf3, O-O (castle), exd5 (capture), e8=Q (promote)")
            print("    UCI:  e2e4, g1f3, e1g1, e7e8q")
            print("  Commands: quit, help, board, moves, history")
            print()
            continue

        if move_str.lower() == "board":
            print()
            print(render_board(board))
            print()
            continue

        if move_str.lower() == "moves":
            print()
            print(f"  Legal moves ({len(legal)}): {', '.join(legal)}")
            print()
            continue

        if move_str.lower() == "history":
            print()
            print(format_history(state.history) if state.history else "  (no moves yet)")
            print()
            continue

        success, msg, _ = apply_move(state, move_str)
        if success:
            print(f"  -> {msg}")
            print()
            return move_str
        else:
            print(f"  {msg}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play chess against Mistral Large 4")
    parser.add_argument("--new", action="store_true", help="Start a new game")
    parser.add_argument("--ai-white", action="store_true", help="AI plays White")
    parser.add_argument("--ai-black", action="store_true", help="AI plays Black (default)")
    parser.add_argument("--ai-vs-ai", action="store_true", help="AI vs AI (demo)")
    parser.add_argument("--vision", action="store_true",
                        help="Use Mistral vision model to see the board image")
    parser.add_argument("--model", default="mistral-large-4",
                        help="Mistral model to use (default: mistral-large-4)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose AI output")
    args = parser.parse_args()

    # Check API key if vision mode
    if args.vision and not os.environ.get("MISTRAL_API_KEY"):
        print("ERROR: MISTRAL_API_KEY environment variable not set.")
        print("Get a key at https://console.mistral.ai/ and run:")
        print("  export MISTRAL_API_KEY='your-key-here'")
        sys.exit(1)

    # Determine players
    if args.ai_vs_ai:
        white_player, black_player = "ai", "ai"
    elif args.ai_white:
        white_player, black_player = "ai", "human"
    else:
        white_player, black_player = "human", "ai"

    # Load or create game
    if args.new or not STATE_FILE.exists():
        state = new_game(white_player=white_player, black_player=black_player)
        print("New game started!")
        print(f"  White: {white_player}")
        print(f"  Black: {black_player}")
        if args.vision:
            print(f"  AI mode: Vision ({args.model})")
        print()
    else:
        state = load_state()
        if state is None:
            state = new_game(white_player=white_player, black_player=black_player)
        print("Game loaded.")
        print()

    print(display_game(state))
    print()

    # Game loop
    while True:
        if is_game_over(state):
            board = state.to_board()
            print("=" * 50)
            print(f"  GAME OVER: {game_status(board)}")
            print("=" * 50)
            if STATE_FILE.exists():
                STATE_FILE.unlink()
            break

        turn = whose_turn(state)
        is_ai = is_ai_turn(state)

        if is_ai:
            if args.vision:
                move_str = get_vision_ai_move(state, args.model, verbose=not args.quiet)
            else:
                move_str = get_ai_move_interactive(state)
            # Apply the move
            success, msg, _ = apply_move(state, move_str)
            if not success:
                print(f"  ERROR: AI move failed: {msg}")
                # Try interactive fallback
                move_str = get_ai_move_interactive(state)
                success, msg, _ = apply_move(state, move_str)
                if not success:
                    print(f"  FATAL: Could not apply AI move. Exiting.")
                    sys.exit(1)
        else:
            print(f"--- {turn.capitalize()}'s turn (Human) ---")
            get_human_move(state)

        print(display_game(state))
        print()


if __name__ == "__main__":
    main()
