#!/usr/bin/env python3
"""Vision-based chess game: Human vs Mistral Large 4 (computer vision).

The AI renders the board as an image and 'looks at it' using Mistral's
vision model (pixtral-large). No text/FEN is used for move selection —
the AI sees only the visual board.

Usage:
    export MISTRAL_API_KEY='your-key'
    uv run python vision_game.py [--new] [--ai-white] [--model MODEL]

Requires: MISTRAL_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import os
import sys

import chess

from ai_player import VisionChessAI
from board_renderer import board_to_png, save_board_image
from chess_game import (
    STATE_FILE,
    GameState,
    apply_move,
    display_game,
    format_history,
    game_status,
    get_legal_moves_san,
    is_game_over,
    load_state,
    new_game,
    render_board,
    whose_turn,
)


def show_ai_view(board: chess.Board, state: GameState) -> None:
    """Show what the AI 'sees' — the rendered image info."""
    side = "White" if board.turn == chess.WHITE else "Black"
    print(f"  [Vision] Rendering board for {side}...")
    last_move = board.peek() if board.move_stack else None
    png_bytes = board_to_png(board, square_size=80, last_move=last_move)
    img_path = save_board_image(board, "current_position.png", last_move=last_move)
    print(f"  [Vision] Board image saved to: {img_path}")
    print(f"  [Vision] Image size: {len(png_bytes)} bytes")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision chess: Human vs Mistral Large 4")
    parser.add_argument("--new", action="store_true", help="Start a new game")
    parser.add_argument("--ai-white", action="store_true", help="AI plays White")
    parser.add_argument("--ai-black", action="store_true", help="AI plays Black (default)")
    parser.add_argument("--model", default="mistral-large-4",
                        help="Mistral vision model (default: mistral-large-4)")
    parser.add_argument("--ai-vs-ai", action="store_true", help="AI vs AI demo")
    parser.add_argument("--max-moves", type=int, default=200, help="Max half-moves in AI vs AI mode")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: MISTRAL_API_KEY environment variable not set.")
        print()
        print("Get a key at https://console.mistral.ai/ then run:")
        print("  export MISTRAL_API_KEY='your-key-here'")
        print("  uv run python vision_game.py --new")
        sys.exit(1)

    # Determine players
    if args.ai_vs_ai:
        white_player, black_player = "ai", "ai"
    elif args.ai_white:
        white_player, black_player = "ai", "human"
    else:
        white_player, black_player = "human", "ai"

    print("=" * 60)
    print("  VISION CHESS — Mistral Large 4 plays by SEEING the board")
    print("=" * 60)
    print()
    print(f"  Model: {args.model}")
    print(f"  White: {white_player}")
    print(f"  Black: {black_player}")
    print()

    # Create AI player
    ai = VisionChessAI(api_key=api_key, model=args.model)

    # Load or create game
    if args.new or not STATE_FILE.exists():
        state = new_game(white_player=white_player, black_player=black_player)
        print("New game started!")
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
    move_count = 0
    while True:
        if is_game_over(state):
            board = state.to_board()
            print("=" * 60)
            print(f"  GAME OVER: {game_status(board)}")
            print("=" * 60)
            print()
            print("Final position:")
            print(render_board(board))
            print()
            print("Move history:")
            print(format_history(state.history))
            if STATE_FILE.exists():
                STATE_FILE.unlink()
            break

        if args.ai_vs_ai and move_count >= args.max_moves:
            print("Max moves reached.")
            break

        board = state.to_board()
        turn = whose_turn(state)
        is_human = (turn == "white" and white_player == "human") or \
                   (turn == "black" and black_player == "human")

        if is_human:
            # Human turn
            print(f"--- {turn.capitalize()}'s turn (YOU) ---")
            legal = get_legal_moves_san(board)
            print(f"  Legal moves ({len(legal)}): {', '.join(legal)}")
            print()

            while True:
                try:
                    move_str = input("  Your move> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGame saved.")
                    sys.exit(0)

                if not move_str:
                    continue
                if move_str.lower() in ("quit", "q", "exit"):
                    print("Game saved.")
                    sys.exit(0)
                if move_str.lower() == "board":
                    print()
                    print(render_board(board))
                    print()
                    continue

                success, msg, _ = apply_move(state, move_str)
                if success:
                    print(f"  -> {msg}")
                    break
                else:
                    print(f"  {msg}")
            print()

        else:
            # AI turn (vision)
            print(f"--- {turn.capitalize()}'s turn (MISTRAL LARGE 4) ---")
            show_ai_view(board, state)

            result = ai.get_move(
                board,
                move_history=state.history,
                ai_memory=state.ai_memory,
                verbose=True,
            )

            if result.success and result.move_san:
                print(f"  AI sees the board and plays: {result.move_san}")
                success, msg, _ = apply_move(state, result.move_san)
                if success:
                    # Persist the AI's reasoning so it can plan across moves
                    state.add_ai_memory(result.move_san, result.reasoning, turn)
                else:
                    print(f"  ERROR applying move: {msg}")
                    # Try to recover
                    legal = get_legal_moves_san(board)
                    if legal:
                        fallback = legal[len(legal) // 2]  # pick middle move
                        print(f"  Fallback move: {fallback}")
                        success, msg, _ = apply_move(state, fallback)
                print()
            else:
                print(f"  AI failed: {result.error}")
                print(f"  Legal moves: {', '.join(get_legal_moves_san(board))}")
                # Pick a fallback move
                legal = get_legal_moves_san(board)
                if legal:
                    fallback = legal[len(legal) // 2]
                    print(f"  Fallback move: {fallback}")
                    apply_move(state, fallback)
                print()

            move_count += 1

        # Reload state and display
        state = load_state()
        if state is None:
            print("ERROR: Game state lost!")
            sys.exit(1)

        print(display_game(state))
        print()


if __name__ == "__main__":
    main()
