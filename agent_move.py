#!/usr/bin/env python3
"""Agent-friendly chess move interface.

Designed to be called by Mistral Large 4 (or any agent) to:
  1. Initialize a new game
  2. Apply a move (human or AI)
  3. Display the board state for the agent to reason about

Usage:
    uv run python agent_move.py new [--ai-white] [--ai-black]
    uv run python agent_move.py move "<move>"     # e.g. "e4", "Nf3", "e2e4"
    uv run python agent_move.py show               # display board + status
    uv run python agent_move.py legal              # list legal moves
    uv run python agent_move.py ai-view            # full position dump for AI reasoning
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

from chess_game import (
    STATE_FILE,
    GameState,
    apply_move,
    board_to_text_for_ai,
    display_game,
    game_status,
    get_legal_moves_san,
    get_legal_moves_uci,
    is_ai_turn,
    is_game_over,
    load_state,
    new_game,
    whose_turn,
)


def cmd_new(args: argparse.Namespace) -> None:
    if args.ai_white:
        white, black = "ai", "human"
    else:
        white, black = "human", "ai"
    state = new_game(white_player=white, black_player=black)
    print(f"New game: White={white}, Black={black}")
    print()
    print(display_game(state))


def cmd_move(args: argparse.Namespace) -> None:
    state = load_state()
    if state is None:
        print("ERROR: No game in progress. Run 'new' first.")
        sys.exit(1)

    move_str = args.move
    turn_before = whose_turn(state)
    was_ai = is_ai_turn(state)

    success, msg, _ = apply_move(state, move_str)

    if not success:
        print(f"ILLEGAL: {msg}")
        sys.exit(1)

    label = "AI" if was_ai else "Human"
    print(f"[{label} ({turn_before})] {msg}")
    print()

    # Reload state for display
    state = load_state()
    if state is None:
        print("ERROR: State lost after move.")
        sys.exit(1)

    print(display_game(state))

    if is_game_over(state):
        board = state.to_board()
        print()
        print(f"*** GAME OVER: {game_status(board)} ***")


def cmd_show(args: argparse.Namespace) -> None:
    state = load_state()
    if state is None:
        print("No game in progress.")
        sys.exit(1)
    print(display_game(state))


def cmd_legal(args: argparse.Namespace) -> None:
    state = load_state()
    if state is None:
        print("No game in progress.")
        sys.exit(1)
    board = state.to_board()
    san = get_legal_moves_san(board)
    uci = get_legal_moves_uci(board)

    turn = whose_turn(state)
    is_ai = is_ai_turn(state)
    player = "AI" if is_ai else "Human"
    print(f"Turn: {turn} ({player})")
    print(f"Legal moves ({len(san)}):")
    print(f"  SAN: {', '.join(san)}")
    print(f"  UCI: {', '.join(uci)}")


def cmd_ai_view(args: argparse.Namespace) -> None:
    state = load_state()
    if state is None:
        print("No game in progress.")
        sys.exit(1)
    board = state.to_board()
    print(board_to_text_for_ai(board))


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent chess interface")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="Start a new game")
    p_new.add_argument("--ai-white", action="store_true")
    p_new.add_argument("--ai-black", action="store_true", default=True)

    p_move = sub.add_parser("move", help="Apply a move")
    p_move.add_argument("move", help="Move in SAN or UCI notation")

    sub.add_parser("show", help="Display current board")
    sub.add_parser("legal", help="List legal moves")
    sub.add_parser("ai-view", help="Full position dump for AI")

    args = parser.parse_args()

    if args.command == "new":
        cmd_new(args)
    elif args.command == "move":
        cmd_move(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "legal":
        cmd_legal(args)
    elif args.command == "ai-view":
        cmd_ai_view(args)


if __name__ == "__main__":
    main()
