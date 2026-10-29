#!/usr/bin/env python3
"""LeChonk Chess — Mistral Large 4 plays chess with computer vision.

The AI renders the board as a PNG image and 'looks at it' using Mistral's
vision model (mistral-large-4). No text/FEN is used for move selection.

Quick start:
    export MISTRAL_API_KEY='your-key'
    uv run python gui.py                      # Desktop GUI (recommended)
    uv run python vision_game.py --new        # CLI vision: Human (White) vs AI (Black)
    uv run python vision_game.py --new --ai-white   # AI (White) vs Human (Black)
    uv run python vision_game.py --new --ai-vs-ai   # AI vs AI demo
    uv run python text_game.py --new          # CLI text-only: Human vs AI, no images
    uv run python text_game.py --new --ai-vs-ai     # Text-only AI vs AI demo

Other commands:
    uv run python test_chess.py               # Run test suite

Architecture:
    gui.py             — Desktop GUI (tkinter + PIL): visual board, click-to-move,
                         auto-screenshots for the LLM, AI move button
    chess_game.py      — Game state, move validation, board display (python-chess)
    board_renderer.py  — Renders chess positions as PNG images (Pillow + Symbola font)
    ai_player.py       — VisionChessAI: sends board image to Mistral Large 4;
                         TextChessAI: sends a text-only board (no image), parses move
    vision_game.py     — CLI game loop: Human vs Vision AI
    text_game.py       — CLI game loop: Human vs Text AI (no vision)
"""


def main():
    print(__doc__)


if __name__ == "__main__":
    main()
