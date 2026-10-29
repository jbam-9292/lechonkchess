#!/usr/bin/env python3
"""Desktop GUI for LeChonk Chess — Human vs Mistral Large 4 (vision).

The board is rendered as a PNG image (same renderer the LLM sees),
displayed in the GUI via PIL/ImageTk. No font dependency for pieces.

Features:
- Visual chess board (PNG image, identical to LLM screenshots)
- Click-to-move: click a piece, then click a destination square
- SAN/UCI text input as fallback
- Auto-generates PNG screenshot for the LLM on every AI turn
- Shows the screenshot preview in the GUI
- Move history panel

Usage:
    export MISTRAL_API_KEY='your-key'
    uv run python gui.py [--ai-white]
"""

from __future__ import annotations

import argparse
import io
import os
import threading
from pathlib import Path

import chess
from PIL import Image, ImageTk

from chess_game import (
    STATE_FILE,
    apply_move,
    game_status,
    is_ai_turn,
    is_game_over,
    load_state,
    new_game,
    whose_turn,
)
from board_renderer import board_to_png, save_board_image

# ── Constants ──────────────────────────────────────────────────────

# Board display size in the GUI
DISPLAY_SQUARE = 70
DISPLAY_MARGIN = 30  # must match margin in board_renderer.render_board_image
DISPLAY_TITLE_HEIGHT = 40
DISPLAY_BOARD_PX = DISPLAY_SQUARE * 8
DISPLAY_TOTAL = DISPLAY_BOARD_PX + DISPLAY_MARGIN * 2 + DISPLAY_TITLE_HEIGHT

# Screenshot size for the LLM (higher res = better vision)
SCREENSHOT_SQUARE = 80

# ── Mistral brand palette ──────────────────────────────────────────

MISTRAL_ORANGE = "#fa520f"     # Mistral Orange — primary accent
ORANGE_DEEP = "#cc3a05"        # Orange Deep — pressed/active states
SUNSHINE_GOLD = "#ffa110"      # Sunshine Gold — secondary accent
BRIGHT_YELLOW = "#ffd900"      # Bright Yellow — highlights
WARM_CREAM = "#fff8e0"         # Warm Cream — main text
MISTRAL_BLACK = "#1f1f1f"      # Mistral Black — window background

# Semantic aliases
BG_ROOT = MISTRAL_BLACK        # window background
HEADER_BG = "#141414"          # header bar (near-black)
BG_PANEL = "#2a2a2a"           # side panel card
BG_ENTRY = "#141414"           # text inputs
FG_TEXT = WARM_CREAM           # main text
FG_MUTED = "#b5ab93"           # secondary labels
BORDER_SUBTLE = "#3a3a3a"      # card borders / separators

# Highlight colors (canvas overlays drawn over the board image)
SELECTED_COLOR = BRIGHT_YELLOW       # selected square
LEGAL_DOT_COLOR = SUNSHINE_GOLD      # legal move dots
LAST_MOVE_COLOR = MISTRAL_ORANGE     # last move highlight

# Le Chonk portrait shown in the header (first existing file wins)
HEADER_IMAGE_SIZE = 72
ASSET_DIR = Path(__file__).parent
LECHONK_IMAGE_CANDIDATES = [
    ASSET_DIR / "assets" / "lechonk.png",
    ASSET_DIR / "assets" / "lechonk.jpg",
    ASSET_DIR / "assets" / "lechonk.jpeg",
    ASSET_DIR / "assets" / "lechonk.gif",
    ASSET_DIR / "assets" / "lechonk.webp",
    ASSET_DIR / "lechonk.png",
    ASSET_DIR / "lechonk.jpg",
    ASSET_DIR / "lechonk.jpeg",
    ASSET_DIR / "lechonk.gif",
    ASSET_DIR / "lechonk.webp",
]

SCREENSHOT_DIR = Path(__file__).parent / "board_images"
FILES = "abcdefgh"


# ── GUI Application ────────────────────────────────────────────────

class ChessGUI:
    def __init__(self, root, ai_white: bool = False):
        import tkinter as tk
        self.tk = tk
        self.root = root
        self.ai_white = ai_white

        # Game state
        if STATE_FILE.exists():
            self.state = load_state()
        if not hasattr(self, 'state') or self.state is None:
            self.state = new_game(
                white_player="ai" if ai_white else "human",
                black_player="human" if ai_white else "ai",
            )

        # Interaction state
        self.selected_square: chess.Square | None = None
        self.legal_targets: set[chess.Square] = set()
        self.ai_thinking = False
        self._board_photo = None  # Keep reference to prevent GC

        # Setup window
        self.root.title("LeChonk Chess — Mistral Large 4 (Vision)")
        self.root.configure(bg=BG_ROOT)
        self.root.resizable(False, False)

        self._build_ui()
        self._draw_board()
        self._update_status()

        # If AI is White, trigger AI move on start
        if is_ai_turn(self.state) and not is_game_over(self.state):
            self.root.after(500, self._trigger_ai_move)

    def _build_header(self) -> None:
        """Header bar with the Le Chonk portrait and title, plus brand stripe."""
        tk = self.tk
        header = tk.Frame(self.root, bg=HEADER_BG, padx=14, pady=10)
        header.pack(fill=tk.X)

        photo, path = self._load_lechonk_photo(HEADER_IMAGE_SIZE)
        if photo is not None:
            self._lechonk_photo = photo  # keep reference to prevent GC
            portrait = tk.Label(
                header, image=photo, bg=HEADER_BG,
                highlightthickness=2, highlightbackground=MISTRAL_ORANGE,
            )
            portrait.pack(side=tk.LEFT)
            self._set_window_icon(path)
        else:
            # Placeholder monogram until the Le Chonk picture is added to the repo
            placeholder = tk.Canvas(
                header, width=HEADER_IMAGE_SIZE, height=HEADER_IMAGE_SIZE,
                bg=MISTRAL_ORANGE, highlightthickness=2, highlightbackground=ORANGE_DEEP,
            )
            placeholder.create_text(
                HEADER_IMAGE_SIZE // 2, HEADER_IMAGE_SIZE // 2,
                text="LC", font=("Helvetica", 26, "bold"), fill=MISTRAL_BLACK,
            )
            placeholder.pack(side=tk.LEFT)

        titles = tk.Frame(header, bg=HEADER_BG)
        titles.pack(side=tk.LEFT, padx=(14, 0))
        tk.Label(
            titles, text="LeChonk Chess", font=("Helvetica", 22, "bold"),
            bg=HEADER_BG, fg=BRIGHT_YELLOW,
        ).pack(anchor=tk.W)
        tk.Label(
            titles, text="Human vs Mistral Large 4 (Vision)",
            font=("Helvetica", 11), bg=HEADER_BG, fg=FG_MUTED,
        ).pack(anchor=tk.W)

        # Mistral brand stripe under the header
        tk.Frame(self.root, bg=MISTRAL_ORANGE, height=3).pack(fill=tk.X)

    def _load_lechonk_photo(self, size: int):
        """Load the Le Chonk picture as a PhotoImage.

        Returns (photo, source_path), or (None, None) if no image is found.
        """
        for path in LECHONK_IMAGE_CANDIDATES:
            if not path.exists():
                continue
            try:
                img = Image.open(path).convert("RGBA")
                img.thumbnail((size, size))
                return ImageTk.PhotoImage(img), path
            except Exception:
                continue
        return None, None

    def _set_window_icon(self, path: Path) -> None:
        try:
            img = Image.open(path).convert("RGBA")
            img.thumbnail((48, 48))
            self._icon_photo = ImageTk.PhotoImage(img)  # keep reference
            self.root.iconphoto(True, self._icon_photo)
        except Exception:
            pass

    def _add_separator(self, parent) -> None:
        self.tk.Frame(parent, bg=BORDER_SUBTLE, height=1).pack(fill=self.tk.X, pady=8)

    def _section_label(self, parent, text: str):
        return self.tk.Label(
            parent, text=text, font=("Helvetica", 10, "bold"),
            bg=BG_PANEL, fg=SUNSHINE_GOLD,
        )

    def _build_ui(self) -> None:
        tk = self.tk

        self._build_header()

        main = tk.Frame(self.root, bg=BG_ROOT, padx=14, pady=14)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: board canvas
        board_frame = tk.Frame(main, bg=BG_ROOT)
        board_frame.pack(side=tk.LEFT, padx=(0, 14))

        self.canvas = tk.Canvas(
            board_frame,
            width=DISPLAY_TOTAL,
            height=DISPLAY_TOTAL,
            bg=BG_ENTRY,
            highlightthickness=1,
            highlightbackground=BORDER_SUBTLE,
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_square_click)

        # Right: controls (card panel)
        cf = tk.Frame(
            main, bg=BG_PANEL, padx=14, pady=14,
            highlightthickness=1, highlightbackground=BORDER_SUBTLE,
        )
        cf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(
            cf, text="", font=("Helvetica", 14, "bold"), wraplength=300,
            justify=tk.LEFT, bg=BG_PANEL, fg=FG_TEXT,
        )
        self.status_label.pack(anchor=tk.W, pady=(0, 4))
        self._add_separator(cf)

        self._section_label(cf, "Enter move (SAN/UCI):").pack(anchor=tk.W, pady=(0, 2))
        input_frame = tk.Frame(cf, bg=BG_PANEL)
        input_frame.pack(anchor=tk.W, pady=(0, 10))
        self.move_entry = tk.Entry(
            input_frame, font=("Courier", 12), width=15,
            bg=BG_ENTRY, fg=FG_TEXT, insertbackground=BRIGHT_YELLOW,
            relief=tk.FLAT, highlightthickness=1,
            highlightbackground=BORDER_SUBTLE, highlightcolor=MISTRAL_ORANGE,
        )
        self.move_entry.pack(side=tk.LEFT)
        self.move_entry.bind("<Return>", lambda e: self._on_submit_move())
        tk.Button(
            input_frame, text="Play", command=self._on_submit_move,
            font=("Helvetica", 10, "bold"),
            bg=MISTRAL_ORANGE, fg=MISTRAL_BLACK,
            activebackground=ORANGE_DEEP, activeforeground=WARM_CREAM,
            relief=tk.FLAT, padx=10, pady=2, cursor="hand2",
        ).pack(side=tk.LEFT, padx=(6, 0))

        self.ai_button = tk.Button(
            cf, text="AI Move (Mistral Large 4)",
            command=self._trigger_ai_move,
            font=("Helvetica", 11, "bold"),
            bg=SUNSHINE_GOLD, fg=MISTRAL_BLACK,
            activebackground=MISTRAL_ORANGE, activeforeground=MISTRAL_BLACK,
            relief=tk.FLAT, padx=10, pady=5, cursor="hand2",
        )
        self.ai_button.pack(anchor=tk.W, pady=(0, 10))

        tk.Button(
            cf, text="New Game", command=self._new_game,
            font=("Helvetica", 10),
            bg=BORDER_SUBTLE, fg=FG_TEXT,
            activebackground="#4a4a4a", activeforeground=FG_TEXT,
            relief=tk.FLAT, padx=10, pady=3, cursor="hand2",
        ).pack(anchor=tk.W, pady=(0, 4))
        self._add_separator(cf)

        self._section_label(cf, "LLM Screenshot:").pack(anchor=tk.W)
        self.shot_label = tk.Label(
            cf, text="(not yet generated)", font=("Helvetica", 9),
            bg=BG_PANEL, fg=FG_MUTED,
        )
        self.shot_label.pack(anchor=tk.W)
        self.shot_canvas = tk.Canvas(
            cf, width=200, height=200, bg=BG_ENTRY,
            highlightthickness=1, highlightbackground=BORDER_SUBTLE,
        )
        self.shot_canvas.pack(anchor=tk.W, pady=(5, 10))

        self._section_label(cf, "Move History:").pack(anchor=tk.W)
        self.history_text = tk.Text(
            cf, width=35, height=8, font=("Courier", 10), state=tk.DISABLED,
            bg=BG_ENTRY, fg=FG_TEXT, insertbackground=BRIGHT_YELLOW,
            relief=tk.FLAT, highlightthickness=1, highlightbackground=BORDER_SUBTLE,
        )
        self.history_text.pack(anchor=tk.W)
        self._add_separator(cf)

        self.thinking_label = tk.Label(
            cf, text="", font=("Helvetica", 10, "italic"),
            bg=BG_PANEL, fg=SUNSHINE_GOLD,
        )
        self.thinking_label.pack(anchor=tk.W, pady=(0, 2))

        self._section_label(cf, "AI Thinking:").pack(anchor=tk.W)
        thinking_frame = tk.Frame(cf, bg=BG_PANEL)
        thinking_frame.pack(anchor=tk.W, fill=tk.BOTH, expand=True)
        self.thinking_text = tk.Text(
            thinking_frame, width=35, height=10,
            font=("Courier", 9), state=tk.DISABLED,
            wrap=tk.WORD, bg=BG_ENTRY, fg=FG_TEXT,
            relief=tk.FLAT, highlightthickness=1, highlightbackground=BORDER_SUBTLE,
        )
        thinking_scroll = tk.Scrollbar(
            thinking_frame, command=self.thinking_text.yview,
            bg=BG_PANEL, troughcolor=BG_ENTRY,
            activebackground=MISTRAL_ORANGE, highlightthickness=0,
        )
        self.thinking_text.configure(yscrollcommand=thinking_scroll.set)
        thinking_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.thinking_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _render_board_image(self) -> Image.Image:
        """Render the current board position as a PIL Image."""
        from board_renderer import render_board_image
        board = self.state.to_board()
        last_move = self.state.get_last_move()
        status = game_status(board)
        side = "White" if board.turn == chess.WHITE else "Black"
        title = f"{side} to move — {status}"
        return render_board_image(
            board,
            square_size=DISPLAY_SQUARE,
            last_move=last_move,
            title=title,
        )

    def _draw_board(self) -> None:
        """Draw the board as a PNG image with click overlays."""
        tk = self.tk
        self.canvas.delete("all")

        # Render board as image
        img = self._render_board_image()
        self._board_photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self._board_photo, anchor=tk.NW)

        # Draw overlays for selected square and legal moves
        board = self.state.to_board()

        def sq_to_xy(square: chess.Square) -> tuple[int, int, int, int]:
            file_idx = chess.square_file(square)
            rank = chess.square_rank(square)  # 0-7
            row = 7 - rank  # flip: rank 8 at top
            x0 = DISPLAY_MARGIN + file_idx * DISPLAY_SQUARE
            y0 = DISPLAY_MARGIN + DISPLAY_TITLE_HEIGHT + row * DISPLAY_SQUARE
            return x0, y0, x0 + DISPLAY_SQUARE, y0 + DISPLAY_SQUARE

        # Last move highlight
        last = self.state.get_last_move()
        if last is not None:
            for sq in [last.from_square, last.to_square]:
                x0, y0, x1, y1 = sq_to_xy(sq)
                self.canvas.create_rectangle(x0, y0, x1, y1, outline=LAST_MOVE_COLOR, width=3)

        # Selected square highlight
        if self.selected_square is not None:
            x0, y0, x1, y1 = sq_to_xy(self.selected_square)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=SELECTED_COLOR, width=4)

        # Legal move dots
        for sq in self.legal_targets:
            x0, y0, x1, y1 = sq_to_xy(sq)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            piece = board.piece_at(sq)
            if piece is not None:
                r = DISPLAY_SQUARE // 2 - 6
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=LEGAL_DOT_COLOR, width=4)
            else:
                r = DISPLAY_SQUARE // 7
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=LEGAL_DOT_COLOR, outline="")

    def _xy_to_square(self, x: int, y: int) -> chess.Square | None:
        """Convert canvas pixel coordinates to a chess square."""
        bx = x - DISPLAY_MARGIN
        by = y - DISPLAY_MARGIN - DISPLAY_TITLE_HEIGHT
        if bx < 0 or by < 0 or bx >= DISPLAY_BOARD_PX or by >= DISPLAY_BOARD_PX:
            return None
        col = bx // DISPLAY_SQUARE
        row = by // DISPLAY_SQUARE
        rank = 8 - row  # flip
        if not (0 <= col < 8 and 1 <= rank <= 8):
            return None
        return chess.square(col, rank - 1)

    def _on_square_click(self, event) -> None:
        if self.ai_thinking or is_game_over(self.state):
            return
        if is_ai_turn(self.state):
            return

        square = self._xy_to_square(event.x, event.y)
        if square is None:
            return

        board = self.state.to_board()

        if self.selected_square is None:
            piece = board.piece_at(square)
            if piece is not None and piece.color == board.turn:
                self.selected_square = square
                self.legal_targets = {
                    m.to_square for m in board.legal_moves
                    if m.from_square == square
                }
                self._draw_board()
        else:
            if square == self.selected_square:
                self.selected_square = None
                self.legal_targets = set()
                self._draw_board()
                return

            if square in self.legal_targets:
                move = None
                for m in board.legal_moves:
                    if m.from_square == self.selected_square and m.to_square == square:
                        if m.promotion:
                            move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                        else:
                            move = m
                        break

                if move and move in board.legal_moves:
                    san = board.san(move)
                    self._apply_human_move(san)
                    self.selected_square = None
                    self.legal_targets = set()
                    self._draw_board()
                    self._update_status()
                    if is_ai_turn(self.state) and not is_game_over(self.state):
                        self.root.after(300, self._trigger_ai_move)
                    return

            piece = board.piece_at(square)
            if piece is not None and piece.color == board.turn:
                self.selected_square = square
                self.legal_targets = {
                    m.to_square for m in board.legal_moves
                    if m.from_square == square
                }
            else:
                self.selected_square = None
                self.legal_targets = set()
            self._draw_board()

    def _on_submit_move(self) -> None:
        if self.ai_thinking or is_game_over(self.state):
            return
        if is_ai_turn(self.state):
            return

        move_str = self.move_entry.get().strip()
        if not move_str:
            return
        self.move_entry.delete(0, self.tk.END)

        self._apply_human_move(move_str)
        self._draw_board()
        self._update_status()

        if is_ai_turn(self.state) and not is_game_over(self.state):
            self.root.after(300, self._trigger_ai_move)

    def _apply_human_move(self, move_str: str) -> None:
        success, msg, _ = apply_move(self.state, move_str)
        if not success:
            self.status_label.config(text=f"Illegal: {msg[:80]}", fg=MISTRAL_ORANGE)
            return
        self.state = load_state() or self.state

    def _trigger_ai_move(self) -> None:
        if self.ai_thinking or is_game_over(self.state):
            return
        if not is_ai_turn(self.state):
            return

        self.ai_thinking = True
        self.ai_button.config(state=self.tk.DISABLED)
        self.thinking_label.config(text="Mistral Large 4 is looking at the board...")
        self._set_thinking_text("")
        self.root.update()

        thread = threading.Thread(target=self._ai_move_worker, daemon=True)
        thread.start()

    def _ai_move_worker(self) -> None:
        try:
            from ai_player import VisionChessAI

            board = self.state.to_board()

            # Generate screenshot for the LLM
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            move_num = len(self.state.history) + 1
            shot_path = SCREENSHOT_DIR / f"position_{move_num:03d}.png"
            board_to_png(
                board,
                shot_path,
                square_size=SCREENSHOT_SQUARE,
                last_move=self.state.get_last_move(),
            )

            # Call the vision AI
            api_key = os.environ.get("MISTRAL_API_KEY")
            ai = VisionChessAI(api_key=api_key, model="mistral-large-4")
            result = ai.get_move(board, move_history=self.state.history, verbose=False)

            if result.success and result.move_san:
                self.root.after(0, lambda: self._apply_ai_move(
                    result.move_san, str(shot_path), result.reasoning
                ))
            else:
                self.root.after(0, lambda: self._ai_move_failed(result.error or "Unknown error"))

        except Exception as e:
            self.root.after(0, lambda: self._ai_move_failed(str(e)))

    def _apply_ai_move(self, move_san: str, screenshot_path: str, reasoning: str | None = None) -> None:
        apply_move(self.state, move_san)
        self.state = load_state() or self.state

        self.ai_thinking = False
        self.ai_button.config(state=self.tk.NORMAL)
        self.thinking_label.config(text=f"AI played: {move_san}")

        self.shot_label.config(text=f"Saved: {Path(screenshot_path).name}", fg=BRIGHT_YELLOW)
        self._show_screenshot(screenshot_path)

        if reasoning:
            self._set_thinking_text(reasoning)
        else:
            self._set_thinking_text("(no reasoning returned)")

        self._draw_board()
        self._update_status()

        if is_game_over(self.state):
            board = self.state.to_board()
            self.thinking_label.config(text=f"GAME OVER: {game_status(board)}", fg=MISTRAL_ORANGE)

    def _ai_move_failed(self, error: str) -> None:
        self.ai_thinking = False
        self.ai_button.config(state=self.tk.NORMAL)
        self.thinking_label.config(text=f"AI Error: {error[:80]}", fg=MISTRAL_ORANGE)
        self._set_thinking_text(f"Error: {error}")

    def _set_thinking_text(self, text: str) -> None:
        """Update the AI thinking panel with new text."""
        tk = self.tk
        self.thinking_text.config(state=tk.NORMAL)
        self.thinking_text.delete(1.0, tk.END)
        self.thinking_text.insert(tk.END, text)
        self.thinking_text.config(state=tk.DISABLED)
        self.thinking_text.see(tk.END)

    def _show_screenshot(self, path: str) -> None:
        try:
            img = Image.open(path)
            img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(img)
            self.shot_canvas.delete("all")
            self.shot_canvas.create_image(100, 100, image=photo)
            self.shot_canvas.image = photo
        except Exception:
            pass

    def _update_status(self) -> None:
        tk = self.tk
        board = self.state.to_board()
        status = game_status(board)
        turn = whose_turn(self.state)

        if is_game_over(self.state):
            self.status_label.config(text=f"GAME OVER\n{status}", fg=MISTRAL_ORANGE)
        elif is_ai_turn(self.state):
            self.status_label.config(text=f"{turn.capitalize()}: Mistral Large 4 (AI)", fg=SUNSHINE_GOLD)
        else:
            self.status_label.config(text=f"{turn.capitalize()}: You (Human)", fg=FG_TEXT)

        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        history = self.state.history
        for i in range(0, len(history), 2):
            move_num = i // 2 + 1
            white = history[i]
            black = history[i + 1] if i + 1 < len(history) else "..."
            self.history_text.insert(tk.END, f"{move_num}. {white}  {black}\n")
        self.history_text.config(state=tk.DISABLED)
        self.history_text.see(tk.END)

    def _new_game(self) -> None:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        self.state = new_game(
            white_player="ai" if self.ai_white else "human",
            black_player="human" if self.ai_white else "ai",
        )
        self.selected_square = None
        self.legal_targets = set()
        self.shot_canvas.delete("all")
        self.shot_label.config(text="(not yet generated)", fg=FG_MUTED)
        self.thinking_label.config(text="")
        self._set_thinking_text("")
        self._draw_board()
        self._update_status()

        if is_ai_turn(self.state) and not is_game_over(self.state):
            self.root.after(500, self._trigger_ai_move)


def main() -> None:
    import tkinter as tk

    parser = argparse.ArgumentParser(description="LeChonk Chess GUI")
    parser.add_argument("--ai-white", action="store_true", help="AI plays White")
    args = parser.parse_args()

    if not os.environ.get("MISTRAL_API_KEY"):
        print("WARNING: MISTRAL_API_KEY not set. AI moves will fail.")
        print("  export MISTRAL_API_KEY='your-key'")
        print()

    root = tk.Tk()
    app = ChessGUI(root, ai_white=args.ai_white)
    root.mainloop()


if __name__ == "__main__":
    main()
