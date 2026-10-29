"""Render chess boards as PNG images for vision-based AI play.

Uses Pillow to draw a graphical chess board with Unicode chess pieces,
coordinate labels, last-move highlighting, and turn indicators.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import chess
from PIL import Image, ImageDraw, ImageFont

# ── Mistral brand palette (RGB) ────────────────────────────────────

MISTRAL_ORANGE = (250, 82, 15)     # #fa520f
ORANGE_DEEP = (204, 58, 5)         # #cc3a05
SUNSHINE_GOLD = (255, 161, 16)     # #ffa110
BRIGHT_YELLOW = (255, 217, 0)      # #ffd900
WARM_CREAM = (255, 248, 224)       # #fff8e0
MISTRAL_BLACK = (31, 31, 31)       # #1f1f1f

# ── Configuration ──────────────────────────────────────────────────

# Board colors (Mistral theme; dark squares reverted to the original)
LIGHT_SQUARE = WARM_CREAM                       # #fff8e0
DARK_SQUARE = (181, 136, 99)                    # #B58863 (original)
HIGHLIGHT_FROM = (*BRIGHT_YELLOW, 90)           # last-move source wash
HIGHLIGHT_TO = (*SUNSHINE_GOLD, 120)            # last-move target wash
COORD_COLOR_LIGHT = WARM_CREAM                  # labels on dark squares
COORD_COLOR_DARK = (181, 136, 99)               # #B58863, labels on light squares
BORDER_COLOR = MISTRAL_BLACK                    # image background / frame
TEXT_COLOR = BRIGHT_YELLOW                      # title text on the frame

# Find a font with chess glyphs
FONT_CANDIDATES = [
    "/usr/share/fonts/gdouros-symbola/Symbola.ttf",
    "/usr/share/fonts/google-noto/NotoSansSymbols2-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSansSymbols-Regular.ttf",
]

# Find a font for coordinates/text
TEXT_FONT_CANDIDATES = [
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
    "/usr/share/fonts/open-sans/OpenSans-Bold.ttf",
    "/usr/share/fonts/google-droid-sans-fonts/DroidSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


def _find_font(candidates: list[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    # Fallback: search common locations
    for root in ["/usr/share/fonts", "/usr/local/share/fonts"]:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith((".ttf", ".otf")):
                    full = os.path.join(dirpath, fn)
                    # Quick check: try loading it
                    try:
                        ImageFont.truetype(full, 12)
                        return full
                    except Exception:
                        continue
    raise RuntimeError("No suitable font found")


PIECE_FONT_PATH = _find_font(FONT_CANDIDATES)
TEXT_FONT_PATH = _find_font(TEXT_FONT_CANDIDATES)

# Unicode chess pieces: (white, black) for each piece type
PIECE_CHARS: dict[chess.PieceType, tuple[str, str]] = {
    chess.KING:   ("♔", "♚"),
    chess.QUEEN:  ("♕", "♛"),
    chess.ROOK:   ("♖", "♜"),
    chess.BISHOP: ("♗", "♝"),
    chess.KNIGHT: ("♘", "♞"),
    chess.PAWN:   ("♙", "♟"),
}

# ── Custom piece images ────────────────────────────────────────────

# Drop PNGs into assets/pieces/ to replace the Unicode glyphs.
# First matching name wins, case-insensitive:
#   Short:  wK.png wQ.png wR.png wB.png wN.png wP.png  (b... for black)
#   Long:   white_king.png white_queen.png ... black_pawn.png
PIECE_IMAGE_DIR = Path(__file__).parent / "assets" / "pieces"

PIECE_IMAGE_NAMES: dict[tuple[bool, chess.PieceType], tuple[str, ...]] = {
    (chess.WHITE, chess.KING):   ("wK.png", "white_king.png", "w_king.png"),
    (chess.WHITE, chess.QUEEN):  ("wQ.png", "white_queen.png", "w_queen.png"),
    (chess.WHITE, chess.ROOK):   ("wR.png", "white_rook.png", "w_rook.png"),
    (chess.WHITE, chess.BISHOP): ("wB.png", "white_bishop.png", "w_bishop.png"),
    (chess.WHITE, chess.KNIGHT): ("wN.png", "white_knight.png", "w_knight.png"),
    (chess.WHITE, chess.PAWN):   ("wP.png", "white_pawn.png", "w_pawn.png"),
    (chess.BLACK, chess.KING):   ("bK.png", "black_king.png", "b_king.png"),
    (chess.BLACK, chess.QUEEN):  ("bQ.png", "black_queen.png", "b_queen.png"),
    (chess.BLACK, chess.ROOK):   ("bR.png", "black_rook.png", "b_rook.png"),
    (chess.BLACK, chess.BISHOP): ("bB.png", "black_bishop.png", "b_bishop.png"),
    (chess.BLACK, chess.KNIGHT): ("bN.png", "black_knight.png", "b_knight.png"),
    (chess.BLACK, chess.PAWN):   ("bP.png", "black_pawn.png", "b_pawn.png"),
}

# Fraction of the square the piece image may occupy
PIECE_IMAGE_SCALE = 0.82

_piece_image_cache: dict[tuple[bool, chess.PieceType, int], Image.Image | None] = {}


def clear_piece_image_cache() -> None:
    """Forget loaded piece images (e.g. after swapping PNGs on disk)."""
    _piece_image_cache.clear()


def _find_piece_image_file(color: bool, piece_type: chess.PieceType) -> Path | None:
    if not PIECE_IMAGE_DIR.is_dir():
        return None
    try:
        existing = {p.name.lower(): p for p in PIECE_IMAGE_DIR.iterdir() if p.is_file()}
    except OSError:
        return None
    for name in PIECE_IMAGE_NAMES.get((color, piece_type), ()):
        path = existing.get(name.lower())
        if path is not None:
            return path
    return None


def load_piece_image(
    color: bool, piece_type: chess.PieceType, size: int
) -> Image.Image | None:
    """Load a custom piece image resized for a square of `size` px.

    Returns None when no PNG is found, so callers fall back to glyphs.
    Results are cached per (color, piece_type, size).
    """
    key = (color, piece_type, size)
    if key in _piece_image_cache:
        return _piece_image_cache[key]

    img: Image.Image | None = None
    path = _find_piece_image_file(color, piece_type)
    if path is not None:
        try:
            raw = Image.open(path).convert("RGBA")
            target = max(8, int(size * PIECE_IMAGE_SCALE))
            # Aspect-fit into the target box, upscaling small sources
            # (Image.thumbnail only shrinks, which left icons tiny).
            w, h = raw.size
            scale = target / max(w, h)
            if scale != 1.0:
                new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
                raw = raw.resize(new_size, Image.Resampling.LANCZOS)
            img = raw
        except Exception:
            img = None

    _piece_image_cache[key] = img
    return img


OUTPUT_DIR = Path(__file__).parent / "board_images"


def render_board_image(
    board: chess.Board,
    *,
    square_size: int = 80,
    show_coords: bool = True,
    flip: bool = False,
    last_move: chess.Move | None = None,
    highlight_legal: chess.Move | None = None,
    title: str | None = None,
) -> Image.Image:
    """Render a chess board to a PIL Image.

    Args:
        board: The chess position to render.
        square_size: Pixel size of each square.
        show_coords: Whether to draw file/rank labels.
        flip: Render from Black's perspective.
        last_move: Highlight the from/to squares of this move.
        highlight_legal: If a Move is given, highlight squares for a selected piece.
        title: Optional title text at the top.
    """
    margin = 30 if show_coords else 10
    title_height = 40 if title else 0
    board_px = square_size * 8
    total_w = board_px + margin * 2
    total_h = board_px + margin * 2 + title_height

    img = Image.new("RGB", (total_w, total_h), BORDER_COLOR)
    draw = ImageDraw.Draw(img, "RGBA")

    # Title
    if title:
        try:
            title_font = ImageFont.truetype(TEXT_FONT_PATH, 22)
        except Exception:
            title_font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(
            ((total_w - tw) // 2, 8),
            title,
            fill=TEXT_COLOR,
            font=title_font,
        )

    # Piece font
    piece_font_size = int(square_size * 0.7)
    try:
        piece_font = ImageFont.truetype(PIECE_FONT_PATH, piece_font_size)
    except Exception:
        piece_font = ImageFont.load_default()

    # Coord font
    coord_font_size = max(12, square_size // 5)
    try:
        coord_font = ImageFont.truetype(TEXT_FONT_PATH, coord_font_size)
    except Exception:
        coord_font = ImageFont.load_default()

    # Highlight squares
    highlights: dict[chess.Square, tuple[int, int, int, int]] = {}
    if last_move is not None:
        highlights[last_move.from_square] = HIGHLIGHT_FROM
        highlights[last_move.to_square] = HIGHLIGHT_TO

    files = "abcdefgh"
    ranks = range(8, 0, -1) if not flip else range(1, 9)
    rank_list = list(ranks)
    file_list = list(range(8)) if not flip else list(range(7, -1, -1))

    for row_idx, rank in enumerate(rank_list):
        for col_idx, file_idx in enumerate(file_list):
            x0 = margin + col_idx * square_size
            y0 = margin + title_height + row_idx * square_size
            x1 = x0 + square_size
            y1 = y0 + square_size

            square = chess.square(file_idx, rank - 1)
            is_light = (file_idx + rank) % 2 == 1
            base_color = LIGHT_SQUARE if is_light else DARK_SQUARE

            # Draw square
            draw.rectangle([x0, y0, x1, y1], fill=base_color)

            # Highlight overlay
            if square in highlights:
                draw.rectangle([x0, y0, x1, y1], fill=highlights[square])

            # Draw piece (custom PNG if available, Unicode glyph otherwise)
            piece = board.piece_at(square)
            if piece is not None:
                piece_img = load_piece_image(piece.color, piece.piece_type, square_size)
                if piece_img is not None:
                    pw, ph = piece_img.size
                    px = x0 + (square_size - pw) // 2
                    py = y0 + (square_size - ph) // 2
                    img.paste(piece_img, (px, py), piece_img)
                else:
                    char = PIECE_CHARS[piece.piece_type][0 if piece.color == chess.WHITE else 1]
                    # Center the piece character in the square
                    bbox = draw.textbbox((0, 0), char, font=piece_font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    tx = x0 + (square_size - tw) // 2 - bbox[0]
                    ty = y0 + (square_size - th) // 2 - bbox[1]
                    # Draw with slight shadow for black pieces visibility
                    draw.text((tx, ty), char, fill=MISTRAL_BLACK, font=piece_font)

            # Coordinate labels (inside the square, top-left for files, bottom-right for ranks)
            coord_color = DARK_SQUARE if is_light else LIGHT_SQUARE
            file_char = files[file_idx]
            rank_char = str(rank)
            if col_idx == 0:
                draw.text((x0 + 3, y0 + 2), rank_char, fill=coord_color, font=coord_font)
            if row_idx == 7:
                bbox = draw.textbbox((0, 0), file_char, font=coord_font)
                fw = bbox[2] - bbox[0]
                draw.text((x1 - fw - 3, y1 - coord_font_size - 2), file_char, fill=coord_color, font=coord_font)

    # Outer border (brand frame around the board, on the black background)
    draw.rectangle(
        [margin - 2, title_height + margin - 2,
         margin + board_px + 2, title_height + margin + board_px + 2],
        outline=SUNSHINE_GOLD,
        width=2,
    )

    return img


def board_to_png(
    board: chess.Board,
    output_path: str | Path | None = None,
    *,
    square_size: int = 80,
    flip: bool = False,
    last_move: chess.Move | None = None,
    show_status: bool = True,
) -> bytes:
    """Render a chess board to PNG bytes (and optionally save to file).

    Returns the PNG image as bytes.
    """
    from chess_game import game_status

    title = None
    if show_status:
        status = game_status(board)
        side = "White" if board.turn == chess.WHITE else "Black"
        title = f"{side} to move — {status}"

    img = render_board_image(
        board,
        square_size=square_size,
        flip=flip,
        last_move=last_move,
        title=title,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(png_bytes)

    return png_bytes


def save_board_image(
    board: chess.Board,
    filename: str = "board.png",
    **kwargs,
) -> Path:
    """Render and save a board image. Returns the file path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    board_to_png(board, path, **kwargs)
    return path
