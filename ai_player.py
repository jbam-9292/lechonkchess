"""Vision-based chess AI player using Mistral Large 4.

Renders the chess board as a PNG image, sends it to Mistral's vision model,
and parses the returned move. Falls back to text-only mode if vision fails.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess
from mistralai.client import Mistral
from mistralai.client.models.systemmessage import SystemMessage
from mistralai.client.models.usermessage import UserMessage
from mistralai.client.models.textchunk import TextChunk
from mistralai.client.models.imageurlchunk import ImageURLChunk
from mistralai.client.models.imageurl import ImageURL

from board_renderer import board_to_png
from chess_game import get_legal_moves_san, get_legal_moves_uci, parse_move

# ── Configuration ──────────────────────────────────────────────────

DEFAULT_MODEL = "mistral-large-4"  # Mistral Large 4 with vision
FALLBACK_MODEL = "mistral-large-latest"

SYSTEM_PROMPT = """You are a chess grandmaster playing a game. You will be shown an image of a chess board.
The board has coordinate labels: files (columns) are labeled a-h from left to right, and ranks (rows) are labeled 1-8 from bottom to top.
White pieces are at the bottom, Black pieces are at the top.
White pieces are shown as outlined/light symbols, Black pieces as filled/dark symbols.

Your task:
1. Carefully examine the board image to identify all pieces and their positions.
2. Determine whose turn it is (stated in the title bar).
3. Choose the best legal move for the side to move.
4. Respond with ONLY the move in Standard Algebraic Notation (SAN), e.g. "e4", "Nf3", "O-O", "exd5", "Qxf7+".

Rules:
- Use standard SAN notation.
- For castling, use "O-O" (kingside) or "O-O-O" (queenside).
- For pawn promotion, append "=Q", "=R", "=B", or "=N" (e.g., "e8=Q").
- For captures, use "x" (e.g., "exd5", "Nxf7").
- Include "+" for check and "#" for checkmate.
- Respond with ONLY the move. No explanation, no extra text.
"""


@dataclass
class AIMoveResult:
    """Result of an AI move attempt."""
    move_san: str | None
    move_uci: str | None
    raw_response: str
    model_used: str
    attempts: int
    success: bool
    error: str | None = None
    reasoning: str | None = None


class VisionChessAI:
    """Chess AI that uses Mistral's vision model to 'see' the board."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        fallback_model: str = FALLBACK_MODEL,
        max_retries: int = 3,
        temperature: float = 0.1,
        reasoning_effort: str | None = "max",
    ):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable "
                "or pass api_key to VisionChessAI."
            )
        self.model = model
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.client = Mistral(api_key=self.api_key)

    def _encode_image(self, png_bytes: bytes) -> str:
        """Encode PNG bytes to base64 data URL."""
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _build_messages(
        self,
        board: chess.Board,
        png_bytes: bytes,
        move_history: list[str] | None = None,
        use_text_fallback: bool = False,
    ) -> list:
        """Build the messages for the Mistral API call."""
        side = "White" if board.turn == chess.WHITE else "Black"
        legal_san = get_legal_moves_san(board)

        # Build context text
        context_parts = [
            f"It is {side}'s turn.",
            f"FEN: {board.fen()}",
        ]
        if move_history:
            context_parts.append(f"Move history: {' '.join(move_history[-20:])}")
        context_parts.append(f"Legal moves: {', '.join(legal_san)}")
        context_parts.append("")
        context_parts.append(
            "Look at the board image carefully. Identify each piece and its square. "
            "Then choose the best move for the side to move. "
            "Respond with ONLY the move in SAN notation."
        )
        context_text = "\n".join(context_parts)

        system_msg = SystemMessage(content=SYSTEM_PROMPT)

        if use_text_fallback:
            # Text-only mode: ASCII board
            ascii_board = self._board_to_ascii(board)
            context_text = (
                f"{context_text}\n\n"
                f"Board (uppercase = White, lowercase = black, . = empty):\n"
                f"{ascii_board}\n"
            )
            return [
                system_msg,
                UserMessage(content=context_text),
            ]

        # Vision mode: send the image
        image_url = self._encode_image(png_bytes)
        return [
            system_msg,
            UserMessage(content=[
                TextChunk(text=context_text),
                ImageURLChunk(image_url=ImageURL(url=image_url)),
            ]),
        ]

    def _board_to_ascii(self, board: chess.Board) -> str:
        """Simple ASCII board for text fallback."""
        lines = []
        for rank in range(7, -1, -1):
            row = f"  {rank + 1} "
            for file in range(8):
                sq = chess.square(file, rank)
                piece = board.piece_at(sq)
                row += (piece.symbol() if piece else ".") + " "
            lines.append(row)
        lines.append("    a b c d e f g h")
        return "\n".join(lines)

    @staticmethod
    def _extract_text(content) -> str:
        """Extract only the final text (non-thinking) from a Mistral response.

        Content can be a plain string, or a list of chunks (TextChunk, ThinkChunk).
        ThinkChunk content is excluded — use _extract_thinking for that.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for chunk in content:
                ctype = type(chunk).__name__
                if ctype == "TextChunk" and hasattr(chunk, "text"):
                    parts.append(chunk.text)
            return "".join(parts)
        return str(content)

    @staticmethod
    def _extract_thinking(content) -> str:
        """Extract only the reasoning/thinking text from a Mistral response.

        Returns the concatenated text from ThinkChunk entries, or empty string.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return ""
        if isinstance(content, list):
            parts = []
            for chunk in content:
                ctype = type(chunk).__name__
                if ctype == "ThinkChunk" and hasattr(chunk, "thinking"):
                    for tc in chunk.thinking:
                        if hasattr(tc, "text"):
                            parts.append(tc.text)
            return "".join(parts)
        return ""

    def _parse_move_from_response(
        self, response: str, board: chess.Board
    ) -> tuple[str | None, str | None]:
        """Extract a legal move from the AI's text response.

        Returns (san, uci) or (None, None) if no legal move found.
        """
        text = response.strip()

        # Try to extract a move from the response
        # Look for patterns like: e4, Nf3, O-O, exd5, e8=Q, Qxf7+
        wb = '\\' + "b"  # word boundary
        patterns = [
            wb + "([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)" + wb,
            wb + "(O-O-O|O-O|0-0-0|0-0)" + wb,
            wb + "([a-h][1-8][a-h][1-8][qrbn]?)" + wb,
        ]

        candidates = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                candidates.append(match.group(1))

        # Also try the whole response if it's short
        if len(text) <= 10:
            candidates.insert(0, text)

        # Try each candidate
        for candidate in candidates:
            candidate = candidate.strip().replace("0", "O")
            move = parse_move(board, candidate)
            if move is not None:
                san = board.san(move)
                return san, move.uci()

        return None, None

    def get_move(
        self,
        board: chess.Board,
        move_history: list[str] | None = None,
        square_size: int = 80,
        verbose: bool = False,
    ) -> AIMoveResult:
        """Get the AI's move for the given board position.

        Renders the board as an image and sends it to Mistral's vision model.
        Retries on failure, falls back to text-only mode if vision fails.
        """
        # Render board image
        last_move = board.peek() if board.move_stack else None
        png_bytes = board_to_png(
            board,
            square_size=square_size,
            last_move=last_move,
        )

        legal_san = get_legal_moves_san(board)
        if not legal_san:
            return AIMoveResult(
                move_san=None,
                move_uci=None,
                raw_response="",
                model_used=self.model,
                attempts=0,
                success=False,
                error="No legal moves available (game over)",
            )

        last_error = None
        models_to_try = [self.model, self.fallback_model]
        use_vision = True

        for attempt in range(1, self.max_retries + 1):
            for model in models_to_try:
                try:
                    if verbose:
                        mode = "vision" if use_vision else "text"
                        print(f"  [AI] Attempt {attempt} with {model} ({mode} mode)...", file=sys.stderr)

                    messages = self._build_messages(
                        board, png_bytes, move_history,
                        use_text_fallback=not use_vision,
                    )

                    # reasoning_effort is only accepted by Magistral reasoning
                    # models; sending it to others (e.g. mistral-large) is a 400.
                    request_kwargs = {}
                    if self.reasoning_effort is not None and model.startswith("magistral"):
                        request_kwargs["reasoning_effort"] = self.reasoning_effort

                    response = self.client.chat.complete(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=1024,
                        **request_kwargs,
                    )

                    content = response.choices[0].message.content
                    raw_text = self._extract_text(content)
                    thinking = self._extract_thinking(content)
                    if verbose:
                        print(f"  [AI] Raw response: {raw_text!r}", file=sys.stderr)
                        if thinking:
                            print(f"  [AI] Thinking: {thinking[:500]}...", file=sys.stderr)

                    san, uci = self._parse_move_from_response(raw_text, board)

                    if san is not None:
                        if verbose:
                            print(f"  [AI] Parsed move: {san} ({uci})", file=sys.stderr)
                        return AIMoveResult(
                            move_san=san,
                            move_uci=uci,
                            raw_response=raw_text,
                            model_used=model,
                            attempts=attempt,
                            success=True,
                            reasoning=thinking or None,
                        )
                    else:
                        last_error = f"Could not parse move from response: {raw_text!r}"
                        if verbose:
                            print(f"  [AI] {last_error}", file=sys.stderr)

                except Exception as e:
                    last_error = str(e)
                    if verbose:
                        print(f"  [AI] Error with {model}: {e}", file=sys.stderr)
                    # If vision model fails, try text fallback
                    if use_vision and "vision" in str(e).lower():
                        use_vision = False
                        continue

            # If vision failed on all models, switch to text-only
            if use_vision:
                use_vision = False
                if verbose:
                    print("  [AI] Falling back to text-only mode...", file=sys.stderr)
                continue

            time.sleep(1)  # Brief pause before retry

        return AIMoveResult(
            move_san=None,
            move_uci=None,
            raw_response="",
            model_used=self.model,
            attempts=self.max_retries,
            success=False,
            error=last_error or "Unknown error",
        )


def get_ai_player(api_key: str | None = None, **kwargs) -> VisionChessAI:
    """Factory function to create a VisionChessAI instance."""
    return VisionChessAI(api_key=api_key, **kwargs)
