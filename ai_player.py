"""Vision-based chess AI player using Mistral's Responses API.

Renders the chess board as a PNG image, sends it to Mistral's vision model
via the Responses API (client.responses.create), and parses the returned
move. Falls back to text-only mode if vision fails.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass

import chess
from mistralai.client import Mistral
from mistralai.client.models.responseseasyinputmessage import ResponsesEasyInputMessage
from mistralai.client.models.responsesinputtext import ResponsesInputText
from mistralai.client.models.responsesinputimage import ResponsesInputImage

from board_renderer import board_to_png
from chess_game import get_legal_moves_san, get_legal_moves_uci, parse_move

# ── Configuration ──────────────────────────────────────────────────

DEFAULT_MODEL = "mistral-large-4"  # Mistral Large 4 with vision
FALLBACK_MODEL = "mistral-large-4"

SYSTEM_PROMPT = """You are a chess grandmaster playing a game. You will be shown an image of a chess board.
The board has coordinate labels: files (columns) are labeled a-h from left to right, and ranks (rows) are labeled 1-8 from bottom to top.
White pieces are at the bottom, Black pieces are at the top.
White pieces are shown as outlined/light symbols, Black pieces as filled/dark symbols.

Your task:
1. Carefully examine the board image to identify all pieces and their positions.
2. Determine whose turn it is (stated in the title bar).
3. Choose the best legal move for the side to move.
4. Respond with a JSON object: {"move": "<SAN>", "reasoning": "<why>"}.

Rules:
- The "move" field holds the move in standard SAN notation, e.g. "e4", "Nf3", "O-O", "exd5", "Qxf7+".
- For castling, use "O-O" (kingside) or "O-O-O" (queenside).
- For pawn promotion, append "=Q", "=R", "=B", or "=N" (e.g., "e8=Q").
- For captures, use "x" (e.g., "exd5", "Nxf7").
- Include "+" for check and "#" for checkmate.
- The "reasoning" field briefly explains why you chose the move (1-3 sentences).
- Respond with ONLY the JSON object. No other text.
"""

# Structured output schema: the model returns the move AND its reasoning as
# a JSON object, so the caller always gets both (used for the GUI's
# "AI Thinking" panel). Requested via the `text.format` parameter.
MOVE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "move": {
            "type": "string",
            "description": (
                "The chosen move in Standard Algebraic Notation (SAN), "
                'e.g. "e4", "Nf3", "O-O", "exd5", "Qxf7+", "e8=Q".'
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of why this move was chosen (1-3 sentences).",
        },
    },
    "required": ["move", "reasoning"],
    "additionalProperties": False,
}


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


def _extract_json_object(text: str) -> dict | None:
    """Parse the first JSON object found in a response text.

    Handles a raw JSON object and markdown-fenced ```json blocks.
    Returns None if no JSON object is present.
    """
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to scanning for the first {...} block in the text
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


class VisionChessAI:
    """Chess AI that uses Mistral's Responses API to 'see' the board."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        fallback_model: str = FALLBACK_MODEL,
        max_retries: int = 3,
        temperature: float = 0.1,
        reasoning_effort: str | None = None,
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

    def _build_input(
        self,
        board: chess.Board,
        png_bytes: bytes,
        move_history: list[str] | None = None,
        use_text_fallback: bool = False,
    ) -> str | list:
        """Build the `input` for the Responses API call.

        Returns a plain string in text-only mode, or a single user message
        (text + image) in vision mode. The system prompt is sent separately
        via the `instructions` parameter.
        """
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

        if use_text_fallback:
            # Text-only mode: ASCII board
            context_parts.append(
                "Read the board below. Choose the best move for the side to "
                'move. Respond with a JSON object: {"move": "<SAN>", "reasoning": "<why>"}.'
            )
            context_text = "\n".join(context_parts)
            ascii_board = self._board_to_ascii(board)
            return (
                f"{context_text}\n\n"
                f"Board (uppercase = White, lowercase = black, . = empty):\n"
                f"{ascii_board}\n"
            )

        context_parts.append(
            "Look at the board image carefully. Identify each piece and its square. "
            "Then choose the best move for the side to move. "
            'Respond with a JSON object: {"move": "<SAN>", "reasoning": "<why>"}.'
        )
        context_text = "\n".join(context_parts)

        # Vision mode: send the image
        image_url = self._encode_image(png_bytes)
        return [
            ResponsesEasyInputMessage(
                role="user",
                content=[
                    ResponsesInputText(text=context_text),
                    ResponsesInputImage(image_url=image_url, detail="high"),
                ],
            )
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
    def _extract_text(output) -> str:
        """Extract the assistant's final text from a Responses API output list.

        The output list holds items of type "message" (assistant turns),
        "reasoning", and "function_call". Only the message items' output_text
        chunks are collected — use _extract_thinking for reasoning items.
        """
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            parts = []
            for item in output:
                if getattr(item, "type", None) != "message":
                    continue
                content = getattr(item, "content", None)
                if isinstance(content, str):
                    parts.append(content)
                    continue
                for chunk in content or []:
                    if getattr(chunk, "type", None) == "output_text" and hasattr(chunk, "text"):
                        parts.append(chunk.text)
            return "".join(parts)
        return str(output)

    @staticmethod
    def _extract_thinking(output) -> str:
        """Extract the reasoning text from a Responses API output list.

        Reasoning items (type="reasoning") carry summary_text entries and
        optional reasoning_text content. Returns empty string if there is none.
        """
        if output is None:
            return ""
        if isinstance(output, str):
            return ""
        if isinstance(output, list):
            parts = []
            for item in output:
                if getattr(item, "type", None) != "reasoning":
                    continue
                for summary in getattr(item, "summary", None) or []:
                    if hasattr(summary, "text"):
                        parts.append(summary.text)
                for chunk in getattr(item, "content", None) or []:
                    if hasattr(chunk, "text"):
                        parts.append(chunk.text)
            return "".join(parts)
        return ""

    def _parse_move_from_response(
        self, response: str, board: chess.Board
    ) -> tuple[str | None, str | None, str | None]:
        """Extract a legal move (and reasoning, if present) from the AI's response.

        Prefers the structured JSON output {"move": ..., "reasoning": ...};
        falls back to regex move extraction from free text. Returns
        (san, uci, reasoning) — any missing part is None.
        """
        text = response.strip()

        # Preferred: structured JSON response
        data = _extract_json_object(text)
        if data is not None:
            move_str = data.get("move")
            if isinstance(move_str, str) and move_str.strip():
                move = parse_move(board, move_str.strip().replace("0", "O"))
                if move is not None:
                    san = board.san(move)
                    reasoning = data.get("reasoning")
                    if isinstance(reasoning, str) and reasoning.strip():
                        reasoning = reasoning.strip()
                    else:
                        reasoning = None
                    return san, move.uci(), reasoning

        # Fallback: look for patterns like e4, Nf3, O-O, exd5, e8=Q, Qxf7+
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
                return san, move.uci(), None

        return None, None, None

    def get_move(
        self,
        board: chess.Board,
        move_history: list[str] | None = None,
        square_size: int = 80,
        verbose: bool = False,
    ) -> AIMoveResult:
        """Get the AI's move for the given board position.

        Renders the board as an image and sends it to Mistral's vision model
        via the Responses API. Retries on failure, falls back to text-only
        mode if vision fails.
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

                    input_items = self._build_input(
                        board, png_bytes, move_history,
                        use_text_fallback=not use_vision,
                    )

                    # Reasoning effort is only accepted by Magistral reasoning
                    # models; sending it to others (e.g. mistral-large) is a 400.
                    request_kwargs = {}
                    if self.reasoning_effort is not None and model.startswith("magistral"):
                        request_kwargs["reasoning"] = {"effort": self.reasoning_effort}

                    # Structured output: the model returns the move AND its
                    # reasoning as a JSON object, so the caller always gets both.
                    request_kwargs["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": "chess_move",
                            "schema": MOVE_RESPONSE_SCHEMA,
                            "strict": True,
                        }
                    }

                    response = self.client.responses.create(
                        model=model,
                        input=input_items,
                        instructions=SYSTEM_PROMPT,
                        temperature=self.temperature,
                        max_output_tokens=2048,
                        **request_kwargs,
                    )

                    raw_text = self._extract_text(response.output)
                    thinking = self._extract_thinking(response.output)
                    if verbose:
                        print(f"  [AI] Raw response: {raw_text!r}", file=sys.stderr)
                        if thinking:
                            print(f"  [AI] Thinking: {thinking[:500]}...", file=sys.stderr)

                    san, uci, json_reasoning = self._parse_move_from_response(raw_text, board)

                    if san is not None:
                        if verbose:
                            print(f"  [AI] Parsed move: {san} ({uci})", file=sys.stderr)
                        # Prefer the reasoning from the structured JSON output;
                        # fall back to reasoning items from the API, if any.
                        reasoning = json_reasoning or thinking
                        return AIMoveResult(
                            move_san=san,
                            move_uci=uci,
                            raw_response=raw_text,
                            model_used=model,
                            attempts=attempt,
                            success=True,
                            reasoning=reasoning or None,
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
