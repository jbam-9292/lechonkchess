"""Chess AI player using Mistral's Responses API.

Sends the board position to Mistral's model via the Responses API
(client.responses.create) and parses the returned move. Three input
modes are supported (see MODES):
- "text":   the board as text only (Unicode rendering + FEN + legal moves)
- "image":  the board as a PNG image only
- "hybrid": both the image and the text description
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
from chess_game import (
    format_ai_memory,
    get_legal_moves_san,
    get_legal_moves_uci,
    parse_move,
    render_board,
)

# ── Configuration ──────────────────────────────────────────────────

DEFAULT_MODEL = "mistral-large-4"  # Mistral Large 4 with vision
FALLBACK_MODEL = "mistral-large-4"

IMAGE_SYSTEM_PROMPT = """You are a chess grandmaster playing a game. You will be shown an image of a chess board.
The board has coordinate labels: files (columns) are labeled a-h from left to right, and ranks (rows) are labeled 1-8 from bottom to top.
White pieces are at the bottom, Black pieces are at the top.
White pieces are shown as outlined/light symbols, Black pieces as filled/dark symbols.

Your task:
1. Carefully examine the board image to identify all pieces and their positions.
2. Determine whose turn it is (stated in the title bar).
3. Review your MEMORY from previous moves (provided below) to recall your strategy and plans.
4. Think step by step: assess the position, recall your plan, check if it still works, adapt if needed.
5. Choose the best legal move for the side to move.
6. Respond with a JSON object: {"move": "<SAN>", "reasoning": "<why>"}.

MEMORY & PLANNING:
- Your previous reasoning is saved and shown to you on every future turn.
- Use this memory to maintain a consistent long-term strategy across moves.
- If you started a plan (e.g., "I will attack the kingside"), continue executing it.
- If the opponent disrupted your plan, acknowledge it and adapt.
- Remember key features: weak squares, pawn structure, piece placement, threats.
- Think in terms of multi-move plans, not just single moves.
- Your reasoning will be saved and shown to you later — be concise but strategic.

Rules:
- The "move" field holds the move in standard SAN notation, e.g. "e4", "Nf3", "O-O", "exd5", "Qxf7+".
- For castling, use "O-O" (kingside) or "O-O-O" (queenside).
- For pawn promotion, append "=Q", "=R", "=B", or "=N" (e.g., "e8=Q").
- For captures, use "x" (e.g., "exd5", "Nxf7").
- Include "+" for check and "#" for checkmate.
- The "reasoning" field briefly explains why you chose the move (1-3 sentences).
  Reference your memory if you are continuing or adapting a previous plan.
- Respond with ONLY the JSON object. No other text.
"""

TEXT_SYSTEM_PROMPT = """You are a chess grandmaster playing a game. You will be given the board position as TEXT: a text rendering of the board with Unicode chess pieces plus the FEN string. No image is involved.

The text board has coordinate labels: files (columns) are labeled a-h from left to right, and ranks (rows) are labeled 1-8 from top to bottom (rank 8 at the top, rank 1 at the bottom, i.e. White's perspective).
White pieces are shown as outlined/light symbols (♔♕♖♗♘♙), Black pieces as filled/dark symbols (♚♛♜♝♞♟). Empty squares are shown as "·".

Your task:
1. Read the text board and the FEN carefully to identify all pieces and their positions.
2. Determine whose turn it is (stated explicitly).
3. Review your MEMORY from previous moves (provided below) to recall your strategy and plans.
4. Think step by step: assess the position, recall your plan, check if it still works, adapt if needed.
5. Choose the best legal move for the side to move. The list of legal moves is provided — pick one of them.
6. Respond with a JSON object: {"move": "<SAN>", "reasoning": "<why>"}.

MEMORY & PLANNING:
- Your previous reasoning is saved and shown to you on every future turn.
- Use this memory to maintain a consistent long-term strategy across moves.
- If you started a plan (e.g., "I will attack the kingside"), continue executing it.
- If the opponent disrupted your plan, acknowledge it and adapt.
- Remember key features: weak squares, pawn structure, piece placement, threats.
- Think in terms of multi-move plans, not just single moves.
- Your reasoning will be saved and shown to you later — be concise but strategic.

Rules:
- The "move" field holds the move in standard SAN notation, e.g. "e4", "Nf3", "O-O", "exd5", "Qxf7+".
- For castling, use "O-O" (kingside) or "O-O-O" (queenside).
- For pawn promotion, append "=Q", "=R", "=B", or "=N" (e.g., "e8=Q").
- For captures, use "x" (e.g., "exd5", "Nxf7").
- Include "+" for check and "#" for checkmate.
- The "reasoning" field briefly explains why you chose the move (1-3 sentences).
  Reference your memory if you are continuing or adapting a previous plan.
- Respond with ONLY the JSON object. No other text.
"""

# Backwards-compatible alias for the image-only prompt.
SYSTEM_PROMPT = IMAGE_SYSTEM_PROMPT

HYBRID_SYSTEM_PROMPT = """You are a chess grandmaster playing a game. You will be shown an image of a chess board AND a text rendering of the same position (Unicode pieces plus the FEN string). Use both: cross-check piece positions between the image and the text.

The board image has coordinate labels: files (columns) are labeled a-h from left to right, and ranks (rows) are labeled 1-8 from bottom to top. White pieces are at the bottom, Black pieces are at the top. White pieces are shown as outlined/light symbols, Black pieces as filled/dark symbols.
The text board has coordinate labels too: files a-h from left to right, ranks 1-8 from top to bottom (rank 8 at the top, rank 1 at the bottom, i.e. White's perspective). White pieces are outlined/light symbols (♔♕♖♗♘♙), Black pieces filled/dark symbols (♚♛♜♝♞♟). Empty squares are shown as "·".

Your task:
1. Examine the board image and the text board/FEN to identify all pieces and their positions, cross-checking both.
2. Determine whose turn it is (stated explicitly).
3. Review your MEMORY from previous moves (provided below) to recall your strategy and plans.
4. Think step by step: assess the position, recall your plan, check if it still works, adapt if needed.
5. Choose the best legal move for the side to move. The list of legal moves is provided — pick one of them.
6. Respond with a JSON object: {"move": "<SAN>", "reasoning": "<why>"}.

MEMORY & PLANNING:
- Your previous reasoning is saved and shown to you on every future turn.
- Use this memory to maintain a consistent long-term strategy across moves.
- If you started a plan (e.g., "I will attack the kingside"), continue executing it.
- If the opponent disrupted your plan, acknowledge it and adapt.
- Remember key features: weak squares, pawn structure, piece placement, threats.
- Think in terms of multi-move plans, not just single moves.
- Your reasoning will be saved and shown to you later — be concise but strategic.

Rules:
- The "move" field holds the move in standard SAN notation, e.g. "e4", "Nf3", "O-O", "exd5", "Qxf7+".
- For castling, use "O-O" (kingside) or "O-O-O" (queenside).
- For pawn promotion, append "=Q", "=R", "=B", or "=N" (e.g., "e8=Q").
- For captures, use "x" (e.g., "exd5", "Nxf7").
- Include "+" for check and "#" for checkmate.
- The "reasoning" field briefly explains why you chose the move (1-3 sentences).
  Reference your memory if you are continuing or adapting a previous plan.
- Respond with ONLY the JSON object. No other text.
"""

# LLM input modes. The GUI slider maps positions 0..2 onto these.
MODES = ("text", "hybrid", "image")

MODE_SYSTEM_PROMPTS = {
    "text": TEXT_SYSTEM_PROMPT,
    "image": IMAGE_SYSTEM_PROMPT,
    "hybrid": HYBRID_SYSTEM_PROMPT,
}

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
    """Chess AI that uses Mistral's Responses API to play chess.

    The ``mode`` selects what the model receives for the board position:
    - ``"text"``:   text-only — Unicode board rendering + FEN + legal moves,
      no image is ever sent.
    - ``"image"``:  image-only — the board PNG plus side-to-move and memory;
      no FEN/legal-moves/board text.
    - ``"hybrid"``: both the image and the full text description.
    """

    # Default mode for this class; subclasses override via default_mode.
    default_mode = "hybrid"

    # Sent as the `instructions` parameter; set per-instance from the mode
    # in __init__, subclasses may also override at class level.
    system_prompt: str = HYBRID_SYSTEM_PROMPT

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        fallback_model: str = FALLBACK_MODEL,
        max_retries: int = 3,
        temperature: float = 0.1,
        reasoning_effort: str | None = "low",
        mode: str | None = None,
    ):
        self.mode = mode or self.default_mode
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        self.system_prompt = MODE_SYSTEM_PROMPTS[self.mode]
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

    def _render_board_image(self, board: chess.Board, square_size: int) -> bytes | None:
        """Render the board as a PNG image.

        Returns None in text-only mode (no image is sent).
        """
        if self.mode == "text":
            return None
        last_move = board.peek() if board.move_stack else None
        return board_to_png(
            board,
            square_size=square_size,
            last_move=last_move,
        )

    def _build_input(
        self,
        board: chess.Board,
        png_bytes: bytes | None,
        move_history: list[str] | None = None,
        ai_memory: list[dict] | None = None,
    ) -> list:
        """Build the `input` for the Responses API call.

        Returns a single user message. Which parts it contains depends on
        ``self.mode``: "text" sends only text, "image" sends the image plus
        minimal text, "hybrid" sends both the image and the full text
        description. The system prompt is sent separately via the
        `instructions` parameter.

        ``ai_memory`` is a list of dicts (from GameState.ai_memory) holding
        the AI's previous reasoning. It is formatted and injected so the
        model can maintain strategic continuity across moves.
        """
        side = "White" if board.turn == chess.WHITE else "Black"
        with_text_board = self.mode in ("text", "hybrid")

        # Build context text
        context_parts = [f"It is {side}'s turn."]
        if with_text_board:
            context_parts.append(f"FEN: {board.fen()}")
            if move_history:
                context_parts.append(f"Move history: {' '.join(move_history[-20:])}")
            context_parts.append(f"Legal moves: {', '.join(get_legal_moves_san(board))}")
            context_parts.append("")
            context_parts.append(
                "BOARD (text rendering — rank 8 at the top, rank 1 at the bottom, "
                "White's perspective; files a-h left to right):"
            )
            context_parts.append(render_board(board))
            context_parts.append("")

        # Inject AI memory so the model can plan across moves
        if ai_memory:
            memory_block = format_ai_memory(ai_memory)
            context_parts.append(memory_block)
            context_parts.append("")

        if self.mode == "image":
            context_parts.append(
                "Look at the board image carefully. Identify each piece and its square. "
                "Then choose the best move for the side to move. "
                'Respond with a JSON object: {"move": "<SAN>", "reasoning": "<why>"}.'
            )
        elif self.mode == "hybrid":
            context_parts.append(
                "Use both the board image and the text board/FEN. Cross-check piece "
                "positions between the two. Then choose the best legal move for the "
                "side to move. "
                'Respond with a JSON object: {"move": "<SAN>", "reasoning": "<why>"}.'
            )
        else:  # text
            context_parts.append(
                "Read the board text and FEN carefully. Identify each piece and its "
                "square. Then choose the best legal move for the side to move. "
                'Respond with a JSON object: {"move": "<SAN>", "reasoning": "<why>"}.'
            )
        context_text = "\n".join(context_parts)

        content: list = [ResponsesInputText(text=context_text)]
        # Vision modes (image/hybrid): also send the board image
        if png_bytes is not None:
            content.append(
                ResponsesInputImage(
                    image_url=self._encode_image(png_bytes), detail="high"
                )
            )
        return [
            ResponsesEasyInputMessage(role="user", content=content),
        ]

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
        ai_memory: list[dict] | None = None,
        square_size: int = 80,
        verbose: bool = False,
    ) -> AIMoveResult:
        """Get the AI's move for the given board position.

        Sends the board to Mistral's model via the Responses API, in the
        form selected by ``self.mode`` (text / image / hybrid). Retries on
        failure.

        ``ai_memory`` is the list of previous reasoning entries (from
        GameState.ai_memory). It is injected into the prompt so the model
        can maintain strategic continuity and execute multi-move plans.
        The returned AIMoveResult.reasoning should be saved back into
        GameState via ``state.add_ai_memory()`` by the caller.
        """
        # Render the board in whatever form this player mode uses
        # (PNG image for vision, nothing for text)
        png_bytes = self._render_board_image(board, square_size)

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

        for attempt in range(1, self.max_retries + 1):
            for model in models_to_try:
                try:
                    if verbose:
                        print(f"  [AI] Attempt {attempt} with {model}...", file=sys.stderr)

                    input_items = self._build_input(
                        board, png_bytes, move_history,
                        ai_memory=ai_memory,
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
                        instructions=self.system_prompt,
                        temperature=self.temperature,
                        max_output_tokens=2048,
                        **request_kwargs,
                    )

                    raw_text = self._extract_text(response.output)
                    thinking = self._extract_thinking(response.output)
                    # Always print the LLM's raw output to stdout so it is
                    # visible every turn, in every entry point (GUI and CLI).
                    if raw_text:
                        print(f"[LLM {model}] Raw output: {raw_text}", flush=True)
                    else:
                        print(f"[LLM {model}] Raw output: (empty — will retry)", flush=True)

                    san, uci, json_reasoning = self._parse_move_from_response(raw_text, board)

                    if san is not None:
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

            time.sleep(1)  # Brief pause before retry

        error = last_error or "Unknown error"
        return AIMoveResult(
            move_san=None,
            move_uci=None,
            raw_response="",
            model_used=self.model,
            attempts=self.max_retries,
            success=False,
            error=error,
        )


def get_ai_player(api_key: str | None = None, **kwargs) -> VisionChessAI:
    """Factory function to create a VisionChessAI instance."""
    return VisionChessAI(api_key=api_key, **kwargs)


class TextChessAI(VisionChessAI):
    """Chess AI that plays from a text-only description of the board.

    Same Responses API flow and move parsing as VisionChessAI, but with
    ``mode="text"``: instead of a board image the model receives a text
    rendering of the board (Unicode pieces with coordinate labels), the
    FEN, the move history, and the list of legal moves. No vision is
    involved — no image is ever sent.
    """

    default_mode = "hybrid"
    system_prompt: str = TEXT_SYSTEM_PROMPT


def get_text_ai_player(api_key: str | None = None, **kwargs) -> TextChessAI:
    """Factory function to create a TextChessAI instance."""
    return TextChessAI(api_key=api_key, **kwargs)
