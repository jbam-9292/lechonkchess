# Custom chess piece icons

Drop PNG files here to replace the default Unicode chess glyphs on the
board. Transparent backgrounds work best.

Name each file after the piece it represents (first match wins,
case-insensitive):

| Piece        | Short name | Long name         |
|--------------|------------|-------------------|
| White king   | `wK.png`   | `white_king.png`  |
| White queen  | `wQ.png`   | `white_queen.png` |
| White rook   | `wR.png`   | `white_rook.png`  |
| White bishop | `wB.png`   | `white_bishop.png`|
| White knight | `wN.png`   | `white_knight.png`|
| White pawn   | `wP.png`   | `white_pawn.png`  |
| Black king   | `bK.png`   | `black_king.png`  |
| Black queen  | `bQ.png`   | `black_queen.png` |
| Black rook   | `bR.png`   | `black_rook.png`  |
| Black bishop | `bB.png`   | `black_bishop.png`|
| Black knight | `bN.png`   | `black_knight.png`|
| Black pawn   | `bP.png`   | `black_pawn.png`  |

Any piece without a PNG falls back to the Unicode glyph. Images are
scaled to fit within ~82% of a square and are cached at startup, so
restart the app after adding or changing files.
