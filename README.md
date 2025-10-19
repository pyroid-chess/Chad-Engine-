# Chad Engine

Chad Engine is a lightweight, Python-based chess engine focused on being easy to extend and integrate with online play. It includes a simple UCI-style interface for local play and is already connected to Lichess using the berserk client library so it can play online games and accept challenges.

Note: This repository welcomes contributions — see the Contributing section below.

## Features

- Simple, easy-to-read engine core (designed for learning and experimentation)
- UCI-style local interface for GUI integration and testing
- Lichess integration via the berserk library (play games, accept challenges, stream events)
- Designed to be extended: evaluation, search, book, and multiprocessing hooks

## Quickstart

Prerequisites
- Python 3.8+
- pip

Install dependencies (example)
```bash
pip install berserk python-chess
```

Run engine locally (example)
```bash
# Run the engine entrypoint (replace with actual module/script name in repo)
python -m chad_engine
```

## Lichess integration (berserk)

Chad Engine can connect to Lichess via the berserk client library. You need a Lichess API token with the `Play online` scope.

Environment variable:
- LIChess_TOKEN — your Lichess API token

Example player loop (simplified)
```python
import os
import berserk
from chad_engine import get_best_move  # placeholder: adapt to actual API in this repo

TOKEN = os.environ.get("LICHESS_TOKEN") or os.environ.get("LIChess_TOKEN")
if not TOKEN:
    raise SystemExit("Set the LICHESS_TOKEN or LIChess_TOKEN environment variable")

session = berserk.TokenSession(TOKEN)
client = berserk.Client(session=session)

def on_event(event):
    # event processing (gameStart, gameFinish, gameState, challenge, etc.)
    if event["type"] == "gameStart":
        game_id = event["id"]
        print(f"Started game {game_id}")
    if event["type"] == "gameState":
        # get FEN from event payload or build from moves
        fen = event.get("fen")
        move = get_best_move(fen)  # adapt to engine API
        client.bots.make_move(game_id, move)

client.bots.stream_incoming_events(on_event)
```

This repo contains (or can be extended with) helper modules that:
- Keep track of game state and move history
- Convert engine moves to Lichess SAN/uci notation
- Handle challenge acceptance and rating-limited play

Adjust the example to the actual module and function names present in this repository.

## Usage examples

- Play locally against the engine using a GUI that supports UCI.
- Run the lichess bot script to connect with your token and play rated/unrated games.
- Import engine functions in your own projects for automated analyses.

## Contributing

Contributions are open — everyone is welcome!

How to contribute:
1. Open an issue to discuss major changes or proposals.
2. Fork the repository and create a feature branch:
   - git checkout -b feature/your-feature
3. Make your changes, add tests where appropriate.
4. Open a pull request describing your changes.

Guidelines:
- Write clear commit messages and PR descriptions.
- Add or update tests for new functionality.
- Follow existing code style; use linters if present.
- Be respectful and constructive in code reviews.

If you'd like help picking an issue to work on, open an issue or mention "help wanted" and maintainers will assist.

## Roadmap / Ideas

- Improve evaluation function
- Add opening book support
- Implement multi-threaded search
- Better Lichess bot features: auto-accept challenges, configurable time-controls, tournament play

## License

This project is provided under the MIT License — see LICENSE for details. If a LICENSE file is not present, please contact the repository owner to confirm the intended license before contributing.

## Getting help

- Open an issue on GitHub for bugs, feature requests, or questions.
- Use pull requests for code changes.

Happy hacking and good luck on the boards!
