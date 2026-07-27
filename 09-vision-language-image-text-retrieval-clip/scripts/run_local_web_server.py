from __future__ import annotations

import argparse
import http.server
import socketserver
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the static CLIP web application locally.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(PROJECT_ROOT / "web"))
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Serving {PROJECT_ROOT / 'web'} at http://localhost:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
