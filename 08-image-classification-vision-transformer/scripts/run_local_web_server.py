from __future__ import annotations
import argparse
import http.server
import socketserver
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the static web application locally.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    web = Path(__file__).resolve().parents[1] / "web"
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(web), **kw)
    with socketserver.TCPServer(("", args.port), handler) as server:
        print(f"Serving {web} at http://localhost:{args.port}")
        server.serve_forever()

if __name__ == "__main__":
    main()
