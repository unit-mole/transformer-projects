#!/usr/bin/env python
"""Serve the static web directory locally."""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    web_dir = PROJECT_ROOT / "web"
    os.chdir(web_dir)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", args.port), handler) as server:
        print(f"Serving {web_dir} at http://localhost:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")


if __name__ == "__main__":
    main()
