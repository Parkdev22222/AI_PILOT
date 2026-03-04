#!/usr/bin/env python
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from command.commander_db import CommanderCombatDB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    return parser.parse_args()


class Handler(BaseHTTPRequestHandler):
    db = None
    run_id = ""
    index_path = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._serve_index()
            return
        if parsed.path == "/api/live_state":
            self._serve_state()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        return

    def _serve_index(self):
        body = Path(self.index_path).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_state(self):
        payload = self.db.get_live_state(self.run_id)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    args = parse_args()
    Handler.db = CommanderCombatDB(args.db_path)
    Handler.run_id = args.run_id
    Handler.index_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        "web",
        "command_dashboard",
        "index.html",
    )
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Commander web server at http://{args.host}:{args.port} run_id={args.run_id}")
    server.serve_forever()


if __name__ == "__main__":
    main()
