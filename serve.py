"""Quick static server for demo without backend dependencies."""
from __future__ import annotations

import http.server
import os
import threading
import webbrowser

PORT = 8080
DIR = os.path.join(os.path.dirname(__file__), "web")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def log_message(self, format, *args):  # noqa: A002
        pass


if __name__ == "__main__":
    srv = http.server.HTTPServer(("127.0.0.1", PORT), _QuietHandler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"CryptoGuard AI dashboard: {url}")
    print("Press Ctrl+C to stop.")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        srv.server_close()
