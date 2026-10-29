"""Quick static server for demo without backend dependencies."""
import http.server, os, webbrowser, threading

PORT = 8080
DIR = os.path.join(os.path.dirname(__file__), "web")

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw): super().__init__(*a, directory=DIR, **kw)
    def log_message(self, *a): pass

if __name__ == "__main__":
    srv = http.server.HTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}"
    print(f"CryptoGuard AI dashboard: {url}")
    print("Press Ctrl+C to stop.")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\nStopped."); srv.server_close()
