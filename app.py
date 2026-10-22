"""Loopback-only HTTP transport; all model mathematics live in network.py."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback
from urllib.parse import urlsplit

from network import train

STATIC = Path(__file__).resolve().parent / "static"
TRAINING_LOCK = threading.Lock()
MAX_BODY = 8192


class Handler(BaseHTTPRequestHandler):
    # Keep request framing simple: one response closes its HTTP/1.0 connection.
    server_version = "NumPyLab/1.0"

    def respond(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, dict):
            body = json.dumps(body, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; img-src 'self'; object-src 'none'; "
                         "base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(body)

    def allowed_request(self):
        """Reject cross-origin requests and unexpected Hosts (DNS rebinding)."""
        port = self.server.server_address[1]
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in allowed_hosts:
            self.respond(403, {"error": "Use the printed localhost URL."})
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{host}" for host in allowed_hosts}:
            self.respond(403, {"error": "Cross-origin requests are not allowed."})
            return False
        return True

    def do_GET(self):
        if not self.allowed_request():
            return
        # A whitelist intentionally prevents user-controlled filesystem traversal.
        routes = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        path = urlsplit(self.path).path
        if path == "/api/health":
            self.respond(200, {"status": "ok"})
        elif path in routes:
            filename, content_type = routes[path]
            try:
                self.respond(200, (STATIC / filename).read_bytes(), content_type)
            except OSError:
                self.respond(500, {"error": "An application asset is missing. Restore the static folder."})
        else:
            self.respond(404, {"error": "Route not found."})

    def do_POST(self):
        if not self.allowed_request():
            return
        if self.path != "/api/train":
            self.respond(404, {"error": "Route not found."})
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self.respond(400, {"error": "Transfer-Encoding is unsupported."})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.respond(415, {"error": "Content-Type must be application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.respond(400, {"error": "Invalid Content-Length."})
            return
        if not 0 < length <= MAX_BODY:
            self.respond(413, {"error": "Request body must be 1–8192 bytes."})
            return
        # One fit at a time avoids competing CPU-heavy requests. No worker queue.
        if not TRAINING_LOCK.acquire(blocking=False):
            self.respond(429, {"error": "A training run is already in progress. Try again shortly."})
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request body.")
            payload = json.loads(raw.decode("utf-8"))
            result = train(payload)
            status, response = 200, result
        except (ValueError, UnicodeDecodeError, RecursionError) as error:
            status, response = 400, {"error": str(error) or "Invalid JSON settings."}
        except (BrokenPipeError, ConnectionResetError):
            # A closed tab should not turn a completed experiment into a traceback.
            return
        except Exception:
            traceback.print_exc()
            status, response = 500, {"error": "Unexpected training failure. Check the server terminal."}
        finally:
            TRAINING_LOCK.release()
        # A client can start its next request as soon as it reads this response.
        # Release the completed fit's gate first, including on invalid payloads.
        try:
            self.respond(status, response)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def setup(self):
        super().setup()
        # Slow/incomplete local requests must not retain a thread indefinitely.
        self.connection.settimeout(15)


def main():
    parser = argparse.ArgumentParser(description="Offline NumPy neural-network tutorial")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.exit(1, f"Could not start local server: {error}\nTry --port 8001.\n")
    print(f"NumPy Neural Lab: http://127.0.0.1:{args.port}", flush=True)
    print("Offline, loopback only. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
