"""
dashboard/serve.py
Serve the results dashboard on localhost.

  python dashboard/serve.py                 # http://localhost:8560/dashboard/
  python dashboard/serve.py --port 9000 --no-browser
  python dashboard/serve.py --rebuild       # re-pack the showcase seed (attempt_13, seed 41) first

Opening dashboard/index.html directly from disk also works; the server is
only needed for --rebuild convenience or to share the page on the network.
The project root is served so the dashboard can show the figures in
results/attempt_N/ as well as its own data file. Nothing here needs a GPU.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve the AMS 560 results dashboard")
    ap.add_argument("--port", type=int, default=8560)
    ap.add_argument("--attempt", default=os.path.join(ROOT, "results", "attempt_13"))
    ap.add_argument("--seed", type=int, default=41, help="Showcase seed (use -1 for every seed)")
    ap.add_argument("--rebuild", action="store_true", help="Rebuild the data file first")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    seed = None if args.seed < 0 else args.seed
    name = os.path.basename(os.path.normpath(args.attempt)) + ("" if seed is None else f"_s{seed}")
    data = os.path.join(HERE, "data", name + ".js")
    if args.rebuild or not os.path.exists(data):
        if not os.path.isdir(args.attempt):
            sys.exit(f"No results folder at {args.attempt} and no packed data at {data}")
        import build_data
        build_data.main(["--attempt", args.attempt] + ([] if seed is None else ["--seed", str(seed)]))

    handler = functools.partial(_Handler, directory=ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}/dashboard/"
        print(f"Dashboard: {url}   (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
