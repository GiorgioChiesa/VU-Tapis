#!/usr/bin/env python3
"""
Static server for the viewers, like ``python -m http.server`` plus HTTP Range requests,
which browsers need to seek inside the montage mp4s. Loopback only: the viewers hold
patient frames.

Example (detached, as the 8765 server):
    setsid nohup python tools/serve_viewer.py --port 8767 \
        --directory /home/gchie/workspace/viewer/eventi_MSTCN_dilate3_ftenc_capchain \
        > /home/gchie/workspace/viewer/server_eventi.log 2>&1 < /dev/null &

``--api`` serves the label-review app (``tools/label_review.{html,js,css}``, at ``/``) over a
directory written by ``tools/export_label_review.py`` and adds two JSON endpoints:
    GET  /api/items    items.json plus "reviews": the effective review of each item id
    POST /api/review   {id, new_class, offset_frames_4fps, skipped} -> one line appended to
                       revisions.jsonl (file lock + fsync); answers the stored line
    setsid nohup python tools/serve_viewer.py --api --port 8768 \
        --directory /home/gchie/workspace/viewer/revisione_etichette \
        > /home/gchie/workspace/viewer/revisione_etichette/server.log 2>&1 < /dev/null &
"""

import argparse
import datetime
import ipaddress
import json
import os
import re
import threading
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHUNK = 1 << 16
TOOLS = Path(__file__).resolve().parent
APP_FILES = {"/": "label_review.html", "/label_review.html": "label_review.html",
             "/label_review.js": "label_review.js", "/label_review.css": "label_review.css"}
MAX_BODY = 1 << 14


class RangeHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        self._remaining = None
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or os.path.isdir(path):
            return super().send_head()
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        size = os.fstat(f.fileno()).st_size
        if not m or m.groups() == ("", ""):
            f.close()
            return super().send_head()  # multi-range or malformed: whole file
        first, last = m.groups()
        if first == "":  # suffix range: the last N bytes
            start, end = max(0, size - int(last)), size - 1
        else:
            start, end = int(first), min(int(last) if last else size - 1, size - 1)
        if start >= size or start > end:
            f.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(os.fstat(f.fileno()).st_mtime))
        self.end_headers()
        f.seek(start)
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        if self._remaining is None:
            return super().copyfile(source, outputfile)
        while self._remaining > 0:
            buf = source.read(min(CHUNK, self._remaining))
            if not buf:
                break
            outputfile.write(buf)
            self._remaining -= len(buf)


class ReviewHandler(RangeHandler):
    """Label-review app: static frames from ``directory``, app files from ``tools/``, JSON api."""

    items = None  # id -> item, set by main()
    meta = None
    lock = threading.Lock()

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean in APP_FILES:
            return str(TOOLS / APP_FILES[clean])
        return super().translate_path(path)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/api/items":
            from build_revised_labels import effective_revisions, read_revisions

            data = dict(self.meta, reviews=effective_revisions(read_revisions(self.revisions_path())))
            return self._json(HTTPStatus.OK, data)
        if self.path.startswith("/api/"):
            return self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
        return super().do_GET()

    def do_POST(self):
        from build_revised_labels import append_revision, make_record

        if self.path.split("?", 1)[0] != "/api/review":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= MAX_BODY:
                raise ValueError("bad body size")
            req = json.loads(self.rfile.read(n))
            item = self.items[req["id"]]
            rec = make_record(item, req.get("new_class", item["true"]), req.get("offset_frames_4fps", 0),
                              bool(req.get("skipped", False)), len(self.meta["names"]),
                              max_offset=(-self.meta["key_index"], self.meta["n_frames"] - 1 - self.meta["key_index"]),
                              timestamp=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        except (KeyError, ValueError, TypeError) as e:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": f"{type(e).__name__}: {e}"})
        with self.lock:  # threads of this server; the file lock covers other processes
            append_revision(self.revisions_path(), rec)
        return self._json(HTTPStatus.OK, rec)

    def revisions_path(self):
        return Path(self.directory) / "revisions.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--api", action="store_true", help="label-review app and its JSON api")
    args = parser.parse_args()
    assert ipaddress.ip_address(args.bind).is_loopback, "the viewers hold patient frames: bind to loopback only"
    cls = RangeHandler
    if args.api:
        cls = ReviewHandler
        cls.meta = json.loads((Path(args.directory) / "items.json").read_text())
        cls.items = {it["id"]: it for it in cls.meta["items"]}
    handler = partial(cls, directory=args.directory)
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"serving {args.directory} on http://{args.bind}:{args.port}", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
