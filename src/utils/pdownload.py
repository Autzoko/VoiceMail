"""Parallel HTTP range downloader (stdlib only) for throttled single connections.

    python src/utils/pdownload.py URL OUT [--workers 16] [--piece-mb 64]

Pieces are written in place into a pre-sized file; finished pieces are recorded in OUT.pieces so an
interrupted download resumes. Falls back to a single stream if the server ignores Range.
"""
import argparse
import os
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "pdownload/1.0"}


def head_size(url):
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        if r.status == 206 and cr:
            return int(cr.split("/")[-1]), True
        return int(r.headers.get("Content-Length", 0)), False


def fetch_piece(url, out, start, end, tries=30):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(out, "r+b") as f:
                assert r.status == 206, r.status
                f.seek(start)
                pos = start
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    pos += len(buf)
                assert pos == end + 1, f"short read {pos} != {end + 1}"
            return
        except Exception as e:  # noqa: BLE001
            print(f"retry piece {start}-{end} ({t}): {e}", file=sys.stderr, flush=True)
            time.sleep(min(60, 2 ** t))
    raise RuntimeError(f"piece {start}-{end} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--piece-mb", type=int, default=64)
    a = ap.parse_args()
    if os.path.exists(a.out + ".done"):
        print("already done", a.out)
        return
    size, ranged = head_size(a.url)
    print(f"{a.url} size={size / 1e9:.2f}GB ranged={ranged}", flush=True)
    if not ranged or size == 0:
        with urllib.request.urlopen(urllib.request.Request(a.url, headers=UA), timeout=120) as r, \
                open(a.out, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
    else:
        if not os.path.exists(a.out) or os.path.getsize(a.out) != size:
            with open(a.out, "wb") as f:
                f.truncate(size)
        log = a.out + ".pieces"
        done = set(open(log).read().split()) if os.path.exists(log) else set()
        step = a.piece_mb << 20
        pieces = [(s, min(s + step, size) - 1) for s in range(0, size, step) if str(s) not in done]
        lock, n0, t0 = threading.Lock(), len(done), time.time()
        total = (size + step - 1) // step

        def job(p):
            fetch_piece(a.url, a.out, *p)
            with lock:
                done.add(str(p[0]))
                with open(log, "a") as f:
                    f.write(f"{p[0]}\n")
                k = len(done)
                if k % 10 == 0 or k == total:
                    rate = (k - n0) * step / max(time.time() - t0, 1e-6) / 1e6
                    print(f"  {k}/{total} pieces, {rate:.1f} MB/s", flush=True)

        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(job, pieces))
        os.remove(log)
    open(a.out + ".done", "w").close()
    print("done", a.out, flush=True)


if __name__ == "__main__":
    main()
