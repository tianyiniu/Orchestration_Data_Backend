"""
Concurrency benchmark for the Flask retrieval server.

Sends a fixed number of /query_search requests at several concurrency levels
and reports throughput plus latency percentiles. The point is to see whether
the server's throughput plateaus (i.e. the GPU encoder / FAISS search is the
bottleneck) as concurrency rises, or scales roughly linearly.

Reads [server] / [test_server] from config.toml for the base URL, same as
scripts/test_server.py.

Usage:
    python benchmark.py
    python benchmark.py --requests 200 --concurrency 1,2,4,8,16
    python benchmark.py --endpoint url_search
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List

import requests


QUERIES = [
    "ordinary differential equations",
    "who built the Brooklyn Bridge",
    "history of the Roman Empire",
    "photosynthesis in C4 plants",
    "transformer neural network architecture",
    "the French Revolution causes",
    "Mount Everest first ascent",
    "quantum entanglement experiments",
    "Apollo 11 moon landing",
    "Shakespeare's tragedies overview",
    "Black holes event horizon",
    "Industrial Revolution in Britain",
    "Maya civilization collapse",
    "Pacific Ocean trenches",
    "Mariana Trench depth",
    "Renaissance Italian painters",
    "DNA replication mechanism",
    "climate change carbon cycle",
    "Cold War proxy conflicts",
    "ancient Mesopotamian writing",
]

URLS = [
    "https://en.wikipedia.org/wiki/List%20of%20parks%20in%20New%20York%20City",
]


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_config() -> dict:
    with open(os.path.join(repo_root(), "config.toml"), "rb") as f:
        return tomllib.load(f)


def base_url(config: dict) -> str:
    server_config = config.get("server", {})
    test_config = config.get("test_server", {})
    if test_config.get("base_url"):
        return test_config["base_url"]
    host = server_config.get("client_host", "localhost")
    port = int(server_config.get("port", 8000))
    return f"http://{host}:{port}"


@dataclass
class Result:
    ok: bool
    elapsed_ms: float
    status: int
    error: str = ""


def make_query_call(base: str, top_n: int, timeout: float) -> Callable[[int], Result]:
    session_local = requests.Session()

    def call(i: int) -> Result:
        query = QUERIES[i % len(QUERIES)]
        t0 = time.perf_counter()
        try:
            r = session_local.post(
                f"{base}/query_search",
                json={"query": query, "n": top_n},
                timeout=timeout,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            ok = r.ok
            return Result(ok=ok, elapsed_ms=elapsed_ms, status=r.status_code,
                          error="" if ok else r.text[:200])
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return Result(ok=False, elapsed_ms=elapsed_ms, status=0, error=repr(e))

    return call


def make_url_call(base: str, timeout: float) -> Callable[[int], Result]:
    session_local = requests.Session()

    def call(i: int) -> Result:
        url = URLS[i % len(URLS)]
        t0 = time.perf_counter()
        try:
            r = session_local.post(
                f"{base}/url_search",
                json={"url": url},
                timeout=timeout,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            # 404 is a "successful" response from the server's POV
            ok = r.status_code in (200, 404)
            return Result(ok=ok, elapsed_ms=elapsed_ms, status=r.status_code,
                          error="" if ok else r.text[:200])
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return Result(ok=False, elapsed_ms=elapsed_ms, status=0, error=repr(e))

    return call


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def run_level(
    call: Callable[[int], Result],
    n_requests: int,
    concurrency: int,
) -> None:
    results: List[Result] = []
    wall_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(call, i) for i in range(n_requests)]
        for f in as_completed(futures):
            results.append(f.result())
    wall = time.perf_counter() - wall_t0

    ok_results = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    latencies = sorted(r.elapsed_ms for r in ok_results)

    throughput = len(ok_results) / wall if wall > 0 else float("nan")
    mean_ms = statistics.fmean(latencies) if latencies else float("nan")
    p50 = percentile(latencies, 0.50)
    p90 = percentile(latencies, 0.90)
    p99 = percentile(latencies, 0.99)
    mn = latencies[0] if latencies else float("nan")
    mx = latencies[-1] if latencies else float("nan")

    print(
        f"conc={concurrency:>3}  "
        f"req={n_requests:>4}  "
        f"ok={len(ok_results):>4}  "
        f"err={len(errors):>3}  "
        f"wall={wall:6.2f}s  "
        f"thr={throughput:6.1f} req/s  "
        f"mean={mean_ms:7.1f}ms  "
        f"p50={p50:7.1f}  p90={p90:7.1f}  p99={p99:7.1f}  "
        f"min={mn:6.1f}  max={mx:7.1f}"
    )

    if errors:
        sample = errors[0]
        print(f"     first error: status={sample.status} {sample.error}")


def warmup(call: Callable[[int], Result], n: int = 3) -> None:
    print(f"warmup: {n} serial requests...")
    for i in range(n):
        r = call(i)
        if not r.ok:
            print(f"  warmup error: status={r.status} {r.error}")
            sys.exit(1)
        print(f"  [{r.elapsed_ms:7.1f} ms] ok")


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    config = load_config()
    test_config = config.get("test_server", {})
    server_config = config.get("server", {})
    default_top_n = int(server_config.get("default_top_n", 10))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        choices=["query_search", "url_search"],
        default="query_search",
    )
    parser.add_argument("--requests", type=int, default=64,
                        help="requests per concurrency level")
    parser.add_argument("--concurrency", type=parse_int_list,
                        default=[1, 2, 4, 8, 16, 32],
                        help="comma-separated concurrency levels")
    parser.add_argument("--top-n", type=int, default=default_top_n)
    parser.add_argument("--timeout", type=float,
                        default=float(test_config.get("timeout_seconds", 60)))
    parser.add_argument("--base-url", default=None,
                        help="override config base url")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    base = args.base_url or base_url(config)
    print(f"target: {base}  endpoint: /{args.endpoint}")

    try:
        r = requests.get(f"{base}/healthz", timeout=args.timeout)
        r.raise_for_status()
        print(f"health: {r.json()}")
    except Exception as e:
        print(f"server not reachable at {base}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.endpoint == "query_search":
        call = make_query_call(base, args.top_n, args.timeout)
    else:
        call = make_url_call(base, args.timeout)

    if not args.no_warmup:
        warmup(call, n=3)

    print()
    print(f"running benchmark: {args.requests} requests per level, "
          f"concurrency={args.concurrency}")
    print()

    for c in args.concurrency:
        run_level(call, args.requests, c)


if __name__ == "__main__":
    main()
