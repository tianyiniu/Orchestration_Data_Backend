"""
Test client for the Flask retrieval server.

Runtime configuration is read from the repository-level "config.toml". The client uses "[test_server].base_url" when provided; otherwise it builds the URL from "[server].client_host" and "[server].port". Request timeout, query result count, sample query, and sample URL all come from "[test_server]", with "query_top_n" falling back to "[server].default_top_n".

The client always checks "GET /healthz" first. It then exercises "POST /query_search" only when "sample_query" is configured, and "POST /url_search" only when "sample_url" is configured. CUDA settings are owned by the running server process, not by this client.
"""

import os
import sys
import time
import tomllib
from textwrap import shorten

import requests


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def health(base: str, timeout: float) -> None:
    r = requests.get(f"{base}/healthz", timeout=timeout)
    r.raise_for_status()
    print(f"[health] {r.json()}")


def query_search(base: str, query: str, n: int, timeout: float) -> None:
    t0 = time.monotonic()
    r = requests.post(
        f"{base}/query_search",
        json={"query": query, "n": n},
        timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    hits = r.json()["hits"]
    print(f"\n[{elapsed*1000:.0f} ms] query: {query!r} ({len(hits)} hits)")
    for idx, hit in enumerate(hits, 1):
        print(
            f"{idx:>2}. [{hit['score']:.4f}] "
            f"{hit['id']} ({hit['source']}) {hit['title']}"
        )
        print(f"      url: {hit['url']}")
        print(f"  snippet: {shorten(hit['snippet'].replace(chr(10), ' '), 220)}")


def url_search(base: str, url: str, timeout: float) -> None:
    t0 = time.monotonic()
    r = requests.post(f"{base}/url_search", json={"url": url}, timeout=timeout)
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    doc = r.json()
    print(f"\n[{elapsed*1000:.0f} ms] url: {url!r}")
    print(f"      id: {doc['id']}")
    print(f"   title: {doc['title']}")
    print(f"    text: {shorten(doc['text'].replace(chr(10), ' '), 360)}")


def main() -> None:
    config = load_config()
    test_config = config.get("test_server", {})
    base = base_url(config)
    timeout = float(test_config.get("timeout_seconds", 60))
    n = int(test_config.get("query_top_n", config.get("server", {}).get("default_top_n", 5)))

    try:
        health(base, timeout)
    except Exception as e:
        print(f"server not reachable at {base}: {e}", file=sys.stderr)
        sys.exit(1)

    sample_query = test_config.get("sample_query")
    if sample_query:
        query_search(base, sample_query, n, timeout)

    sample_url = test_config.get("sample_url")
    if sample_url:
        url_search(base, sample_url, timeout)

    if not sample_query and not sample_url:
        print("No sample_query or sample_url configured in [test_server].")


if __name__ == "__main__":
    main()
