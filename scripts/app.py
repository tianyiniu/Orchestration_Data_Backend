"""
Flask backend for semantic query search and exact URL document lookup.

Runtime configuration is read from the repository-level "config.toml". The server uses "[corpus]" paths for "leads.jsonl" and "documents.sqlite", "[build]" paths for FAISS-compatible embedding shards, "[model]" settings for the Harrier query encoder, and "[server]" settings for bind address, result limits, and CUDA visibility. "[server].cuda_devices" is applied by setting "CUDA_VISIBLE_DEVICES" before the Harrier encoder is imported or initialized.

Concurrent /query_search calls are merged server-side by a BatchScheduler:
a single background thread owns the encoder and the FAISS index, draining up
to "[server].batch_max_size" pending requests within "[server].batch_max_wait_ms"
and serving them with one GPU encode and one FAISS search. The FAISS index is
moved to GPU when "[server].faiss_gpu" is true (default) and faiss-gpu is
available; otherwise it stays on CPU.

Exposed endpoints:
    GET  /healthz         report server readiness and loaded corpus paths
    POST /query_search    body: {"query": str, "n": int}
    POST /url_search      body: {"url": str}

NOTE: THIS SERVER WILL LIKELY TAKE >2 MIN TO START.
Terminal output will hang after finishing loading model weights.

For real load, prefer Gunicorn over app.run:
    gunicorn -w 1 --threads 32 --timeout 600 -b 0.0.0.0:7470 'scripts.app:build_app()'
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import queue
import sqlite3
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from flask import Flask, jsonify, request

DEFAULT_MODEL = "microsoft/harrier-oss-v1-0.6b"


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config() -> dict:
    with open(os.path.join(repo_root(), "config.toml"), "rb") as f:
        return tomllib.load(f)


def apply_cuda_visibility(config: dict) -> None:
    cuda_devices = config.get("server", {}).get("cuda_devices")
    if cuda_devices is None:
        return
    if isinstance(cuda_devices, (str, int)):
        cuda_devices = [cuda_devices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in cuda_devices)


def corpus_path(config: dict, key: str, default: str) -> str:
    data_dir = config["corpus"]["dataset_build_dir"]
    path = config["corpus"].get(key, default)
    if os.path.isabs(path):
        return path
    return os.path.join(data_dir, path)


def build_path(config: dict, key: str, default: str) -> str:
    data_dir = config["corpus"]["dataset_build_dir"]
    path = config.get("build", {}).get(key, default)
    if os.path.isabs(path):
        return path
    return os.path.join(data_dir, path)


@dataclass
class Lead:
    id: str
    source: str
    docid: str
    title: str
    url: str
    snippet: str


class SearchStore:
    def __init__(self, config: dict):
        server_config = config.get("server", {})
        model_config = config.get("model", {})

        self.lead_jsonl = corpus_path(config, "lead_jsonl", "leads.jsonl")
        self.db_path = corpus_path(config, "documents_db", "documents.sqlite")
        self.index_glob = os.path.join(
            build_path(config, "shard_dir", "indexes/harrier-oss-v1-0.6b"),
            "corpus.*.pkl",
        )
        self.default_top_n = int(server_config.get("default_top_n", 10))
        self.max_top_n = int(server_config.get("max_top_n", 50))
        self.max_query_length = int(server_config.get("max_query_length", 512))
        self.faiss_gpu = bool(server_config.get("faiss_gpu", True))

        sys.path.insert(0, repo_root())
        from retrievers.encoder import HarrierEncoder

        self.encoder = HarrierEncoder(
            model_name=model_config.get("name", DEFAULT_MODEL),
            cache_dir=model_config.get("cache_dir"),
            max_length=self.max_query_length,
        )

        self._load_leads()
        self._load_index()
        self._validate()

    def _load_leads(self) -> None:
        self.leads: Dict[str, Lead] = {}
        with open(self.lead_jsonl) as f:
            for line in f:
                row = json.loads(line)
                self.leads[row["id"]] = Lead(
                    id=row["id"],
                    source=row["source"],
                    docid=row["docid"],
                    title=row.get("title") or "",
                    url=row.get("url") or "",
                    snippet=row.get("text") or "",
                )

    def _load_index(self) -> None:
        shards = sorted(glob.glob(self.index_glob))
        if not shards:
            raise FileNotFoundError(f"No embedding shards matched: {self.index_glob}")

        all_reps: List[np.ndarray] = []
        self.index_ids: List[str] = []
        for path in shards:
            with open(path, "rb") as f:
                reps, ids = pickle.load(f)
            all_reps.append(np.asarray(reps, dtype=np.float32))
            self.index_ids.extend(ids)

        reps = np.vstack(all_reps)
        cpu_index = faiss.IndexFlatIP(reps.shape[1])
        cpu_index.add(reps)
        self.index = self._maybe_to_gpu(cpu_index)

    def _maybe_to_gpu(self, cpu_index: faiss.Index) -> faiss.Index:
        if not self.faiss_gpu:
            self.index_device = "cpu"
            return cpu_index
        if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() == 0:
            print(
                "[warn] faiss_gpu=true in config but faiss-gpu is not available; "
                "falling back to CPU index. Install faiss-gpu-cu12 to enable.",
                file=sys.stderr,
            )
            self.index_device = "cpu"
            return cpu_index
        self._gpu_resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
        self.index_device = "gpu:0"
        return gpu_index

    def _validate(self) -> None:
        missing = [doc_id for doc_id in self.index_ids[:64] if doc_id not in self.leads]
        if missing:
            raise RuntimeError(
                f"Embedding index references ids absent from leads: {missing[:5]}..."
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def search(self, query: str, n: Optional[int] = None) -> List[dict]:
        top_n = self.default_top_n if n is None else n
        top_n = max(1, min(int(top_n), self.max_top_n))

        q = self.encoder.encode_query(query)
        scores, indices = self.index.search(q, top_n)

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            lead = self.leads[self.index_ids[idx]]
            hits.append(
                {
                    "id": lead.id,
                    "source": lead.source,
                    "docid": lead.docid,
                    "title": lead.title,
                    "url": lead.url,
                    "snippet": lead.snippet,
                    "score": float(score),
                }
            )
        return hits

    def get_by_url(self, url: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, source, docid, title, url, text
                FROM documents
                WHERE url = ?
                """,
                (url,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def search_batch(self, queries: List[str], top_ns: List[int]) -> List[List[dict]]:
        assert len(queries) == len(top_ns)
        clamped = [max(1, min(int(n), self.max_top_n)) for n in top_ns]
        max_n = max(clamped)

        embs = self.encoder.encode_queries(queries)
        scores, indices = self.index.search(embs, max_n)

        results: List[List[dict]] = []
        for i, n in enumerate(clamped):
            hits = []
            for score, idx in zip(scores[i][:n], indices[i][:n]):
                if idx < 0:
                    continue
                lead = self.leads[self.index_ids[idx]]
                hits.append(
                    {
                        "id": lead.id,
                        "source": lead.source,
                        "docid": lead.docid,
                        "title": lead.title,
                        "url": lead.url,
                        "snippet": lead.snippet,
                        "score": float(score),
                    }
                )
            results.append(hits)
        return results


@dataclass
class _PendingRequest:
    query: str
    n: int
    event: threading.Event
    result: Optional[List[dict]] = None
    error: Optional[BaseException] = None


class BatchScheduler:
    """
    Collect concurrent /query_search calls into one GPU encode + one FAISS
    search. A single background thread owns the model and the index, so
    request handler threads never contend on them.
    """

    def __init__(
        self,
        store: SearchStore,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
    ):
        self.store = store
        self.max_batch_size = int(max_batch_size)
        self.max_wait_s = float(max_wait_ms) / 1000.0
        self._queue: "queue.Queue[_PendingRequest]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="batch-scheduler", daemon=True
        )
        self._thread.start()

    def submit(self, query: str, n: Optional[int]) -> List[dict]:
        req_n = self.store.default_top_n if n is None else int(n)
        req = _PendingRequest(query=query, n=req_n, event=threading.Event())
        self._queue.put(req)
        req.event.wait()
        if req.error is not None:
            raise req.error
        return req.result or []

    def stop(self) -> None:
        self._stop.set()

    def _drain_batch(self) -> List[_PendingRequest]:
        try:
            first = self._queue.get(timeout=0.1)
        except queue.Empty:
            return []
        batch = [first]
        deadline = time.monotonic() + self.max_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._drain_batch()
            if not batch:
                continue
            try:
                results = self.store.search_batch(
                    [r.query for r in batch],
                    [r.n for r in batch],
                )
                for req, hits in zip(batch, results):
                    req.result = hits
                    req.event.set()
            except BaseException as e:
                for req in batch:
                    req.error = e
                    req.event.set()


def create_app(store: SearchStore, scheduler: BatchScheduler) -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify(
            {
                "ok": True,
                "n_docs": len(store.index_ids),
                "lead_jsonl": store.lead_jsonl,
                "documents_db": store.db_path,
                "index_device": getattr(store, "index_device", "cpu"),
                "batch_max_size": scheduler.max_batch_size,
                "batch_max_wait_ms": scheduler.max_wait_s * 1000.0,
            }
        )

    @app.post("/query_search")
    def query_search() -> Any:
        body: Dict[str, Any] = request.get_json(force=True) or {}
        query = body.get("query")
        if not query or not isinstance(query, str):
            return jsonify({"error": "missing or invalid 'query'"}), 400
        try:
            n = None if body.get("n") is None else int(body["n"])
        except (TypeError, ValueError):
            return jsonify({"error": "missing or invalid 'n'"}), 400

        hits = scheduler.submit(query, n)
        return jsonify({"hits": hits})

    @app.post("/url_search")
    def url_search() -> Any:
        body: Dict[str, Any] = request.get_json(force=True) or {}
        url = body.get("url")
        if not url or not isinstance(url, str):
            return jsonify({"error": "missing or invalid 'url'"}), 400

        doc = store.get_by_url(url)
        if doc is None:
            return jsonify({"url": url, "text": None}), 404
        return jsonify(doc)

    return app


def _make_scheduler(config: dict, store: SearchStore) -> BatchScheduler:
    server_config = config.get("server", {})
    return BatchScheduler(
        store=store,
        max_batch_size=int(server_config.get("batch_max_size", 32)),
        max_wait_ms=float(server_config.get("batch_max_wait_ms", 10.0)),
    )


def build_app() -> Flask:
    config = load_config()
    apply_cuda_visibility(config)
    store = SearchStore(config)
    scheduler = _make_scheduler(config, store)
    return create_app(store, scheduler)


def main() -> None:
    config = load_config()
    apply_cuda_visibility(config)
    server_config = config.get("server", {})
    store = SearchStore(config)
    scheduler = _make_scheduler(config, store)
    app = create_app(store, scheduler)
    app.run(
        host=server_config.get("host", "0.0.0.0"),
        port=int(server_config.get("port", 8000)),
        threaded=bool(server_config.get("threaded", True)),
        debug=bool(server_config.get("debug", False)),
    )


if __name__ == "__main__":
    main()
