"""
Flask backend for query search and URL document lookup.

The dataset downloader owns `leads.jsonl` and `documents.sqlite`; the corpus
builder owns embedding shards. This server loads those configured artifacts
and exposes two primary endpoints:

    POST /query_search    body: {"query": str, "n": int}
    POST /url_search      body: {"url": str}
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sqlite3
import sys
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
        self.index = faiss.IndexFlatIP(reps.shape[1])
        self.index.add(reps)

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


def create_app(store: SearchStore) -> Flask:
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

        hits = store.search(query, n=n)
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


def build_app() -> Flask:
    return create_app(SearchStore(load_config()))


def main() -> None:
    config = load_config()
    server_config = config.get("server", {})
    app = create_app(SearchStore(config))
    app.run(
        host=server_config.get("host", "0.0.0.0"),
        port=int(server_config.get("port", 8000)),
        threaded=bool(server_config.get("threaded", True)),
        debug=bool(server_config.get("debug", False)),
    )


if __name__ == "__main__":
    main()
