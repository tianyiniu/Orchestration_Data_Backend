import glob
import json
import pickle
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

import faiss
import numpy as np

from .encoder import HarrierEncoder

Source = Literal["bcp", "wiki"]


@dataclass
class SearchResult:
    id: str
    docid: str
    source: Source
    title: str
    url: str
    score: float
    snippet: str
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Retriever:
    """Local FAISS retriever over cached leads with SQLite full-text lookup."""

    def __init__(
        self,
        index_glob: str,
        lead_jsonl: str,
        documents_db: str,
        encoder: Optional[HarrierEncoder] = None,
        model_cache_dir: Optional[str] = None,
    ):
        self.encoder = encoder or HarrierEncoder(cache_dir=model_cache_dir)
        self.documents_db = documents_db

        self._load_index(index_glob)
        self._load_leads(lead_jsonl)
        self._validate()

    def _load_index(self, index_glob: str) -> None:
        shards = sorted(glob.glob(index_glob))
        if not shards:
            raise FileNotFoundError(f"No FAISS shards matched: {index_glob}")
        all_reps: List[np.ndarray] = []
        self.index_ids: List[str] = []
        for path in shards:
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            all_reps.append(np.asarray(reps, dtype=np.float32))
            self.index_ids.extend(lookup)
        reps = np.vstack(all_reps)
        self.index = faiss.IndexFlatIP(reps.shape[1])
        self.index.add(reps)

    def _load_leads(self, lead_jsonl: str) -> None:
        self.leads: Dict[str, dict] = {}
        with open(lead_jsonl) as f:
            for line in f:
                row = json.loads(line)
                self.leads[row["id"]] = row

    def _validate(self) -> None:
        missing = [d for d in self.index_ids[:64] if d not in self.leads]
        if missing:
            raise RuntimeError(
                f"FAISS index references ids absent from lead JSONL: {missing[:5]}..."
            )

    def search(
        self,
        query: str,
        k: int = 10,
        include_full: bool = True,
    ) -> List[SearchResult]:
        q = self.encoder.encode_query(query)
        scores, indices = self.index.search(q, k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            lead_id = self.index_ids[idx]
            lead = self.leads[lead_id]
            full = (
                self.get_document(lead["source"], lead["docid"]) if include_full else None
            )
            results.append(
                SearchResult(
                    id=lead_id,
                    docid=lead["docid"],
                    source=lead["source"],
                    title=lead.get("title") or "",
                    url=lead.get("url") or "",
                    score=float(score),
                    snippet=lead.get("text") or "",
                    text=full,
                )
            )
        return results

    def get_document(self, source: Source, docid: str) -> Optional[str]:
        with sqlite3.connect(self.documents_db) as conn:
            row = conn.execute(
                """
                SELECT text
                FROM documents
                WHERE source = ? AND docid = ?
                """,
                (source, docid),
            ).fetchone()
        if row is None:
            return None
        return row[0]

    def get_document_by_url(self, url: str) -> Optional[dict]:
        with sqlite3.connect(self.documents_db) as conn:
            conn.row_factory = sqlite3.Row
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


HybridRetriever = Retriever
