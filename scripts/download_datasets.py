"""
Materialize configured source datasets into the local retrieval corpus.

Runtime configuration is read from the repository-level "config.toml". The "[corpus]" section controls the output directory, whether BrowseComp-Plus and/or English Wikipedia are downloaded, the combined lead JSONL path, and how many characters from each document are retained for retrieval leads.

This script writes two server/build inputs under "[corpus].dataset_build_dir": the combined "leads.jsonl" file used by "build_corpus.py" for embedding, and the "documents.sqlite" store used by "app.py" for exact URL lookups. Relative corpus paths are resolved under "dataset_build_dir". BrowseComp-Plus is loaded from "Tevatron/browsecomp-plus-corpus", filtered with a lightweight English heuristic, and assigned generated article URLs. Wikipedia is downloaded from "wikimedia/wikipedia" using the "20231101.en" parquet shards.
"""

import os
import re
import sys
import json
import string
import sqlite3
import tomllib
import glob
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import snapshot_download



ENGLISH_STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "on",
    "with", "as", "was", "at", "by", "from", "it", "this", "be"
}

def looks_english(text: str) -> bool:
    sample = text[:2000].lower()

    letters = [c for c in sample if c.isalpha()]
    if not letters:
        return False

    ascii_letters = [c for c in letters if c in string.ascii_lowercase]
    ascii_ratio = len(ascii_letters) / len(letters)

    words = re.findall(r"[a-z]+", sample)
    if not words:
        return False

    stopword_hits = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    stopword_ratio = stopword_hits / len(words)

    return ascii_ratio > 0.85 and stopword_ratio > 0.03


def extract_title(text: str) -> str:
    match = re.search(r"title:\s*(.*?)\s*\[Archives:", text)
    if match:
        return match.group(1).strip()

    fallback = text.removeprefix("---\ntitle:").strip()
    return " ".join(fallback.split()[:3])


def make_url(article_title: str) -> str: 
    """Mock a url for BCP article"""
    url = "_".join(article_title.split(" "))
    return url


def make_filename(name: str, max_chars: int = 180) -> str:
    """Convert a title/url-like string into a filesystem-safe filename stem."""
    name = re.sub(r"[^\w.-]+", "_", name)
    name = name.strip("._")
    return name[:max_chars] or "untitled"


def init_documents_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            source TEXT NOT NULL,
            docid TEXT NOT NULL,
            id TEXT NOT NULL,
            url TEXT,
            title TEXT,
            text TEXT NOT NULL,
            PRIMARY KEY (source, docid)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_url
        ON documents(url)
        """
    )
    return conn


def lead_jsonl_path(config):
    dataset_build_dir = config["corpus"]["dataset_build_dir"]
    path = config["corpus"].get("lead_jsonl", "leads.jsonl")
    if os.path.isabs(path):
        return path
    return os.path.join(dataset_build_dir, path)


def download_bcp(config):
    dataset_build_dir = config["corpus"]["dataset_build_dir"]
    num_lead_chars = config["corpus"]["lead_chars"]
    dataset = "Tevatron/browsecomp-plus-corpus"
    source_name = "bcp"
    lead_path = lead_jsonl_path(config)
    db_path = os.path.join(dataset_build_dir, "documents.sqlite")
    os.makedirs(dataset_build_dir, exist_ok=True)

    ds = load_dataset(dataset, split="train")

    conn = init_documents_db(db_path)
    with open(lead_path, "a") as f:
        for idx, row in enumerate(tqdm(ds, desc="Writing BCP full corpus")):

            text = row["text"]
            if not looks_english(text):
                continue
        
            title = extract_title(row["text"])
            url = make_url(title)
            docid = row["docid"]
            article_id = f"{source_name}+{idx}"

            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (source, docid, id, url, title, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_name, docid, article_id, url, title, text),
            )

            f.write(
                json.dumps({
                    "id": article_id,
                    "docid": docid,
                    "source": source_name,
                    "url": url,
                    "title": title,
                    "text": text[:num_lead_chars]}, ensure_ascii=False)
                + "\n"
            )
    conn.commit()
    conn.close()
    print(f"Wrote {len(ds)} BrowseComp-Plus docs at {dataset_build_dir}")


def download_wikipedia(config):
    dataset_build_dir = config["corpus"]["dataset_build_dir"]
    num_lead_chars = config["corpus"]["lead_chars"]
    dataset = "wikimedia/wikipedia"
    config_name = "20231101.en"
    source_name = "wiki"
    lead_path = lead_jsonl_path(config)
    db_path = os.path.join(dataset_build_dir, "documents.sqlite")
    output_dir = os.path.join(dataset_build_dir, "wikipedia/raw")

    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        allow_patterns=[f"{config_name}/*.parquet"],
        local_dir=output_dir,
    )
    print(f"Downloaded {config_name} parquet shards -> {output_dir}")

    conn = init_documents_db(db_path)
    shards = sorted(glob.glob(os.path.join(output_dir, "**/*.parquet"), recursive=True))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {output_dir}")

    idx = 0
    with open(lead_path, "a") as f:
        for shard in tqdm(shards, desc="Writing Wikipedia shards"):
            table = pq.read_table(shard, columns=["id", "url", "title", "text"])
            ids = table.column("id").to_pylist()
            urls = table.column("url").to_pylist()
            titles = table.column("title").to_pylist()
            texts = table.column("text").to_pylist()

            for docid, url, title, text in zip(ids, urls, titles, texts):
                if not text:
                    continue

                article_id = f"{source_name}+{idx}"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents
                    (source, docid, id, url, title, text)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (source_name, docid, article_id, url, title, text),
                )

                f.write(
                    json.dumps({
                        "id": article_id,
                        "docid": docid,
                        "source": source_name,
                        "url": url,
                        "title": title,
                        "text": f"{title}\n\n{text[:num_lead_chars]}"}, ensure_ascii=False)
                    + "\n"
                )
                idx += 1
            conn.commit()
    conn.close()
    print(f"Wrote {idx} Wikipedia docs at {dataset_build_dir}")


if __name__ == "__main__":
    toml_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(toml_path+"/config.toml", "rb") as f:
        config = tomllib.load(f)

    dataset_build_dir = config["corpus"]["dataset_build_dir"]
    os.makedirs(dataset_build_dir, exist_ok=True)
    lead_path = lead_jsonl_path(config)
    open(lead_path, "w").close()

    if config["corpus"]["download_bcp"]:
        print("--- Downloading BCP ---")
        download_bcp(config)

    if config["corpus"]["download_wiki"]:
        print("--- Downloading Wikipedia ---")
        download_wikipedia(config)
