# Local RAG Retriever: BrowseComp-Plus + English Wikipedia

This project builds a fully local retrieval backend over BrowseComp-Plus and
English Wikipedia. It indexes cached lead text with Harrier embeddings and
serves full documents from a local SQLite database.

- **Retriever model**: `microsoft/harrier-oss-v1-0.6b`
- **Vector index**: FAISS `IndexFlatIP` over L2-normalized embeddings
- **Indexed text**: cached lead/snippet text, currently first `lead_chars`
  characters
- **Full text**: local `documents.sqlite`
- **Configuration**: `config.toml`
- **No live Wikipedia API dependency at serving time**

## Pipeline

```text
scripts/download_datasets.py
  -> leads.jsonl
  -> documents.sqlite
  -> wikipedia/raw/*.parquet

scripts/build_corpus.py
  -> indexes/harrier-oss-v1-0.6b/corpus.*.pkl

scripts/app.py
  -> Flask server over FAISS + SQLite
```

All generated corpus artifacts are offloaded under:

```toml
[corpus]
dataset_build_dir = "/nas-ssd2/tianyin4/DataDump"
```

Relative paths in `config.toml` are resolved under `dataset_build_dir`, so:

```toml
lead_jsonl = "leads.jsonl"
documents_db = "documents.sqlite"
shard_dir = "indexes/harrier-oss-v1-0.6b"
log_dir = "logs"
```

become:

```text
/nas-ssd2/tianyin4/DataDump/leads.jsonl
/nas-ssd2/tianyin4/DataDump/documents.sqlite
/nas-ssd2/tianyin4/DataDump/indexes/harrier-oss-v1-0.6b
/nas-ssd2/tianyin4/DataDump/logs
```

## Layout

```text
config.toml
requirements.txt
retrievers/
  encoder.py          # HarrierEncoder wrapper
  retriever.py        # local FAISS + SQLite retriever helper
scripts/
  download_datasets.py
  build_corpus.py
  app.py
  test_server.py
```

Generated data lives outside the project directory:

```text
/nas-ssd2/tianyin4/DataDump/
  leads.jsonl
  documents.sqlite
  wikipedia/raw/...
  indexes/harrier-oss-v1-0.6b/corpus.*.pkl
  logs/encode.rank*.log
```

The Harrier model cache is explicit:

```toml
[model]
cache_dir = "/nas-ssd2/tianyin4/cache/huggingface"
```

## Setup

```bash
pip install -r requirements.txt
```

Review `config.toml` before running. Important sections:

```toml
[corpus]
dataset_build_dir = "/nas-ssd2/tianyin4/DataDump"
download_bcp = false
download_wiki = true
lead_jsonl = "leads.jsonl"
documents_db = "documents.sqlite"
lead_chars = 1000

[model]
name = "microsoft/harrier-oss-v1-0.6b"
cache_dir = "/nas-ssd2/tianyin4/cache/huggingface"
max_length = 1024
batch_size = 1024
dtype = "float16"

[build]
shard_dir = "indexes/harrier-oss-v1-0.6b"
log_dir = "logs"
cuda_devices = [4, 5, 6, 7]

[server]
host = "0.0.0.0"
client_host = "localhost"
port = 8000
threaded = true
default_top_n = 10
max_top_n = 50
max_query_length = 512
```

## End-to-End

```bash
# 1. Download/cache datasets and build local document store.
python scripts/download_datasets.py

# 2. Encode cached leads into FAISS-compatible embedding shards.
python scripts/build_corpus.py

# 3. Start the Flask server.
python scripts/app.py

# 4. Optional smoke test using [test_server] settings in config.toml.
python scripts/test_server.py
```

There are no command-line flags for the main pipeline scripts. Change
`config.toml` instead.

## Dataset Build

`scripts/download_datasets.py` writes:

```text
leads.jsonl
documents.sqlite
wikipedia/raw/...
```

Lead rows use this shape:

```json
{
  "id": "wiki+0",
  "docid": "original dataset id",
  "source": "wiki",
  "url": "https://...",
  "title": "Article title",
  "text": "cached snippet text"
}
```

`id` is the retrieval/index key. `docid` is the original dataset identifier:

```text
BCP:       row["docid"]
Wikipedia: parquet row["id"]
```

Full text is stored in SQLite:

```sql
CREATE TABLE documents (
    source TEXT NOT NULL,
    docid TEXT NOT NULL,
    id TEXT NOT NULL,
    url TEXT,
    title TEXT,
    text TEXT NOT NULL,
    PRIMARY KEY (source, docid)
);
```

The database also has an index on `url` for fast URL lookup:

```sql
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
```

## Embedding Build

`scripts/build_corpus.py` reads `leads.jsonl` and writes:

```text
indexes/harrier-oss-v1-0.6b/corpus.000.pkl
indexes/harrier-oss-v1-0.6b/corpus.001.pkl
...
```

Each shard is:

```python
(reps, ids)
```

where:

```text
reps[i] -> embedding vector
ids[i]  -> lead row id, such as "bcp+0" or "wiki+0"
```

CUDA devices are configured in TOML:

```toml
[build]
cuda_devices = [4, 5, 6, 7]
```

Worker count is inferred from this list. The parent process starts one worker
per device and sets:

```text
CUDA_VISIBLE_DEVICES=<configured device>
ENCODE_RANK=<worker index>
ENCODE_WORLD_SIZE=<number of workers>
```

`ENCODE_RANK` and `ENCODE_WORLD_SIZE` are internal worker coordination values.
They are not user config.

## Server API

Start the server:

```bash
python scripts/app.py
```

Health:

```bash
curl localhost:8000/healthz
```

Query search:

```bash
curl -X POST localhost:8000/query_search \
     -H 'content-type: application/json' \
     -d '{"query":"who built the Brooklyn Bridge","n":5}'
```

Response:

```json
{
  "hits": [
    {
      "id": "wiki+123",
      "source": "wiki",
      "docid": "123456",
      "title": "Brooklyn Bridge",
      "url": "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
      "snippet": "Brooklyn Bridge\n\nThe Brooklyn Bridge is...",
      "score": 0.83
    }
  ]
}
```

URL search:

```bash
curl -X POST localhost:8000/url_search \
     -H 'content-type: application/json' \
     -d '{"url":"https://en.wikipedia.org/wiki/Brooklyn_Bridge"}'
```

Response:

```json
{
  "id": "wiki+123",
  "source": "wiki",
  "docid": "123456",
  "title": "Brooklyn Bridge",
  "url": "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
  "text": "Full local article text..."
}
```

## Testing

`scripts/test_server.py` reads `[test_server]` from `config.toml`:

```toml
[test_server]
base_url = ""
query_top_n = 5
timeout_seconds = 60
sample_query = ""
sample_url = ""
```

If `base_url` is empty, it uses:

```text
http://{server.client_host}:{server.port}
```

Set `sample_query` and/or `sample_url`, then run:

```bash
python scripts/test_server.py
```

## Serving Notes

The Flask development server uses:

```toml
[server]
threaded = true
debug = false
```

## Storage Notes

- Harrier model cache: `/nas-ssd2/tianyin4/cache/huggingface`
- Full document store: `documents.sqlite`
- Retrieval index: `indexes/harrier-oss-v1-0.6b/corpus.*.pkl`
- Raw Wikipedia parquet shards are used to build the SQLite store and leads.
- Project-local `indexes/` and `logs/` are old outputs unless `config.toml`
  points back into the project directory.
