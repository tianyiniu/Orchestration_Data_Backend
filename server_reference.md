# Flask Retrieval Server Reference

This server exposes a local retrieval API over a FAISS embedding index and a
SQLite document store. It is configured by `config.toml`.

Start it with:

```bash
python scripts/app.py
```

Default server settings from `config.toml`:

```text
Host: 0.0.0.0
Port: 8000
Client URL: http://localhost:8000
```

## GET `/healthz`

Checks whether the server is running and reports loaded corpus paths.

Example:

```bash
curl http://localhost:8000/healthz
```

Successful response:

```json
{
  "ok": true,
  "n_docs": 123456,
  "lead_jsonl": "/path/to/leads.jsonl",
  "documents_db": "/path/to/documents.sqlite"
}
```

## POST `/query_search`

Runs semantic search over indexed lead/snippet text.

Request body:

```json
{
  "query": "ordinary differential equations",
  "n": 3
}
```

`query` is required. `n` is optional; if omitted, the server uses
`default_top_n` from `config.toml`. Results are capped by `max_top_n`.

Example:

```bash
curl -X POST http://localhost:8000/query_search \
  -H "Content-Type: application/json" \
  -d '{"query":"ordinary differential equations","n":3}'
```

Successful response:

```json
{
  "hits": [
    {
      "id": "wiki+123",
      "source": "wiki",
      "docid": "456789",
      "title": "Example title",
      "url": "https://example.com/page",
      "snippet": "Indexed lead/snippet text...",
      "score": 0.8123
    }
  ]
}
```

Bad request responses:

```json
{"error": "missing or invalid 'query'"}
```

```json
{"error": "missing or invalid 'n'"}
```

## POST `/url_search`

Looks up a full document by exact URL in the SQLite database.

Request body:

```json
{
  "url": "https://en.wikipedia.org/wiki/List%20of%20parks%20in%20New%20York%20City"
}
```

Example:

```bash
curl -X POST http://localhost:8000/url_search \
  -H "Content-Type: application/json" \
  -d '{"url":"https://en.wikipedia.org/wiki/List%20of%20parks%20in%20New%20York%20City"}'
```

Successful response:

```json
{
  "id": "wiki+123",
  "source": "wiki",
  "docid": "456789",
  "title": "List of parks in New York City",
  "url": "https://en.wikipedia.org/wiki/List%20of%20parks%20in%20New%20York%20City",
  "text": "Full document text..."
}
```

If the URL is not found, the server returns HTTP `404`:

```json
{
  "url": "https://example.com/missing",
  "text": null
}
```

Bad request response:

```json
{"error": "missing or invalid 'url'"}
```

## Testing client

There is also a test client:

```bash
python scripts/test_server.py
```

It reads `[test_server]` settings from `config.toml` and exercises `/healthz`,
`/query_search`, and `/url_search`.
