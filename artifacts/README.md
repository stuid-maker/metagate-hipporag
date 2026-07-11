# Generated state

This directory holds resumable, machine-generated state and is not committed:

- `batch/`: OpenAI Batch request files, job IDs, outputs and parse reports.
- `cache/`: schema-validated LLM responses and embeddings.
- `indexes/`: HippoRAG OpenIE files, Parquet vectors and graph pickles.
- `runs/`: one JSONL record per dataset, method and query plus a run manifest.
- `tmp/`: atomic-write staging files.

No API key or authorization header may be written here.

