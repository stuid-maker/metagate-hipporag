# Data contract

`data/raw/` and `data/processed/` are generated locally and ignored by Git. Every downloaded source file must have its URL, byte size and SHA-256 recorded in `data/manifest.json`. Fixed sample IDs are written once to `data/splits/<dataset>.json`; reruns must load these files rather than resample.

Canonical dataset IDs are `nq_rear`, `musique`, and `2wikimultihopqa`. The source is the official `osunlp/HippoRAG_2` Hugging Face dataset used by the pinned HippoRAG repository.

