"""
embedding_profile.py

Text formats and defaults for the embedding models used by these scripts
(D-061). The topic catalog and the seeds use Qwen3-Embedding-4B (dedicated
Ollama `ollama-qwen`, port 11436). `hs_products` is the only thing left on
Nomic (768 dims, shared Ollama on port 11434): the HS scripts pin the Nomic
constants below, keep their own `search_query:` / `search_document:` prefixes
and ignore OLLAMA_MODEL.
"""
from __future__ import annotations

QWEN_MODEL = "qwen3-embedding:4b"
QWEN_URL = "http://localhost:11436"

# Used only by the HS scripts (load_hs_products.py, poc_hs_candidates.py).
NOMIC_MODEL = "nomic-embed-text-v2-moe"
NOMIC_URL = "http://localhost:11434"

# Same task text as quyca (quyca/infrastructure/ollama.py) and the B19-B21 tests.
QUERY_TASK = "Given a scientific topic description, retrieve relevant research topics"


def require_qwen(model: str) -> None:
    """The topic catalog is Qwen-only; stop instead of mixing models."""
    if not model.startswith("qwen3-embedding"):
        raise SystemExit(
            f"ERROR: model '{model}' is not valid for the topic catalog (Qwen3-Embedding only, D-061)."
        )


def format_document(text: str) -> str:
    """Catalog text: Qwen takes documents without any prefix."""
    return text


def format_query(text: str) -> str:
    """Seed text: Qwen takes an instruction."""
    return f"Instruct: {QUERY_TASK}\nQuery:{text}"
