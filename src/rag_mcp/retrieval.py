"""Гибридный поиск: BM25 (sparse) + вектор (Chroma) → слияние через RRF.

Возвращает ранжированные чанки без вызова LLM. Используется узлом retrieve
в графе Corrective RAG и MCP-инструментом find_relevant_docs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .config import settings
from .indexer import get_collection

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    id: str
    document: str
    source: str
    chunk_index: int
    score: float


def _load_corpus(collection) -> tuple[list[str], list[str], list[dict]]:
    got = collection.get(include=["documents", "metadatas"])
    return got["ids"], got["documents"], got["metadatas"]


def _bm25_ranking(query: str, ids: list[str], docs: list[str], limit: int) -> list[str]:
    bm25 = BM25Okapi([_tokenize(d) for d in docs])
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [ids[i] for i in order[:limit]]


def _vector_ranking(query: str, collection, limit: int) -> list[str]:
    res = collection.query(query_texts=[query], n_results=limit)
    return res["ids"][0] if res["ids"] else []


def _rrf_merge(rankings: list[list[str]], rrf_k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


def find_relevant_docs(
    query: str,
    top_k: int | None = None,
    collection=None,
) -> list[RetrievedChunk]:
    """Найти релевантные чанки гибридным поиском (BM25 + вектор → RRF)."""
    top_k = top_k or settings.top_k
    col = collection or get_collection()

    ids, docs, metas = _load_corpus(col)
    if not docs:
        return []

    candidate_k = settings.candidate_k
    bm25_rank = _bm25_ranking(query, ids, docs, candidate_k)
    vector_rank = _vector_ranking(query, col, candidate_k)

    fused = _rrf_merge([bm25_rank, vector_rank], settings.rrf_k)

    by_id = {i: (d, m) for i, d, m in zip(ids, docs, metas)}
    ranked_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]

    result: list[RetrievedChunk] = []
    for doc_id in ranked_ids:
        doc, meta = by_id[doc_id]
        result.append(
            RetrievedChunk(
                id=doc_id,
                document=doc,
                source=meta.get("source", ""),
                chunk_index=meta.get("chunk_index", 0),
                score=fused[doc_id],
            )
        )
    return result
