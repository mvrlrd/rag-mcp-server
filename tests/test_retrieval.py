from rag_mcp import indexer, retrieval


def test_rrf_merge_rewards_agreement():
    # Документ, высоко ранжированный обоими ретриверами, должен победить.
    r1 = ["a", "b", "c"]
    r2 = ["b", "a", "c"]
    fused = retrieval._rrf_merge([r1, r2], rrf_k=60)
    assert fused["b"] > fused["c"]
    assert fused["a"] > fused["c"]


def test_rrf_merge_uses_rank_not_membership():
    # "a" на первом месте в обоих списках — сумма выше, чем у "b".
    fused = retrieval._rrf_merge([["a", "b"], ["a", "b"]], rrf_k=60)
    assert fused["a"] > fused["b"]


def test_find_relevant_docs_empty_collection(collection):
    assert retrieval.find_relevant_docs("что угодно", collection=collection) == []


def test_find_relevant_docs_returns_ranked_chunks(sample_docs, collection):
    indexer.index_folder(str(sample_docs), collection=collection)
    hits = retrieval.find_relevant_docs("бюджет попугаев Гидра", top_k=3, collection=collection)
    assert hits
    assert len(hits) <= 3
    # Результаты отсортированы по убыванию RRF-скора.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_find_relevant_docs_finds_seeded_fact(sample_docs, collection):
    indexer.index_folder(str(sample_docs), collection=collection)
    hits = retrieval.find_relevant_docs("42 попугая бюджет", top_k=2, collection=collection)
    joined = " ".join(h.document for h in hits)
    assert "попуга" in joined.lower()


def test_find_relevant_docs_chunk_has_source(sample_docs, collection):
    indexer.index_folder(str(sample_docs), collection=collection)
    hits = retrieval.find_relevant_docs("Гидра", top_k=1, collection=collection)
    assert hits[0].source.endswith((".md", ".py"))
