from pathlib import Path

import pytest

from rag_mcp import indexer

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "sample_docs"


def test_index_folder_counts_files_and_chunks(sample_docs, collection):
    stats = indexer.index_folder(str(sample_docs), collection=collection)
    # note.md и main.py — валидны; empty.txt пуст, ignore.bin не поддержан
    assert stats["files"] == 2
    assert stats["chunks"] >= 2
    assert collection.count() == stats["chunks"]


def test_chunk_metadata_has_source_and_index(sample_docs, collection):
    indexer.index_folder(str(sample_docs), collection=collection)
    got = collection.get(include=["metadatas"])
    metas = got["metadatas"]
    assert all("source" in m and "chunk_index" in m for m in metas)
    assert any(m["source"].endswith("note.md") for m in metas)


def test_index_folder_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        indexer.index_folder("/no/such/folder/xyz")


def test_split_file_respects_supported_only(sample_docs, collection):
    indexer.index_folder(str(sample_docs), collection=collection)
    got = collection.get(include=["metadatas"])
    sources = {m["source"] for m in got["metadatas"]}
    assert not any(s.endswith(".bin") for s in sources)


def test_content_is_searchable(sample_docs, collection):
    indexer.index_folder(str(sample_docs), collection=collection)
    res = collection.query(query_texts=["Квадратов институт космология"], n_results=1)
    assert res["documents"][0]


def test_sample_docs_index_contains_seeded_fact(collection):
    stats = indexer.index_folder(str(SAMPLE_DOCS), collection=collection)
    assert stats["chunks"] > 0
    got = collection.get(include=["documents"])
    joined = "\n".join(got["documents"])
    assert "Квадратов" in joined
    assert "Абсцисса" in joined
    assert "Хордова" in joined
