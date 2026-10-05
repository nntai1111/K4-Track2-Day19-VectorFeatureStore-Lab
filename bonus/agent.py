"""HybridMemoryAgent — episodic memory (Qdrant) + stable profile / recent activity (Feast).

remember(): chunk -> embed -> upsert into ONE collection, payload user_id (filtered ANN).
recall():   Feast profile + activity  ->  3-way RRF (BM25, vector, profile-affinity)
            restricted to the user's own chunks  ->  context string (no LLM call).
"""
from __future__ import annotations

import re
import sys
import tempfile
import time
import unicodedata
import uuid
from collections import deque
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchValue,
                                  PointStruct, VectorParams)
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root -> app/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.embeddings import Embedder  # noqa: E402
from profile_store import build_store, push_activity, read_features  # noqa: E402

CHUNK_WORDS = 60      # ~80 tokens: one idea per chunk, still fits 5 chunks in a small context
RRF_K = 60
DEPTH = 20
# Weighted RRF: the profile is a tie-breaker, not a retriever of equal standing.
# With k=60 and a few dozen chunks per user, rank 1 vs rank 2 differs by only
# ~0.0003, so at weight 1.0 (or even 0.3) the affinity list outvoted the query
# itself (asking about Kubernetes returned cloud notes first). Must stay < ~0.05.
WEIGHTS = (1.0, 1.0, 0.03)  # BM25, vector, profile-affinity

# Function words that carry no topic. Without this, "về"/"gì"/"tôi" match everything.
VI_STOPWORDS = set("tôi mình bạn đã đang sẽ gì về cho của là và có không các những được "
                   "này đó nào thì với để một trong khi theo gần đây hãy giúp".split())


def strip_accents(s: str) -> str:
    """'tự động' -> 'tu dong' (đ is not a combining mark, map it by hand)."""
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    """Vietnamese-aware BM25 tokens (no extra dependency):
    * syllable bigrams ('tài_liệu') approximate word segmentation, so the
      syllable 'liệu' in 'tài liệu' no longer equals 'dữ liệu' on its own;
    * stopwords dropped from unigrams;
    * every token also emitted without diacritics, so 'tu dong mo rong'
      (typed without Telex) still hits 'tự động mở rộng'."""
    syl = re.findall(r"\w+", unicodedata.normalize("NFC", text.lower()))
    toks = [w for w in syl if w not in VI_STOPWORDS]
    toks += [f"{a}_{b}" for a, b in zip(syl, syl[1:])
             if a not in VI_STOPWORDS and b not in VI_STOPWORDS]
    return toks + [strip_accents(t) for t in toks if strip_accents(t) != t]


def chunk(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """Sentence-packing: never split mid-sentence, start a new chunk past max_words."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s.strip()]
    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class HybridMemoryAgent:
    def __init__(self, workdir: Path | None = None, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(
            "episodic", vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE))
        self.fs = build_store(workdir or Path(tempfile.mkdtemp(prefix="bonus_feast_")))
        self.chunks: dict[str, list[dict]] = {}          # user_id -> payloads (for BM25)
        self.events: dict[str, deque] = {}               # user_id -> (ts, topic) query log

    # ── write path ────────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001", topic: str = "general",
                 source: str = "note") -> int:
        pieces = chunk(text)
        vectors = list(self.embedder.embed(pieces))
        points = []
        for piece, vec in zip(pieces, vectors):
            payload = {"user_id": user_id, "text": piece, "topic": topic,
                       "source": source, "ts": time.time()}
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload=payload))
            self.chunks.setdefault(user_id, []).append(payload)
        self.qdrant.upsert("episodic", points=points)
        return len(points)

    # ── read path ─────────────────────────────────────────────────────────
    def _vector(self, query: str, user_id: str) -> list[str]:
        qv = next(self.embedder.embed([query])).tolist()
        only_me = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        res = self.qdrant.query_points("episodic", query=qv, query_filter=only_me, limit=DEPTH)
        return [p.payload["text"] for p in res.points]

    def _keyword(self, query: str, user_id: str) -> list[str]:
        docs = self.chunks.get(user_id, [])
        if not docs:
            return []
        scores = BM25Okapi([tokenize(d["text"]) for d in docs]).get_scores(tokenize(query))
        ranked = sorted(zip(scores, docs), key=lambda x: -x[0])
        return [d["text"] for s, d in ranked[:DEPTH] if s > 0]

    def _affinity(self, user_id: str, topic: str | None) -> list[str]:
        """3rd retriever: the user's chunks on their profile topic, newest first."""
        docs = [d for d in self.chunks.get(user_id, []) if topic and d["topic"] == topic]
        return [d["text"] for d in sorted(docs, key=lambda d: -d["ts"])[:DEPTH]]

    def _log_query(self, user_id: str, topic: str) -> None:
        now = time.time()
        log = self.events.setdefault(user_id, deque())
        log.append((now, topic))
        while log and now - log[0][0] > 3600:
            log.popleft()
        topics = list(dict.fromkeys(t for _, t in reversed(log)))[:3]
        late = sum(1 for ts, _ in log if time.localtime(ts).tm_hour >= 22
                   or time.localtime(ts).tm_hour < 5) / len(log)
        push_activity(self.fs, user_id, len(log), topics, late)

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        feats = read_features(self.fs, user_id)        # read BEFORE logging this query
        lists = [self._keyword(query, user_id), self._vector(query, user_id),
                 self._affinity(user_id, feats.get("topic_affinity"))]
        rrf: dict[str, float] = {}
        for weight, ranked in zip(WEIGHTS, lists):
            for rank, text in enumerate(ranked, start=1):
                rrf[text] = rrf.get(text, 0.0) + weight / (RRF_K + rank)
        top = sorted(rrf, key=lambda t: -rrf[t])[:top_k]
        topic_of = {d["text"]: d["topic"] for d in self.chunks.get(user_id, [])}
        self._log_query(user_id, topic_of.get(top[0], "general") if top else "general")

        recent = feats.get("recent_topics") or "(chưa có — activity TTL 1h đã hết hoặc user mới)"
        lines = [
            f"User {user_id} likes {feats.get('topic_affinity') or 'unknown'}, "
            f"reads at {feats.get('reading_speed_wpm') or '?'} wpm, "
            f"language={feats.get('preferred_language') or '?'}, active {feats.get('active_hours') or '?'}h.",
            f"Recent activity: {feats.get('queries_last_hour') or 0} queries in the last hour; "
            f"topics: {recent}.",
            "Top memories:",
        ]
        lines += [f"  {i}. [{topic_of.get(t, '?')}] {t}" for i, t in enumerate(top, 1)]
        return "\n".join(lines)
