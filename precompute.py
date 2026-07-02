#!/usr/bin/env python3
"""
HybridRank precompute step — Team Abhijayati
Generates semantic embeddings for all candidates using a compact local model.

Run ONCE before rank.py:
    pip install sentence-transformers numpy
    python precompute.py --candidates ./candidates.jsonl

Outputs (in same directory as candidates.jsonl):
    candidate_embeddings.npy   (~38 MB for 100K candidates)
    cand_ids.txt               (ordered candidate IDs)
    jd_embedding.npy           (1 x 384 JD query vector)

rank.py will auto-detect these files and add a semantic dimension.
Pre-computation may take 3-5 minutes on CPU (within allowed pre-compute window).
The ranking step (rank.py) remains <5 minutes as required.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: Missing dependencies. Run:")
    print("  pip install sentence-transformers numpy")
    sys.exit(1)


# Distilled JD query — encodes the IDEAL candidate for this specific role.
# Deliberately dense with IR/retrieval/ranking terminology so the embedding
# space separates "built a search engine" from "used a search engine at work".
JD_QUERY = (
    "Production NLP and information retrieval engineer. "
    "Experience building embedding-based retrieval systems: sentence-transformers, "
    "BGE, E5, OpenAI embeddings, dense retrieval, bi-encoder, cross-encoder. "
    "Vector database operations: FAISS, Pinecone, Weaviate, Qdrant, Milvus, pgvector. "
    "Hybrid search combining BM25 sparse retrieval with dense embeddings. "
    "Elasticsearch, OpenSearch, Solr for production search at scale. "
    "Ranking systems: learning to rank, LTR, XGBoost ranking, LambdaMART. "
    "Recommendation systems, collaborative filtering, personalization, CTR prediction. "
    "Evaluation frameworks: NDCG, MRR, MAP, Precision@K, offline-to-online correlation, A/B testing. "
    "LLM integration: RAG pipelines, fine-tuning, RLHF, LoRA, PEFT. "
    "Python, PyTorch, MLOps, model serving, feature stores. "
    "Applied ML at product companies, shipping to real users, production deployment. "
    "5-9 years experience, India-based, strong retrieval and ranking background."
)


def build_candidate_text(candidate):
    """
    Build a rich text representation of each candidate for embedding.
    Combines the most informative fields: title, headline, summary, top skills,
    and key career descriptions — without hitting token limits.
    """
    p = candidate.get("profile", {})
    parts = []

    # Title and current context (high signal)
    title = p.get("current_title", "")
    company = p.get("current_company", "")
    if title:
        parts.append(title)
    if company:
        parts.append("at " + company)

    # Headline (usually the most curated self-description)
    headline = p.get("headline", "")
    if headline:
        parts.append(headline)

    # Summary (rich narrative, truncated to avoid embedding noise)
    summary = p.get("summary", "")
    if summary:
        parts.append(summary[:500])

    # Skills as a flat list (high-density signal)
    skills = candidate.get("skills", [])
    skill_names = [s.get("name", "") for s in skills if s.get("name")]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names[:20]))

    # Most recent 2 career descriptions (current role is most relevant)
    career = candidate.get("career_history", [])
    career_sorted = sorted(career, key=lambda j: j.get("is_current", False), reverse=True)
    for j in career_sorted[:2]:
        desc = j.get("description", "")
        if desc:
            job_line = j.get("title", "") + " at " + j.get("company", "") + ": " + desc[:200]
            parts.append(job_line)

    return " | ".join(p for p in parts if p.strip())


def main():
    parser = argparse.ArgumentParser(description="Precompute semantic embeddings for HybridRank")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="SentenceTransformers model (default: all-MiniLM-L6-v2, ~23MB)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for encoding (higher = faster, more RAM)")
    args = parser.parse_args()

    cand_path = Path(args.candidates)
    out_dir = cand_path.parent

    print(f"[precompute] Loading model: {args.model} ...")
    model = SentenceTransformer(args.model)

    print(f"[precompute] Reading {cand_path} ...")
    cand_ids = []
    texts = []
    with open(cand_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cand_ids.append(c["candidate_id"])
                texts.append(build_candidate_text(c))
            except Exception:
                pass

    print(f"[precompute] Encoding {len(texts):,} candidates (batch_size={args.batch_size}) ...")
    print(f"[precompute] Estimated time: ~{len(texts) // 1000 // 3} minutes on CPU")

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize so dot product = cosine similarity
    )

    print(f"[precompute] Encoding JD query ...")
    jd_emb = model.encode(
        [JD_QUERY],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    emb_path = out_dir / "candidate_embeddings.npy"
    ids_path = out_dir / "cand_ids.txt"
    jd_path = out_dir / "jd_embedding.npy"

    np.save(emb_path, embeddings)
    np.save(jd_path, jd_emb)
    with open(ids_path, "w") as f:
        f.write("\n".join(cand_ids))

    print(f"[precompute] Done!")
    print(f"  Embeddings: {emb_path}  ({os.path.getsize(emb_path) / 1e6:.1f} MB)")
    print(f"  Candidate IDs: {ids_path}")
    print(f"  JD embedding: {jd_path}")
    print(f"\n[precompute] Shape: {embeddings.shape}  (candidates x dims)")

    # Quick sanity check: top 10 by cosine similarity to JD
    scores = embeddings @ jd_emb.T
    top_idx = scores[:, 0].argsort()[::-1][:10]
    print("\n[precompute] Top 10 by semantic similarity (sanity check):")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2}. {cand_ids[idx]}  sim={scores[idx, 0]:.4f}  {texts[idx][:80]}")

    print("\n[precompute] Now run: python rank.py --candidates ./candidates.jsonl --out ./submission.csv")


if __name__ == "__main__":
    main()
