#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, Any, Optional
from dotenv import load_dotenv


if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    sys.exit(1)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_chromadb_openai_embedding_function():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("EMBEDDING_KEY"),
        api_base=os.getenv("EMBEDDING_ENDPOINT"),
        api_type="azure",
        api_version=os.getenv("EMBEDDING_API_VERSION"),
        deployment_id=os.getenv("EMBEDDING_DEPLOYMENT"),
    )


def load_collection(cfg: Dict[str, Any]):
    client = chromadb.PersistentClient(path=cfg["chroma_persist_directory"])
    embedding_function = get_chromadb_openai_embedding_function()
    return client, client.get_or_create_collection(
        name=cfg["collection_name"],
        embedding_function=embedding_function
    )


def get_all_vectors(coll):
    results = coll.get(include=["embeddings", "metadatas", "documents"])
    ids = results["ids"]
    embeddings = results.get("embeddings", [])
    metadatas = results.get("metadatas", [])
    documents = results.get("documents", [])
    return ids, embeddings, metadatas, documents


def _derive_chunk_id(doc_id: str, meta: Optional[Dict[str, Any]]) -> str:
    if meta and isinstance(meta, dict) and meta.get("chunk_id"):
        return meta["chunk_id"]
    if isinstance(doc_id, str) and "_openai_embed" in doc_id:
        return doc_id.split("_openai_embed")[0]
    return doc_id


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else x / n


def build_alignment(
    text_ids, text_embs, text_mds,
    meta_ids, meta_embs, meta_mds
) -> Dict[str, Dict[str, Any]]:
    aligned: Dict[str, Dict[str, Any]] = {}
    for i, emb, md in zip(text_ids, text_embs, text_mds):
        key = _derive_chunk_id(i, md)
        aligned.setdefault(key, {})
        aligned[key]["text"] = np.asarray(emb, dtype=np.float32)
        aligned[key]["md"] = md or {}
    for i, emb, md in zip(meta_ids, meta_embs, meta_mds):
        key = _derive_chunk_id(i, md)
        aligned.setdefault(key, {})
        aligned[key]["meta"] = np.asarray(emb, dtype=np.float32)
        if "md" not in aligned[key] or not aligned[key]["md"]:
            aligned[key]["md"] = md or {}
    return aligned


def fuse_weighted_sum(aligned: Dict[str, Dict[str, Any]], alpha: float):
    ids, embs, mds = [], [], []
    a = float(alpha); b = 1.0 - a
    for cid, payload in aligned.items():
        et = payload.get("text"); em = payload.get("meta")
        if et is None and em is None:
            continue
        if et is None: et = np.zeros_like(em, dtype=np.float32)
        if em is None: em = np.zeros_like(et, dtype=np.float32)
        et = l2n(et); em = l2n(em)
        fused = l2n(a * et + b * em).astype(np.float32)
        ids.append(cid); embs.append(fused.tolist()); mds.append(payload.get("md", {}))
    return ids, embs, mds


def add_as_new_collection(dest_cfg, ids, embeddings, metadatas, batch_size: int):
    dest_client = chromadb.PersistentClient(path=dest_cfg["chroma_persist_directory"])

    if dest_cfg.get("delete_existing", False) and dest_cfg.get("allow_reset", False):
        try:
            print(f"Resetting persist dir: {dest_cfg['chroma_persist_directory']}")
            dest_client.reset()
        except Exception as e:
            print("Warning: reset failed:", e)

    coll = dest_client.get_or_create_collection(
        name=dest_cfg["collection_name"],
        embedding_function=None  # we're adding explicit embeddings
    )

    ids = list(ids)
    embeddings = [(e.tolist() if hasattr(e, "tolist") else list(e)) for e in embeddings]
    metadatas = [dict(m) if isinstance(m, dict) else {} for m in metadatas]

    n = len(ids)
    if not (len(embeddings) == n and len(metadatas) == n):
        raise ValueError(f"Input lengths must match: ids={n}, embeddings={len(embeddings)}, metadatas={len(metadatas)}")

    print(f"Adding {n} vectors to '{dest_cfg['collection_name']}' ...")
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        batch_ids = ids[start:end]
        batch_embs = embeddings[start:end]
        batch_mds = metadatas[start:end]

        if not (len(batch_ids) == len(batch_embs) == len(batch_mds)):
            raise ValueError(f"Batch length mismatch at {start}:{end} "
                             f"ids={len(batch_ids)}, embeds={len(batch_embs)}, metas={len(batch_mds)}")

        coll.add(ids=batch_ids, embeddings=batch_embs, metadatas=batch_mds)
        start = end


def main():
    ap = argparse.ArgumentParser(
        description="Fuse text-only and meta-only Chroma indexes into unified embeddings (weighted sum)."
    )
    # Removed all defaults; these are required now.
    ap.add_argument("--config-meta-only", required=True, help="YAML for the META-ONLY source collection.")
    ap.add_argument("--config-text-only", required=True, help="YAML for the TEXT-ONLY source collection.")
    ap.add_argument("--config-out-sum", required=True, help="YAML for the DEST (weighted-sum) collection.")
    ap.add_argument("--alpha", type=float, required=True, help="Weight for TEXT side (0..1). Meta is (1-alpha).")
    ap.add_argument("--batch-size", type=int, required=True, help="Add batch size.")
    args = ap.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print("alpha must be in [0,1]")
        sys.exit(1)

    cfg_meta  = load_config(args.config_meta_only)
    cfg_text  = load_config(args.config_text_only)
    cfg_out_s = load_config(args.config_out_sum)

    print("Opening TEXT-ONLY collection...")
    _, coll_text = load_collection(cfg_text)
    print("Opening META-ONLY collection...")
    _, coll_meta = load_collection(cfg_meta)

    print("Fetching TEXT-ONLY vectors...")
    t_ids, t_embs, t_mds, _ = get_all_vectors(coll_text)
    print(f"Loaded {len(t_ids)} text vectors.")

    print("Fetching META-ONLY vectors...")
    m_ids, m_embs, m_mds, _ = get_all_vectors(coll_meta)
    print(f"Loaded {len(m_ids)} meta vectors.")

    print("Aligning by chunk_id ...")
    aligned = build_alignment(t_ids, t_embs, t_mds, m_ids, m_embs, m_mds)
    print(f"Aligned keys: {len(aligned)}")

    ids_s, embs_s, mds_s = fuse_weighted_sum(aligned, args.alpha)

    add_as_new_collection(cfg_out_s, ids_s, embs_s, mds_s, batch_size=args.batch_size)
    print("DONE: merged weighted-sum collection.")

    dim_t = len(t_embs[0]) if t_embs else 0
    print("\nSummary:")
    print(f"  Text dim: {dim_t}")
    print(f"  Sum dim:  {len(embs_s[0]) if embs_s else 0} (should be {dim_t})")


if __name__ == "__main__":
    main()
