
import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=10)

import json
import yaml
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from utils import *
import os
import sys
from tqdm import tqdm
from dotenv import load_dotenv

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_chromadb_openai_embedding_function():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("EMBEDDING_KEY"),
        api_base=os.getenv("EMBEDDING_ENDPOINT"),
        api_type="azure",
        api_version=os.getenv("EMBEDDING_API_VERSION"),
        deployment_id=os.getenv("EMBEDDING_DEPLOYMENT"),
    )

def load_collection(cfg):
    client = chromadb.PersistentClient(path=cfg['chroma_persist_directory'])
    embedding_function = get_chromadb_openai_embedding_function()
    return client.get_or_create_collection(
        name=cfg['collection_name'],
        embedding_function=embedding_function
    )

def get_chunk_id(doc_id, meta):
    if meta and "chunk_id" in meta:
        return meta["chunk_id"]
    return doc_id.split("_openai_embed")[0]

def dual_retrieve(query, top_k, chunk_coll, meta_coll, alpha=0.5, chunk_cfg=None, meta_cfg=None):
    if "bge" in chunk_cfg['embedding_model']:
        BGE_PREFIX = "Represent this sentence for searching relevant passages: "
        query_input = BGE_PREFIX + query
    else:
        query_input = query

    chunk_res = chunk_coll.query(query_texts=[query_input], n_results=top_k)
    meta_res = meta_coll.query(query_texts=[query_input], n_results=top_k)

    scores_dict = {}

    for doc, meta, dist, doc_id in zip(
        chunk_res['documents'][0],
        chunk_res['metadatas'][0],
        chunk_res['distances'][0],
        chunk_res['ids'][0]
    ):
        chunk_id = get_chunk_id(doc_id, meta)
        scores_dict[chunk_id] = {
            'chunk_score': dist,
            'meta_score': None,
            'text': doc,
            'metadata': meta
        }

    for doc, meta, dist, doc_id in zip(
        meta_res['documents'][0],
        meta_res['metadatas'][0],
        meta_res['distances'][0],
        meta_res['ids'][0]
    ):
        chunk_id = get_chunk_id(doc_id, meta)
        if chunk_id not in scores_dict:
            scores_dict[chunk_id] = {
                'chunk_score': None,
                'meta_score': dist,
                'text': "",
                'metadata': meta
            }
        else:
            scores_dict[chunk_id]['meta_score'] = dist

    for chunk_id, entry in scores_dict.items():
        chunk_score = entry['chunk_score'] if entry['chunk_score'] is not None else 1.0
        meta_score = entry['meta_score'] if entry['meta_score'] is not None else 1.0
        entry['final_score'] = alpha * chunk_score + (1 - alpha) * meta_score

    sorted_entries = sorted(scores_dict.items(), key=lambda x: x[1]['final_score'])

    return [{
        'id': chunk_id,
        'text': entry['text'],
        'metadata': entry['metadata'],
        'score': entry['final_score']
    } for chunk_id, entry in sorted_entries[:top_k]]

def run_alpha_sweep(chunk_cfg_path, meta_cfg_path, sweep_dir="alpha_sweep_outputs", alphas=np.linspace(0, 1, 11)):
    chunk_cfg = load_config(chunk_cfg_path)
    meta_cfg = load_config(meta_cfg_path)

    os.makedirs(sweep_dir, exist_ok=True)

    chunk_collection = load_collection(chunk_cfg)
    meta_collection = load_collection(meta_cfg)

    with open(chunk_cfg['rag_ques_file'], 'r') as f:
        questions_data = json.load(f)

    top_k = chunk_cfg.get('top_k', 25)

    for alpha in alphas:
        print(f"Running alpha={alpha:.2f}")
        responses = []
        for category in ['general', 'deeper']:
            for ticker, company_dict in questions_data[category].items():
                company_name = company_dict.get("company_name")
                year = company_dict.get("year")
                questions = company_dict.get("questions", [])
                for question in questions:
                    results = dual_retrieve(
                        query=question,
                        top_k=top_k,
                        chunk_coll=chunk_collection,
                        meta_coll=meta_collection,
                        alpha=alpha,
                        chunk_cfg=chunk_cfg,
                        meta_cfg=meta_cfg
                    )
                    responses.append({
                        'category': category,
                        'ticker': ticker,
                        'company_name': company_name,
                        'year': year,
                        'question': question,
                        'results': results
                    })

        full_output = {
            'chunk_collection': chunk_cfg['collection_name'],
            'meta_collection': meta_cfg['collection_name'],
            'retrieval_config': {
                'alpha': alpha,
                'top_k': top_k
            },
            'responses': responses
        }

        filename = f"retrieval_alpha_{alpha:.2f}.json"
        with open(os.path.join(sweep_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(full_output, f, indent=2)

        print(f"Saved: {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python alpha_sweep.py <chunk_config.yaml> <meta_config.yaml>")
        sys.exit(1)
    run_alpha_sweep(sys.argv[1], sys.argv[2])
