import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=10)

import os
import json
import sys
import yaml
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
if not load_dotenv():
    print(
        "Could not load .env file or it is empty. Please check if it exists and is readable."
    )
    exit(1)

def sanitize_metadata(md):
    clean_md = {}
    for k, v in md.items():
        if isinstance(v, (list, dict)):
            clean_md[k] = str(v)
        else:
            clean_md[k] = v
    return clean_md

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_documents(doc_dir):
    docs = []
    for fname in os.listdir(doc_dir):
        if fname.endswith(".json") and fname.startswith("chunk_"):
            with open(os.path.join(doc_dir, fname), 'r') as f:
                doc = json.load(f)
                docs.append(doc)
    return docs

def prepare_text(doc, use_metadata, metadata_position="top"):
    base_text = doc['text']

    if not use_metadata:
        return base_text

    metadata = doc.get('metadata', {})
    doc_meta = metadata.get('doc_metadata', {})
    section = metadata.get('section', None)

    meta_parts = []
    if section:
        meta_parts.append(f"section: {section}")

    meta_parts.extend([f"{k}: {v}" for k, v in doc_meta.items()])
    meta_str = "\n".join(meta_parts)

    if not meta_str:
        return base_text

    full_meta_block = (
        "--- DOCUMENT METADATA ---\n"
        f"{meta_str}\n"
        "--- End of document metadata ---\n"
    )

    if metadata_position == "top":
        return f"{full_meta_block}\n{base_text}"
    elif metadata_position == "bottom":
        return f"{base_text}\n\n{full_meta_block}"
    elif metadata_position == "middle":
        words = base_text.split()
        midpoint = len(words) // 2
        first_half = " ".join(words[:midpoint])
        second_half = " ".join(words[midpoint:])
        return f"{first_half}\n\n{full_meta_block}\n\n{second_half}"
    else:
        return f"{full_meta_block}\n{base_text}"

def get_chromadb_openai_embedding_function():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("EMBEDDING_KEY"),
        api_base=os.getenv("EMBEDDING_ENDPOINT"),
        api_type="azure",
        api_version=os.getenv("EMBEDDING_API_VERSION"),
        # model_name=os.getenv("EMBEDDING_DEPLOYMENT"),
        deployment_id=os.getenv("EMBEDDING_DEPLOYMENT"),
    )

def get_chromadb_bge_embedding_function(model_name="BAAI/bge-m3"):
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

import re

def extract_company_name_and_year(doc):
    metadata = doc.get("metadata", {})
    doc_meta = metadata.get("doc_metadata", {})
    company_name = doc_meta.get("company_name") or metadata.get("company_name")

    # Try to extract year from source_doc or chunk_id
    source_doc = metadata.get("source_doc", "")
    chunk_id = metadata.get("chunk_id", "")

    # Use regex to find a four-digit year (e.g. 2024) in either string
    year_match = re.search(r'(\d{4})', source_doc)
    if not year_match:
        year_match = re.search(r'(\d{4})', chunk_id)

    year = year_match.group(1) if year_match else None

    return company_name, year

def main():
    if len(sys.argv) != 2:
        print("Usage: python filename.py <config_filename.yaml>")
        sys.exit(1)

    config_filename = sys.argv[1]
    cfg = load_config(config_filename)

    chroma_dir = cfg['chroma_persist_directory']
    collection_name = cfg['collection_name']
    delete_existing = cfg.get('delete_existing', False)
    allow_reset = cfg.get('allow_reset', False)

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(allow_reset=allow_reset)
    )

    if delete_existing and allow_reset:
        print(f"Resetting ChromaDB in directory: {chroma_dir}")
        client.reset()

    # alternatively can use the BGE embedding function
    embedding_function = get_chromadb_openai_embedding_function()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    docs = load_documents(cfg['chunked_document_dir'])

    # checking if script is running in a test environment
    # docs = docs[0:2]

    metadata_position = cfg.get('metadata_position', 'top')
    texts = [
        prepare_text(doc, cfg['use_metadata'], metadata_position)
        for doc in docs
    ]


    suffix = cfg.get("id_suffix", "")
    ids = [f"{doc['metadata']['chunk_id']}_{suffix}" for doc in docs]

    metadatas = []
    for doc in docs:
        chunk_id = doc["metadata"]["chunk_id"]
        company_name, year = extract_company_name_and_year(doc)
        metadatas.append({
            "chunk_id": chunk_id,
            "company_name": company_name,
            "year": year
        })

    batch_size = cfg.get('indexing_batch_size', 200)

    print("Starting indexing in batches...")

    for i in tqdm(range(0, len(ids), batch_size), desc="Indexing batches"):
        batch_ids = ids[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas
        )

    print(f"Indexed {len(docs)} chunks into ChromaDB collection '{collection_name}' in persist directory '{chroma_dir}'.")

if __name__ == "__main__":
    main()
