import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=10)  # Only print small pieces
import json
import yaml
import chromadb
from chromadb.config import Settings
from utils import *
import os
from chromadb.utils import embedding_functions
import sys
from datasets import load_dataset
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

def retrieve(query, top_k, collection, cfg, without_meta_presence=False):
    """
    Retrieve top_k documents for a given query.
    
    Args:
        query (str): The input query string.
        top_k (int): Number of top documents to retrieve.
        collection: The vector database collection.
        cfg (dict): Configuration containing embedding model info.
        without_meta_presence (bool): If True, removes metadata from document text before scoring.

    Returns:
        list of dicts: Each with id, cleaned text (optional), metadata, and score.
    """
    if "bge" in cfg['embedding_model']:
        BGE_PREFIX = "Represent this sentence for searching relevant passages: "
        query_input = BGE_PREFIX + query
    else:
        query_input = query
        
    results = collection.query(
        query_texts=[query_input],
        n_results=top_k
    )

    output = []
    for doc, metadata, score, doc_id in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0],
        results['ids'][0]
    ):
        cleaned_text = doc
        if without_meta_presence:
            if "--- DOCUMENT METADATA ---" in doc and len(doc.split("--- DOCUMENT METADATA ---")[0]) > 200:
                cleaned_text = doc.split("--- DOCUMENT METADATA ---")[0]
            elif "--- DOCUMENT METADATA ---" in doc:
                cleaned_text = doc.split("--- End of document metadata ---")[1]

        output.append({
            'id': doc_id,
            'text': cleaned_text,
            'metadata': metadata,
            'score': score
        })

    # Score the (optionally cleaned) retrieved chunks using encoder similarity
    # embedding_model = cfg['embedding_model']
    # output = score_retrieved_chunks(question=query_input, retrieved_chunks=output, embedding_model=embedding_model)

    return output


def main():
    if len(sys.argv) != 2:
        print("Usage: python your_script.py <config_filename.yaml>")
        sys.exit(1)

    config_filename = sys.argv[1]
    cfg = load_config(config_filename)
    os.makedirs(cfg['output_dir'], exist_ok=True)

    collection = load_collection(cfg)
    
    # --- Load questions ---
    with open(cfg['rag_ques_file'], 'r') as f:
        questions_data = json.load(f)
        
    top_k = cfg.get('top_k', 25)

    # --- Retrieval results ---
    responses = []
    for category in ['general', 'deeper']:
        print(f"\nProcessing category: {category}")
        for ticker, company_dict in questions_data[category].items():
            company_name = company_dict.get("company_name")
            year = company_dict.get("year")
            questions = company_dict.get("questions", [])
            for question in questions:
                print(f"Retrieving for: {question}")
                results = retrieve(question, top_k, collection, cfg, without_meta_presence=False)
                responses.append({
                    'category': category,
                    'ticker': ticker,
                    'company_name': company_name,
                    'year': year,
                    'question': question,
                    'results': results
                })

    # Save results along with metadata about the run
    full_output = {
        'collection_name': cfg['collection_name'],
        'retrieval_config': cfg,
        'responses': responses
    }

    output_filename = f"retrieval_results_own.json"
    with open(os.path.join(cfg['output_dir'], output_filename), 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2)

    print(f"Saved results to {output_filename}")

if __name__ == "__main__":
    main()
