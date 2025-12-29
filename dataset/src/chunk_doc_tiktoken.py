import os
import json
import argparse
import yaml
import shutil
import sys
import hashlib
from collections import defaultdict

from langchain.text_splitter import TokenTextSplitter

def load_docs(input_dir):
    docs = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".json"):
            with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if isinstance(doc, list):
                    for section in doc:
                        if isinstance(section, dict):
                            section['source'] = fname
                            docs.append(section)
                        else:
                            print(f"Skipping non-dict section in {fname}: {section}")
                elif isinstance(doc, dict):
                    doc['source'] = fname
                    docs.append(doc)
                else:
                    print(f"Skipping unexpected structure in {fname}: {type(doc)}")
    return docs

def get_tokenizer(model_name):
    if any(
        m in model_name.lower()
        for m in [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "openai",
            "azure"
        ]
    ):
        try:
            import tiktoken
        except ImportError:
            raise ImportError("Please install tiktoken for OpenAI tokenization: pip install tiktoken")
        # Use OpenAI's embedding tokenizer
        enc = tiktoken.get_encoding("cl100k_base")
        return enc
    else:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)

def chunk_docs(
    docs, tokenizer, chunk_size, chunk_overlap, section_metadata_included, use_tiktoken=False
):
    """
    Split documents into chunks.
    Returns: chunks, tokens_per_doc, chunks_per_doc, doc_to_chunkids
    """
    chunks = []
    tokens_per_doc = defaultdict(int)
    chunks_per_doc = defaultdict(int)
    doc_to_chunkids = defaultdict(list)  # NEW: map source_doc -> list of chunk_ids

    for doc_idx, doc in enumerate(docs):
        text = doc['text']
        metadata = doc.get('metadata', {})
        doc_metadata = metadata.get('metadata', metadata)
        section_title = metadata.get('section', "no_section")
        source_doc = doc.get("source", f"doc_{doc_idx}")

        section_hash = hashlib.sha1(section_title.encode("utf-8")).hexdigest()[:8]

        if use_tiktoken:
            input_ids = tokenizer.encode(text)
        else:
            input_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0].tolist()
        total_tokens = len(input_ids)

        # Add to document's token sum
        tokens_per_doc[source_doc] += total_tokens

        start = 0
        chunk_counter = 0

        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_ids = input_ids[start:end]

            if use_tiktoken:
                chunk_text_piece = tokenizer.decode(chunk_ids)
            else:
                chunk_text_piece = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

            # For safety, double-check length
            if use_tiktoken:
                chunk_piece_ids = tokenizer.encode(chunk_text_piece)
                if len(chunk_piece_ids) > 512:
                    chunk_piece_ids = chunk_piece_ids[:512]
                    chunk_text_piece = tokenizer.decode(chunk_piece_ids)
            else:
                chunk_piece_ids = tokenizer(chunk_text_piece, return_tensors="pt", truncation=False)["input_ids"][0]
                if len(chunk_piece_ids) > 512:
                    chunk_piece_ids = chunk_piece_ids[:512]
                    chunk_text_piece = tokenizer.decode(chunk_piece_ids, skip_special_tokens=True).strip()

            chunk_id = f"{source_doc}_sec_{section_hash}_chunk_{chunk_counter}"

            chunk_obj = {
                "text": chunk_text_piece,
                "metadata": {
                    "source_doc": source_doc,
                    "chunk_id": chunk_id,
                    "section": section_title,
                    "doc_metadata": doc_metadata
                }
            }
            chunks.append(chunk_obj)

            # Track chunk id for this doc
            doc_to_chunkids[source_doc].append(chunk_id)
            chunks_per_doc[source_doc] += 1

            start += chunk_size - chunk_overlap
            chunk_counter += 1

    return chunks, tokens_per_doc, chunks_per_doc, doc_to_chunkids

def save_chunks(chunks, output_dir, params):
    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(output_dir, f"chunk_{i:05d}.json"), 'w', encoding='utf-8') as f:
            json.dump(chunk, f)
    chunk_counts = {}
    for chunk in chunks:
        source = chunk['metadata'].get('source_doc', 'unknown')
        chunk_counts[source] = chunk_counts.get(source, 0) + 1
    params['chunk_counts_by_source_doc'] = chunk_counts
    with open(os.path.join(output_dir, "chunking_params.json"), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    if len(sys.argv) != 2:
        print("Usage: python your_script.py <config_filename.yaml>")
        sys.exit(1)

    config_filename = sys.argv[1]
    cfg = load_config(config_filename)

    input_dir = cfg['input_document_dir']
    output_dir = cfg['chunked_document_dir']
    chunk_size = cfg.get('chunk_size')
    chunk_overlap = cfg.get('chunk_overlap')
    tokenizer_model = cfg.get('tokenizer_model', "text-embedding-3-small")
    section_metadata_included = cfg.get('section_metadata_included', False)

    # make the output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    docs = load_docs(input_dir)

    use_tiktoken = any(
        m in tokenizer_model.lower()
        for m in [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "openai",
            "azure"
        ]
    )
    tokenizer = get_tokenizer(tokenizer_model)

    # Get chunks, tokens per doc, chunks per doc, and doc->chunkids mapping
    chunks, tokens_per_doc, chunks_per_doc, doc_to_chunkids = chunk_docs(
        docs,
        tokenizer,
        chunk_size,
        chunk_overlap,
        section_metadata_included,
        use_tiktoken=use_tiktoken
    )

    # Print token and chunk counts
    print("\nToken counts and chunk counts per document:\n")
    all_sources = sorted(set(list(tokens_per_doc.keys()) + list(chunks_per_doc.keys())))
    for source in all_sources:
        tcount = tokens_per_doc.get(source, 0)
        ccount = chunks_per_doc.get(source, 0)
        print(f"{source} - Tokens: {tcount:,} | Chunks: {ccount}")

    # Save the mapping of doc -> list of chunk_ids
    doc_to_chunkids_path = os.path.join(output_dir, "doc_to_chunkids.json")
    with open(doc_to_chunkids_path, "w", encoding="utf-8") as f:
        json.dump(doc_to_chunkids, f, indent=2)
    print(f"\nSaved doc to chunk id mapping to {doc_to_chunkids_path}")

    chunking_params = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "tokenizer_model": tokenizer_model,
        "section_metadata_included": section_metadata_included
    }

    save_chunks(chunks, output_dir, chunking_params)
    shutil.copyfile(config_filename, os.path.join(output_dir, "used_config.yaml"))
    print(f"\nSaved {len(chunks)} chunks to {output_dir}")
    print("Chunking parameters logged in chunking_params.json")

if __name__ == "__main__":
    main()
