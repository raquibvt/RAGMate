# Anonymous codebase for the paper "Utilizing Metadata for Better Retrieval-Augmented Generation"

## Install a new conda environment using the following:
Note that we are using python=3.10.0 for now.
```console
conda create --prefix env python=3.10.0
conda activate env/
pip install -r requirements.txt
```

## Directory Structure
`dataset` directory contains the RAGMATE-10K dataset and code files to process the dataset.
`10K_forms` has the preprocessed documents. Ground truth has already been provided.

## Running the experiments
Workflow: 
1. run script `dataset/src/chunk_doc_tiktoken.py` to chunkify `dataset/10K_forms`.
2. run script `indexing.py` to make the databases. Here is the list of variables we used for openAI text embedding model. Place in .env file. Feel free to change as needed. These are used in the `get_chromadb_openai_embedding_function()` to load the embedding function into chromaDB. 

    ```
    # Embedding Model Environment Variables
    EMBEDDING_ENDPOINT=
    EMBEDDING_DEPLOYMENT=
    EMBEDDING_KEY=
    EMBEDDING_API_VERSION=
    ```
    Can also run `get_chromadb_bge_embedding_function` function to use the BAAI/bge-m3 model. Model names can updated in the config files.

3. Run `dual_encoder_unified.py` to make the dual encoder based unified embedding from the embeddings.
4. `retrieve_regular.py` has the regular retrieval pipeline code. `retrieve_dual_sweep.py` has the dual encoder late-fusion code. 

## Config files
These `yaml` config files that define parameters for document processing and retrieval.  
Each config can vary, but typically covers:

- **Chunking** – how documents are split into token chunks (size, overlap, input/output dirs).  
- **Embeddings** – which model is used for tokenization and vector generation.  
- **Database** – storage settings for the vector database (collection name, persist directory, batch size, reset options).  
- **Retrieval** – number of results to return and where outputs are saved.  
- **RAG** – settings for retrieval-augmented generation (e.g. top-k chunks, query file).  

Multiple YAML files can exist to experiment with different chunk sizes, models, and storage options without modifying code.



