from sentence_transformers import SentenceTransformer
import numpy as np
import os

import re

def extract_chunk_ids_from_answer(answer_text):
    """Extract string-style chunk IDs like [XYZ_2022_chunk_001] from LLM answer."""
    return list(set(re.findall(r"\[([^\[\]]+?)\]", answer_text)))

def normalize(text):
    """Lowercase and remove punctuation for token comparison."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def token_overlap_score(a, b):
    """
    Compute Jaccard similarity between token sets of two strings.
    Returns a float between 0 and 1.
    """
    tokens_a = set(normalize(a).split())
    tokens_b = set(normalize(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

def compute_context_accuracy_token_based(question_dicts, retrieved_chunks_per_query, k=5, threshold=0.7):
    """
    Token-based Context Accuracy @k using Jaccard similarity over tokens.

    Args:
        question_dicts: List of dicts with 'evidence' field.
        retrieved_chunks_per_query: List of top-k retrieved chunks per query (each chunk has 'text').
        k: Evaluate top-k retrieved chunks.
        threshold: Jaccard similarity threshold (e.g., 0.7 = 70% token overlap).

    Returns:
        context_accuracy: Fraction of queries with successful retrievals in top-k.
    """
    assert len(question_dicts) == len(retrieved_chunks_per_query), "Mismatch in query and retrieval lengths."

    successful = 0
    total = len(question_dicts)

    for qdict, retrieved_chunks in zip(question_dicts, retrieved_chunks_per_query):
        gold_evidence = qdict.get("evidence", "")
        top_k_chunks = retrieved_chunks[:k]

        match_found = any(
            token_overlap_score(gold_evidence, chunk["text"]) >= threshold
            for chunk in top_k_chunks
        )

        if match_found:
            successful += 1

    return successful / total if total > 0 else 0.0


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def score_retrieved_chunks(question, retrieved_chunks, embedding_model, model_cache={}):
    """
    Adds encoder-based similarity scores to retrieved chunks.
    Updates each chunk in-place with a 'encoder_similarity' field.
    """
    # Cache the model if not already loaded
    if 'model' not in model_cache:
        model_cache['model'] = SentenceTransformer(embedding_model)

    model = model_cache['model']

    # Embed query once
    query_emb = model.encode(question, normalize_embeddings=True)

    for result in retrieved_chunks:
        chunk_text = result['text']
        chunk_emb = model.encode(chunk_text, normalize_embeddings=True)
        sim = cosine_similarity(query_emb, chunk_emb)
        result['encoder_similarity'] = float(sim)  # ensure JSON serializable

    return retrieved_chunks

def verify_embedding_compatibility(collection, query_embedding):
    """
    Verifies that the dimensionality of the query embedding matches
    the dimensionality of the embeddings stored in the Chroma collection.
    """
    if collection.count() == 0:
        print("Warning: Collection is empty. Cannot verify embedding dimensionality.")
        return

    # Peek at a stored embedding
    sample = collection.peek(n=1)
    if not sample['ids'][0]:
        print("Warning: No data in collection to verify embedding dimensionality.")
        return

    sample_id = sample['ids'][0]
    sample_data = collection.get(ids=[sample_id])

    stored_embedding = sample_data['embeddings'][0]
    stored_dim = len(stored_embedding)
    query_dim = len(query_embedding)

    print(f"Stored embedding dimension: {stored_dim}")
    print(f"Query embedding dimension: {query_dim}")

    if stored_dim != query_dim:
        raise ValueError(
            f"Embedding dimension mismatch! Stored: {stored_dim}, Query: {query_dim}.\n"
            "Check that the same embedding model is used for indexing and querying."
        )
    else:
        print("Embedding dimensions match.")

# for helper utils
# import logging
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

# for loading models
import torch
# from auto_gptq import AutoGPTQForCausalLM
# from huggingface_hub import hf_hub_download

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    pipeline
)
# from constants import MODELS_PATH, MODEL_ID, MAX_NEW_TOKENS, MODEL_BASENAME
from langchain.llms import HuggingFacePipeline
from langchain_openai import AzureOpenAIEmbeddings


def load_context(docs, top):
    context_chunks = []
    for doc in docs[:top]:
        chunk_id = doc['id']
        chunk_text = doc['text']
        # removing meta, first for bottom case, second is for top case
        if "--- DOCUMENT METADATA ---" in chunk_text and len(chunk_text.split("--- DOCUMENT METADATA ---")[0]) > 200:
            chunk_text = chunk_text.split("--- DOCUMENT METADATA ---")[0]
        elif "--- DOCUMENT METADATA ---" in chunk_text:
            chunk_text = chunk_text.split("--- End of document metadata ---")[1]
        
        # also remove the embedding prompts, if any
        if "Represent this sentence for searching relevant passages:" in chunk_text:
            chunk_text = chunk_text.replace("Represent this sentence for searching relevant passages:", "")

        context_chunks.append(f"[{chunk_id}]\n{chunk_text}")
    return "\n\n".join(context_chunks)

def generate_answer_with_citation(question, context, llm):
    prompt = f"""
You are a helpful AI assistant. You will answer a question using a list of retrieved chunks from SEC 10-K filings. Each chunk is prefixed by a unique chunk ID in the format "[chunk_id] chunk_text".

<question>
{question}
</question>

<context>
{context}
</context>

Your tasks are:

1. **Answer the question** using only the information in the chunks.
2. **Cite each chunk you use** with its [chunk_id], placed at the end of the sentence it supports.
3. If none of the chunks provide an answer, reply with: "Sorry, I don't know."

Please follow this answer format exactly:

Answer: <your concise and accurate answer here>

Citations: [chunk_id1], [chunk_id2], ... (only include chunk_ids actually used in the answer)

Guidelines:
- Do not cite chunks that are irrelevant.
- Be precise and faithful to the chunk content.
- Do not hallucinate facts not present in the context.
    """.strip()

    connection_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", prompt),
        ],
    )
    messages = connection_prompt_template.format_messages(question=question, context=context)
    response = llm.invoke(messages).content.strip()
    return response


def generate_answer(question, context, llm):

    prompt = f"""
You are tasked with answering a question using provided chunks of information. Your goal is to provide an accurate answer while citing your sources using a specific markdown format.

Here is the question you need to answer:
<question>
{question}
</question>

Below are chunks of information that you can use to answer the question. Each chunk is preceded by a 
source identifier in the format "[chunk_id] chunk_text".

<chunks>
{context}
</chunks>

Your task is to answer the question using the information provided in these chunks. 
When you use information from a specific chunk in your answer, you must cite it using the chunk_id, enclosed in square parenthes. The citation should appear at the end of the sentence where the information is used.

If you cannot answer the question using the provided chunks, say "Sorry I don't know".
""".strip()

    # Using azure_openai
    connection_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", prompt),
        ],
    )
    messages = connection_prompt_template.format_messages(question=question, context=context)
    # number_of_tokens = llm['openAI']['gpt4'].get_num_tokens_from_messages(messages)
    # print(number_of_tokens)

    #calling the prompt
    resp = llm.invoke(messages).content.strip()
    # number_of_tokens = llm_gpt35['openAI'].get_num_tokens_from_messages(resp)
    return resp


def load_model(model_type="openAI", device_type="cuda", model_id=None, model_basename=None, openai_temperature=0.2, local_temperature=0.7):
    # logging.info(f"Loading model_type: {model_type}")
    llm = None

    if model_type in ["llama", "local"]:
        model, tokenizer = load_full_model(model_id, model_basename, device_type)

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(model_id)

        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=2000,
            temperature=local_temperature,
            top_p=0.1,
            top_k=40,
            repetition_penalty=1.176,
            generation_config=generation_config,
            truncation=True
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        # logging.info(f"Local LLM Loaded with model_type: {model_type}")
        
    else:
        llm = AzureChatOpenAI(
            temperature=openai_temperature,
            api_key=os.getenv("GPT4o_KEY"),
            api_version=os.getenv("GPT4o_API_VERSION"),
            azure_deployment=os.getenv("GPT4o_DEPLOYMENT"),
            azure_endpoint=os.getenv("GPT4o_ENDPOINT"),
        )
        # logging.info(f"OpenAI model Loaded with model_type: {model_type}")
    
    return llm
