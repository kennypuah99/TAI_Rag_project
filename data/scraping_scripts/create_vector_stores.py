"""
Vector Store Creation Script

Purpose:
This script processes various data sources (e.g., transformers, peft, trl, llama_index, openai_cookbooks, langchain)
to create vector stores using Chroma and LlamaIndex. It reads data from JSONL files, creates document embeddings,
and stores them in persistent Chroma databases for efficient retrieval.

Usage:
python script_name.py <source1> <source2> ...

Example:
python script_name.py transformers peft llama_index

The script accepts one or more source names as command-line arguments. Valid source names are:
transformers, peft, trl, llama_index, openai_cookbooks, langchain

For each specified source, the script will:
1. Read data from the corresponding JSONL file
2. Create document embeddings
3. Store the embeddings in a Chroma vector database
4. Save a dictionary of documents for future reference

Note: Ensure that the input JSONL files are present in the 'data' directory.
"""

import argparse
import json
import os
import pdb
import pickle
import shutil

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# Configuration for different sources
SOURCE_CONFIGS = {
    "transformers": {
        "input_file": "data/transformers_data.jsonl",
        "db_name": "chroma-db-transformers",
    },
    "peft": {"input_file": "data/peft_data.jsonl", "db_name": "chroma-db-peft"},
    "trl": {"input_file": "data/trl_data.jsonl", "db_name": "chroma-db-trl"},
    "llama_index": {
        "input_file": "data/llama_index_data.jsonl",
        "db_name": "chroma-db-llama_index",
    },
    "openai_cookbooks": {
        "input_file": "data/openai_cookbooks_data.jsonl",
        "db_name": "chroma-db-openai_cookbooks",
    },
    "langchain": {
        "input_file": "data/langchain_data.jsonl",
        "db_name": "chroma-db-langchain",
    },
    "tai_blog": {
        "input_file": "data/tai_blog_data.jsonl",
        "db_name": "chroma-db-tai_blog",
    },
    "all_sources": {
        "input_file": "data/all_sources_data.jsonl",
        "db_name": "chroma-db-all_sources",
    },
}


def create_docs(input_file: str) -> list[Document]:
    with open(input_file, "r") as f:
        documents = []
        for line in f:
            data = json.loads(line)
            documents.append(
                Document(
                    doc_id=data["doc_id"],
                    text=data["content"],
                    metadata={  # type: ignore
                        "url": data["url"],
                        "title": data["name"],
                        "tokens": data["tokens"],
                        "retrieve_doc": data["retrieve_doc"],
                        "source": data["source"],
                    },
                    excluded_llm_metadata_keys=[  # url is included in LLM context
                        "title",
                        "tokens",
                        "retrieve_doc",
                        "source",
                    ],
                    excluded_embed_metadata_keys=[  # title is embedded along the content
                        "url",
                        "tokens",
                        "retrieve_doc",
                        "source",
                    ],
                )
            )
    return documents


def process_source(source: str):
    config = SOURCE_CONFIGS[source]

    input_file = config["input_file"]
    db_name = config["db_name"]
    db_path = f"data/{db_name}"

    print(f"Processing source: {source}")

    documents: list[Document] = create_docs(input_file)
    print(f"Created {len(documents)} documents")

    # Check if the folder exists and delete it
    if os.path.exists(db_path):
        print(f"Existing database found at {db_path}. Deleting...")
        shutil.rmtree(db_path)
        print(f"Deleted existing database at {db_path}")

    # Create Chroma client and collection
    chroma_client = chromadb.PersistentClient(path=f"data/{db_name}")
    chroma_collection = chroma_client.create_collection(db_name)

    # Create vector store and storage context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Save document dictionary
    document_dict: dict[str, Document] = {doc.doc_id: doc for doc in documents}
    document_dict_file = f"data/{db_name}/document_dict_{source}.pkl"
    with open(document_dict_file, "wb") as f:
        pickle.dump(document_dict, f)
    print(f"Saved document dictionary to {document_dict_file}")

    # Load nodes with context
    with open("data/all_sources_contextual_nodes.pkl", "rb") as f:
        nodes_with_context: list[TextNode] = pickle.load(f)

    print(f"Loaded {len(nodes_with_context)} nodes with context")
    # pdb.set_trace()
    # exit()

    # Create vector store index
    index = VectorStoreIndex(
        nodes=nodes_with_context,
        # embed_model=OpenAIEmbedding(model="text-embedding-3-large", mode="similarity"),
        embed_model=CohereEmbedding(
            api_key=os.environ["COHERE_API_KEY"],
            model_name="embed-english-v3.0",
            input_type="search_document",
        ),
        show_progress=True,
        use_async=True,
        storage_context=storage_context,
    )
    llm = OpenAI(
        temperature=1,
        model="gpt-4o-mini",
        # model="gpt-4o",
        max_tokens=5000,
        max_retries=3,
    )
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query("How to fine-tune an llm?")
    print(response)
    for src in response.source_nodes:
        print("Node ID\t", src.node_id)
        print("Title\t", src.metadata["title"])
        print("Text\t", src.text)
        print("Score\t", src.score)
        print("-_" * 20)

    # # Create vector store index
    # index = VectorStoreIndex.from_documents(
    #     documents,
    #     # embed_model=OpenAIEmbedding(model="text-embedding-3-large", mode="similarity"),
    #     embed_model=CohereEmbedding(
    #         api_key=os.environ["COHERE_API_KEY"],
    #         model_name="embed-english-v3.0",
    #         input_type="search_document",
    #     ),
    #     transformations=[SentenceSplitter(chunk_size=800, chunk_overlap=0)],
    #     show_progress=True,
    #     use_async=True,
    #     storage_context=storage_context,
    # )
    print(f"Created vector store index for {source}")


def main(sources: list[str]):
    for source in sources:
        if source in SOURCE_CONFIGS:
            process_source(source)
        else:
            print(f"Unknown source: {source}. Skipping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process sources and create vector stores."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        choices=SOURCE_CONFIGS.keys(),
        help="Specify one or more sources to process",
    )
    args = parser.parse_args()

    main(args.sources)
