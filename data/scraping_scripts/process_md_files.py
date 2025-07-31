"""
Markdown Document Processor for Documentation Sources

This script processes Markdown (.md) and MDX (.mdx) files from various documentation sources
(such as Hugging Face Transformers, PEFT, TRL, LlamaIndex, and OpenAI Cookbook) and converts
them into a standardized JSONL format for further processing or indexing.

Key features:
1. Configurable for multiple documentation sources
2. Extracts titles, generates URLs, and counts tokens for each document
3. Supports inclusion/exclusion of specific directories and root files
4. Removes copyright headers from content
5. Generates a unique ID for each document
6. Determines if a whole document should be retrieved based on token count
7. Handles special cases like openai-cookbook repo by adding .ipynb extensions
8. Processes multiple sources in a single run

Usage:
    python process_md_files.py <source1> <source2> ...

Where <source1>, <source2>, etc. are one or more of the predefined sources in SOURCE_CONFIGS
(e.g., 'transformers', 'llama_index', 'openai_cookbooks').

The script processes all Markdown files in the specified input directories (and their subdirectories),
applies the configured filters, and saves the results in JSONL files. Each line in the output
files represents a single document with metadata and content.

To add or modify sources, update the SOURCE_CONFIGS dictionary at the top of the script.
"""

import argparse
import json
import logging
import os
import re
import uuid
from typing import Dict, List

import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for different sources
SOURCE_CONFIGS = {
    "transformers": {
        "base_url": "https://huggingface.co/docs/transformers/",
        "input_directory": "data/transformers_md_files",
        "output_file": "data/transformers_data.jsonl",
        "source_name": "transformers",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": ["internal", "main_classes"],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "peft": {
        "base_url": "https://huggingface.co/docs/peft/",
        "input_directory": "data/peft_md_files",
        "output_file": "data/peft_data.jsonl",
        "source_name": "peft",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "trl": {
        "base_url": "https://huggingface.co/docs/trl/",
        "input_directory": "data/trl_md_files",
        "output_file": "data/trl_data.jsonl",
        "source_name": "trl",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "llama_index": {
        "base_url": "https://docs.llamaindex.ai/en/stable/",
        "input_directory": "data/llama_index_md_files",
        "output_file": "data/llama_index_data.jsonl",
        "source_name": "llama_index",
        "use_include_list": True,
        "included_dirs": [
            "getting_started",
            "understanding",
            "use_cases",
            "examples",
            "module_guides",
            "optimizing",
        ],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": ["index.md"],
        "url_extension": "",
    },
    "openai_cookbooks": {
        "base_url": "https://github.com/openai/openai-cookbook/blob/main/examples/",
        "input_directory": "data/openai-cookbook_md_files",
        "output_file": "data/openai_cookbooks_data.jsonl",
        "source_name": "openai_cookbooks",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": ".ipynb",
    },
    "langchain": {
        "base_url": "https://python.langchain.com/docs/",
        "input_directory": "data/langchain_md_files",
        "output_file": "data/langchain_data.jsonl",
        "source_name": "langchain",
        "use_include_list": True,
        "included_dirs": ["how_to", "versions", "turorials", "integrations"],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": ["security.md", "concepts.mdx", "introduction.mdx"],
        "url_extension": "",
    },
    "tai_blog": {
        "base_url": "",
        "input_directory": "",
        "output_file": "data/tai_blog_data.jsonl",
        "source_name": "tai_blog",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "8-hour_primer": {
        "base_url": "",
        "input_directory": "data/8-hour_primer",  # Path to the directory that contains the Markdown files
        "output_file": "data/8-hour_primer_data.jsonl",  # 8-hour Generative AI Primer
        "source_name": "8-hour_primer",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "llm_developer": {
        "base_url": "",
        "input_directory": "data/llm_developer",  # Path to the directory that contains the Markdown files
        "output_file": "data/llm_developer_data.jsonl",  # From Beginner to Advanced LLM Developer
        "source_name": "llm_developer",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "python_primer": {
        "base_url": "",
        "input_directory": "data/python_primer",  # Path to the directory that contains the Markdown files
        "output_file": "data/python_primer_data.jsonl",  # From Beginner to Advanced LLM Developer
        "source_name": "python_primer",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
}


def extract_title(content: str):
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()

    lines = content.split("\n")
    for line in lines:
        if line.strip():
            return line.strip()

    return None


def generate_url(file_path: str, config: Dict) -> str:
    """
    Return an empty string if base_url is empty;
    otherwise return the constructed URL as before.
    """
    if not config["base_url"]:
        return ""

    path_without_extension = os.path.splitext(file_path)[0]
    path_with_forward_slashes = path_without_extension.replace("\\", "/")
    return config["base_url"] + path_with_forward_slashes + config["url_extension"]


def should_include_file(file_path: str, config: Dict) -> bool:
    if os.path.dirname(file_path) == "":
        if config["use_include_list"]:
            return os.path.basename(file_path) in config["included_root_files"]
        else:
            return os.path.basename(file_path) not in config["excluded_root_files"]

    if config["use_include_list"]:
        return any(file_path.startswith(dir) for dir in config["included_dirs"])
    else:
        return not any(file_path.startswith(dir) for dir in config["excluded_dirs"])


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def remove_copyright_header(content: str) -> str:
    header_pattern = re.compile(r"<!--Copyright.*?-->\s*", re.DOTALL)
    cleaned_content = header_pattern.sub("", content, count=1)
    return cleaned_content.strip()


def process_md_files(directory: str, config: Dict) -> List[Dict]:
    jsonl_data = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md") or file.endswith(".mdx"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)

                if should_include_file(relative_path, config):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    title = extract_title(content)
                    token_count = num_tokens_from_string(content, "cl100k_base")

                    # Skip very small or extremely large files
                    if token_count < 100 or token_count > 200_000:
                        logger.info(
                            f"Skipping {relative_path} due to token count {token_count}"
                        )
                        continue

                    cleaned_content = remove_copyright_header(content)

                    json_object = {
                        "tokens": token_count,
                        "doc_id": str(uuid.uuid4()),
                        "name": (title if title else file),
                        "url": generate_url(relative_path, config),
                        "retrieve_doc": (token_count <= 8000),
                        "source": config["source_name"],
                        "content": cleaned_content,
                    }

                    jsonl_data.append(json_object)

    return jsonl_data


def save_jsonl(data: List[Dict], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def combine_all_sources(sources: List[str]) -> None:
    """
    Combine JSONL files from multiple sources, preserving existing sources not being processed.
    
    For example, if sources = ['transformers'], this will:
    1. Load data from transformers_data.jsonl
    2. Load data from all other source JSONL files that exist (course files, etc.)
    3. Combine them all into all_sources_data.jsonl
    """
    all_data = []
    output_file = "data/all_sources_data.jsonl"
    
    # Track which sources we're processing
    processed_sources = set()
    
    # First, add data from sources we're explicitly processing
    for source in sources:
        if source not in SOURCE_CONFIGS:
            logger.error(f"Unknown source '{source}'. Skipping.")
            continue
            
        processed_sources.add(source)
        input_file = SOURCE_CONFIGS[source]["output_file"]
        logger.info(f"Processing updated source: {source} from {input_file}")
        
        try:
            source_data = []
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    source_data.append(json.loads(line))
            
            logger.info(f"Added {len(source_data)} documents from {source}")
            all_data.extend(source_data)
        except Exception as e:
            logger.error(f"Error loading {input_file}: {e}")
    
    # Now add data from all other sources not being processed
    for source_name, config in SOURCE_CONFIGS.items():
        # Skip sources we already processed
        if source_name in processed_sources:
            continue
            
        # Try to load the individual source file
        source_file = config["output_file"]
        if os.path.exists(source_file):
            logger.info(f"Preserving existing source: {source_name} from {source_file}")
            try:
                source_data = []
                with open(source_file, "r", encoding="utf-8") as f:
                    for line in f:
                        source_data.append(json.loads(line))
                
                logger.info(f"Preserved {len(source_data)} documents from {source_name}")
                all_data.extend(source_data)
            except Exception as e:
                logger.error(f"Error loading {source_file}: {e}")
    
    logger.info(f"Total documents combined: {len(all_data)}")
    save_jsonl(all_data, output_file)
    logger.info(f"Combined data saved to {output_file}")


def process_source(source: str) -> None:
    if source not in SOURCE_CONFIGS:
        logger.error(f"Unknown source '{source}'. Skipping.")
        return

    config = SOURCE_CONFIGS[source]
    logger.info(f"\n\nProcessing source: {source}")
    jsonl_data = process_md_files(config["input_directory"], config)
    save_jsonl(jsonl_data, config["output_file"])
    logger.info(
        f"Processed {len(jsonl_data)} files and saved to {config['output_file']}"
    )


def main(sources: List[str]) -> None:
    for source in sources:
        process_source(source)

    if len(sources) > 1:
        combine_all_sources(sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Markdown files from specified sources."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        choices=SOURCE_CONFIGS.keys(),
        help="Specify one or more sources to process",
    )
    args = parser.parse_args()

    main(args.sources)
