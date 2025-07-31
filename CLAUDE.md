# AI Tutor App Instructions for Claude

## Project Overview
This is an AI tutor application that uses RAG (Retrieval Augmented Generation) to provide accurate responses about AI concepts by searching through multiple documentation sources. The application has a Gradio UI and uses ChromaDB for vector storage.

## Key Repositories and URLs
- Main code: https://github.com/towardsai/ai-tutor-app
- Live demo: https://huggingface.co/spaces/towardsai-tutors/ai-tutor-chatbot
- Vector database: https://huggingface.co/datasets/towardsai-tutors/ai-tutor-vector-db
- Private JSONL repo: https://huggingface.co/datasets/towardsai-tutors/ai-tutor-data

## Architecture Overview
- Frontend: Gradio-based UI in `scripts/main.py`
- Retrieval: Custom retriever using ChromaDB vector stores
- Embedding: Cohere embeddings for vector search
- LLM: OpenAI models (GPT-4o, etc.) for context addition and responses
- Storage: Individual JSONL files per source + combined file for retrieval

## Data Update Workflows

### 1. Adding a New Course
```bash
python data/scraping_scripts/add_course_workflow.py --course [COURSE_NAME]
```
- This requires the course to be configured in `process_md_files.py` under `SOURCE_CONFIGS`
- The workflow will pause for manual URL addition after processing markdown files
- Only new content will have context added by default (efficient)
- Use `--process-all-context` if you need to regenerate context for all documents
- Both database and data files are uploaded to HuggingFace by default
- Use `--skip-data-upload` if you don't want to upload data files

### 2. Updating Documentation from GitHub
```bash
python data/scraping_scripts/update_docs_workflow.py
```
- Updates all supported documentation sources (or specify specific ones with `--sources`)
- Downloads fresh documentation from GitHub repositories
- Only new content will have context added by default (efficient)
- Use `--process-all-context` if you need to regenerate context for all documents
- Both database and data files are uploaded to HuggingFace by default
- Use `--skip-data-upload` if you don't want to upload data files

### 3. Data File Management
```bash
# Upload both JSONL and PKL files to private HuggingFace repository
python data/scraping_scripts/upload_data_to_hf.py
```

## Data Flow and File Relationships

### Document Processing Pipeline
1. **Markdown Files** → `process_md_files.py` → **Individual JSONL files** (e.g., `transformers_data.jsonl`)
2. Individual JSONL files → `combine_all_sources()` → `all_sources_data.jsonl`
3. `all_sources_data.jsonl` → `add_context_to_nodes.py` → `all_sources_contextual_nodes.pkl`
4. `all_sources_contextual_nodes.pkl` → `create_vector_stores.py` → ChromaDB vector stores

### Important Files and Their Purpose
- `all_sources_data.jsonl` - Combined raw document data without context
- Source-specific JSONL files (e.g., `transformers_data.jsonl`) - Raw data for individual sources
- `all_sources_contextual_nodes.pkl` - Processed nodes with added context
- `chroma-db-all_sources` - Vector database directory containing embeddings
- `document_dict_all_sources.pkl` - Dictionary mapping document IDs to full documents

## Configuration Details

### Adding a New Course Source
1. Update `SOURCE_CONFIGS` in `process_md_files.py`:
```python
"new_course": {
    "base_url": "",
    "input_directory": "data/new_course",
    "output_file": "data/new_course_data.jsonl",
    "source_name": "new_course",
    "use_include_list": False,
    "included_dirs": [],
    "excluded_dirs": [],
    "excluded_root_files": [],
    "included_root_files": [],
    "url_extension": "",
},
```

2. Update UI configurations in:
   - `setup.py`: Add to `AVAILABLE_SOURCES` and `AVAILABLE_SOURCES_UI`
   - `main.py`: Add mapping in `source_mapping` dictionary

## Deployment and Publishing

### GitHub Actions Workflow
The application is automatically deployed to HuggingFace Spaces when changes are pushed to the main branch (excluding documentation and scraping scripts).

### Manual Deployment
```bash
git push --force https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/towardsai-tutors/ai-tutor-chatbot main:main
```

## Development Environment Setup

### Required Environment Variables
- `OPENAI_API_KEY` - For LLM processing
- `COHERE_API_KEY` - For embeddings
- `HF_TOKEN` - For HuggingFace uploads
- `GITHUB_TOKEN` - For accessing documentation via the GitHub API

### Running the Application Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the Gradio UI
python scripts/main.py
```

## Important Notes

1. When adding new courses, make sure to:
   - Place markdown files exported from Notion in the appropriate directory
   - Add URLs manually from the live course platform 
   - Example URL format: `https://academy.towardsai.net/courses/take/python-for-genai/multimedia/62515980-course-structure`
   - Configure the course in `process_md_files.py`
   - Verify it appears in the UI after deployment

2. For updating documentation:
   - The GitHub API is used to fetch the latest documentation
   - The workflow handles updating existing sources without affecting course data

3. For efficient context addition:
   - Only new content gets processed by default
   - Old nodes for updated sources are removed from the PKL file
   - This ensures no duplicate content in the vector database

## Technical Details for Debugging

### Node Removal Logic
- When adding context, the workflow now removes existing nodes for sources being updated
- This prevents duplication of content in the vector database
- The source of each node is extracted from either `node.source_node.metadata` or `node.metadata`

### Performance Considerations
- Context addition is the most time-consuming step (uses OpenAI API)
- The new default behavior only processes new content
- For large updates, consider running in batches