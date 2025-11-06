# GIGO Ingestion & Indexing Pipeline

**Garbage In, Garbage Out** - A tutorial project demonstrating state-of-the-art document ingestion and indexing for RAG systems over technical documents.

## 🎯 Project Overview

The Ingestion & Indexing Pipeline transforms unstructured PDF documents into a richly structured, multi-modal knowledge base optimized for semantic search and AI consumption. This is the foundation of any intelligent RAG (Retrieval-Augmented Generation) system.

**The Core Problem**: Traditional document processing treats PDFs as flat text blobs, losing crucial context about structure, relationships, and visual information. This leads to poor retrieval quality and the dreaded "GIGO" (Garbage In, Garbage Out) problem.

**Our Solution**: A multi-stage pipeline that extracts not just text, but **meaning, structure, context, and relationships** using state-of-the-art OCR, vision-capable AI, and multi-vector embeddings.

### Why This Matters

**Without this pipeline:**
- Search for "brake valve location" → Returns "See page 235"
- Search for "Model 1055 oil capacity" → Finds the table but can't extract the value
- Retrieval accuracy: 60-65%

**With this pipeline:**
- Search for "brake valve location" → Returns "The manual release valve for Model 1055 front axle is located on the front-left side of the axle housing. See Figure 235-1 for exact location. Related troubleshooting procedures are on page 236. WARNING: Release hydraulic pressure before service."
- Search for "Model 1055 oil capacity" → Returns "Model 1055 Front Axle requires 5.2 liters, Rear Axle requires 7.8 liters. Use SAE 80W-90 oil."
- Retrieval accuracy: 85-90%

## 🏗️ Architecture

The pipeline consists of 4 sequential stages, each building upon the previous:

![Architecture Diagram](docs/assets/Data%20North%20Complex%20RAG%20Agent.png)

### Stage 1: OCR & Document Extraction

**Purpose**: Transform PDF from pixels to structured elements

**Technologies**:
- **Docling OCR** with EasyOCR backend for high-accuracy text recognition
- **PyMuPDF** for PDF parsing and 2x resolution page image extraction
- **TableFormer** (accurate mode) for precise table detection and extraction
- **PictureItem detection** for figure/diagram extraction

**Output**: From 1 unstructured PDF page → 4-6 structured, typed elements (tables, figures, text blocks)

**Impact**: This is the difference between "page 45 exists" and "page 45 contains 2 tables, 3 diagrams, and 5 text blocks—each with identity and type."

### Stage 2: Context-Aware Metadata Extraction

**Purpose**: Extract deep semantic understanding using vision-capable AI

**Why Context Matters**: Technical manuals don't exist in isolated pages. A procedure on page 45 might:
- Start on page 44
- Continue on page 46
- Reference a table on page 44
- Relate to a diagram on page 43

**Processing Strategy**: Vision-capable LLMs (Claude Sonnet 3.7 or GPT-4o) analyze **three consecutive page images** (N-1, N, N+1) simultaneously.

**Extracted Metadata**:
- **Document-Level**: Title, manufacturer, models covered, document type
- **Section Hierarchy**: Section numbers, titles, hierarchical structure
- **Page Visual Description**: Layout, visual emphasis, diagram types
- **Content Elements**: Type, ID, title, detailed summary, keywords, entities, safety warnings
- **Relationships**: Within-page and cross-page references

**Impact**: Transforms isolated pages into interconnected knowledge with semantic understanding.

### Stage 3: Table Flattening

**Purpose**: Make tables searchable by AI while preserving human readability

**The Challenge**: Tables are excellent for humans but problematic for vector embeddings. Spatial relationships (same row/column) are lost in pure text.

**The Solution**: Transform tables into dense, searchable text that makes all relationships explicit.

**Example Transformation**:

| Model | Front Axle | Rear Axle | Oil Type |
|-------|-----------|-----------|----------|
| 1055  | 5.2 L     | 7.8 L     | SAE 80W-90 |

Becomes:

```
Model 1055 Front Axle capacity: 5.2 liters
Model 1055 Rear Axle capacity: 7.8 liters
Model 1055 Oil Type: SAE 80W-90
```

**Impact**: AI can now understand "How much oil does Model 1055 need?" by finding exact semantic matches.

### Stage 4: Multi-Vector Embedding Generation

**Purpose**: Create mathematical representations optimized for different search strategies

**Why Multiple Embeddings?** Different questions need different approaches:
- "What's the hydraulic capacity?" → Exact term matching
- "Why won't my brakes work?" → Semantic understanding
- "Detailed Model 1055 brake troubleshooting" → Both + precision

**5 Complementary Embeddings Per Page**:

1. **Dense Embedding (Semantic)**
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Dimensions: 384
   - Purpose: Understands meaning, not just words
   - Example: "brake failure" → matches "brake malfunction", "stuck brakes"

2. **Sparse Embedding (Keyword)**
   - Model: `Qdrant/bm42-all-minilm-l6-v2` (BM42)
   - Type: Sparse indices/values
   - Purpose: Exact term matching with context
   - Example: "Model 1055" → ONLY exact model matches

3. **ColBERT Embedding (Token-Level)**
   - Model: `colbert-ir/colbertv2.0`
   - Dimensions: 128 per token (multi-vector)
   - Purpose: Fine-grained token-level matching for re-ranking
   - Innovation: Compares query tokens to document tokens individually

4. **Small OpenAI Embedding (Fast)**
   - Model: `text-embedding-3-small`
   - Dimensions: 128 (Matryoshka embedding)
   - Purpose: Quick initial filtering

5. **Large OpenAI Embedding (Precise)**
   - Model: `text-embedding-3-large`
   - Dimensions: 1024 (Matryoshka embedding)
   - Purpose: High-precision final ranking

**Strategy**: Cast both a wide net (dense) and a precise spear (sparse), then use a magnifying glass (ColBERT) to examine catches at multiple zoom levels (small/large).

**Impact**: Single embedding: 60-65% accuracy → Multi-vector hybrid: 85-90% accuracy

## 🚀 Features

- ✅ **High-Accuracy OCR**: Docling with EasyOCR backend
- ✅ **Table Extraction**: TableFormer in accurate mode
- ✅ **Vision-AI Metadata**: Claude Sonnet 3.7 / GPT-4o
- ✅ **Context-Aware Processing**: 3-page sliding window
- ✅ **Table Flattening**: AI-powered table-to-text transformation
- ✅ **Multi-Vector Embeddings**: 5 complementary embedding types
- ✅ **Qdrant Integration**: Production-ready vector database
- ✅ **Modular Architecture**: Easy to extend and customize
- ✅ **Comprehensive Logging**: Track every step of the pipeline
- ✅ **Batch Processing**: Efficient processing of large documents

## 📋 Prerequisites

- **Python**: 3.9 or higher
- **UV**: Fast Python package installer ([installation guide](https://github.com/astral-sh/uv))
- **Docker & Docker Compose**: For running Qdrant vector database
- **OpenAI API Key**: For GPT-4o and text-embedding models (or use Claude via LiteLLM)
- **Memory**: Minimum 8GB RAM recommended
- **Disk Space**: ~2-5GB per document (varies by document size)

## 🛠️ Local Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd garbage-in-garbage-out/gigo_ingestion
```

### 2. Install UV (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### 3. Install Dependencies

Using UV:

```bash
# Install project dependencies
uv pip install -e .
```

Or with development dependencies:

```bash
uv pip install -e ".[dev]"
```

Alternatively, sync all dependencies:

```bash
uv sync
```

### 4. Start Qdrant Vector Database

```bash
# From the repository root
docker-compose up -d
```

This starts Qdrant on:
- HTTP: `http://localhost:6333`
- gRPC: `localhost:6334`

Verify it's running:

```bash
curl http://localhost:6333/collections
```

### 5. Configure Environment Variables

Create a `.env` file in the `gigo_ingestion` directory:

```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional, leave empty for local development
QDRANT_COLLECTION_NAME=hybrid_collection

# LLM Configuration (choose one)
# Option 1: OpenAI
OPENAI_API_KEY=your-openai-api-key-here

# Option 2: Anthropic (Claude)
# ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Option 3: Other LiteLLM supported providers
# See: https://docs.litellm.ai/docs/providers
```

### 6. Verify Installation

```bash
python -c "from src.parser import DocumentParser; print('✓ Installation successful')"
```

## 📁 Folder Structure

```
gigo_ingestion/
├── data/                          # Data directory
│   └── processed/                 # Processed documents output
│       └── {document_name}/       # Per-document directory
│           └── page_{N}/          # Per-page directory
│               ├── page_{N}_full.png          # Full page image
│               ├── metadata_page_{N}.json     # Basic metadata
│               ├── context_metadata.json      # Rich AI-extracted metadata
│               ├── tables/        # Extracted tables
│               │   ├── table-{N}-{M}.html
│               │   ├── table-{N}-{M}.png
│               │   └── table-{N}-{M}_flattened.txt
│               ├── images/        # Extracted figures/diagrams
│               │   ├── image-{N}-{M}.png
│               │   └── image-{N}-{M}_interpreted.json
│               └── text/          # Extracted text blocks
│                   └── page_{N}_text.txt
├── src/
│   ├── algorithms/                # Search and ranking algorithms
│   ├── db/                        # Vector database integration
│   │   ├── manager.py            # Qdrant connection manager
│   │   └── collection.py         # Collection with multi-vector embeddings
│   ├── metadata_extractors/      # Metadata extraction modules
│   │   ├── page_context_extractor.py  # Context-aware page metadata
│   │   ├── table_extractor.py         # Table metadata extraction
│   │   └── text_extractor.py          # Text metadata extraction
│   ├── processors/                # Content processors
│   │   ├── table.py              # Table flattening processor
│   │   ├── text_blocks.py        # Text block processor
│   │   ├── image.py              # Image interpretation processor
│   │   └── question_mapping.py   # Question-answer mapping
│   ├── prompts/                   # LLM prompt templates
│   │   ├── context_metadata.py   # Page context prompts
│   │   ├── table_metadata.py     # Table metadata prompts
│   │   ├── table_flattening.py   # Table flattening prompts
│   │   └── image_interpretation.py  # Image interpretation prompts
│   ├── config.py                  # Configuration management
│   ├── llm.py                     # LLM client (LiteLLM)
│   ├── parser.py                  # PDF parser (Docling + PyMuPDF)
│   ├── schemas.py                 # Pydantic schemas
│   └── utils.py                   # Utility functions
├── scripts/                       # Utility scripts
├── test/                          # Test suite
├── main.py                        # Main pipeline entry point
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## 💻 Usage

### Basic Usage

Process a PDF document through the complete pipeline:

```bash
python main.py path/to/document.pdf
```

### With Vector Database Indexing

Process and index to Qdrant:

```bash
python main.py path/to/document.pdf --index
```

### Advanced Options

```bash
python main.py path/to/document.pdf \
    --output-dir data/processed \
    --collection my_documents \
    --model openai/gpt-4o \
    --index \
    --start-page 10 \
    --end-page 20
```

**Arguments**:
- `pdf_path`: Path to the PDF file (required)
- `--output-dir`: Output directory for processed files (default: `data/processed`)
- `--collection`: Qdrant collection name (default: `documents`)
- `--model`: LLM model to use (default: `openai/gpt-4o`)
  - OpenAI: `openai/gpt-4o`, `openai/gpt-4o-mini`
  - Anthropic: `anthropic/claude-3-7-sonnet`, `anthropic/claude-3-5-sonnet`
  - See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for more
- `--no-metadata`: Skip metadata extraction (faster, but less intelligent)
- `--index`: Index to vector database
- `--start-page`: Start processing from this page
- `--end-page`: End processing at this page

```

## ⚙️ Configuration

### Qdrant Configuration

Edit `src/config.py` or set environment variables:

```python
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key  # Optional
QDRANT_COLLECTION_NAME=hybrid_collection
```

### LLM Configuration

The pipeline uses [LiteLLM](https://docs.litellm.ai/) for unified LLM access. Supports 100+ LLM providers.

**OpenAI**:
```bash
OPENAI_API_KEY=sk-...
```

**Anthropic (Claude)**:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**Azure OpenAI**:
```bash
AZURE_API_KEY=...
AZURE_API_BASE=https://...
AZURE_API_VERSION=2023-05-15
```

### OCR Configuration

Edit `src/parser.py` to customize OCR settings:

```python
self.pipeline_options.do_ocr = True
self.pipeline_options.ocr_options = EasyOcrOptions()
self.pipeline_options.images_scale = 2  # 2x resolution
self.pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
```

### Embedding Models

Edit `src/db/collection.py` to customize embedding models:

```python
self.dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
self.sparse_embedding_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")
self.late_interaction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
```

## 🔍 How Multi-Vector Search Works

When you query the system:

1. **Query Embedding**: Your query is embedded using all 5 models
2. **Parallel Search**: 
   - Dense embedding finds semantically similar content
   - Sparse embedding finds exact term matches
   - ColBERT performs token-level comparison
   - Small/Large embeddings provide multi-resolution matching
3. **Hybrid Scoring**: Results are combined using weighted scores
4. **Re-ranking**: ColBERT re-ranks top candidates for precision
5. **Return**: Best matches with rich metadata and context

**Example Query Flow**:

```
Query: "How do I troubleshoot brake failure on Model 1055?"

Dense → Finds: "brake malfunction", "brake issues", "hydraulic problems"
Sparse → Finds: Exact "Model 1055", "brake"
ColBERT → Re-ranks: Prioritizes "brake failure" over "brake maintenance"
Large → Refines: Prioritizes troubleshooting over installation
Result → Page 236: "Model 1055 Brake Troubleshooting Procedure"
```

## 📝 License

See [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **Docling**: IBM's amazing document understanding library
- **Qdrant**: High-performance vector database
- **LiteLLM**: Unified LLM API
- **FastEmbed**: Fast embedding generation
- **ColBERT**: Token-level retrieval innovation

---

**Happy Ingesting! 🚀**

For questions or issues, please open a GitHub issue.
