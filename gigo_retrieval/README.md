# GIGO Retrieval & Question Answering

**Garbage In, Garbage Out** - A tutorial project demonstrating state-of-the-art retrieval and question-answering for RAG systems over technical documents.

## 🎯 Project Overview

The Retrieval & Question Answering system transforms user questions into accurate, context-rich answers by intelligently searching through a multi-vector knowledge base. This is the query side of any intelligent RAG (Retrieval-Augmented Generation) system.

**The Core Problem**: Simple keyword search or single-vector retrieval fails on complex technical questions. Users ask multi-part questions, need specific data from tables, and require references to diagrams—but traditional retrieval returns generic, context-poor results.

**Our Solution**: A multi-stage retrieval pipeline that decomposes complex questions, uses hybrid search strategies, extracts structured references, and generates comprehensive answers with proper citations.

### Why This Matters

**Without this pipeline:**
- Question: "How do I troubleshoot brake failure on Model 1055?" → Returns generic "brake" pages with no specific procedure
- Question: "What's the oil capacity for Model 1055?" → Returns page 45 but can't extract the exact value from the table
- Answer quality: 60-65% accuracy, missing critical details

**With this pipeline:**
- Question: "How do I troubleshoot brake failure on Model 1055?" → Returns "Model 1055 Brake Troubleshooting Procedure (Page 236): 1) Check hydraulic pressure... See Figure 236-2 for valve location. WARNING: Release pressure before service."
- Question: "What's the oil capacity for Model 1055?" → Returns "Model 1055 Front Axle requires 5.2 liters, Rear Axle requires 7.8 liters. Use SAE 80W-90 oil. See table on page 45."
- Answer quality: 85-90% accuracy with proper citations and references

## 🏗️ Architecture

The retrieval pipeline consists of 4 sequential stages:

![Chat Retrieval System](docs/assets/Chat%20Retrieval%20System.png)

### Stage 1: Query Decomposition

**Purpose**: Break complex questions into focused sub-questions

**The Challenge**: Users ask multi-part questions like "How do I troubleshoot brake failure on Model 1055 and what's the oil capacity?" This needs to be split into:
- "How do I troubleshoot brake failure on Model 1055?"
- "What is the oil capacity for Model 1055?"

**The Solution**: Vision-capable LLMs (Claude Sonnet 3.7 or GPT-4o) analyze the original question and decompose it into structured sub-questions with:
- Original question preservation
- Sub-question mapping with section numbers
- Matched chapters for context

**Output**: Structured `QueryDecompositionResponse` with multiple focused sub-questions

**Impact**: Each sub-question can be searched independently, improving precision and recall.

### Stage 2: Hybrid Multi-Vector Retrieval

**Purpose**: Retrieve relevant content using multiple complementary search strategies

**Why Multiple Strategies?** Different questions need different approaches:
- "What's the hydraulic capacity?" → Exact term matching (sparse)
- "Why won't my brakes work?" → Semantic understanding (dense)
- "Detailed Model 1055 brake troubleshooting" → Both + precision (hybrid)

**Available Retrieval Strategies**:

1. **Hybrid Retriever** (Recommended)
   - Combines dense (semantic) + sparse (keyword) + ColBERT (token-level)
   - Uses prefetch for wide net, then ColBERT for precise re-ranking
   - Best for: General questions requiring both semantic and exact matching

2. **ColBERT Retriever**
   - Token-level late interaction model
   - Fine-grained matching for precise answers
   - Best for: Specific technical queries requiring exact terminology

3. **Matrioska Retriever**
   - Uses OpenAI embeddings (small + large)
   - Multi-resolution matching with Matryoshka embeddings
   - Best for: Questions requiring high precision

4. **Fusion Retriever**
   - Combines multiple strategies with weighted fusion
   - Maximum coverage across different query types
   - Best for: Complex questions with multiple aspects

**How It Works**:
1. Each sub-question is embedded using the selected strategy
2. Parallel search across all vector types in Qdrant
3. Results are scored and ranked
4. Top-k results returned with rich metadata

**Impact**: Single strategy: 60-65% accuracy → Hybrid multi-vector: 85-90% accuracy

### Stage 3: Reference Extraction

**Purpose**: Extract structured references to tables and figures from retrieved content

**The Challenge**: Retrieved pages contain tables and figures, but we need to:
- Identify which tables/figures are relevant
- Extract their identifiers and page numbers
- Correlate with actual files on disk
- Deduplicate across multiple sub-questions

**The Solution**: Extractor pattern with specialized extractors for different data sources:

- **ContentElementsExtractor**: Extracts from `content_elements` (tables and figures)
- **FlattenedTablesExtractor**: Extracts from `flattened_tables`
- **TableMetadataExtractor**: Extracts from `table_metadata`
- **ContentSummaryExtractor**: Extracts from `content_summary.figures`
- **WithinPageRelationsExtractor**: Extracts related figures from `within_page_relations`

**Processing Pipeline**:
1. All extractors run on each retrieved result
2. References are correlated with actual files on disk
3. Duplicates are removed based on element_id/page_number or label/page_number

**Output**: Structured references with file paths for tables (PNG, HTML) and figures (PNG)

**Impact**: Answers can include "See Table 45-2" or "Refer to Figure 236-1" with actual file references.

### Stage 4: Answer Generation

**Purpose**: Synthesize retrieved content into comprehensive, cited answers

**The Challenge**: Raw retrieval results are fragmented. Users need:
- Coherent answers that address all parts of the question
- Proper citations to sources
- References to tables and figures
- Safety warnings and important notes

**The Solution**: Vision-capable LLMs synthesize retrieved points into structured answers:
- Addresses each sub-question clearly
- Cites sources (page numbers, sections)
- References tables and figures appropriately
- Preserves technical language and specificity
- Includes warnings and important notes

**Output**: Complete answer with:
- `answer`: Full text response
- `references`: Structured table and figure references with file paths

**Impact**: Transforms fragmented retrieval into professional, comprehensive answers.

## 🚀 Features

- ✅ **Query Decomposition**: Intelligent multi-part question splitting
- ✅ **Hybrid Retrieval**: Multiple search strategies (Hybrid, ColBERT, Matrioska, Fusion)
- ✅ **Multi-Vector Search**: Dense, sparse, and token-level embeddings
- ✅ **Reference Extraction**: Automatic table and figure identification
- ✅ **File Correlation**: Links references to actual files on disk
- ✅ **Answer Synthesis**: LLM-powered answer generation with citations
- ✅ **Modular Architecture**: Strategy pattern for easy extension
- ✅ **Type Safety**: Pydantic models throughout
- ✅ **Comprehensive Logging**: Track every step of retrieval

## 📋 Prerequisites

- **Python**: 3.11 or higher
- **UV**: Fast Python package installer ([installation guide](https://github.com/astral-sh/uv))
- **Qdrant Vector Database**: Running instance (local or remote)
- **LLM API Key**: OpenAI, Anthropic, or other LiteLLM-supported provider
- **Memory**: Minimum 4GB RAM recommended
- **Indexed Documents**: Documents must be indexed using `gigo_ingestion` first

## 🛠️ Local Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd garbage-in-garbage-out/gigo_retrieval
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

Or sync all dependencies:

```bash
uv sync
```

### 4. Ensure Qdrant is Running

Make sure Qdrant is running (from ingestion setup or standalone):

```bash
# From repository root
docker-compose up -d
```

Verify it's running:

```bash
curl http://localhost:6333/collections
```

### 5. Configure Environment Variables

Create a `.env` file in the `gigo_retrieval` directory:

```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional, leave empty for local development
QDRANT_COLLECTION_NAME=hybrid_collection

# LLM Configuration (choose one)
# Option 1: OpenAI
OPENAI_API_KEY=your-openai-api-key-here

# Option 2: Anthropic (Claude) - Recommended
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Option 3: Other LiteLLM supported providers
# See: https://docs.litellm.ai/docs/providers

# LLM Model (optional, defaults to Claude Sonnet 3.7)
LLM_MODEL_NAME=anthropic/claude-3-7-sonnet-20250219
```

### 6. Verify Installation

```bash
python -c "from src.services import RetrievalService; print('✓ Installation successful')"
```

## 📁 Folder Structure

```
gigo_retrieval/
├── src/
│   ├── config.py                  # Configuration management
│   ├── db/                        # Vector database integration
│   │   ├── manager.py            # Qdrant connection manager
│   │   └── collection.py         # Collection utilities
│   ├── llm.py                     # LLM client (LiteLLM)
│   ├── prompts/                   # LLM prompt templates
│   │   ├── query_decomposition.py  # Query decomposition prompts
│   │   └── answer_question.py      # Answer generation prompts
│   ├── references/                # Reference extraction system
│   │   ├── models.py             # Pydantic models (TableReference, FigureReference)
│   │   ├── extractor.py          # Main extraction + correlation + deduplication
│   │   └── extractors/            # Extractor pattern implementations
│   │       ├── base.py           # Abstract base extractor
│   │       ├── content_elements.py
│   │       ├── flattened_tables.py
│   │       ├── table_metadata.py
│   │       ├── content_summary.py
│   │       └── within_page_relations.py
│   ├── schemas.py                 # Pydantic schemas
│   ├── services.py                # Main services (QueryDecomposition, Retrieval)
│   └── strategies/                # Retrieval strategies
│       ├── base.py               # Base retriever interface
│       ├── hybrid.py             # Hybrid retriever (dense + sparse + ColBERT)
│       ├── colbert.py             # ColBERT-only retriever
│       ├── matrioska.py           # Matrioska retriever (OpenAI embeddings)
│       └── fusion.py              # Fusion retriever (multiple strategies)
├── test/                          # Test suite
├── docs/                          # Documentation
├── main.py                        # Main entry point
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
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

The system uses [LiteLLM](https://docs.litellm.ai/) for unified LLM access. Supports 100+ LLM providers.

**Anthropic (Claude)** - Recommended:
```bash
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL_NAME=anthropic/claude-3-7-sonnet-20250219
```

**OpenAI**:
```bash
OPENAI_API_KEY=sk-...
LLM_MODEL_NAME=openai/gpt-4o
```

**Azure OpenAI**:
```bash
AZURE_API_KEY=...
AZURE_API_BASE=https://...
AZURE_API_VERSION=2023-05-15
```

### Retrieval Strategy Configuration

Customize retrieval parameters:

```python
# Hybrid retriever with custom limits
results = retriever.retrieve(
    query="your question",
    limit=10,                    # Final results
    prefetch_limit=30,           # Initial candidates
    score_threshold=5,           # Minimum score
    collection_name="my_collection"
)
```

### Reference Extraction Configuration

Customize scratch path for file correlation:

```python
from src.references.extractor import correlate_references_with_files

references = correlate_references_with_files(
    references,
    scratch_path="path/to/processed/documents"
)
```

## 🔍 How the Retrieval Pipeline Works

### Example Query Flow

```
User Question: "How do I troubleshoot brake failure on Model 1055 and what's the oil capacity?"

Stage 1: Query Decomposition
├─ Sub-question 1: "How do I troubleshoot brake failure on Model 1055?"
└─ Sub-question 2: "What is the oil capacity for Model 1055?"

Stage 2: Hybrid Retrieval (for each sub-question)
├─ Dense Embedding → Finds: "brake malfunction", "brake issues", "hydraulic problems"
├─ Sparse Embedding → Finds: Exact "Model 1055", "brake"
├─ ColBERT → Re-ranks: Prioritizes "brake failure" over "brake maintenance"
└─ Results: 
   ├─ Page 236: "Model 1055 Brake Troubleshooting Procedure" (score: 0.92)
   └─ Page 45: "Model 1055 Specifications Table" (score: 0.88)

Stage 3: Reference Extraction
├─ Table References: 
│  └─ Table 45-2: "Oil Capacity Specifications" (page 45)
└─ Figure References:
   └─ Figure 236-2: "Brake Valve Location Diagram" (page 236)

Stage 4: Answer Generation
└─ Answer: 
   "To troubleshoot brake failure on Model 1055:
    1) Check hydraulic pressure (should be 2000-2500 PSI)
    2) Inspect brake valve (see Figure 236-2 for location)
    3) ...
    
    WARNING: Release hydraulic pressure before service.
    
    Oil Capacity for Model 1055:
    - Front Axle: 5.2 liters
    - Rear Axle: 7.8 liters
    - Oil Type: SAE 80W-90
    (See Table 45-2 on page 45)
    
    References:
    - Table 45-2: Oil Capacity Specifications
    - Figure 236-2: Brake Valve Location"
```
## 📝 License

See [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **Qdrant**: High-performance vector database
- **LiteLLM**: Unified LLM API
- **FastEmbed**: Fast embedding generation
- **ColBERT**: Token-level retrieval innovation
- **Pydantic**: Type-safe data validation

---

**Happy Retrieving! 🚀**

For questions or issues, please open a GitHub issue.

