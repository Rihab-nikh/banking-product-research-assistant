# Banking Product Research Assistant

An agentic banking research prototype that combines **RAG, structured financial data, tool calling, and external APIs** behind a FastAPI service.

The system helps users research treasury and banking products by routing each request to the appropriate data source instead of relying on a single LLM context.

## What It Does

The assistant can:

- Search banking documentation using semantic retrieval
- Explain treasury and foreign-exchange products
- Compare banking products
- Perform document-grounded compliance research
- Query a structured PostgreSQL product catalog
- Retrieve external FX reference rates
- Automatically select the appropriate tool based on the user's request

## Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/493e247b-cd01-456b-9f66-f7bdfdd9cbcf" />


The architecture deliberately separates three kinds of knowledge.

### 1. Unstructured Document Knowledge

Banking documents are processed using:

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/aa1645c0-959a-4750-973e-f4bfbc290f2a" />


The current knowledge base is built from a treasury-products document and contains embedded document chunks used for semantic retrieval.

### 2. Structured Internal Data

Structured product information is stored separately in PostgreSQL.

The agent accesses this data through **Model Context Protocol (MCP)** tools such as:

- `list_products`
- `get_product`
- `get_compliance_rules`

This separates deterministic structured queries from semantic document retrieval.

### 3. External Financial Data

The MCP layer also exposes external services.

For example:

```text
User asks for EUR/USD
        ↓
LangChain Agent
        ↓
mcp_get_exchange_rate
        ↓
MCP Server
        ↓
Frankfurter FX API
        ↓
Latest available reference rate
```

This allows internal databases and external services to be exposed to the agent through a consistent tool interface.

## Agentic Workflow

The LangChain agent decides which capability to use depending on the question.

Example:

```text
"How can a company hedge FX exposure?"
        ↓
Document RAG

"List the products in the internal database."
        ↓
PostgreSQL through MCP

"What is the latest available EUR/USD rate?"
        ↓
External FX API through MCP
```

The application therefore uses the LLM primarily for **reasoning, routing and synthesis**, while factual information is retrieved from explicit data sources.

## RAG Pipeline

The document pipeline uses:

- LlamaIndex for document ingestion and retrieval
- NVIDIA Nemotron embeddings
- PostgreSQL with pgvector for vector storage
- Semantic similarity retrieval
- Grounded answer generation

This keeps banking-document answers tied to retrieved context rather than relying solely on the model's parametric knowledge.

## MCP Integration

The project includes an MCP server exposing structured capabilities to the agent.

```text
LangChain Agent
       |
       v
   MCP Client
       |
       v
   MCP Server
     /     \
    v       v
PostgreSQL  External API
```

This reduces coupling between agent orchestration and individual integrations.

## Evaluation & Observability

The project uses **LangSmith** for tracing agent and tool interactions during development.

A small evaluation suite also checks whether the agent selects the expected tool for representative requests such as:

- product research
- product comparison
- compliance research

This currently evaluates **tool-routing behavior** rather than claiming comprehensive model or retrieval-quality evaluation.

## API

The assistant is exposed through FastAPI.

### Health check

```http
GET /
```

### Ask the banking assistant

```http
POST /ask
Content-Type: application/json
```

Example:

```json
{
  "question": "How can a company protect itself against foreign exchange risk?"
}
```

FastAPI automatically exposes interactive API documentation through Swagger UI.

## Technology Stack

**AI / Agentic Systems**

- LangChain
- LlamaIndex
- NVIDIA Nemotron
- Model Context Protocol (MCP)
- LangSmith

**Backend**

- Python
- FastAPI
- Pydantic

**Data**

- PostgreSQL
- pgvector

**Infrastructure**

- Docker

**External Data**

- Frankfurter FX API

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Rihab-nikh/banking-product-research-assistant.git
cd banking-product-research-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file locally containing the required database and API configuration.

Do **not** commit credentials to Git.

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Then open the FastAPI Swagger interface at:

```text
http://localhost:8000/docs
```

## Docker

The repository includes a Dockerfile for packaging the FastAPI application.

```bash
docker build -t banking-assistant .
```

Environment variables should be supplied at runtime rather than embedded inside the image.

## Current Scope

This repository is an engineering prototype rather than a production banking system.

In particular:

- the source banking document is historical reference material
- FX data represents reference rates rather than executable trading prices
- compliance functionality is document-grounded research, not automated regulatory approval
- financial outputs should not be interpreted as financial advice

## Why This Project?

The goal was not simply to build another chatbot.

The project explores how an AI application can combine:

```text
Unstructured knowledge  → RAG
Structured enterprise data → MCP + PostgreSQL
External information → MCP + APIs
Reasoning/orchestration → LangChain Agent
Observability → LangSmith
Serving → FastAPI
Packaging → Docker
```

This architecture is designed around a common production AI problem:

**giving an LLM controlled access to multiple trustworthy sources while keeping retrieval, structured data and external integrations separated.**
