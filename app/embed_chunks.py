import os
from pathlib import Path

from dotenv import load_dotenv

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import StorageContext, VectorStoreIndex


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Configure NVIDIA Nemotron embeddings
# --------------------------------------------------

embed_model = NVIDIAEmbedding(
    model="nvidia/nemotron-3-embed-1b",
    api_key=os.getenv("NVIDIA_API_KEY"),
)


# --------------------------------------------------
# 3. Configure PostgreSQL + pgvector
# --------------------------------------------------

vector_store = PGVectorStore.from_params(
    database=os.getenv("DB_NAME"),
    host=os.getenv("DB_HOST"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    table_name="banking_chunks_llama",
    embed_dim=2048,
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# --------------------------------------------------
# 4. Find PDF
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

pdf_path = (
    BASE_DIR
    / "data"
    / "Treasury_termekismerteto_EN.pdf"
)


# --------------------------------------------------
# 5. PDF -> LlamaIndex Documents
# --------------------------------------------------

reader = PDFReader()

documents = reader.load_data(
    file=pdf_path
)

print(f"Documents loaded: {len(documents)}")


# --------------------------------------------------
# 6. Documents -> Chunks
# --------------------------------------------------

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

chunks = splitter.get_nodes_from_documents(
    documents
)

print(f"Chunks created: {len(chunks)}")


# --------------------------------------------------
# 7. Chunks -> Embeddings -> PostgreSQL/pgvector
# --------------------------------------------------

index = VectorStoreIndex(
    chunks,
    storage_context=storage_context,
    embed_model=embed_model,
)

print("LlamaIndex ingestion completed successfully!")