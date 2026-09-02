import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from langchain_core.tools import tool


# --------------------------------------------------
# 0. UTF-8 output
# --------------------------------------------------

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. NVIDIA embedding model
# --------------------------------------------------

embed_model = NVIDIAEmbedding(
    model="nvidia/nemotron-3-embed-1b",
    api_key=os.getenv("NVIDIA_API_KEY"),
)


# --------------------------------------------------
# 3. NVIDIA LLM client
# --------------------------------------------------

llm_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)


# --------------------------------------------------
# 4. PostgreSQL + pgvector
# --------------------------------------------------

vector_store = PGVectorStore.from_params(
    database=os.getenv("DB_NAME"),
    host=os.getenv("DB_HOST"),
    password=os.getenv("DB_PASSWORD"),
    port=int(os.getenv("DB_PORT", "5432")),
    user=os.getenv("DB_USER"),
    table_name="banking_chunks_llama",
    embed_dim=2048,
)


# --------------------------------------------------
# 5. Load existing LlamaIndex vector index
# --------------------------------------------------

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)


# --------------------------------------------------
# 6. Retriever
# --------------------------------------------------

retriever = index.as_retriever(
    similarity_top_k=3
)


# --------------------------------------------------
# 7. Main RAG function
# --------------------------------------------------

def ask_banking_assistant(question: str) -> str:

    results = retriever.retrieve(question)

    context = "\n\n".join(
        result.node.get_content()
        for result in results
    )

    system_prompt = """
You are a banking product research assistant.

Answer questions using ONLY the supplied banking document context.

Rules:
- Give only the final answer.
- Do not show your reasoning or thinking process.
- Do not describe how you analyzed the context.
- Do not invent information.
- Do not invent banking products, rates, conditions, or recommendations.
- If the context is insufficient, explicitly say so.
- Keep the answer concise and practical.
"""

    user_prompt = f"""
QUESTION:
{question}

BANKING DOCUMENT CONTEXT:
{context}

Answer the question using only this context.
"""

    response = llm_client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=600,
        extra_body={
            "reasoning_budget": 0
        },
    )

    return response.choices[0].message.content


# --------------------------------------------------
# 8. LangChain tool
# --------------------------------------------------

@tool
def search_products(question: str) -> str:
    """
    Search the banking product knowledge base for information
    relevant to the user's question.

    Use this tool when the user asks about banking products,
    treasury products, foreign exchange risk, hedging,
    or product characteristics.
    """

    return ask_banking_assistant(question)


# --------------------------------------------------
# 9. Direct test
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "How can I protect my company "
        "against foreign exchange risk?"
    )

    print("\nQUESTION:")
    print(question)

    answer = ask_banking_assistant(question)

    print("\n" + "=" * 80)
    print("FINAL RAG ANSWER")
    print("=" * 80)
    print(answer)