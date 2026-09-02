import os
import asyncio
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from mcp import Client

from app.mcp_server import mcp
from app.search_chunks import ask_banking_assistant, search_products


load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.1,
)


# ============================================================
# EXISTING RAG TOOLS
# ============================================================

@tool
def compare_products(product_a: str, product_b: str) -> str:
    """
    Compare two banking products using only the
    internal banking knowledge base.
    """

    question = f"""
Compare:

{product_a}

and

{product_b}

Use ONLY information retrieved from the banking
knowledge base.

Include:
- similarities
- differences
- mechanism
- risks
- relevant use cases

If information is unavailable, explicitly say so.

Do not add general banking knowledge.
"""

    return ask_banking_assistant(question)


@tool
def check_compliance(question: str) -> str:
    """
    Check whether a banking compliance claim,
    eligibility condition, restriction, suitability
    condition or target-market statement is supported
    by the internal documentation.
    """

    query = f"""
STRICT COMPLIANCE CHECK

Claim/question:

{question}

Use ONLY retrieved banking documentation.

Classify it as exactly one of:

SUPPORTED
UNSUPPORTED
INSUFFICIENT_INFORMATION

SUPPORTED means the documentation explicitly
supports the claim.

UNSUPPORTED means the documentation explicitly
contradicts the claim.

INSUFFICIENT_INFORMATION means there is not
enough evidence.

Do not infer missing compliance rules.
Do not use general banking knowledge.

Then provide:
- classification
- explanation
- evidence
"""

    return ask_banking_assistant(query)


# ============================================================
# MCP CLIENT
# ============================================================

async def call_mcp_tool(
    tool_name: str,
    arguments: dict | None = None,
):
    """
    Connect to our MCP server and execute one tool.
    """

    async with Client(mcp) as client:

        result = await client.call_tool(
            tool_name,
            arguments or {},
        )

        # Prefer structured MCP output when available.
        if result.structured_content is not None:
            return result.structured_content

        # Fall back to text output.
        if result.content:
            first = result.content[0]

            if hasattr(first, "text"):
                return first.text

        return None


# ============================================================
# MCP → LANGCHAIN TOOLS
# ============================================================

@tool
def mcp_list_products() -> str:
    """
    Retrieve the list of products from the internal
    PostgreSQL product database through MCP.
    """

    result = asyncio.run(
        call_mcp_tool("list_products")
    )

    return str(result)


@tool
def mcp_get_product(product_name: str) -> str:
    """
    Retrieve structured information about one banking
    product from PostgreSQL through MCP.
    """

    result = asyncio.run(
        call_mcp_tool(
            "get_product",
            {
                "product_name": product_name,
            },
        )
    )

    return str(result)


@tool
def mcp_get_compliance_rules(
    product_name: str,
) -> str:
    """
    Retrieve structured compliance rules for a banking
    product from PostgreSQL through MCP.
    """

    result = asyncio.run(
        call_mcp_tool(
            "get_compliance_rules",
            {
                "product_name": product_name,
            },
        )
    )

    return str(result)


# ============================================================
# AGENT
# ============================================================
@tool
def mcp_get_exchange_rate(
    base_currency: str,
    target_currency: str,
) -> str:
    """
    Retrieve the latest available exchange rate
    between two currencies through MCP using
    an external FX API.
    """

    result = asyncio.run(
        call_mcp_tool(
            "get_exchange_rate",
            {
                "base_currency": base_currency,
                "target_currency": target_currency,
            },
        )
    )

    return str(result)
SYSTEM_PROMPT = """
You are a banking product research agent.

You have two different data layers.

DOCUMENT KNOWLEDGE BASE
Use:
- search_products
- compare_products
- check_compliance

These tools use document retrieval through
LlamaIndex and pgvector.

STRUCTURED INTERNAL DATABASE
Use:
- mcp_list_products
- mcp_get_product
- mcp_get_compliance_rules

These tools access structured PostgreSQL data
through Model Context Protocol (MCP).

RULES:

Use search_products for general banking product
research from documents.

Use compare_products when comparing two products.

Use check_compliance for document-grounded
compliance questions.

Use MCP tools when structured product or
compliance database information is requested.

EXTERNAL MARKET DATA

Use mcp_get_exchange_rate when the user asks for
the latest available exchange rate between currencies.

This tool accesses an external FX API through MCP.

Never invent or estimate an exchange rate.

Never invent:
- banking products
- rates
- product conditions
- eligibility requirements
- compliance requirements
- regulations
- costs
- risk limits

If a tool returns no information, explicitly say
that the requested information is unavailable.

Never convert missing information into an
assumption.

Do not expose hidden reasoning.

Final answers must be based only on evidence
returned by the tools.
"""


agent = create_agent(
    model=llm,
    tools=[
        search_products,
        compare_products,
        check_compliance,
        mcp_list_products,
        mcp_get_product,
        mcp_get_compliance_rules,
        mcp_get_exchange_rate,
    ],
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# PUBLIC AGENT FUNCTION
# ============================================================

def ask_agent(
    question: str,
    show_trace: bool = False,
) -> str:

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    if show_trace:

        print("\nAGENT TRACE")

        for message in result["messages"]:

            print("\n" + "-" * 80)
            print("TYPE:", type(message).__name__)
            print("CONTENT:", message.content)

            if hasattr(message, "tool_calls"):
                print(
                    "TOOL CALLS:",
                    message.tool_calls,
                )

    return result["messages"][-1].content


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "List the banking products stored in our "
        "structured internal product database."
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(
        ask_agent(
            question,
            show_trace=True,
        )
    )
if __name__ == "__main__":

    question = (
        "What is the latest available EUR to USD exchange rate?"
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(
        ask_agent(
            question,
            show_trace=True,
        )
    )