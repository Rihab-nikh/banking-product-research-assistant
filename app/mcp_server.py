import os

import psycopg
from dotenv import load_dotenv
from mcp.server import MCPServer

load_dotenv()

mcp = MCPServer(
    "Banking Internal Product Database"
)


def get_connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode=os.getenv("DB_SSLMODE", "prefer"),
    )

@mcp.tool()
def list_products() -> list[dict]:
    """
    List banking products stored in the internal
    PostgreSQL product database.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    name,
                    category,
                    description,
                    source
                FROM banking_products
                WHERE active = TRUE
                ORDER BY name
                """
            )

            rows = cur.fetchall()

    return [
        {
            "name": row[0],
            "category": row[1],
            "description": row[2],
            "source": row[3],
        }
        for row in rows
    ]


@mcp.tool()
def get_product(product_name: str) -> dict:
    """
    Retrieve one banking product from the internal
    PostgreSQL product database.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    name,
                    category,
                    description,
                    source
                FROM banking_products
                WHERE LOWER(name) = LOWER(%s)
                  AND active = TRUE
                LIMIT 1
                """,
                (product_name,),
            )

            row = cur.fetchone()

    if row is None:
        return {
            "found": False,
            "product": product_name,
        }

    return {
        "found": True,
        "name": row[0],
        "category": row[1],
        "description": row[2],
        "source": row[3],
    }


@mcp.tool()
def get_compliance_rules(
    product_name: str,
) -> list[dict]:
    """
    Retrieve documented compliance rules for a banking
    product from the internal PostgreSQL database.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    product_name,
                    rule_type,
                    rule_text,
                    client_type,
                    risk_profile,
                    source,
                    effective_date
                FROM compliance_rules
                WHERE LOWER(product_name) = LOWER(%s)
                  AND active = TRUE
                ORDER BY id
                """,
                (product_name,),
            )

            rows = cur.fetchall()

    return [
        {
            "product_name": row[0],
            "rule_type": row[1],
            "rule_text": row[2],
            "client_type": row[3],
            "risk_profile": row[4],
            "source": row[5],
            "effective_date": (
                str(row[6]) if row[6] else None
            ),
        }
        for row in rows
    ]

@mcp.tool()
def get_exchange_rate(
    base_currency: str,
    target_currency: str,
) -> dict:
    """
    Retrieve the latest available exchange rate
    from the external Frankfurter FX API.
    """

    import httpx

    base = base_currency.upper().strip()
    target = target_currency.upper().strip()

    response = httpx.get(
        f"https://api.frankfurter.dev/v2/rate/{base}/{target}",
        timeout=10.0,
    )

    response.raise_for_status()

    data = response.json()

    return {
        "base_currency": data["base"],
        "target_currency": data["quote"],
        "rate": data["rate"],
        "date": data["date"],
        "source": "Frankfurter FX API",
    }
if __name__ == "__main__":
    mcp.run()