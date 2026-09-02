from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import ask_agent


app = FastAPI(
    title="Banking Product Research Assistant",
    description=(
        "Agentic banking research service using "
        "LangChain, LlamaIndex and pgvector."
    ),
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3
    )


class AnswerResponse(BaseModel):
    answer: str


@app.get("/")
def health_check():

    return {
        "status": "ok",
        "service":
            "Banking Product Research Assistant",
    }


@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask(
    request: QuestionRequest,
):

    try:

        answer = ask_agent(
            request.question
        )

        return AnswerResponse(
            answer=answer
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process banking "
                "research request."
            ),
        ) from exc