from __future__ import annotations

from langchain_groq import ChatGroq

from app.config import Settings


def build_llm(settings: Settings, temperature: float = 0.1) -> ChatGroq:
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and set your key."
        )
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=temperature,
    )


def llm_text(llm: ChatGroq, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()
