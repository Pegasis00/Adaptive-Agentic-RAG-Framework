from __future__ import annotations


GENERATION_PROMPT = """You answer questions using only the uploaded document context.

Question: {question}
Query type: {query_type}

Context passages:
{context}

Instructions:
- Use only facts that are explicitly present in the context.
- Do not include source names, page numbers, chunk numbers, or retrieval metadata in the answer.
- If the answer is not present, say exactly: "I could not find that in the uploaded document."
- Only provide code if the user explicitly asks for code, pseudocode, a script, or implementation.
- Answer the question directly. Do not add unrelated background.

Answer:"""


GROUNDEDNESS_PROMPT = """You are a Self-RAG groundedness verifier.

Question: {question}
Context:
{context}

Proposed answer:
{answer}

Is every factual claim in the answer directly supported by the context?
If the answer explicitly says context is missing instead of inventing facts, treat it as supported.
Inline citations are not required.

Respond with exactly one token:
- supported
- unsupported"""


UTILITY_PROMPT = """You are a Self-RAG utility verifier.

Question: {question}
Query type: {query_type}

Proposed answer:
{answer}

Does the answer directly address the question using only the available context?
Reward concise, complete answers. Penalize unrelated content, unsupported claims,
generic advice, or code when the user did not ask for code.

Respond with exactly one token:
- useful
- notuseful"""


REFINE_PROMPT = """Revise the answer using the verification feedback.

Question: {question}
Query type: {query_type}
Context:
{context}

Previous answer:
{answer}

Issues to fix:
{issues}

Rules:
- Use only facts present in the context.
- Do not mention the revision process, feedback, context, chunks, pages, or sources.
- Do not add unrelated details.
- If the answer is unavailable, say exactly: "I could not find that in the uploaded document."

Answer:"""


CODE_FOCUS_PROMPT = """You answer explicit implementation requests using only the uploaded document context.

Question: {question}
Context:
{context}

Rules:
- Only write code if the user explicitly asks for code, pseudocode, a script, or implementation.
- If the user asks a factual/list/summary question, answer as plain text instead.
- Do not include source names, page numbers, chunk numbers, or retrieval metadata in the answer.
- If a required detail is absent, name the missing detail and choose a conservative default.
- Keep the answer concise and directly aligned with the question."""


RESEARCH_FOCUS_PROMPT = """You answer questions directly using only the uploaded document context.

Question: {question}
Context:
{context}

Rules:
- Answer only what the user asked.
- Use only facts explicitly present in the context. Do not infer beyond it.
- Do not include source names, page numbers, chunk numbers, or retrieval metadata in the answer.
- Do not mention "provided context", "chunk", "section", "part", or retrieval process unless the user asks.
- Do not add unrelated suggestions, limitations, experiments, reading, or general knowledge.
- For list questions, return a clean bullet list.
- For comparison questions, use a compact table if it helps.
- For yes/no questions, answer "Yes" or "No" first, then give the minimal supporting detail found in the document.
- For detailed questions, provide a structured answer with only relevant details from the document.
- If the answer is not in the uploaded document, say: "I could not find that in the uploaded document."
- Keep the answer concise and factual."""
