"""
Amazon product Q&A agents.

Two-agent system:
  Agent 1 — KBDirectAgent : TF-IDF keyword search, zero Gemini calls.
  Agent 2 — ReasoningAgent: Gemini flash-lite for comparative / inferential queries.

All LLM dependencies are lazy-imported so this package is safe to import
without google-genai installed.
"""
