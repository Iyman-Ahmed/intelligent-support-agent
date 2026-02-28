"""
Agent 1 — KB Direct Agent.

TF-IDF search over the Amazon product dataset.
Returns a formatted answer and a confidence score [0.0, 1.0].
Makes ZERO Gemini API calls — pure Python, no external dependencies.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

CONFIDENCE_THRESHOLD = 0.35          # minimum cosine similarity to return an answer
DATA_FILE = Path(__file__).parent.parent.parent / "data" / "amazon_products.json"

# Stopwords to ignore during tokenisation
_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "of", "for", "to",
    "and", "or", "but", "with", "that", "this", "are", "was", "be", "has",
    "its", "by", "do", "my", "me", "we", "you", "your", "they", "their",
    "can", "will", "how", "what", "which", "does", "has", "have", "get",
    "use", "used", "good", "best", "like", "also", "more", "than", "from",
}


# ---------------------------------------------------------------------------
# TF-IDF helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 2 and w not in _STOPWORDS]


def _product_to_text(product: dict) -> str:
    """Concatenate all searchable fields into a single string for indexing."""
    parts = [
        product.get("title", ""),
        product.get("brand", ""),
        product.get("description", ""),
        product.get("category", ""),
        " ".join(product.get("features", [])),
        " ".join(product.get("tags", [])),
        " ".join(
            qa["question"] + " " + qa["answer"]
            for qa in product.get("qa_pairs", [])
        ),
        " ".join(r["body"] for r in product.get("reviews", [])),
    ]
    return " ".join(filter(None, parts))


# ---------------------------------------------------------------------------
# KBDirectAgent
# ---------------------------------------------------------------------------

class KBDirectAgent:
    """
    Agent 1: answers product questions directly from the dataset.

    Uses TF-IDF + cosine similarity to find the best-matching product.
    If confidence >= CONFIDENCE_THRESHOLD, returns a formatted answer.
    Otherwise returns ("", 0.0) to signal hand-off to Agent 2.
    """

    def __init__(self, products: list[dict]):
        self.products = products
        self._idf: dict[str, float] = {}
        self._index: dict[str, dict[str, float]] = {}   # asin → normalised tfidf vec
        self._build_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str) -> tuple[str, float]:
        """
        Returns (formatted_answer, confidence_score).
        Returns ("", 0.0) when confidence < CONFIDENCE_THRESHOLD.
        """
        scores = self._score_query(question)
        if not scores:
            return ("", 0.0)

        top_asin, top_score = scores[0]
        if top_score < CONFIDENCE_THRESHOLD:
            return ("", 0.0)

        product = self._get_product(top_asin)
        if not product:
            return ("", 0.0)

        return self._format_answer(product, question), round(top_score, 3)

    def get_top_products(self, question: str, n: int = 5) -> list[dict]:
        """Return top-n matching products for use as context in Agent 2."""
        scores = self._score_query(question)
        top_asins = {asin for asin, _ in scores[:n]}
        # Preserve score order
        ordered = []
        for asin, _ in scores[:n]:
            for p in self.products:
                if p["asin"] == asin and p not in ordered:
                    ordered.append(p)
        return ordered

    # ------------------------------------------------------------------
    # TF-IDF index
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        n_docs = len(self.products)
        raw_tf: dict[str, dict[str, int]] = {}
        df: dict[str, int] = {}

        # First pass — term frequency per document + document frequency
        for product in self.products:
            asin = product["asin"]
            terms = _tokenize(_product_to_text(product))
            tf: dict[str, int] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            raw_tf[asin] = tf
            for unique_term in set(terms):
                df[unique_term] = df.get(unique_term, 0) + 1

        # IDF: log-smoothed
        self._idf = {
            term: math.log((n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }

        # TF-IDF vectors, L2-normalised
        for product in self.products:
            asin = product["asin"]
            tf = raw_tf[asin]
            doc_len = sum(tf.values()) or 1
            vec = {
                t: (count / doc_len) * self._idf.get(t, 1.0)
                for t, count in tf.items()
            }
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._index[asin] = {t: v / norm for t, v in vec.items()}

    def _score_query(self, query: str) -> list[tuple[str, float]]:
        """Return list of (asin, cosine_similarity) sorted descending."""
        terms = _tokenize(query)
        if not terms:
            return []

        doc_len = len(terms)
        q_tf: dict[str, int] = {}
        for t in terms:
            q_tf[t] = q_tf.get(t, 0) + 1

        q_vec = {
            t: (count / doc_len) * self._idf.get(t, 1.0)
            for t, count in q_tf.items()
        }
        norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {t: v / norm for t, v in q_vec.items()}

        scores: list[tuple[str, float]] = []
        for asin, doc_vec in self._index.items():
            score = sum(q_vec.get(t, 0.0) * w for t, w in doc_vec.items())
            scores.append((asin, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ------------------------------------------------------------------
    # Answer formatting
    # ------------------------------------------------------------------

    def _format_answer(self, product: dict, question: str) -> str:
        q_lower = question.lower()
        q_words = set(w for w in q_lower.split() if len(w) > 3)

        # Check for matching Q&A pair first
        for qa in product.get("qa_pairs", []):
            qa_words = set(_tokenize(qa["question"]))
            if len(q_words & qa_words) >= 2:
                return (
                    f"**{product['title']}**\n"
                    f"💰 ${product['price']}  |  ⭐ {product['rating']}/5 "
                    f"({product['review_count']:,} reviews)\n\n"
                    f"**Q:** {qa['question']}\n"
                    f"**A:** {qa['answer']}\n\n"
                    f"**Key features:** {' • '.join(product['features'][:3])}"
                )

        # Highlight relevant reviews if question is about opinions/experience
        opinion_words = {"good", "worth", "recommend", "review", "opinion", "experience"}
        if q_words & opinion_words and product.get("reviews"):
            review = product["reviews"][0]
            stars = "⭐" * int(review["rating"])
            return (
                f"**{product['title']}**\n"
                f"💰 ${product['price']}  |  ⭐ {product['rating']}/5 "
                f"({product['review_count']:,} reviews)\n\n"
                f"**Top customer review:** {stars}\n"
                f"*\"{review['title']}\"* — {review['body'][:280]}\n\n"
                f"**Features:** {' • '.join(product['features'][:4])}"
            )

        # Default: structured product summary
        features_md = "\n".join(f"  • {f}" for f in product["features"][:5])
        return (
            f"**{product['title']}**\n"
            f"💰 ${product['price']}  |  ⭐ {product['rating']}/5 "
            f"({product['review_count']:,} reviews)  |  Brand: {product['brand']}\n\n"
            f"**Features:**\n{features_md}\n\n"
            f"_{product.get('description', '')}_"
        )

    def _get_product(self, asin: str) -> Optional[dict]:
        for p in self.products:
            if p["asin"] == asin:
                return p
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_kb_agent(data_file: Path = DATA_FILE) -> KBDirectAgent:
    """Load the dataset and return a ready-to-use KBDirectAgent."""
    with open(data_file, "r", encoding="utf-8") as f:
        products = json.load(f)
    return KBDirectAgent(products)
