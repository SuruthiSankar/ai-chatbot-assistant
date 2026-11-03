"""AI-Powered Chatbot Assistant.

A hybrid chatbot that combines:
  1. Automated FAQ handling  - TF-IDF + cosine similarity, written from scratch.
  2. LLM responses           - OpenAI Chat Completions API via the standard library.
  3. Prompt engineering      - role prompt, grounding rules and FAQ context injection.
  4. Conversation memory     - a sliding window of recent turns.

No third-party packages are required (only Python 3.9+).

Usage:
    export OPENAI_API_KEY="sk-..."      # optional: without it, FAQ-only mode is used
    python chatbot.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class Settings:
    api_key: str = ""
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.3
    max_tokens: int = 400
    timeout: int = 30
    max_retries: int = 3
    max_turns: int = 6  # how many user/assistant pairs to remember
    faq_path: Path = Path(__file__).with_name("faq.json")
    faq_direct_threshold: float = 0.60  # >= this: answer straight from the FAQ
    faq_context_threshold: float = 0.15  # >= this: pass FAQ entry to the LLM as context
    faq_fallback_threshold: float = 0.25  # used when the LLM is unavailable

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        )


# --------------------------------------------------------------------------- #
# NLP preprocessing
# --------------------------------------------------------------------------- #

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with about as into from up
    down out over under again further is are was were be been being am do does did doing
    have has had having i me my we our you your he him his she her it its they them their
    this that these those what which who whom how when where why can could should would
    will shall may might must please tell give want need let there here so than too very
    just also not no nor only own same some such any all both each few more most other""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ing", "ed", "es", "s")


def stem(word: str) -> str:
    """Very light suffix stripping so 'refunds' and 'refund' match."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> List[str]:
    """Lowercase -> split into word tokens -> drop stop words -> stem."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]


# --------------------------------------------------------------------------- #
# TF-IDF retrieval (FAQ handling)
# --------------------------------------------------------------------------- #


class TfidfIndex:
    """Tiny TF-IDF index with cosine similarity, implemented from scratch."""

    def __init__(self, documents: List[str]) -> None:
        self._docs = [tokenize(d) for d in documents]
        n_docs = len(self._docs)
        doc_freq: Counter = Counter()
        for tokens in self._docs:
            doc_freq.update(set(tokens))
        # Smoothed IDF: never zero, never negative.
        self._idf: Dict[str, float] = {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0 for term, df in doc_freq.items()
        }
        self._vectors = [self._vectorize(tokens) for tokens in self._docs]

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        total = len(tokens)
        vec = {
            term: (count / total) * self._idf[term]
            for term, count in counts.items()
            if term in self._idf
        }
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0:
            return {}
        return {term: v / norm for term, v in vec.items()}

    def query(self, text: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Return [(document_index, cosine_score)] sorted best-first."""
        q_vec = self._vectorize(tokenize(text))
        if not q_vec:
            return []
        scored: List[Tuple[int, float]] = []
        for idx, d_vec in enumerate(self._vectors):
            small, big = (q_vec, d_vec) if len(q_vec) < len(d_vec) else (d_vec, q_vec)
            score = sum(w * big.get(term, 0.0) for term, w in small.items())
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


@dataclass
class FAQEntry:
    question: str
    answer: str


class FAQStore:
    """Loads FAQ entries from JSON and finds the closest ones to a user query."""

    def __init__(self, entries: List[FAQEntry]) -> None:
        self.entries = entries
        # The question is repeated so it weighs more than the answer text.
        docs = [f"{e.question} {e.question} {e.answer}" for e in entries]
        self._index = TfidfIndex(docs) if entries else None

    @classmethod
    def load(cls, path: Path) -> "FAQStore":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            entries = [FAQEntry(item["question"], item["answer"]) for item in raw]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[warning] Could not load FAQ file ({exc}). Continuing without FAQs.")
            entries = []
        return cls(entries)

    def search(self, text: str, top_k: int = 3) -> List[Tuple[FAQEntry, float]]:
        if self._index is None:
            return []
        return [(self.entries[i], s) for i, s in self._index.query(text, top_k)]


# --------------------------------------------------------------------------- #
# Prompt engineering
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a friendly, concise customer-support assistant.\n"
    "Rules:\n"
    "1. Answer in 2-4 short sentences unless the user asks for detail.\n"
    "2. If a 'Knowledge base' section is provided, treat it as the source of truth and "
    "do not contradict it.\n"
    "3. If you do not know the answer, say so honestly and suggest contacting human support. "
    "Never invent policies, prices or links.\n"
    "4. Stay polite, neutral and on topic. Ask a clarifying question if the request is vague."
)


def build_context_message(matches: List[Tuple[FAQEntry, float]], threshold: float) -> Optional[str]:
    relevant = [(e, s) for e, s in matches if s >= threshold]
    if not relevant:
        return None
    lines = ["Knowledge base (use if relevant):"]
    for entry, _ in relevant:
        lines.append(f"Q: {entry.question}\nA: {entry.answer}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# OpenAI client (standard library only)
# --------------------------------------------------------------------------- #


class ChatError(Exception):
    """Raised when the language model could not produce an answer."""


class LLMClient(Protocol):
    def complete(self, messages: List[Dict[str, str]]) -> str: ...


class OpenAIClient:
    """Minimal Chat Completions client using urllib, with retry + backoff."""

    RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(self, messages: List[Dict[str, str]]) -> str:
        s = self.settings
        payload = json.dumps(
            {
                "model": s.model,
                "messages": messages,
                "temperature": s.temperature,
                "max_tokens": s.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{s.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {s.api_key}",
            },
            method="POST",
        )

        last_error = "unknown error"
        for attempt in range(1, s.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=s.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code not in self.RETRYABLE:
                    raise ChatError(f"API rejected the request ({last_error}).") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = f"network problem: {exc}"
            except (KeyError, IndexError, ValueError) as exc:
                raise ChatError("Unexpected response format from the API.") from exc

            if attempt < s.max_retries:
                time.sleep(2 ** (attempt - 1))  # 1s, 2s, ...
        raise ChatError(f"API unavailable after {s.max_retries} attempts ({last_error}).")


# --------------------------------------------------------------------------- #
# Chatbot core
# --------------------------------------------------------------------------- #

FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find a reliable answer to that. "
    "Please rephrase your question or contact human support."
)


@dataclass
class Reply:
    text: str
    source: str  # "faq", "llm", "faq-fallback" or "fallback"
    confidence: float = 0.0


class Chatbot:
    def __init__(self, settings: Settings, faq: FAQStore, client: Optional[LLMClient]) -> None:
        self.settings = settings
        self.faq = faq
        self.client = client
        self.history: List[Dict[str, str]] = []

    # -- memory ------------------------------------------------------------- #
    def reset(self) -> None:
        self.history.clear()

    def _remember(self, user_text: str, bot_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": bot_text})
        limit = self.settings.max_turns * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    # -- routing ------------------------------------------------------------ #
    def reply(self, user_text: str) -> Reply:
        user_text = user_text.strip()
        if not user_text:
            return Reply("Please type a question.", "fallback")

        matches = self.faq.search(user_text)
        best_score = matches[0][1] if matches else 0.0

        # 1) High-confidence FAQ hit: answer instantly, no API cost.
        if matches and best_score >= self.settings.faq_direct_threshold:
            answer = matches[0][0].answer
            self._remember(user_text, answer)
            return Reply(answer, "faq", best_score)

        # 2) Otherwise ask the LLM, grounded with any related FAQ entries.
        if self.client is not None:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            context = build_context_message(matches, self.settings.faq_context_threshold)
            if context:
                messages.append({"role": "system", "content": context})
            messages.extend(self.history)
            messages.append({"role": "user", "content": user_text})
            try:
                answer = self.client.complete(messages)
                self._remember(user_text, answer)
                return Reply(answer, "llm", best_score)
            except ChatError as exc:
                print(f"[warning] {exc}")

        # 3) LLM missing or failed: use a weaker FAQ match if there is one.
        if matches and best_score >= self.settings.faq_fallback_threshold:
            answer = matches[0][0].answer
            self._remember(user_text, answer)
            return Reply(answer, "faq-fallback", best_score)

        return Reply(FALLBACK_MESSAGE, "fallback", best_score)


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #

HELP_TEXT = """Commands:
  /help      show this message
  /history   show the remembered conversation
  /reset     forget the conversation
  /save      save the conversation to a JSON transcript
  /quit      exit"""


def save_transcript(history: List[Dict[str, str]]) -> Path:
    path = Path(f"transcript_{datetime.now():%Y%m%d_%H%M%S}.json")
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    settings = Settings.from_env()
    faq = FAQStore.load(settings.faq_path)
    client: Optional[LLMClient] = OpenAIClient(settings) if settings.api_key else None
    bot = Chatbot(settings, faq, client)

    mode = f"LLM + FAQ ({settings.model})" if client else "FAQ-only (set OPENAI_API_KEY for LLM mode)"
    print(f"AI Chatbot Assistant - mode: {mode}")
    print("Type /help for commands.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if user_text == "/quit":
            print("Goodbye!")
            return 0
        if user_text == "/help":
            print(HELP_TEXT)
            continue
        if user_text == "/reset":
            bot.reset()
            print("Conversation cleared.")
            continue
        if user_text == "/history":
            for turn in bot.history:
                print(f"  {turn['role']}: {turn['content']}")
            if not bot.history:
                print("  (empty)")
            continue
        if user_text == "/save":
            print(f"Saved to {save_transcript(bot.history)}")
            continue

        result = bot.reply(user_text)
        print(f"Bot [{result.source}]: {result.text}\n")


if __name__ == "__main__":
    sys.exit(main())
